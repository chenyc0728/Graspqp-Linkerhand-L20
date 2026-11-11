"""验收校验 for ``manual_review.py`` 的输出（HANDOFF round-2 强制项 1–5）。

对 ``--dataset``（v3.pt）与 ``--manual``（人工审核后的 .pt）跑 5 项检查并把原始结果写进
``--out``（默认 ``logs/verify_v5_manual.json``）：

1. **结构**：``torch.load`` 两个文件，顶层 key 相同、``q`` 的 dtype/shape 相同、
   ``robot_name`` 相同、``len(metadata) == info.num_total``、``object_name`` 前缀 ⊆
   ``{ycb+, contactdb+}``、``num_per_object`` / ``num_upper_object`` 自洽、metadata 顺序是
   v3 的子序列、且**没有**任何 ``reject`` 样本混进来；
2. **逐位一致**：随机 20 条人工样本，``torch.equal`` 比对同 uid 的 v3 样本（uid 用
   ``sha256:grasp_index``，与 ``convert_to_cmap`` 的去重键一致）；
3. **q → FK 往返**：20 条样本用 numpy FK 推出 ``hand_base_link``，与原始 GraspQP
   ``root_pose`` 比（期望 rot ~1e-7 rad、trans ~1e-8 m）；顺带记录本机 pytorch_kinematics
   fork 的 ``±0.01 rad`` 偏置（只记录，不修）；
4. **真实 ``CMapDataset.py`` 装载**：``metadata`` 可迭代、``len(ds)`` 等于
   ``info.num_per_object`` 的物体数、``__getitem__`` 的 robot/shape 正确，且参考的
   allegro 521 条不受影响；
5. **dry-run 可复现**：用同一份 decisions 跑两遍 ``--dry-run``，逐张量 ``torch.equal``
   （不比 md5 —— ``torch.save`` 的 zip 容器不是逐字节可复现的，见 REPORT_v4 §6）。

用法::

    python new_grasp_filter/verify_v5_manual.py \\
        --dataset new_grasp_filter/cmap_dataset_from_graspqp_v3.pt \\
        --manual  new_grasp_filter/cmap_dataset_from_graspqp_v3_manual.pt \\
        --decisions new_grasp_filter/manual_review_decisions.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import convert_to_cmap as c2c  # noqa: E402
import grasp_io as gio  # noqa: E402
import manual_review as mr  # noqa: E402
import validate_cmap_dataset as vcd  # noqa: E402

ALLOWED_PREFIXES = {"ycb", "contactdb"}
VN = ("virtual_joint_x", "virtual_joint_y", "virtual_joint_z")
RN = ("virtual_joint_roll", "virtual_joint_pitch", "virtual_joint_yaw")


def subset_index_map(dataset_meta, subset_meta):
    """把人工样本映射回 v3 的 metadata 下标。

    ``(object_name, q)`` 在 v3 里 **不唯一**（实测 771 组重复：同一物体不同 grasp 文件里
    有逐位相同的 q），所以用**按 dataset 顺序贪心匹配**的方式定位，并报告有多少样本落在
    重复键上。输出顺序是否真的是 v3 顺序，另由「dry-run 重放结果与交付文件逐张量相同」保证。
    """
    lut: dict = {}
    for i, m in enumerate(dataset_meta):
        lut.setdefault((m[1], m[0].numpy().tobytes()), []).append(i)
    idx, cursor, dup_hits = [], 0, 0
    for m in subset_meta:
        key = (m[1], m[0].numpy().tobytes())
        cands = lut.get(key, [])
        if len(cands) > 1:
            dup_hits += 1
        pick = next((c for c in cands if c >= cursor), None)
        if pick is None:
            raise SystemExit(
                f"[error] 人工样本 {m[1]} 在 v3.pt 里找不到（逐位相同的 q）或顺序不对；"
                f"该 (object_name, q) 在 v3 中的下标为 {cands[:5]}")
        idx.append(pick)
        cursor = pick + 1
    return idx, {"n_ambiguous_q_keys": sum(1 for v in lut.values() if len(v) > 1),
                 "n_samples_on_ambiguous_key": dup_hits}


def check_structure(dataset: Path, manual: Path, decisions: Path) -> dict:
    a = torch.load(dataset, map_location="cpu", weights_only=False)
    b = torch.load(manual, map_location="cpu", weights_only=False)
    out: dict = {
        "dataset_keys": list(a.keys()), "manual_keys": list(b.keys()),
        "keys_equal": list(a.keys()) == list(b.keys()),
        "dataset_info_keys": list(a["info"].keys()),
        "manual_info_keys": list(b["info"].keys()),
    }
    ra = a["info"][mr.ROBOT_NAME]
    rb = b["info"][mr.ROBOT_NAME]
    out["manual_robot_block_keys"] = list(rb.keys())
    out["robot_name_equal"] = ra["robot_name"] == rb["robot_name"]
    out["robot_name"] = rb["robot_name"]
    ma, mb = a["metadata"], b["metadata"]
    out["n_metadata"] = [len(ma), len(mb)]
    out["num_total_matches"] = len(mb) == rb["num_total"]
    out["tuple_len_set"] = sorted({len(m) for m in mb})
    out["q_dtype_set"] = sorted({str(m[0].dtype) for m in mb})
    out["q_shape_set"] = sorted({tuple(m[0].shape) for m in mb})
    out["q_float32_22"] = (out["q_dtype_set"] == ["torch.float32"] and out["q_shape_set"] == [(22,)])
    out["q_dtype_equal_to_dataset"] = ({str(m[0].dtype) for m in ma} == {str(m[0].dtype) for m in mb})
    out["q_shape_equal_to_dataset"] = ({tuple(m[0].shape) for m in ma} == {tuple(m[0].shape) for m in mb})
    out["robot_names_equal"] = ({m[2] for m in ma} == {m[2] for m in mb})
    prefixes = sorted({m[1].split("+", 1)[0] for m in mb})
    out["prefixes"] = prefixes
    out["prefixes_allowed"] = set(prefixes) <= ALLOWED_PREFIXES
    per_object: dict = {}
    for m in mb:
        per_object[m[1]] = per_object.get(m[1], 0) + 1
    out["per_object_matches_info"] = per_object == rb["num_per_object"]
    out["num_upper_object_matches"] = rb["num_upper_object"] == (max(per_object.values()) if per_object else 0)
    out["num_per_object"] = per_object
    out["num_upper_object"] = rb["num_upper_object"]
    out["has_manual_review_block"] = "manual_review" in rb
    if "manual_review" in rb:
        mrev = rb["manual_review"]
        out["manual_review"] = {k: mrev[k] for k in
                                ("num_before", "num_after", "num_rejected", "num_unreviewed",
                                 "kept_by_prefix", "rejected_by_prefix", "decisions_file")
                                if k in mrev}

    # 把人工样本映射回 v3 下标，再用与 convert_to_cmap 相同的口径算出 uid
    specs = mr.resolve_index_specs(["ycb", "contactdb"], None)
    a_samples = mr.build_samples(ma, specs, "sha", dataset)
    uid_of = {s.index: s.uid for s in a_samples}
    idx, amb = subset_index_map(ma, mb)
    out["q_key_ambiguity"] = amb
    b_uids = [uid_of[i] for i in idx]
    out["manual_uid_examples"] = b_uids[:3]
    out["all_uids_found_in_dataset"] = True
    out["metadata_order_is_dataset_subsequence"] = idx == sorted(idx) and len(set(idx)) == len(idx)
    out["v3_indices"] = idx
    b_samples = [mr.Sample(index=i, uid=u, object_name=mb[k][1], robot_name=mb[k][2],
                           q_tensor=mb[k][0]) for k, (i, u) in enumerate(zip(idx, b_uids))]

    # no reject sample may survive; count must match the decision file
    store = mr.DecisionStore(decisions).load()
    kept_uids = {u for u, r in store.by_uid.items() if r["decision"] == mr.KEEP}
    rejected_uids = {u for u, r in store.by_uid.items() if r["decision"] == mr.REJECT}
    out["decisions_file"] = str(decisions)
    out["decisions_lines"] = len(store.records)
    out["decisions_keep"] = len(kept_uids)
    out["decisions_reject"] = len(rejected_uids)
    out["no_rejected_sample_in_output"] = not (rejected_uids & set(b_uids))
    out["output_uids_equal_keep_decisions"] = (set(b_uids) == kept_uids)

    # every output sample must exist verbatim in the dataset (full sweep, not only 20)
    ok_all = True
    for i, s in zip(idx, b_samples):
        ok_all &= bool(torch.equal(ma[i][0], s.q_tensor)) and ma[i][1] == s.object_name
    out["all_samples_bit_identical_to_dataset"] = ok_all
    out["ok"] = all(bool(out[k]) for k in (
        "keys_equal", "robot_name_equal", "num_total_matches", "q_float32_22",
        "q_dtype_equal_to_dataset", "q_shape_equal_to_dataset", "robot_names_equal",
        "prefixes_allowed", "per_object_matches_info", "num_upper_object_matches",
        "all_uids_found_in_dataset", "metadata_order_is_dataset_subsequence",
        "no_rejected_sample_in_output", "output_uids_equal_keep_decisions",
        "all_samples_bit_identical_to_dataset"))
    return out


def check_bit_identical(dataset: Path, manual: Path, n: int = 20, seed: int = 0) -> dict:
    ma = torch.load(dataset, weights_only=False)["metadata"]
    mb = torch.load(manual, weights_only=False)["metadata"]
    specs = mr.resolve_index_specs(["ycb", "contactdb"], None)
    a_samples = mr.build_samples(ma, specs, "sha", dataset)
    uid_of = {s.index: s.uid for s in a_samples}
    idx, _amb = subset_index_map(ma, mb)
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(len(mb), size=min(n, len(mb)), replace=False).tolist())
    rows = []
    for k in picks:
        q0, q1 = ma[idx[k]][0], mb[k][0]
        rows.append({"manual_index": k, "dataset_index": idx[k], "uid": uid_of[idx[k]],
                     "object_name": mb[k][1],
                     "torch_equal": bool(torch.equal(q0, q1)),
                     "max_abs_diff": float((q0 - q1).abs().max())})
    return {"n_checked": len(rows), "n_expected": min(n, len(mb)),
            "all_equal": all(r["torch_equal"] for r in rows), "samples": rows}


def check_fk(dataset: Path, manual: Path, n: int = 20, seed: int = 0) -> dict:
    """q -> repaired CMap left URDF -> hand_base_link vs the original GraspQP root_pose."""
    specs = mr.resolve_index_specs(["ycb", "contactdb"], None)
    pairs = list(c2c.iter_pairs(specs, False, True))
    ma = torch.load(dataset, weights_only=False)["metadata"]
    mb = torch.load(manual, weights_only=False)["metadata"]
    a_samples = mr.build_samples(ma, specs, "sha", dataset)
    uid_of = {s.index: s.uid for s in a_samples}
    idx, _amb = subset_index_map(ma, mb)
    b_samples = [mr.Sample(index=i, uid=uid_of[i], object_name=mb[k][1], robot_name=mb[k][2],
                           q_tensor=mb[k][0]) for k, i in enumerate(idx)]
    uid2pair = {f"{p[0]['sha256']}:{p[1]}": p for p in pairs}
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(len(b_samples), size=min(n, len(b_samples)), replace=False).tolist())

    res = {}
    for backend in ("numpy", "torch"):
        gio.set_backend(backend)
        left = gio.UrdfModel(gio.cmap_left_urdf(repaired=True))
        names = (list(left.zero_angles().keys()) if backend == "numpy"
                 else list(left.pk_chain_on("cpu").get_joint_parameter_names()))
        worst_rot = worst_tr = 0.0
        rows = []
        for k in picks:
            s = b_samples[k]
            row, gi, gpath, _, _ = uid2pair[s.uid]
            g = gio.load_grasp_file(gpath)
            R0 = gio.quat_xyzw_to_matrix(g["root_pose"][gi, 3:7])
            t0 = g["root_pose"][gi, :3] + R0 @ gio.ROOT_TO_HAND_BASE
            qv = s.q_tensor.numpy().astype(np.float64)
            full21 = gio.build_full_angles(qv[6:22][None, :])[0]
            angles = {}
            for name in names:
                if name in VN:
                    angles[name] = float(qv[VN.index(name)])
                elif name in RN:
                    angles[name] = float(qv[3 + RN.index(name)])
                elif name in gio.CHAIN_JOINT_NAMES:
                    angles[name] = float(full21[gio.CHAIN_JOINT_NAMES.index(name)])
            T = left.fk(angles)["hand_base_link"]
            rot = float(np.abs(T[:3, :3] - R0).max())
            tr = float(np.abs(T[:3, 3] - t0).max())
            worst_rot, worst_tr = max(worst_rot, rot), max(worst_tr, tr)
            rows.append({"uid": s.uid, "object_name": s.object_name, "grasp_index": int(gi),
                         "rotation_error": rot, "translation_error_m": tr})
        res["numpy" if backend == "numpy" else "pytorch_kinematics"] = {
            "n_checked": len(rows), "max_abs_rotation_error": worst_rot,
            "max_abs_translation_error_m": worst_tr,
            "rotation_within_1e-6": worst_rot < 1e-6,
            "translation_within_1e-7": worst_tr < 1e-7,
            "samples": rows,
        }
    gio.set_backend("torch")
    res["note"] = ("pytorch_kinematics of this env is a fork that adds ±1e-2 rad per revolute "
                   "joint; the resulting ~4e-2 rotation / ~4e-2 m offset is a known upstream bias, "
                   "recorded only (REPORT_v4 §5). manual_review.py therefore renders with numpy FK "
                   "by default (--fk-backend pk reproduces the biased hand).")
    res["ok"] = bool(res["numpy"]["rotation_within_1e-6"] and res["numpy"]["translation_within_1e-7"])
    return res


def check_cmapdataset(manual: Path) -> dict:
    ref_mod = vcd.load_cmap_module()
    our_mod = vcd.load_cmap_module(dataset_override=manual)
    data = torch.load(manual, weights_only=False)
    objects = sorted(data["info"][mr.ROBOT_NAME]["num_per_object"].keys())
    ours = vcd.check_dataset(our_mod, manual, [mr.ROBOT_NAME], objects, is_train=False,
                             label=f"{manual.name} (manual keep)", n_getitem=1)
    ours["n_objects_in_info"] = len(objects)
    ours["len_equals_n_objects"] = ours.get("len") == len(objects)
    ours["n_metadata_matches_info"] = ours.get("n_metadata") == data["info"][mr.ROBOT_NAME]["num_total"]
    ours["getitem_robot_ok"] = ours.get("getitem_robot") == mr.ROBOT_NAME
    ours["getitem_shapes_ok"] = (ours.get("getitem_initial_q") == [1, 27]
                                 and ours.get("getitem_object_pc") == [1, 512, 3])
    ref = vcd.check_dataset(ref_mod, gio.CMAP_ROOT / "CMapDataset_filtered/cmap_dataset.pt",
                            ["allegro"], ["ycb+lemon", "ycb+baseball", "contactdb+apple"],
                            is_train=True, label="reference cmap_dataset.pt (allegro)")
    ref["unaffected"] = (ref.get("n_metadata") == 521 and ref.get("robots_in_metadata") == ["allegro"])
    return {"manual": ours, "reference_allegro": ref,
            "ok": bool(ours["len_equals_n_objects"] and ours["n_metadata_matches_info"]
                       and ours["getitem_robot_ok"] and ours["getitem_shapes_ok"]
                       and ref["unaffected"])}


def check_dry_run(dataset: Path, decisions: Path, shipped: Path | None = None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="mr_dryrun_"))
    outs = []
    for tag in ("a", "b"):
        out = tmp / f"dry_{tag}.pt"
        cmd = [sys.executable, str(HERE / "manual_review.py"), "--dry-run",
               "--dataset", str(dataset), "--decisions", str(decisions),
               "--output", str(out), "--summary-json", str(tmp / f"sum_{tag}.json"),
               "--rejected-output", ""]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return {"ok": False, "cmd": cmd, "returncode": proc.returncode,
                    "stderr": proc.stderr[-2000:]}
        outs.append(out)
    a = torch.load(outs[0], weights_only=False)
    b = torch.load(outs[1], weights_only=False)
    same_n = len(a["metadata"]) == len(b["metadata"])
    same_names = [m[1] for m in a["metadata"]] == [m[1] for m in b["metadata"]]
    same_q = same_n and all(torch.equal(x[0], y[0]) for x, y in zip(a["metadata"], b["metadata"]))
    same_info = a["info"][mr.ROBOT_NAME]["num_per_object"] == b["info"][mr.ROBOT_NAME]["num_per_object"]
    matches_shipped = None
    if shipped is not None and Path(shipped).exists():
        c = torch.load(shipped, weights_only=False)
        matches_shipped = bool(
            len(c["metadata"]) == len(a["metadata"])
            and [m[1] for m in c["metadata"]] == [m[1] for m in a["metadata"]]
            and all(torch.equal(x[0], y[0]) for x, y in zip(c["metadata"], a["metadata"]))
            and c["info"][mr.ROBOT_NAME]["num_per_object"] == a["info"][mr.ROBOT_NAME]["num_per_object"])
    return {"ok": bool(same_n and same_names and same_q and same_info and matches_shipped is not False),
            "matches_shipped_file": matches_shipped,
            "n_metadata": [len(a["metadata"]), len(b["metadata"])],
            "same_object_names": same_names, "all_q_torch_equal": same_q,
            "same_num_per_object": same_info,
            "note": "比较张量而不是 md5：torch.save 的 zip 容器不保证逐字节可复现",
            "dry_run_outputs": [str(p) for p in outs]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(HERE / "cmap_dataset_from_graspqp_v3.pt"))
    ap.add_argument("--manual", default=str(HERE / "cmap_dataset_from_graspqp_v3_manual.pt"))
    ap.add_argument("--decisions", default=str(HERE / "manual_review_decisions.jsonl"))
    ap.add_argument("--n", type=int, default=20, help="逐位一致 / FK 往返的抽样条数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "logs" / "verify_v5_manual.json"))
    ap.add_argument("--skip-dry-run", action="store_true", help="跳过第 5 项（省时间）")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    manual = Path(args.manual).resolve()
    decisions = Path(args.decisions).resolve()
    for p in (dataset, manual, decisions):
        if not p.exists():
            raise SystemExit(f"[error] 缺文件: {p}")

    report = {"dataset": str(dataset), "manual": str(manual), "decisions": str(decisions),
              "n_requested": args.n, "seed": args.seed}
    report["check1_structure"] = check_structure(dataset, manual, decisions)
    print("[1/5] structure:", "OK" if report["check1_structure"]["ok"] else "FAIL", flush=True)
    report["check2_bit_identical"] = check_bit_identical(dataset, manual, args.n, args.seed)
    print("[2/5] bit-identical:", "OK" if report["check2_bit_identical"]["all_equal"] else "FAIL", flush=True)
    report["check3_fk_roundtrip"] = check_fk(dataset, manual, args.n, args.seed)
    print("[3/5] fk roundtrip:", "OK" if report["check3_fk_roundtrip"]["ok"] else "FAIL", flush=True)
    report["check4_cmapdataset_load"] = check_cmapdataset(manual)
    print("[4/5] CMapDataset load:", "OK" if report["check4_cmapdataset_load"]["ok"] else "FAIL", flush=True)
    if args.skip_dry_run:
        report["check5_dry_run_determinism"] = {"skipped": True}
    else:
        report["check5_dry_run_determinism"] = check_dry_run(dataset, decisions, manual)
        print("[5/5] dry-run determinism:", "OK" if report["check5_dry_run_determinism"]["ok"] else "FAIL",
              flush=True)
    report["ok"] = all(report[k].get("ok", report[k].get("all_equal", False))
                       for k in ("check1_structure", "check2_bit_identical", "check3_fk_roundtrip",
                                 "check4_cmapdataset_load", "check5_dry_run_determinism"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nALL OK = {report['ok']}\nwritten -> {out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
