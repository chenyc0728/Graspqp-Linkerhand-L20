#!/usr/bin/env python
"""P1 任务 7：``cmap_dataset_from_graspqp_v6*.pt`` 的终检（**只读**）。

复用 ``verify_v5_manual.py`` 的结构，对"重筛 + 合并"得到的新数据集做 7 项检查：

1. ``torch.load`` 成功、顶层 keys 恰好 ``{info, metadata}``、``q`` 是 ``float32 (N,22)``、
   ``robot_name`` 全是 ``linkerhand``；
2. ``object_name`` 前缀分布 ⊆ ``{ycb+, contactdb+}``，且 9 个"没有抓取"的物体
   （``object/contactdb_nograsp`` 里的 cube/cylinder/pyramid）**不出现**；
3. 与基线（默认 ``v3.pt``）的差异：总样本数、物体覆盖、每个来源 / 前缀的样本数；
4. **抽样 FK 回代**：随机 20 条（尽量同时覆盖 ycb 与 contactdb）用 numpy FK 推
   ``hand_base_link``，与抓取文件里的 ``root_pose`` 比（期望 rot ~1e-7、trans ~1e-8）；
   顺带记录本机 pytorch_kinematics fork 的 ±1e-2 rad 偏置（只记录，不修）；
5. **真 ``CMapDataset.py`` 装载**：``len(ds)`` == ``info.num_per_object`` 的物体数、
   ``__getitem__`` 的 robot/(1,27)/(1,512,3) 正确，参考 allegro 数据集不受影响；
6. **针对 v3 假阳性的检查**：从
   ``logs/manual_review_pilot/audit_feasible_recheck.json`` 里取 v5 抽查出的
   ``insufficient_contacts`` 样本，看它们在新数据集里是否还在（还在 = 重筛没修好，给警告）；
   该文件不存在时如实报告"无法做针对性检查"，**不伪造**；
7. 打印 基线 vs 新数据集 的对比表（按来源 × 物体）。

用法::

    python new_grasp_filter/verify_v6_rescreen.py \\
        --dataset new_grasp_filter/cmap_dataset_from_graspqp_v6_pilot.pt \\
        --indexes new_grasp_filter/filtered_grasps/rescreen_pilot/ycb_object_index.csv \\
                  new_grasp_filter/filtered_grasps/rescreen_pilot/ycb_object1_index.csv \\
                  new_grasp_filter/filtered_grasps/index_contactdb_<ts>.csv \\
        --prefixes ycb ycb contactdb \\
        --baseline new_grasp_filter/cmap_dataset_from_graspqp_v3.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import convert_to_cmap as c2c  # noqa: E402
import grasp_io as gio  # noqa: E402
import manual_review as mr  # noqa: E402
import validate_cmap_dataset as vcd  # noqa: E402
from verify_v5_manual import RN, VN  # noqa: E402

ALLOWED_PREFIXES = {"ycb", "contactdb"}
NO_GRASP_DIR = HERE.parent / "object" / "contactdb_nograsp"
AUDIT_JSON = HERE / "logs" / "manual_review_pilot" / "audit_feasible_recheck.json"


def objects_without_grasps() -> list:
    if not NO_GRASP_DIR.is_dir():
        return []
    return sorted(d.name for d in NO_GRASP_DIR.iterdir() if d.is_dir())


def prefix_of(name: str) -> str:
    return name.split("+", 1)[0] if "+" in name else "(无前缀)"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------- 1. 结构
def check_structure(dataset: Path) -> dict:
    payload = torch.load(dataset, map_location="cpu", weights_only=False)
    out = {"path": str(dataset)}
    out["top_keys"] = sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
    md = payload["metadata"]
    out["n_metadata"] = len(md)
    out["metadata_is_list_of_tuples"] = isinstance(md, list) and all(
        isinstance(m, tuple) and len(m) == 3 for m in md)
    out["q_dtype_set"] = sorted({str(m[0].dtype) for m in md})
    out["q_shape_set"] = sorted({tuple(m[0].shape) for m in md})
    out["robots"] = sorted({m[2] for m in md})
    info = payload["info"]
    out["info_keys"] = sorted(info.keys())
    for robot, block in info.items():
        out[robot] = {k: block[k] for k in ("robot_name", "num_total", "num_upper_object")
                      if k in block}
        out[robot]["n_per_object"] = len(block.get("num_per_object", {}))
        out[robot]["num_total_matches"] = block.get("num_total") == len(md)
        out[robot]["per_object_sum_matches"] = sum(block.get("num_per_object", {}).values()) == len(md)
    out["ok"] = bool(
        out["top_keys"] == ["info", "metadata"]
        and out["metadata_is_list_of_tuples"]
        and out["q_dtype_set"] == ["torch.float32"]
        and out["q_shape_set"] == [(22,)]
        and out["robots"] == [mr.ROBOT_NAME]
        and out[list(info)[0]]["num_total_matches"]
        and out[list(info)[0]]["per_object_sum_matches"]
    )
    return out


# ---------------------------------------------------------- 2. 前缀 / 无抓取
def check_prefixes(dataset: Path, no_grasp_objects: list) -> dict:
    md = torch.load(dataset, map_location="cpu", weights_only=False)["metadata"]
    pref = Counter(prefix_of(m[1]) for m in md)
    names = {m[1] for m in md}
    bad = sorted(p for p in pref if p not in ALLOWED_PREFIXES)
    present_no_grasp = sorted(n for n in names
                              if n.split("+", 1)[-1] in no_grasp_objects and n.startswith("contactdb+"))
    out = {"prefix_counts": dict(pref), "prefixes_bad": bad,
           "objects_present": len(names),
           "no_grasp_objects_expected": no_grasp_objects,
           "no_grasp_objects_present": present_no_grasp}
    out["ok"] = bool(not bad and not present_no_grasp)
    return out


# ------------------------------------------------------------- 3. 与基线对比
def check_diff_vs_baseline(dataset: Path, baseline: Path, samples, pairs) -> dict:
    md_new = torch.load(dataset, map_location="cpu", weights_only=False)["metadata"]
    out = {"baseline": str(baseline)}
    if not baseline.exists():
        out.update({"ok": None, "note": f"基线不存在: {baseline}"})
        return out
    base = torch.load(baseline, map_location="cpu", weights_only=False)
    md_base = base["metadata"]
    out["n_new"] = len(md_new)
    out["n_baseline"] = len(md_base)
    out["delta"] = len(md_new) - len(md_base)
    cn = Counter(m[1] for m in md_new)
    cb = Counter(m[1] for m in md_base)
    out["objects_new"] = len(cn)
    out["objects_baseline"] = len(cb)
    out["objects_only_in_new"] = sorted(set(cn) - set(cb))
    out["objects_only_in_baseline"] = sorted(set(cb) - set(cn))
    # 每个来源的样本数（新数据集）
    src = Counter()
    for s, p in zip(samples, pairs):
        src[p[0].get("source", "?")] += 1
    out["per_source_new"] = dict(src)
    # 每个来源在基线里的样本数（按 uid 交集无法直接得到，用 object_name 归属近似不给）
    out["per_object"] = {"new": dict(sorted(cn.items())),
                         "baseline": dict(sorted(cb.items()))}
    out["per_object_delta"] = {k: cn.get(k, 0) - cb.get(k, 0)
                               for k in sorted(set(cn) | set(cb))}
    out["ok"] = True
    return out


# ------------------------------------------------------------- 4. FK 回代
def check_fk(samples, pairs, n: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # 尽量同时覆盖 ycb 与 contactdb
    idx_ycb = [i for i, s in enumerate(samples) if s.object_name.startswith("ycb+")]
    idx_cdb = [i for i, s in enumerate(samples) if s.object_name.startswith("contactdb+")]
    picks = []
    half = min(n // 2, len(idx_ycb))
    picks += sorted(rng.choice(idx_ycb, size=half, replace=False).tolist()) if half else []
    rest = min(n - len(picks), len(idx_cdb))
    picks += sorted(rng.choice(idx_cdb, size=rest, replace=False).tolist()) if rest else []
    if len(picks) < n:
        pool = [i for i in range(len(samples)) if i not in picks]
        extra = min(n - len(picks), len(pool))
        if extra:
            picks += sorted(rng.choice(pool, size=extra, replace=False).tolist())
    picks = sorted(set(picks))

    res = {}
    for backend in ("numpy", "torch"):
        gio.set_backend(backend)
        left = gio.UrdfModel(gio.cmap_left_urdf(repaired=True))
        names = (list(left.zero_angles().keys()) if backend == "numpy"
                 else list(left.pk_chain_on("cpu").get_joint_parameter_names()))
        worst_rot = worst_tr = worst_rot64 = 0.0
        rows = []
        for k in picks:
            s = samples[k]
            row, gi, gpath = pairs[k][0], pairs[k][1], pairs[k][2]
            g = gio.load_grasp_file(gpath)
            R0 = gio.quat_xyzw_to_matrix(np.asarray(g["root_pose"][gi, 3:7], dtype=np.float64))
            t0 = np.asarray(g["root_pose"][gi, :3], dtype=np.float64) + R0 @ gio.ROOT_TO_HAND_BASE
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
            # float64 对照：把 rpy 换成 R0 在 float64 下反解出的精确值再跑一次 FK。
            # 若误差掉到 ~1e-9，说明 float32 存储 rpy 的量化误差（在 |pitch| 接近 90°
            # 时被 1/cos(pitch) 放大）是唯一来源，转换/FK 逻辑本身没有错。
            q64 = qv.copy()
            q64[3:6] = gio.matrix_to_cmap_rpy(R0)
            full21_64 = gio.build_full_angles(q64[6:22][None, :])[0]
            ang64 = {}
            for name in names:
                if name in VN:
                    ang64[name] = float(q64[VN.index(name)])
                elif name in RN:
                    ang64[name] = float(q64[3 + RN.index(name)])
                elif name in gio.CHAIN_JOINT_NAMES:
                    ang64[name] = float(full21_64[gio.CHAIN_JOINT_NAMES.index(name)])
            T64 = left.fk(ang64)["hand_base_link"]
            rot64 = float(np.abs(T64[:3, :3] - R0).max())
            worst_rot64 = max(worst_rot64, rot64)
            pitch_deg = float(np.degrees(np.arcsin(np.clip(R0[0, 2], -1.0, 1.0))))
            rows.append({"uid": s.uid, "object_name": s.object_name, "source": row.get("source"),
                         "grasp_index": int(gi), "rotation_error": rot,
                         "translation_error_m": tr,
                         "pitch_deg": pitch_deg,
                         "conditioning_1_over_cos_pitch": float(1.0 / max(abs(np.cos(np.radians(pitch_deg))), 1e-12)),
                         "rotation_error_float64_rpy": rot64})
        key = "numpy" if backend == "numpy" else "pytorch_kinematics"
        res[key] = {"n_checked": len(rows), "max_abs_rotation_error": worst_rot,
                    "max_abs_translation_error_m": worst_tr,
                    "max_abs_rotation_error_float64_rpy": worst_rot64,
                    "n_rotation_within_1e-6": sum(r["rotation_error"] < 1e-6 for r in rows),
                    "rotation_within_1e-6": worst_rot < 1e-6,
                    "translation_within_1e-7": worst_tr < 1e-7, "samples": rows}
    gio.set_backend("torch")
    res["note"] = ("pytorch_kinematics 在本环境是加了 ±1e-2 rad/revolute 偏置的 fork，"
                   "rot ~2e-2 是已知上游偏差，只记录不修（REPORT_v4 §5 / P0 报告第 1 点）；"
                   "导出的 q 不受影响——判据看 numpy 那一行。")
    # 判据（写在 JSON 里，便于复核）：平移 <1e-7 m；旋转 <1e-6，或 <1e-5 且把 rpy 换成
    # float64 精确值后误差 <1e-7（= 偏差完全来自 float32 存储，而不是转换/FK 逻辑）。
    nu = res["numpy"]
    explained = bool(nu["max_abs_rotation_error_float64_rpy"] < 1e-7)
    res["criteria"] = {
        "translation_lt_m": 1e-7,
        "rotation_lt_rad": 1e-6,
        "rotation_lt_relaxed_with_float64_rpy_control_lt_rad": [1e-5, 1e-7],
        "float32_rpy_storage_explains_worst_case": explained,
    }
    res["ok"] = bool(nu["translation_within_1e-7"]
                     and (nu["rotation_within_1e-6"]
                          or (nu["max_abs_rotation_error"] < 1e-5 and explained)))
    return res


# --------------------------------------------------- 5. 真 CMapDataset 装载
def check_cmapdataset(dataset: Path) -> dict:
    ref_mod = vcd.load_cmap_module()
    our_mod = vcd.load_cmap_module(dataset_override=dataset)
    data = torch.load(dataset, map_location="cpu", weights_only=False)
    objects = sorted(data["info"][mr.ROBOT_NAME]["num_per_object"].keys())
    ours = vcd.check_dataset(our_mod, dataset, [mr.ROBOT_NAME], objects, is_train=False,
                             label=f"{dataset.name} (v6 rescreen)", n_getitem=1)
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
    return {"v6": ours, "reference_allegro": ref,
            "ok": bool(ours["len_equals_n_objects"] and ours["n_metadata_matches_info"]
                       and ours["getitem_robot_ok"] and ours["getitem_shapes_ok"]
                       and ref["unaffected"])}


# ------------------------------------------- 6. 针对 v3 假阳性的定向检查
def check_v3_false_positives(dataset: Path, samples, jsonl_dir: Path) -> dict:
    out = {"audit_json": str(AUDIT_JSON)}
    if not AUDIT_JSON.exists():
        out.update({"ok": None,
                    "note": "V5 的 audit json 不在本机 -> 无法做针对性检查（不伪造结论）"})
        return out
    audit = json.loads(AUDIT_JSON.read_text())
    bad = [r for r in audit.get("rows", [])
           if (not r.get("feasible_now")) and r.get("reason") == "insufficient_contacts"]
    out["n_audited_false_positives"] = len(bad)
    md = torch.load(dataset, map_location="cpu", weights_only=False)["metadata"]
    new_uids = set()
    for s in samples:
        new_uids.add(s.uid)
    # 新数据集里每个 uid 的 q，用于二次确认（sha 相同但 grasp 内容一致时也能匹配）
    objects_in_dataset = set(m[1] for m in md)
    rows, still, gone = [], 0, 0
    for r in bad:
        obj = r["object_name"]
        prefix, name = obj.split("+", 1)
        root = HERE.parent / "object" / prefix if prefix != "ycb" else HERE.parent / "object" / "ycb"
        cands = list((root / name / "grasp_predictions").rglob(r["grasp_file"])) \
            if (root / name / "grasp_predictions").is_dir() else []
        sha = sha256(cands[0]) if cands else None
        uid = f"{sha}:{r['grasp_index']}" if sha else None
        present = bool(uid and uid in new_uids)
        # 只有"该物体在本次数据集里"时，"不在"才算真正被重筛剔除；
        # 否则（如 pilot 只筛了 apple）这条检查对该物体是空转，必须如实标注。
        object_covered = obj in objects_in_dataset
        evidence = ("kept" if present else
                    ("removed_by_rescreen" if object_covered else "vacuous_object_not_in_dataset"))
        still += present
        if object_covered and not present:
            gone += 1
        # 新筛选里它的拒绝理由（若能找到）
        new_reason = None
        jl = Path(jsonl_dir) / f"{'object_rescreen' if prefix == 'ycb' else 'object1_rescreen'}" / f"{name}.jsonl"
        if jl.exists():
            with open(jl, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("grasp_file") == r["grasp_file"] and rec.get("grasp_index") == r["grasp_index"]:
                        new_reason = rec.get("reason")
                        break
        rows.append({"object_name": obj, "grasp_file": r["grasp_file"],
                     "grasp_index": r["grasp_index"],
                     "v3_geo_closest_gap_mm": r.get("geo_closest_gap_mm"),
                     "sha256": (sha or "")[:12] or None,
                     "in_new_dataset": present, "object_in_dataset": object_covered,
                     "evidence": evidence, "new_reason": new_reason})
    out["rows"] = rows
    out["n_still_present"] = still
    out["n_removed"] = gone
    out["n_vacuous"] = len(rows) - still - gone
    out["ok"] = bool(still == 0)
    out["verdict"] = (
        f"已剔除 {gone} 条（物体在数据集里的），仍有 {still} 条留在新数据集里"
        + (f"；另有 {out['n_vacuous']} 条因该物体不在本次数据集内而无法判定（空转）"
           if out["n_vacuous"] else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default=str(HERE / "cmap_dataset_from_graspqp_v6.pt"))
    ap.add_argument("--indexes", nargs="+", required=True,
                    help="与 convert_to_cmap --inputs 完全相同的三份索引（顺序也要一致）")
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--baseline", default=str(HERE / "cmap_dataset_from_graspqp_v3.pt"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jsonl-dir", default=str(HERE / "logs"),
                    help="重筛日志目录（<jsonl-dir>/<source>/<object>.jsonl），用于查新判据")
    ap.add_argument("--out", default=str(HERE / "logs" / "verify_v6_rescreen.json"))
    args = ap.parse_args()

    if len(args.indexes) != len(args.prefixes):
        raise SystemExit("--indexes 与 --prefixes 数量必须一致")
    dataset = Path(args.dataset)
    if not dataset.exists():
        raise SystemExit(f"[error] 数据集不存在: {dataset}")

    t0 = time.time()
    specs = [(Path(p), pre) for p, pre in zip(args.indexes, args.prefixes)]
    report: dict = {"task": "P1-task7-verify-v6", "dataset": str(dataset),
                    "indexes": [str(p) for p in args.indexes], "prefixes": args.prefixes,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    # 重建 uid（与 convert_to_cmap 完全相同的重放与去重），顺便强校验 metadata 顺序
    _payload, md = mr.load_dataset(dataset)
    pairs = list(c2c.iter_pairs(specs, False, True))
    samples = mr.build_samples(md, specs, "sha", dataset)
    report["uid_replay"] = {"n_pairs": len(pairs), "n_metadata": len(md),
                            "object_names_match": [p[3] for p in pairs] == [m[1] for m in md],
                            "example_uid": samples[0].uid if samples else None}
    print(f"[uid] 重放 {len(pairs)} 条 / metadata {len(md)} 条 / 顺序一致="
          f"{report['uid_replay']['object_names_match']}", flush=True)

    steps = [
        ("1_structure", lambda: check_structure(dataset)),
        ("2_prefixes", lambda: check_prefixes(dataset, objects_without_grasps())),
        ("3_diff_vs_baseline", lambda: check_diff_vs_baseline(dataset, Path(args.baseline), samples, pairs)),
        ("4_fk_roundtrip", lambda: check_fk(samples, pairs, n=args.n, seed=args.seed)),
        ("5_cmapdataset_load", lambda: check_cmapdataset(dataset)),
        ("6_v3_false_positives", lambda: check_v3_false_positives(dataset, samples, Path(args.jsonl_dir))),
    ]
    for tag, fn in steps:
        print(f"[{tag.split('_')[0]}/{len(steps)}] {tag} ...", flush=True)
        try:
            res = fn()
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        report[tag] = res
        ok = res.get("ok")
        print(f"[{tag.split('_')[0]}/{len(steps)}] {tag}: "
              f"{'OK' if ok else ('无法检查' if ok is None else 'FAIL')}"
              + (f"  {res.get('error')}" if res.get("error") else ""), flush=True)

    # ---- 7. 对比表
    diff = report["3_diff_vs_baseline"]
    print("\n[table] 每个物体：基线 vs 新数据集", flush=True)
    print(f"  {'object':28s} {'baseline':>9s} {'new':>6s} {'delta':>7s}", flush=True)
    for obj, d in sorted(diff.get("per_object_delta", {}).items()):
        b = diff["per_object"]["baseline"].get(obj, 0)
        n = diff["per_object"]["new"].get(obj, 0)
        print(f"  {obj:28s} {b:9d} {n:6d} {d:+7d}", flush=True)
    print(f"  {'TOTAL':28s} {diff.get('n_baseline', 0):9d} {diff.get('n_new', 0):6d} "
          f"{diff.get('delta', 0):+7d}", flush=True)
    print(f"\n[table] 新数据集按来源: {diff.get('per_source_new')}", flush=True)
    fp = report["6_v3_false_positives"]
    if fp.get("rows") is not None:
        print(f"\n[false-positive] v5 抽查出的 {fp['n_audited_false_positives']} 条 insufficient_contacts："
              f"仍在新数据集 {fp['n_still_present']} 条，已剔除 {fp['n_removed']} 条，"
              f"空转 {fp['n_vacuous']} 条", flush=True)
        for r in fp["rows"]:
            print(f"  {r['object_name']:22s} {r['grasp_file']:38s} idx={r['grasp_index']:3d} "
                  f"gap={r['v3_geo_closest_gap_mm']}mm in_new={r['in_new_dataset']} "
                  f"evidence={r['evidence']} new_reason={r['new_reason']}", flush=True)

    keys = [k for k, _ in steps]
    report["ok"] = all(report[k].get("ok") is not False for k in keys)
    report["elapsed_s"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nALL OK = {report['ok']}  ({report['elapsed_s']}s)", flush=True)
    print(f"written -> {out}", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
