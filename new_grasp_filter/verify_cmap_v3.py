"""Verify ``cmap_dataset_from_graspqp_v3.pt`` (YCB + ContactDB, left LinkerHand L20).

Checks (see HANDOFF_ubuntu.md §8.1 item 6 / acceptance criteria):

1. ``torch.load`` works, top-level keys are exactly ``['info', 'metadata']``,
   ``metadata`` is a list of 3-tuples, ``q`` is ``float32 (N, 22)``,
   ``robot_name == 'linkerhand'`` everywhere;
2. ``N == 3429 (ycb) + sum(n_feasible over the contactdb index)`` and the object
   count ``== 28 (ycb) + n contactdb objects with feasible grasps``;
3. ``info`` aggregates (``num_total`` / ``num_upper_object`` / ``num_per_object``)
   are consistent with ``metadata``;
4. every contactdb sample is named ``contactdb+<obj>`` and every ycb sample
   ``ycb+<obj>``; the 9 face-poor contactdb objects (``cube_*`` / ``cylinder_*`` /
   ``pyramid_*``) never appear;
5. per-object counts equal the per-index ``n_feasible`` sums;
6. the YCB part is **bit-identical** to ``cmap_dataset_from_graspqp_v2.pt``
   (same ``(sha256, grasp_index)`` -> same ``q``);
7. ``q`` round-trip: 20 random **contactdb** samples are pushed through the
   repaired CMap left-hand URDF and the hand-base pose is compared with the
   original GraspQP ``root_pose`` (expect rotation ~1e-7, translation ~1e-8 m).

Run::

    python new_grasp_filter/verify_cmap_v3.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import convert_to_cmap as c2c  # noqa: E402
import grasp_io as gio  # noqa: E402

FG = HERE / "filtered_grasps"
NO_GRASP = [f"{kind}_{size}" for kind in ("cube", "cylinder", "pyramid")
            for size in ("small", "medium", "large")]
EXPECTED = {
    "ycb_samples": 3429,
    "ycb_objects": 28,
    "robot_name": "linkerhand",
}


def index_stats(csv_path: Path):
    """(n_feasible per object, total feasible, rows, sources) of one index CSV."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    per_obj: "OrderedDict[str, int]" = OrderedDict()
    for r in rows:
        per_obj[r["object"]] = per_obj.get(r["object"], 0) + int(r["n_feasible"])
    return (OrderedDict((k, v) for k, v in per_obj.items() if v),
            sum(int(r["n_feasible"]) for r in rows), len(rows),
            dict(Counter(r.get("source", "") for r in rows)))


def fk_roundtrip(dataset: Path, specs, n: int = 20, seed: int = 0, prefix_filter: str = "contactdb",
                 backend: str = "numpy"):
    """q -> CMap left-hand URDF -> hand_base_link pose vs the original GraspQP pose.

    ``backend="numpy"`` uses ``grasp_io``'s own URDF FK (no external state).  The
    ``pytorch_kinematics`` install of this conda env (fork
    ``renezurbruegg/pytorch_kinematics`` @ 92c3b97) adds ``±1e-2`` rad to every
    revolute joint inside ``Frame.get_transform`` ("ugly fix"), which shows up here
    as a ~1e-2 rotation bias; see REPORT_v4 §5.
    """
    gio.set_backend(backend)
    left = gio.UrdfModel(gio.cmap_left_urdf(repaired=True))
    names = list(left.zero_angles().keys()) if backend == "numpy" else list(
        left.pk_chain_on("cpu").get_joint_parameter_names())
    md = torch.load(dataset, weights_only=False)["metadata"]

    pairs = list(c2c.iter_pairs(specs, False, True))
    assert len(pairs) == len(md), f"metadata {len(md)} != index pairs {len(pairs)}"
    cand = [k for k, (_, _, _, run_name, _) in enumerate(pairs)
            if run_name.startswith(prefix_filter + "+")]
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(cand, size=min(n, len(cand)), replace=False).tolist())

    VN = ("virtual_joint_x", "virtual_joint_y", "virtual_joint_z")
    RN = ("virtual_joint_roll", "virtual_joint_pitch", "virtual_joint_yaw")
    worst_rot = worst_trans = 0.0
    worst_where = None
    per_sample = []
    for k in picks:
        q, obj_name, _ = md[k]
        row, i, path, run_name, _ = pairs[k]
        g = gio.load_grasp_file(path)
        R0 = gio.quat_xyzw_to_matrix(g["root_pose"][i, 3:7])
        t0 = g["root_pose"][i, :3] + R0 @ gio.ROOT_TO_HAND_BASE
        qv = q.numpy().astype(np.float64)
        full21 = gio.build_full_angles(qv[6:22][None, :])[0]
        angles = {}
        for j, name in enumerate(names):
            if name in VN:
                angles[name] = float(qv[VN.index(name)])
            elif name in RN:
                angles[name] = float(qv[3 + RN.index(name)])
            elif name in gio.CHAIN_JOINT_NAMES:
                angles[name] = float(full21[gio.CHAIN_JOINT_NAMES.index(name)])
        T = left.fk(angles)["hand_base_link"]
        rot = float(np.abs(T[:3, :3] - R0).max())
        trans = float(np.abs(T[:3, 3] - t0).max())
        per_sample.append({"metadata_index": int(k), "object_name": obj_name,
                           "grasp_file": row["grasp_file"], "grasp_index": int(i),
                           "max_abs_rotation_error": rot,
                           "max_abs_translation_error_m": trans})
        if rot > worst_rot:
            worst_rot, worst_where = rot, per_sample[-1]
        worst_trans = max(worst_trans, trans)
    gio.set_backend("torch")
    return {"n_checked": len(picks), "prefix_filter": prefix_filter,
            "fk_backend": backend,
            "max_abs_rotation_error": worst_rot,
            "max_abs_translation_error_m": worst_trans,
            "worst_sample": worst_where, "samples": per_sample}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(HERE / "cmap_dataset_from_graspqp_v3.pt"))
    ap.add_argument("--v2", default=str(HERE / "cmap_dataset_from_graspqp_v2.pt"))
    ap.add_argument("--ycb-index", default=str(FG / "index_ycb.csv"))
    ap.add_argument("--contactdb-index", default=str(FG / "index.csv"))
    ap.add_argument("--n-roundtrip", type=int, default=20)
    ap.add_argument("--out", default=str(HERE / "logs" / "verify_cmap_v3.json"))
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    specs = [(Path(args.ycb_index), "ycb"), (Path(args.contactdb_index), "contactdb")]
    ycb_obj, ycb_total, ycb_rows, ycb_sources = index_stats(specs[0][0])
    cdb_obj, cdb_total, cdb_rows, cdb_sources = index_stats(specs[1][0])

    rep: dict = {"dataset": str(dataset),
                 "inputs": {"ycb_index": {"rows": ycb_rows, "sources": ycb_sources,
                                           "feasible": ycb_total, "objects": len(ycb_obj)},
                            "contactdb_index": {"rows": cdb_rows, "sources": cdb_sources,
                                                 "feasible": cdb_total, "objects": len(cdb_obj)}},
                 "checks": {}}
    chk = rep["checks"]

    # ---- 1. structure -----------------------------------------------------
    data = torch.load(dataset, weights_only=False)
    md = data["metadata"]
    chk["top_level_keys"] = sorted(data.keys())
    chk["top_level_keys_ok"] = sorted(data.keys()) == ["info", "metadata"]
    chk["metadata_type"] = type(md).__name__
    chk["metadata_list_of_3tuples"] = isinstance(md, list) and all(
        isinstance(m, tuple) and len(m) == 3 for m in md)
    shapes = {tuple(m[0].shape) for m in md}
    dtypes = {str(m[0].dtype) for m in md}
    chk["q_shapes"] = sorted(shapes)
    chk["q_dtypes"] = sorted(dtypes)
    chk["q_float32_22"] = shapes == {(22,)} and dtypes == {"torch.float32"}
    chk["n_samples"] = len(md)
    chk["robots"] = sorted({m[2] for m in md})
    chk["info_keys"] = sorted(data["info"].keys())
    chk["robot_name_in_info"] = data["info"].get(EXPECTED["robot_name"], {}).get("robot_name")
    chk["robot_name_ok"] = (chk["robots"] == [EXPECTED["robot_name"]]
                            and chk["robot_name_in_info"] == EXPECTED["robot_name"])

    # ---- 2/3. counts + info aggregates -----------------------------------
    n_objects = len({m[1] for m in md})
    chk["n_objects"] = n_objects
    chk["expected_n_samples"] = ycb_total + cdb_total
    chk["expected_n_objects"] = len(ycb_obj) + len(cdb_obj)
    chk["n_samples_ok"] = chk["n_samples"] == ycb_total + cdb_total
    chk["n_objects_ok"] = n_objects == len(ycb_obj) + len(cdb_obj)
    info = data["info"][EXPECTED["robot_name"]]
    actual_per_object = Counter(m[1] for m in md)
    chk["info_num_total"] = info["num_total"]
    chk["info_num_upper_object"] = info["num_upper_object"]
    chk["info_num_total_ok"] = info["num_total"] == len(md)
    chk["info_num_per_object_ok"] = dict(info["num_per_object"]) == dict(actual_per_object)
    chk["info_num_upper_object_ok"] = info["num_upper_object"] == max(actual_per_object.values())

    # ---- 4. prefixes / excluded objects ---------------------------------
    prefixes = Counter(m[1].split("+", 1)[0] for m in md)
    chk["prefix_counts"] = dict(prefixes)
    chk["prefixes_ok"] = set(prefixes) == {"ycb", "contactdb"}
    bad_pref = [m[1] for m in md
                if m[1].split("+", 1)[0] not in ("ycb", "contactdb") or "+" not in m[1]]
    chk["bad_object_names"] = bad_pref[:10]
    chk["all_names_prefixed"] = not bad_pref
    chk["ycb_count"] = prefixes.get("ycb", 0)
    chk["contactdb_count"] = prefixes.get("contactdb", 0)
    chk["ycb_count_ok"] = prefixes.get("ycb", 0) == EXPECTED["ycb_samples"]
    chk["contactdb_count_ok"] = prefixes.get("contactdb", 0) == cdb_total
    present_nograsp = sorted(n for n in {m[1] for m in md}
                             if n.split("+", 1)[1] in NO_GRASP)
    chk["nograsp_objects_present"] = present_nograsp
    chk["nograsp_absent_ok"] = not present_nograsp

    # ---- 5. per-object counts vs n_feasible -----------------------------
    mism = []
    for prefix, per_obj in (("ycb", ycb_obj), ("contactdb", cdb_obj)):
        for obj, n in per_obj.items():
            got = actual_per_object.get(f"{prefix}+{obj}", 0)
            if got != n:
                mism.append({"object_name": f"{prefix}+{obj}", "dataset": got, "index_n_feasible": n})
        for name in actual_per_object:
            if name.startswith(prefix + "+") and name.split("+", 1)[1] not in per_obj:
                mism.append({"object_name": name, "dataset": actual_per_object[name],
                             "index_n_feasible": 0})
    chk["per_object_mismatches"] = mism
    chk["per_object_ok"] = not mism

    # ---- 6. ycb part identical to v2 ------------------------------------
    v2 = torch.load(Path(args.v2).resolve(), weights_only=False)["metadata"]
    chk["v2_n_samples"] = len(v2)
    chk["v2_ycb_identical"] = False
    if len(v2) <= len(md):
        same = all(torch.equal(a[0], b[0]) for a, b in zip(v2, md[: len(v2)]))
        names_same = all(a[1] == b[1] for a, b in zip(v2, md[: len(v2)]))
        max_dq = max((float((a[0] - b[0]).abs().max()) for a, b in zip(v2, md[: len(v2)])),
                     default=0.0)
        chk["v2_vs_v3_prefix_max_abs_dq"] = max_dq
        chk["v2_vs_v3_object_names_equal"] = names_same
        chk["v2_ycb_identical"] = bool(same and names_same and len(v2) == EXPECTED["ycb_samples"])

    # ---- 7. FK round-trip on contactdb samples --------------------------
    rep["q_roundtrip_contactdb"] = fk_roundtrip(dataset, specs, n=args.n_roundtrip,
                                                prefix_filter="contactdb")
    rep["q_roundtrip_ycb_smoke"] = {k: v for k, v in fk_roundtrip(
        dataset, specs, n=3, prefix_filter="ycb").items() if k != "samples"}
    rep["q_roundtrip_contactdb_pk"] = {k: v for k, v in fk_roundtrip(
        dataset, specs, n=args.n_roundtrip, prefix_filter="contactdb",
        backend="torch").items() if k != "samples"}
    rt = rep["q_roundtrip_contactdb"]
    chk["roundtrip_ok"] = (rt["max_abs_rotation_error"] < 1e-5
                           and rt["max_abs_translation_error_m"] < 1e-6)

    # ---- summary --------------------------------------------------------
    failures = [k for k, v in chk.items()
                if (k.endswith("_ok") or k == "all_names_prefixed") and v is False]
    rep["failures"] = failures
    rep["all_ok"] = not failures

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print(f"\nwritten -> {out}")
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
