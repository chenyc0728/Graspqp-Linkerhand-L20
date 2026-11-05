"""Convert the SAPIEN-filtered GraspQP LinkerHand-L20 grasps to CMapDataset format.

Input  : ``filtered_grasps/`` (byte-identical copies of the original ``.pt``
         files) plus ``filtered_grasps/index.csv`` (which lists, per file, the
         grasp indices that passed the SAPIEN verification).
Output : ``cmap_dataset_from_graspqp.pt`` with exactly the top-level structure of
         ``Cmap_dataset/CMapDataset_filtered/cmap_dataset.pt``::

             {"info": {robot_name: {"robot_name", "num_total",
                                    "num_upper_object", "num_per_object"}},
              "metadata": [(q: float32[22], object_name: str, robot_name: str), ...]}

``q`` layout (verified against
``Cmap_dataset/data_urdf/robot/linkerhand/right/linkerhand_l20_right_extended.urdf``):
``[x, y, z, roll, pitch, yaw, j1..j16]`` where the 6 first entries are the
``virtual_joint_{x,y,z,roll,pitch,yaw}`` values and the last 16 are the L20
driven joints in ``grasp_io.DRIVE_NAMES`` order (the same order used by CMap's
linkerhand URDF when the fixed/mimic joints are skipped).

Run::

    python new_grasp_filter/convert_to_cmap.py \
        --input new_grasp_filter/filtered_grasps \
        --output new_grasp_filter/cmap_dataset_from_graspqp.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grasp_io as gio  # noqa: E402

ROBOT_NAME = "linkerhand"  # key used by Cmap_dataset/data_urdf/robot/urdf_assets_meta.json


def matrix_to_rpy_cmap(m: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,3) roll/pitch/yaw such that ``R = Rx(r) @ Ry(p) @ Rz(y)``.

    This is the convention of the CMap "extended" URDF, whose virtual chain is
    ``virtual_joint_x -> y -> z -> roll -> pitch -> yaw -> fixed -> base_link``
    with identity joint origins, i.e. ``R = Rx(roll) @ Ry(pitch) @ Rz(yaw)``
    (note: the *opposite* multiplication order of the usual URDF ``rpy``).
    """
    m = np.asarray(m, dtype=np.float64)
    flat = m.reshape(-1, 3, 3)
    out = np.zeros((flat.shape[0], 3))
    for i, r in enumerate(flat):
        sp = np.clip(r[0, 2], -1.0, 1.0)
        pitch = np.arcsin(sp)
        if abs(sp) < 1.0 - 1e-9:
            roll = np.arctan2(-r[1, 2], r[2, 2])
            yaw = np.arctan2(-r[0, 1], r[0, 0])
        else:  # gimbal lock
            roll = 0.0
            yaw = np.arctan2(r[1, 0], r[1, 1])
        out[i] = [roll, pitch, yaw]
    return out.reshape(*m.shape[:-2], 3)


def grasp_to_q(root_pose: np.ndarray, joints: np.ndarray) -> np.ndarray:
    R = gio.quat_xyzw_to_matrix(root_pose[3:7])
    rpy = matrix_to_rpy_cmap(R)
    return np.concatenate([root_pose[:3], rpy, joints]).astype(np.float32)


def convert(input_dir: Path, output: Path, include_uncertain: bool = False) -> dict:
    input_dir = Path(input_dir).resolve()
    index_path = input_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"missing {index_path}; run run_filter.py first")

    samples = []
    with open(index_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # deterministic order -> idempotent output
    rows.sort(key=lambda r: (r["object"], r["grasp_file"]))
    per_object = OrderedDict()
    for row in rows:
        obj = row["object"]
        run_name = f"ycb+{obj}"
        path = input_dir / row["grasp_file"]
        if not path.exists():
            continue
        idx = [int(x) for x in row["feasible_indices"].split()] if row["feasible_indices"].strip() else []
        if include_uncertain and row.get("uncertain_indices", "").strip():
            idx += [int(x) for x in row["uncertain_indices"].split()]
        if not idx:
            continue
        g = gio.load_grasp_file(path)
        for i in sorted(set(idx)):
            samples.append((grasp_to_q(g["root_pose"][i], g["joints"][i]), run_name, obj))
            per_object[run_name] = per_object.get(run_name, 0) + 1

    metadata = [(torch.from_numpy(q), run_name, ROBOT_NAME) for q, run_name, _ in samples]
    info = {
        ROBOT_NAME: {
            "robot_name": ROBOT_NAME,
            "num_total": len(metadata),
            "num_upper_object": max(per_object.values()) if per_object else 0,
            "num_per_object": dict(per_object),
        }
    }
    payload = {"info": info, "metadata": metadata}
    torch.save(payload, output)
    return {"n_samples": len(metadata), "objects": len(per_object), "per_object": dict(per_object)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(HERE / "filtered_grasps"))
    ap.add_argument("--output", default=str(HERE / "cmap_dataset_from_graspqp.pt"))
    ap.add_argument("--include-uncertain", action="store_true", default=False,
                    help="also export grasps whose verdict is 'uncertain'")
    args = ap.parse_args()
    stats = convert(Path(args.input), Path(args.output), args.include_uncertain)
    print(json.dumps({"output": str(Path(args.output).resolve()), **stats}, indent=2))


if __name__ == "__main__":
    main()
