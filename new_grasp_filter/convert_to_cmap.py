"""Convert the SAPIEN-filtered GraspQP LinkerHand **left-hand** grasps to CMapDataset format (v2).

Changes vs v1:

* ``q`` now describes the **hand base** (``hand_base_link``), which is what CMap's
  virtual chain ends at -- v1 exported the GraspQP ``root_link`` pose directly and
  was therefore offset by ``(-0.03, 0, -0.10)`` m (see REPORT_v2 §2).
* ``robot_name`` is ``linkerhand`` (LEFT hand).  CMap's historical right-hand
  identity is renamed to ``linkerhand_right`` in ``CMapDataset.py``.
* Input is the merged ``filtered_grasps/index.csv`` (``object`` + ``object1``).
* The two sources are byte-identical in this workspace, so samples are deduped by
  ``(sha256, grasp_index)`` by default (``--no-dedupe`` disables it); the source
  breakdown is written to the companion JSON.

Output structure is exactly ``{"info": ..., "metadata": [(q, object_name, robot_name), ...]}``.

Run::

    python new_grasp_filter/convert_to_cmap.py \
        --input new_grasp_filter/filtered_grasps \
        --output new_grasp_filter/cmap_dataset_from_graspqp_v2.pt
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

ROBOT_NAME = "linkerhand"  # LEFT hand (see CMapDataset.ROBOT_ALIASES)
LEGACY_ROBOT_NAME = "linkerhand_right"


def convert(input_dir: Path, output: Path, include_uncertain: bool = False,
            dedupe: bool = True, robot_name: str = ROBOT_NAME) -> dict:
    input_dir = Path(input_dir).resolve()
    index_path = input_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"missing {index_path}; run run_filter.py first")

    with open(index_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r.get("source", ""), r["object"], r["grasp_file"]))

    samples = []
    seen = set()
    per_object = OrderedDict()
    per_source = OrderedDict()
    dropped_dupes = 0
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
            key = (row.get("sha256", ""), i)
            if dedupe and key in seen:
                dropped_dupes += 1
                continue
            seen.add(key)
            q = gio.root_pose_to_cmap_q(g["root_pose"][i], g["joints"][i])
            samples.append((q, run_name, obj, row.get("source", "")))
            per_object[run_name] = per_object.get(run_name, 0) + 1
            src = row.get("source", "") or "unknown"
            per_source[src] = per_source.get(src, 0) + 1

    metadata = [(torch.from_numpy(q), run_name, robot_name) for q, run_name, _, _ in samples]
    info = {
        robot_name: {
            "robot_name": robot_name,
            "num_total": len(metadata),
            "num_upper_object": max(per_object.values()) if per_object else 0,
            "num_per_object": dict(per_object),
        }
    }
    torch.save({"info": info, "metadata": metadata}, output)

    meta = {
        "output": str(output),
        "robot_name": robot_name,
        "n_samples": len(metadata),
        "n_objects": len(per_object),
        "per_object": dict(per_object),
        "per_source": dict(per_source),
        "dedupe": bool(dedupe),
        "dropped_duplicates": dropped_dupes,
        "note": (
            "q = [x, y, z, roll, pitch, yaw, 16 joints]; translation/rotation are the "
            "HAND BASE (hand_base_link) pose in the object frame, R = Rx@Ry@Rz. "
            "GraspQP's root_link pose was converted with t_base = t_root + R @ (-0.03, 0, -0.10)."
        ),
    }
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(HERE / "filtered_grasps"))
    ap.add_argument("--output", default=str(HERE / "cmap_dataset_from_graspqp_v2.pt"))
    ap.add_argument("--include-uncertain", action="store_true", default=False)
    ap.add_argument("--dedupe", dest="dedupe", action="store_true", default=True)
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    ap.add_argument("--robot-name", default=ROBOT_NAME)
    args = ap.parse_args()
    stats = convert(Path(args.input), Path(args.output), args.include_uncertain,
                    args.dedupe, args.robot_name)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_object"}, indent=2))


if __name__ == "__main__":
    main()
