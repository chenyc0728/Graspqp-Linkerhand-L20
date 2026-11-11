"""Convert the SAPIEN-filtered GraspQP LinkerHand **left-hand** grasps to CMapDataset format.

Changes vs v1:

* ``q`` now describes the **hand base** (``hand_base_link``), which is what CMap's
  virtual chain ends at -- v1 exported the GraspQP ``root_link`` pose directly and
  was therefore offset by ``(-0.03, 0, -0.10)`` m (see REPORT_v2 §2).
* ``robot_name`` is ``linkerhand`` (LEFT hand).  CMap's historical right-hand
  identity is renamed to ``linkerhand_right`` in ``CMapDataset.py``.

v3 additions (backwards compatible, see REPORT_v4_contactdb.md):

* ``--inputs <index.csv> [...]`` + ``--prefixes <prefix> [...]``: several index
  files can be merged in one pass, each with its own ``object_name`` prefix
  (``ycb+apple`` for YCB, ``contactdb+alarm_clock`` for ContactDB).
* Without ``--inputs`` the old single-input behaviour is kept: ``--input`` is a
  directory holding ``index.csv`` and the prefix defaults to ``ycb``.
* Samples are deduped by ``(sha256, grasp_index)`` across all inputs
  (``--no-dedupe`` disables it).  Per-source / per-prefix / per-object counts and
  the prefix->index mapping are written to the companion ``*.meta.json``.

Output structure is exactly ``{"info": ..., "metadata": [(q, object_name, robot_name), ...]}``.

Run (v3, YCB + ContactDB)::

    python new_grasp_filter/convert_to_cmap.py \
        --inputs new_grasp_filter/filtered_grasps/index_ycb.csv \
                 new_grasp_filter/filtered_grasps/index.csv \
        --prefixes ycb contactdb \
        --output new_grasp_filter/cmap_dataset_from_graspqp_v3.pt

Run (v2 reproduction, single input)::

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
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grasp_io as gio  # noqa: E402

ROBOT_NAME = "linkerhand"  # LEFT hand (see CMapDataset.ROBOT_ALIASES)
LEGACY_ROBOT_NAME = "linkerhand_right"
DEFAULT_PREFIX = "ycb"

#: ``(index_csv_or_dir, object_name_prefix)``
InputSpec = Tuple[Path, str]


def normalize_inputs(inputs: Union[Path, str, InputSpec, Iterable]) -> List[InputSpec]:
    """Accept a directory / a single index CSV / a list of ``(path, prefix)``.

    A directory means ``<dir>/index.csv`` (v2 behaviour).  A bare path without a
    prefix gets :data:`DEFAULT_PREFIX`.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [(inputs, DEFAULT_PREFIX)]
    specs: List[InputSpec] = []
    for item in inputs:
        if isinstance(item, (str, Path)):
            path, prefix = Path(item), DEFAULT_PREFIX
        else:
            path, prefix = item
            path = Path(path)
        if path.is_dir():
            path = path / "index.csv"
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"missing index file {path}")
        specs.append((path, str(prefix)))
    if not specs:
        raise ValueError("no inputs given")
    return specs


def iter_pairs(specs: Sequence[InputSpec], include_uncertain: bool = False,
               dedupe: bool = True, stats: dict | None = None
               ) -> Iterable[Tuple[dict, int, Path, str, str]]:
    """Yield ``(row, grasp_index, grasp_path, object_name, source)`` in dataset order.

    The order is: inputs in the order given, rows sorted by
    ``(source, object, grasp_file)``, feasible indices ascending, deduped by
    ``(sha256, grasp_index)``.  ``verify_cmap_v3.py`` / ``validate_cmap_dataset.py``
    reproduce the same order to pair every ``q`` with its source grasp.

    Rows whose grasp file is absent are skipped (and counted in ``stats``).
    """
    seen = set()
    if stats is not None:
        stats.setdefault("dropped_duplicates", 0)
        stats.setdefault("missing_grasp_files", [])
    for csv_path, prefix in specs:
        base = csv_path.parent
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: (r.get("source", ""), r["object"], r["grasp_file"]))
        for row in rows:
            idx = [int(x) for x in row["feasible_indices"].split()] if row["feasible_indices"].strip() else []
            if include_uncertain and row.get("uncertain_indices", "").strip():
                idx += [int(x) for x in row["uncertain_indices"].split()]
            if not idx:
                continue
            path = base / row["grasp_file"]
            for i in sorted(set(idx)):
                key = (row.get("sha256", ""), i)
                if key in seen:
                    if dedupe:
                        if stats is not None:
                            stats["dropped_duplicates"] += 1
                        continue
                    # no-dedupe mode: keep the sample, but the same physical grasp
                    # would be counted twice -> still skip missing files below
                seen.add(key)
                if not path.exists():
                    if stats is not None:
                        stats["missing_grasp_files"].append(str(path))
                    continue
                yield row, i, path, f"{prefix}+{row['object']}", row.get("source", "")


def convert(inputs, output: Path, include_uncertain: bool = False,
            dedupe: bool = True, robot_name: str = ROBOT_NAME) -> dict:
    specs = normalize_inputs(inputs)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    per_object: "OrderedDict[str, int]" = OrderedDict()
    per_source: "OrderedDict[str, int]" = OrderedDict()
    per_prefix: "OrderedDict[str, int]" = OrderedDict()
    stats: dict = {}
    for row, i, path, run_name, src in iter_pairs(specs, include_uncertain, dedupe, stats):
        g = gio.load_grasp_file(path)
        q = gio.root_pose_to_cmap_q(g["root_pose"][i], g["joints"][i])
        samples.append((q, run_name, src))
        per_object[run_name] = per_object.get(run_name, 0) + 1
        src_key = src or "unknown"
        per_source[src_key] = per_source.get(src_key, 0) + 1
        prefix = run_name.split("+", 1)[0]
        per_prefix[prefix] = per_prefix.get(prefix, 0) + 1

    metadata = [(torch.from_numpy(q), run_name, robot_name) for q, run_name, _ in samples]
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
        "per_prefix": dict(per_prefix),
        "prefixes": [p for _, p in specs],
        "inputs": [{"index_csv": str(p), "prefix": pre} for p, pre in specs],
        "dedupe": bool(dedupe),
        "dropped_duplicates": stats.get("dropped_duplicates", 0),
        "n_missing_grasp_files": len(stats.get("missing_grasp_files", [])),
        "missing_grasp_files": stats.get("missing_grasp_files", [])[:20],
        "note": (
            "q = [x, y, z, roll, pitch, yaw, 16 joints]; translation/rotation are the "
            "HAND BASE (hand_base_link) pose in the object frame, R = Rx@Ry@Rz. "
            "GraspQP's root_link pose was converted with t_base = t_root + R @ (-0.03, 0, -0.10). "
            "object_name = <prefix>+<object> (ycb+apple / contactdb+alarm_clock)."
        ),
    }
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(HERE / "filtered_grasps"),
                    help="single-input mode: directory holding index.csv (ignored with --inputs)")
    ap.add_argument("--inputs", nargs="+", default=None,
                    help="one or more index CSVs (or directories) to merge, in this order")
    ap.add_argument("--prefixes", nargs="+", default=None,
                    help="object_name prefix per --inputs entry (e.g. ycb contactdb)")
    ap.add_argument("--output", default=str(HERE / "cmap_dataset_from_graspqp_v3.pt"))
    ap.add_argument("--include-uncertain", action="store_true", default=False)
    ap.add_argument("--dedupe", dest="dedupe", action="store_true", default=True)
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    ap.add_argument("--robot-name", default=ROBOT_NAME)
    args = ap.parse_args()

    if args.inputs:
        inputs = args.inputs
        prefixes = args.prefixes or []
        if prefixes and len(prefixes) != len(inputs):
            ap.error(f"--prefixes has {len(prefixes)} entries but --inputs has {len(inputs)}")
        if not prefixes:
            prefixes = [DEFAULT_PREFIX] * len(inputs)
        specs = list(zip(inputs, prefixes))
    else:
        prefix = (args.prefixes or [DEFAULT_PREFIX])[0]
        specs = [(args.input, prefix)]

    stats = convert(specs, Path(args.output), args.include_uncertain,
                    args.dedupe, args.robot_name)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("per_object", "missing_grasp_files")}, indent=2))
    print(f"objects -> {stats['n_objects']}")


if __name__ == "__main__":
    main()
