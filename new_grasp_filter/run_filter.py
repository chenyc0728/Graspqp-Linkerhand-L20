"""Batch filter: run the SAPIEN checker over every GraspQP LinkerHand-L20 grasp.

Usage (from the repository root, in the ``genh2r`` environment)::

    python new_grasp_filter/run_filter.py --limit 4          # smoke test
    python new_grasp_filter/run_filter.py                    # all objects
    python new_grasp_filter/run_filter.py --object apple --grasp apple.dexgrasp.pt

Outputs (all under ``new_grasp_filter/``):

* ``logs/<object>.jsonl``  -- one line per grasp, every metric + verdict
* ``logs/summary.json``    -- aggregate statistics
* ``filtered_grasps/<object>/linkerhand_l20/12_contacts/graspqp/default/*.pt``
  -- **byte-identical copies** (verified with SHA-256) of every ``.pt`` that
  contains at least one feasible grasp
* ``filtered_grasps/index.csv`` -- per-file and per-grasp index with verdicts

A ``.pt`` file holds a batch of 32 grasps, so the copy unit is the file while the
verdict is per grasp; ``index.csv`` lists the feasible grasp indices so a
consumer can select the exact subset (``cmap_dataset_from_graspqp.pt`` does).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grasp_io as gio  # noqa: E402
from sapien_grasp_checker import CheckerConfig, SapienGraspChecker  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_objects(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and gio.list_grasp_files(p.name))


def verdict_of(reason: str, metrics: dict, cfg: CheckerConfig) -> str:
    if reason == "ok":
        return "feasible"
    if reason.startswith("exception") or reason == "unstable_simulation":
        return "uncertain"
    return "infeasible"


def run(args):
    out_dir = Path(args.output_dir).resolve()
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    filtered = out_dir / "filtered_grasps"
    filtered.mkdir(parents=True, exist_ok=True)

    cfg = CheckerConfig()
    if args.max_penetration_mm is not None:
        cfg.max_penetration = args.max_penetration_mm / 1000.0
    if args.lift_min_rise_mm is not None:
        cfg.lift_min_rise = args.lift_min_rise_mm / 1000.0

    objects = [args.object] if args.object else discover_objects(gio.YCB_ROOT)
    index_rows = []
    summary = {"config": cfg.__dict__, "objects": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t_start = time.time()

    for obj in objects:
        files = gio.list_grasp_files(obj)
        if args.grasp:
            files = [f for f in files if f.name == args.grasp]
        if args.max_files_per_object:
            files = files[: args.max_files_per_object]
        if not files:
            print(f"[skip] {obj}: no grasp files")
            continue

        t0 = time.time()
        checker = SapienGraspChecker(obj, cfg)
        obj_stats = {
            "n_files": len(files),
            "n_grasps": 0,
            "n_feasible": 0,
            "n_infeasible": 0,
            "n_uncertain": 0,
            "n_files_copied": 0,
            "reasons": {},
        }
        log_path = log_dir / f"{obj}.jsonl"
        with open(log_path, "w", encoding="utf-8") as logf:
            for fpath in files:
                g = gio.load_grasp_file(fpath)
                n = len(g["values"]) if args.limit is None else min(args.limit, len(g["values"]))
                feas_idx, unc_idx = [], []
                reasons = {}
                for i in range(n):
                    res = checker.check(g["root_pose"][i], g["joints"][i])
                    v = verdict_of(res.reason, res.metrics, cfg)
                    rec = {
                        "object": obj,
                        "grasp_file": fpath.name,
                        "grasp_index": i,
                        "energy": float(g["values"][i]),
                        "verdict": v,
                        "reason": res.reason,
                        "metrics": {k: (float(x) if isinstance(x, (int, float, np.floating)) else x)
                                    for k, x in res.metrics.items() if k != "geo_contact_links"},
                        "contact_links": res.metrics.get("geo_contact_links", []),
                        "sapien_contacts": res.contacts[:60],
                    }
                    logf.write(json.dumps(rec) + "\n")
                    obj_stats["n_grasps"] += 1
                    reasons[res.reason] = reasons.get(res.reason, 0) + 1
                    if v == "feasible":
                        feas_idx.append(i)
                    elif v == "uncertain":
                        unc_idx.append(i)
                obj_stats["reasons"] = {k: obj_stats["reasons"].get(k, 0) + v for k, v in reasons.items()}
                obj_stats["n_feasible"] += len(feas_idx)
                obj_stats["n_uncertain"] += len(unc_idx)
                obj_stats["n_infeasible"] += n - len(feas_idx) - len(unc_idx)

                copy_it = len(feas_idx) > 0 or (args.include_uncertain and len(unc_idx) > 0)
                dst = filtered / obj / gio.GRASP_SUBDIR / fpath.name
                src_hash = sha256(fpath)
                if copy_it:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fpath, dst)
                    if sha256(dst) != src_hash:
                        raise RuntimeError(f"hash mismatch after copy: {dst}")
                    obj_stats["n_files_copied"] += 1
                index_rows.append(
                    {
                        "object": obj,
                        "grasp_file": str(Path(obj) / gio.GRASP_SUBDIR / fpath.name),
                        "n_grasps": n,
                        "n_feasible": len(feas_idx),
                        "n_uncertain": len(unc_idx),
                        "verdict": "feasible" if feas_idx else ("uncertain" if (args.include_uncertain and unc_idx) else "infeasible"),
                        "feasible_indices": " ".join(map(str, feas_idx)),
                        "uncertain_indices": " ".join(map(str, unc_idx)),
                        "reason_counts": json.dumps(reasons),
                        "copied": int(copy_it),
                        "sha256": src_hash,
                    }
                )
                logf.flush()
        summary["objects"][obj] = obj_stats
        print(
            f"[{obj}] files={len(files)} grasps={obj_stats['n_grasps']} "
            f"feasible={obj_stats['n_feasible']} uncertain={obj_stats['n_uncertain']} "
            f"copied={obj_stats['n_files_copied']} ({time.time()-t0:.1f}s)",
            flush=True,
        )

    index_path = filtered / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()) if index_rows else ["object"])
        w.writeheader()
        w.writerows(index_rows)

    summary["total"] = {
        "n_grasps": sum(v["n_grasps"] for v in summary["objects"].values()),
        "n_feasible": sum(v["n_feasible"] for v in summary["objects"].values()),
        "n_uncertain": sum(v["n_uncertain"] for v in summary["objects"].values()),
        "n_files_copied": sum(v["n_files_copied"] for v in summary["objects"].values()),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    with open(log_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("TOTAL", json.dumps(summary["total"]), flush=True)
    print("index:", index_path, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", help="single object name (e.g. apple)")
    ap.add_argument("--grasp", help="single .pt file name (e.g. apple.dexgrasp.pt)")
    ap.add_argument("--limit", type=int, default=None, help="max grasps per .pt file")
    ap.add_argument("--max-files-per-object", type=int, default=None)
    ap.add_argument("--output-dir", default=str(HERE))
    ap.add_argument("--max-penetration-mm", type=float, default=None)
    ap.add_argument("--lift-min-rise-mm", type=float, default=None)
    ap.add_argument("--include-uncertain", dest="include_uncertain", action="store_true", default=True)
    ap.add_argument("--no-include-uncertain", dest="include_uncertain", action="store_false")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":  # Windows: required guard for any multiprocessing future use
    main()
