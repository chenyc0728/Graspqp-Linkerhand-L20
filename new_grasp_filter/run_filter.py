"""Batch filter v2: SAPIEN verification over one or more grasp roots.

New in v2 (all backwards compatible):

* ``--grasp-path <root>``   arbitrary root laid out like ``<root>/<object>/grasp_predictions/...``
  (e.g. ``object``, ``object1``, or any future run);
* ``--source-tag <str>``    tag written to the output tree + index (default: root dir name);
* ``--merge-from <csv>``    merge a previous run's ``index.csv`` (re-uses its
  verdicts instead of re-simulating) and copies/verifies any missing files;
* ``--legacy-numpy``        use the v1 numpy maths instead of roma/pytorch_kinematics;
* ``--device cpu|cuda``     device for the torch FK backend;
* ``--gpu``                 try to enable SAPIEN PhysX GPU (falls back to CPU with a log line).

Outputs (under ``--output-dir``, default ``new_grasp_filter/``)::

    filtered_grasps/<source>/<object>/linkerhand_l20/12_contacts/graspqp/default/*.pt
    filtered_grasps/index.csv          # one row per .pt, all sources
    filtered_grasps/index_grasps.csv   # one row per feasible grasp
    logs/<source>/<object>.jsonl       # one line per grasp, all metrics
    logs/<source>/summary.json

Run::

    python new_grasp_filter/run_filter.py --grasp-path object1 --source-tag object1 \
        --merge-from new_grasp_filter/filtered_grasps/index_v1_object.csv
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

INDEX_FIELDS = [
    "source", "object", "grasp_file", "n_grasps", "n_feasible", "n_uncertain", "verdict",
    "feasible_indices", "uncertain_indices", "reason_counts",
    "geo_penetration_mm_min", "geo_penetration_mm_median", "geo_penetration_mm_max",
    "lift_rise_mm_max", "mesh_watertight", "copied", "sha256", "merged_from",
]
GRASP_FIELDS = [
    "source", "object", "grasp_file", "grasp_index", "energy", "verdict", "reason",
    "geo_penetration_mm", "geo_min_distance_mm", "geo_contact_links",
    "lift_rise_mm", "lift_drift_mm", "mesh_watertight",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_source_tag(root: Path) -> str:
    return Path(root).name or "grasps"


def discover_objects(root: Path) -> list[str]:
    return gio.list_objects(root)


def verdict_of(reason: str) -> str:
    if reason == "ok":
        return "feasible"
    if reason.startswith("exception") or reason == "unstable_simulation":
        return "uncertain"
    return "infeasible"


def fnum(x, default=""):
    try:
        v = float(x)
        return "" if not np.isfinite(v) else round(v, 3)
    except Exception:
        return default


def run(args):
    out_dir = Path(args.output_dir).resolve()
    root = Path(args.grasp_path).resolve() if args.grasp_path else gio.YCB_ROOT
    if not root.is_dir():
        raise SystemExit(f"--grasp-path {root} does not exist")
    # accept either <root>/<object>/grasp_predictions/... or <root>/ycb/<object>/...
    if not gio.list_objects(root) and (root / "ycb").is_dir():
        root = root / "ycb"
    source = args.source_tag or default_source_tag(root.parent if root.name == "ycb" else root)

    gio.set_backend("numpy" if args.legacy_numpy else "torch")
    if not args.legacy_numpy:
        gio.set_device(args.device)
    print(f"[setup] root={root} source={source} backend={gio.get_backend()} device={gio.get_device()}", flush=True)

    log_dir = out_dir / "logs" / source
    log_dir.mkdir(parents=True, exist_ok=True)
    filtered = out_dir / "filtered_grasps" / source
    filtered.mkdir(parents=True, exist_ok=True)

    from sapien_grasp_checker import CheckerConfig, SapienGraspChecker

    cfg = CheckerConfig()
    if args.max_penetration_mm is not None:
        cfg.max_penetration = args.max_penetration_mm / 1000.0
    if args.lift_min_rise_mm is not None:
        cfg.lift_min_rise = args.lift_min_rise_mm / 1000.0
    cfg.use_gpu = bool(args.gpu)

    objects = [args.object] if args.object else discover_objects(root)
    rows, grasp_rows = [], []
    file_results = {}  # sha256 -> {"feasible": [...], "metrics": {i: row}} for every evaluated file
    summary = {"source": source, "root": str(root), "config": vars(cfg), "objects": {},
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t_start = time.time()

    import trimesh

    for obj in objects:
        files = gio.list_grasp_files(obj, root)
        if args.grasp:
            files = [f for f in files if f.name == args.grasp]
        if args.max_files_per_object:
            files = files[: args.max_files_per_object]
        if not files:
            print(f"[skip] {obj}: no grasp files", flush=True)
            continue

        mesh = trimesh.load(gio.object_mesh_path(obj, root), force="mesh")
        watertight = bool(mesh.is_watertight and mesh.volume > 0)

        t0 = time.time()
        checker = SapienGraspChecker(obj, cfg, root=root)
        stats = {"n_files": len(files), "n_grasps": 0, "n_feasible": 0,
                 "n_infeasible": 0, "n_uncertain": 0, "n_files_copied": 0, "reasons": {},
                 "mesh_watertight": watertight}
        log_path = log_dir / f"{obj}.jsonl"
        with open(log_path, "w", encoding="utf-8") as logf:
            for fpath in files:
                g = gio.load_grasp_file(fpath)
                n = len(g["values"]) if args.limit is None else min(args.limit, len(g["values"]))
                feas_idx, unc_idx, reasons = [], [], {}
                pens, rises = [], []
                grasp_row_start = len(grasp_rows)
                for i in range(n):
                    res = checker.check(g["root_pose"][i], g["joints"][i])
                    v = verdict_of(res.reason)
                    m = res.metrics
                    rec = {
                        "source": source, "object": obj, "grasp_file": fpath.name, "grasp_index": i,
                        "energy": float(g["values"][i]), "verdict": v, "reason": res.reason,
                        "metrics": {k: (float(x) if isinstance(x, (int, float, np.floating)) else x)
                                    for k, x in m.items() if k != "geo_contact_links"},
                        "contact_links": m.get("geo_contact_links", []),
                        "mesh_watertight": watertight,
                        "sapien_contacts": res.contacts[:60],
                    }
                    logf.write(json.dumps(rec) + "\n")
                    stats["n_grasps"] += 1
                    reasons[res.reason] = reasons.get(res.reason, 0) + 1
                    pen = m.get("geo_penetration_mm", float("nan"))
                    if np.isfinite(pen):
                        pens.append(pen)
                    if v == "feasible":
                        feas_idx.append(i)
                        rises.append(m.get("lift_rise_mm", float("nan")))
                        grasp_rows.append({
                            "source": source, "object": obj, "grasp_file": fpath.name,
                            "grasp_index": i, "energy": round(float(g["values"][i]), 4),
                            "verdict": v, "reason": res.reason,
                            "geo_penetration_mm": fnum(pen),
                            "geo_min_distance_mm": fnum(m.get("geo_min_distance_mm")),
                            "geo_contact_links": "|".join(m.get("geo_contact_links", [])),
                            "lift_rise_mm": fnum(m.get("lift_rise_mm")),
                            "lift_drift_mm": fnum(m.get("lift_drift_mm")),
                            "mesh_watertight": int(watertight),
                        })
                    elif v == "uncertain":
                        unc_idx.append(i)
                stats["n_feasible"] += len(feas_idx)
                stats["n_uncertain"] += len(unc_idx)
                stats["n_infeasible"] += n - len(feas_idx) - len(unc_idx)
                for k, c in reasons.items():
                    stats["reasons"][k] = stats["reasons"].get(k, 0) + c

                copy_it = len(feas_idx) > 0 or (args.include_uncertain and len(unc_idx) > 0)
                dst = filtered / obj / gio.GRASP_SUBDIR / fpath.name
                src_hash = sha256(fpath)
                for gr in grasp_rows[grasp_row_start:]:
                    gr["_sha"] = src_hash
                file_results[src_hash] = {
                    "object": obj, "file": fpath.name,
                    "feasible": list(feas_idx),
                    "rows": {gr["grasp_index"]: gr for gr in grasp_rows[grasp_row_start:]},
                }
                if copy_it:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fpath, dst)
                    if sha256(dst) != src_hash:
                        raise RuntimeError(f"hash mismatch after copy: {dst}")
                    stats["n_files_copied"] += 1
                rows.append({
                    "source": source, "object": obj,
                    "grasp_file": (Path(source) / obj / gio.GRASP_SUBDIR / fpath.name).as_posix(),
                    "n_grasps": n, "n_feasible": len(feas_idx), "n_uncertain": len(unc_idx),
                    "verdict": "feasible" if feas_idx else ("uncertain" if (args.include_uncertain and unc_idx) else "infeasible"),
                    "feasible_indices": " ".join(map(str, feas_idx)),
                    "uncertain_indices": " ".join(map(str, unc_idx)),
                    "reason_counts": json.dumps(reasons),
                    "geo_penetration_mm_min": fnum(min(pens)) if pens else "",
                    "geo_penetration_mm_median": fnum(float(np.median(pens))) if pens else "",
                    "geo_penetration_mm_max": fnum(max(pens)) if pens else "",
                    "lift_rise_mm_max": fnum(max([r for r in rises if np.isfinite(r)], default=np.nan)) if rises else "",
                    "mesh_watertight": int(watertight),
                    "copied": int(copy_it),
                    "sha256": src_hash,
                    "merged_from": "",
                })
                logf.flush()
        summary["objects"][obj] = stats
        print(f"[{source}/{obj}] files={len(files)} grasps={stats['n_grasps']} "
              f"feasible={stats['n_feasible']} uncertain={stats['n_uncertain']} "
              f"copied={stats['n_files_copied']} ({time.time()-t0:.1f}s)", flush=True)

    if args.merge_from:
        merged = merge_previous(Path(args.merge_from), out_dir, grasp_rows, file_results)
        rows.extend(merged["rows"])
        grasp_rows.extend(merged["grasps"])
        summary["merged"] = merged["stats"]

    write_indexes(out_dir, rows, grasp_rows)
    summary["total"] = {
        "n_grasps": sum(v["n_grasps"] for v in summary["objects"].values()),
        "n_feasible": sum(v["n_feasible"] for v in summary["objects"].values()),
        "n_uncertain": sum(v["n_uncertain"] for v in summary["objects"].values()),
        "n_files_copied": sum(v["n_files_copied"] for v in summary["objects"].values()),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TOTAL", json.dumps(summary["total"]), flush=True)
    print("index:", out_dir / "filtered_grasps" / "index.csv", flush=True)


def merge_previous(index_path: Path, out_dir: Path, current_grasps=None, file_results=None) -> dict:
    """Merge a previous run's index (default source tag ``object``).

    Files referenced by the old index are copied/verified into
    ``filtered_grasps/<source>/...`` (the v1 layout had no source dir, so a row
    like ``apple/grasp_predictions/...`` is re-rooted under ``object/``).
    Per-grasp geometry metrics are recovered from the previous ``logs/`` so
    ``index_grasps.csv`` stays complete without re-simulating.
    """
    old_dir = index_path.parent
    old_logs = old_dir.parent / "logs"
    # content-addressed reuse: a file that is byte-identical to one already
    # evaluated in this run inherits those verdicts (the v1 file hashes are the
    # same for object/ and object1/, verified in REPORT_v2 §1.1)
    by_sha = {}
    for gr in (current_grasps or []):
        by_sha.setdefault(gr["_sha"], {}).setdefault(gr["grasp_index"], gr)
    # every evaluated file (even with zero feasible grasps) must shadow the old
    # verdicts, otherwise a v1 phantom-positive file could leak back in
    evaluated = set((file_results or {}).keys())
    rows_out, grasps_out = [], []
    stats = {"from": str(index_path), "files": 0, "copied": 0, "grasps": 0}
    with open(index_path, newline="", encoding="utf-8") as f:
        old_rows = list(csv.DictReader(f))
    for r in old_rows:
        src = r.get("source") or "object"
        obj = r["object"]
        old_rel = r["grasp_file"].replace("\\", "/")
        fname = Path(old_rel).name
        parts = old_rel.split("/")
        if parts[0] == src:
            parts = parts[1:]
        src_file = old_dir / Path(*parts)
        if not src_file.exists():  # v1 layout had no <source>/ level yet
            alt = old_dir / src / Path(*parts)
            if alt.exists():
                src_file = alt
        copied = int(r.get("copied", 0) or 0)
        if not src_file.exists():
            if copied:  # only copied files are expected to exist in the old tree
                print(f"[merge] missing source file {src_file}", flush=True)
                continue
            # rows without a copy: hash the original grasp file instead
            orig = gio.REPO_ROOT / "object" / "ycb" / obj / gio.GRASP_SUBDIR / fname
            src_file = orig if orig.exists() else None
        h = sha256(src_file) if src_file is not None else ""
        dst = out_dir / "filtered_grasps" / src / obj / gio.GRASP_SUBDIR / fname
        if copied:
            if not dst.exists() or sha256(dst) != h:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst)
                stats["copied"] += 1
            if sha256(dst) != h:
                raise RuntimeError(f"merge hash mismatch: {dst}")
        stats["files"] += 1
        reuse = by_sha.get(h, {})
        if h in evaluated:
            feas = sorted(reuse.keys())
            stats["reused_from_current"] = stats.get("reused_from_current", 0) + 1
        else:
            feas = [int(x) for x in (r.get("feasible_indices") or "").split()]

        log_recs = {}
        lp = old_logs / f"{obj}.jsonl"
        if lp.exists():
            with open(lp, encoding="utf-8") as lf:
                for line in lf:
                    rec = json.loads(line)
                    if rec.get("grasp_file") == fname:
                        log_recs[rec["grasp_index"]] = rec
        wt = r.get("mesh_watertight", "")
        if wt == "":
            try:
                import trimesh

                m = trimesh.load(gio.object_mesh_path(obj), force="mesh")
                wt = int(bool(m.is_watertight and m.volume > 0))
            except Exception:
                wt = ""
        pens = [log_recs[i]["metrics"].get("geo_penetration_mm") for i in feas if i in log_recs]
        pens = [p for p in pens if p is not None and np.isfinite(p)]
        lifts = [log_recs[i]["metrics"].get("lift_rise_mm") for i in feas if i in log_recs]
        lifts = [x for x in lifts if x is not None and np.isfinite(x)]
        for i in feas:
            if reuse and i in reuse:
                g0 = dict(reuse[i])
                g0.update({"source": src, "object": obj, "grasp_file": fname})
                grasps_out.append(g0)
                stats["grasps"] += 1
                continue
            rec = log_recs.get(i, {})
            m = rec.get("metrics", {})
            grasps_out.append({
                "source": src, "object": obj, "grasp_file": fname, "grasp_index": i,
                "energy": round(float(rec["energy"]), 4) if rec else "",
                "verdict": "feasible", "reason": rec.get("reason", "ok"),
                "geo_penetration_mm": fnum(m.get("geo_penetration_mm")),
                "geo_min_distance_mm": fnum(m.get("geo_min_distance_mm")),
                "geo_contact_links": "|".join(rec.get("contact_links", [])),
                "lift_rise_mm": fnum(m.get("lift_rise_mm")),
                "lift_drift_mm": fnum(m.get("lift_drift_mm")),
                "mesh_watertight": wt,
            })
            stats["grasps"] += 1
        rows_out.append({
            "source": src, "object": obj,
            "grasp_file": (Path(src) / obj / gio.GRASP_SUBDIR / fname).as_posix(),
            "n_grasps": r.get("n_grasps", ""), "n_feasible": len(feas),
            "n_uncertain": r.get("n_uncertain", 0), "verdict": r.get("verdict", ""),
            "feasible_indices": r.get("feasible_indices", ""),
            "uncertain_indices": r.get("uncertain_indices", ""),
            "reason_counts": r.get("reason_counts", ""),
            "geo_penetration_mm_min": fnum(min(pens)) if pens else "",
            "geo_penetration_mm_median": fnum(float(np.median(pens))) if pens else "",
            "geo_penetration_mm_max": fnum(max(pens)) if pens else "",
            "lift_rise_mm_max": fnum(max(lifts)) if lifts else "",
            "mesh_watertight": wt,
            "copied": int(r.get("copied", 0) or 0),
            "sha256": h,
            "merged_from": str(index_path),
        })
    print(f"[merge] {stats}", flush=True)
    return {"rows": rows_out, "grasps": grasps_out, "stats": stats}


def write_indexes(out_dir: Path, rows, grasp_rows):
    fg = out_dir / "filtered_grasps"
    fg.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["source"], r["object"], r["grasp_file"]))
    grasp_rows = sorted(grasp_rows, key=lambda r: (r["source"], r["object"], r["grasp_file"], int(r["grasp_index"])))
    grasp_rows = [{k: v for k, v in r.items() if k != "_sha"} for r in grasp_rows]
    with open(fg / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(fg / "index_grasps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GRASP_FIELDS)
        w.writeheader()
        w.writerows(grasp_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grasp-path", default=None,
                    help="root containing <root>/<object>/grasp_predictions/... (default: object/)")
    ap.add_argument("--source-tag", default=None, help="tag for the output tree + index (default: root dir name)")
    ap.add_argument("--merge-from", default=None, help="previous filtered_grasps/index.csv to merge (no re-simulation)")
    ap.add_argument("--legacy-numpy", action="store_true", help="use the v1 numpy maths instead of roma/pytorch_kinematics")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="device for the torch FK backend")
    ap.add_argument("--gpu", action="store_true", help="try SAPIEN PhysX GPU (falls back to CPU)")
    ap.add_argument("--object", default=None)
    ap.add_argument("--grasp", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-files-per-object", type=int, default=None)
    ap.add_argument("--output-dir", default=str(HERE))
    ap.add_argument("--max-penetration-mm", type=float, default=None)
    ap.add_argument("--lift-min-rise-mm", type=float, default=None)
    ap.add_argument("--include-uncertain", dest="include_uncertain", action="store_true", default=True)
    ap.add_argument("--no-include-uncertain", dest="include_uncertain", action="store_false")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":  # Windows spawn guard
    main()
