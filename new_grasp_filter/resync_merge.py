"""Resync the merged ``object`` rows with the freshly evaluated ``object1`` rows.

The v2 run evaluates ``object1`` and merges ``object`` from the v1 index.  Because
``object`` and ``object1`` are byte-identical (REPORT_v2 §1.1), the correct source
of truth for ``object`` is the *new* evaluation, keyed by SHA-256.  If the merge
ran with an older ``run_filter.py`` (which only shadowed files that had at least
one feasible grasp), some zero-feasible files may have kept their v1 verdicts.

This script rewrites both indexes from the ``object1`` results by content hash and
verifies every copied file.  It is idempotent and needs no simulation.

Run::

    python new_grasp_filter/resync_merge.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_io as gio  # noqa: E402

FG = HERE / "filtered_grasps"
FIELDS_FILE = ["source", "object", "grasp_file", "n_grasps", "n_feasible", "n_uncertain", "verdict",
               "feasible_indices", "uncertain_indices", "reason_counts",
               "geo_penetration_mm_min", "geo_penetration_mm_median", "geo_penetration_mm_max",
               "lift_rise_mm_max", "mesh_watertight", "copied", "sha256", "merged_from"]
FIELDS_GRASP = ["source", "object", "grasp_file", "grasp_index", "energy", "verdict", "reason",
                "geo_penetration_mm", "geo_min_distance_mm", "geo_contact_links",
                "lift_rise_mm", "lift_drift_mm", "mesh_watertight"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    with open(FG / "index.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(FG / "index_grasps.csv", newline="", encoding="utf-8") as f:
        graws = list(csv.DictReader(f))

    by_sha = {r["sha256"]: r for r in rows if r["source"] == "object1"}
    g_by_sha = defaultdict(list)
    for g in graws:
        if g["source"] == "object1":
            g_by_sha[g["sha256"] if "sha256" in g else None].append(g)
    # grasp rows don't carry sha; map via (object, grasp_file) of the object1 file row
    file_by_obj_name = {(r["object"], Path(r["grasp_file"]).name): r for r in rows if r["source"] == "object1"}
    g_by_file = defaultdict(list)
    for g in graws:
        if g["source"] == "object1":
            g_by_file[(g["object"], g["grasp_file"])].append(g)

    changed = 0
    for r in rows:
        if r["source"] != "object":
            continue
        ref = by_sha.get(r["sha256"])
        if not ref:
            continue
        for k in ["n_grasps", "n_feasible", "n_uncertain", "verdict", "feasible_indices",
                  "uncertain_indices", "reason_counts", "geo_penetration_mm_min",
                  "geo_penetration_mm_median", "geo_penetration_mm_max", "lift_rise_mm_max",
                  "mesh_watertight"]:
            if r.get(k) != ref.get(k):
                changed += 1
            r[k] = ref.get(k, "")
        r["copied"] = ref.get("copied", "0")

    # rebuild object grasp rows from the object1 grasp rows (same content)
    new_grasps = [g for g in graws if g["source"] != "object"]
    for g in graws:
        if g["source"] != "object1":
            continue
        ref = file_by_obj_name.get((g["object"], g["grasp_file"]))
        if ref is None:
            continue
        dst_name = Path(ref["grasp_file"]).name
        new_grasps.append({
            "source": "object", "object": g["object"],
            "grasp_file": dst_name, "grasp_index": g["grasp_index"],
            "energy": g["energy"], "verdict": "feasible", "reason": g["reason"],
            "geo_penetration_mm": g["geo_penetration_mm"],
            "geo_min_distance_mm": g["geo_min_distance_mm"],
            "geo_contact_links": g["geo_contact_links"],
            "lift_rise_mm": g["lift_rise_mm"], "lift_drift_mm": g["lift_drift_mm"],
            "mesh_watertight": g["mesh_watertight"],
        })

    # verify every referenced copy
    bad = 0
    for r in rows:
        if int(r.get("copied") or 0):
            p = FG / r["grasp_file"]
            if not p.exists() or sha256(p) != r["sha256"]:
                bad += 1
                print("MISSING/MISMATCH", r["grasp_file"])
    if bad:
        raise SystemExit(f"{bad} copied files failed verification")

    rows.sort(key=lambda r: (r["source"], r["object"], r["grasp_file"]))
    new_grasps.sort(key=lambda r: (r["source"], r["object"], r["grasp_file"], int(r["grasp_index"])))
    with open(FG / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_FILE)
        w.writeheader()
        w.writerows(rows)
    with open(FG / "index_grasps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_GRASP)
        w.writeheader()
        w.writerows(new_grasps)

    n_obj = sum(int(r["n_feasible"] or 0) for r in rows if r["source"] == "object")
    n_o1 = sum(int(r["n_feasible"] or 0) for r in rows if r["source"] == "object1")
    print(json.dumps({"object_feasible": n_obj, "object1_feasible": n_o1,
                      "fields_changed": changed, "grasp_rows": len(new_grasps),
                      "copied_verified": sum(1 for r in rows if int(r.get('copied') or 0))}, indent=2))


if __name__ == "__main__":
    main()
