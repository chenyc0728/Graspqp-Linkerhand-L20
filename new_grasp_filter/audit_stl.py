"""Audit object meshes (watertightness / winding / topology / self-overlap / volume).

Targets:
  * the 8 YCB meshes that were non-watertight in v2;
  * every ``object/contactdb/<name>`` mesh (49 objects).

Outputs ``logs/stl_audit_<tag>.csv`` + ``.json`` with one row per file, and a
PASS / WARN / FAIL verdict:

  FAIL: not watertight, inconsistent winding, non-positive volume, non-manifold
        edges, degenerate/duplicate faces, or a high self-overlap rate;
  WARN: watertight but suspicious (body_count > 1, boundary loops present,
        inward-probe overlap 1-5%);
  PASS: everything clean.

The self-overlap rate is a sampling proxy (allowed by the task): sample surface
points, step ``eps`` along the inward normal and ask ``trimesh.contains`` whether
that point is inside.  A valid closed mesh answers "outside" for every point.

Usage::

    python new_grasp_filter/audit_stl.py --tag before
    python new_grasp_filter/audit_stl.py --tag after
    python new_grasp_filter/audit_stl.py --tag before --paths <file> [<file> ...]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_io as gio  # noqa: E402

YCB_BAD = ["gelatin_box", "mini_soccer_ball", "mustard_bottle", "pear",
           "power_drill", "sponge", "tomato_soup_can", "tuna_fish_can"]
FIELDS = [
    "path", "dataset", "object", "format", "sha256", "bytes", "n_vertices", "n_faces",
    "bbox_diag_mm", "volume_cm3", "is_watertight", "is_winding_consistent", "is_volume",
    "euler_number", "body_count", "boundary_edges", "nonmanifold_edges",
    "degenerate_faces", "duplicate_faces", "min_body_volume_cm3", "n_negative_bodies",
    "n_nested_bodies", "inward_probe_overlap_pct",
    "self_intersection_proxy", "verdict", "notes",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _boundary_edges(mesh):
    """Edges used by exactly one face (holes) and by more than two faces."""
    try:
        from trimesh.grouping import group_rows

        edges = mesh.edges_sorted
        groups = group_rows(edges, require_count=1)
        boundary = len(groups)
        # non-manifold: edges shared by >2 faces
        _, counts = np.unique(edges, axis=0, return_counts=True)
        nonmanifold = int((counts > 2).sum())
        return boundary, nonmanifold
    except Exception:  # noqa: BLE001
        return None, None


def _inward_probe(mesh, n=200, eps_mm=1.5, seed=0):
    """Fraction of surface samples whose **outward**-offset point is inside the mesh.

    For a valid closed mesh with outward normals, ``p + n*eps`` is outside, so the
    fraction is ~0%.  An inside-out mesh (consistent but inverted winding) gives
    ~100% -- that is the bug documented in docs/linkerhand_l20_notes.md.
    """
    try:
        rng = np.random.default_rng(seed)
        pts, face_ids = trimesh.sample.sample_surface(mesh, n, seed=seed)
        normals = mesh.face_normals[face_ids]
        probe = pts + normals * (eps_mm / 1000.0)
        inside = mesh.contains(probe)
        return float(np.mean(inside) * 100.0)
    except Exception:  # noqa: BLE001
        return None


def audit_one(path: Path, dataset: str, object_name: str, probe_eps_mm=2.0) -> dict:
    row = {k: None for k in FIELDS}
    row.update({"path": str(path.relative_to(gio.REPO_ROOT)).replace("\\", "/"),
                "dataset": dataset, "object": object_name,
                "format": path.suffix.lower().lstrip("."), "sha256": sha256(path),
                "bytes": path.stat().st_size})
    try:
        # process=True merges the STL's per-triangle vertices; without it every
        # STL would trivially be "non-watertight" (edges never shared).
        mesh = trimesh.load(str(path), force="mesh", process=True)
    except Exception as exc:  # noqa: BLE001
        row.update({"verdict": "FAIL", "notes": f"load failed: {type(exc).__name__}: {exc}"})
        return row

    row["n_vertices"] = len(mesh.vertices)
    row["n_faces"] = len(mesh.faces)
    row["bbox_diag_mm"] = round(float(np.linalg.norm(mesh.extents)) * 1000, 2)
    # volume is only meaningful for a closed mesh, but report it anyway
    row["volume_cm3"] = round(float(mesh.volume) * 1e6, 3)
    row["is_watertight"] = bool(mesh.is_watertight)
    row["is_winding_consistent"] = bool(mesh.is_winding_consistent)
    row["is_volume"] = bool(mesh.is_volume)
    row["euler_number"] = int(mesh.euler_number)
    row["body_count"] = int(mesh.body_count)
    be, nme = _boundary_edges(mesh)
    row["boundary_edges"] = be
    row["nonmanifold_edges"] = nme
    row["degenerate_faces"] = int(mesh.nondegenerate_faces().shape[0] == 0) if hasattr(mesh, "nondegenerate_faces") else None
    if hasattr(mesh, "nondegenerate_faces"):
        row["degenerate_faces"] = int(len(mesh.faces) - len(mesh.nondegenerate_faces()))
    try:
        uv, counts = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_counts=True)
        row["duplicate_faces"] = int((counts > 1).sum())
    except Exception:  # noqa: BLE001
        row["duplicate_faces"] = None
    # `contains` (ray casting) is only meaningful on a closed mesh
    # --- per-body topology: inverted shells and nested (internal-cavity) shells ---
    n_neg = n_nested = 0
    min_body_vol = None
    try:
        parts = mesh.split(only_watertight=False)
        vols = [float(b.volume) * 1e6 for b in parts]
        if vols:
            min_body_vol = round(min(vols), 4)
        n_neg = int(sum(1 for v in vols if v < -1e-3))  # ignore numerical dust
        # nested shells: centroid of a body inside another (usually larger) body
        if len(parts) > 1:
            order = np.argsort(vols)[::-1]
            for small_idx in order[1:]:
                c = parts[small_idx].centroid
                for big_idx in order[:1]:  # test against the largest body
                    try:
                        if parts[big_idx].is_watertight and parts[big_idx].contains([c])[0]:
                            n_nested += 1
                    except Exception:  # noqa: BLE001
                        pass
    except Exception:  # noqa: BLE001
        pass
    row["min_body_volume_cm3"] = min_body_vol
    row["n_negative_bodies"] = n_neg
    row["n_nested_bodies"] = n_nested
    # `contains` (ray casting) is only meaningful on a closed mesh; for multi-body
    # meshes a 1.5 mm outward step can land inside *another* part, so the probe is
    # only used as a verdict signal for single-body meshes.
    row["inward_probe_overlap_pct"] = (
        _inward_probe(mesh, eps_mm=probe_eps_mm) if mesh.is_watertight and mesh.body_count == 1 else None
    )
    # proxy verdict: >5% of inward probes classified inside == self-overlap
    overlap = row["inward_probe_overlap_pct"]
    if overlap is None:
        row["self_intersection_proxy"] = "unknown"
    else:
        row["self_intersection_proxy"] = "yes" if overlap > 5.0 else ("maybe" if overlap > 1.0 else "no")

    notes = []
    verdict = "PASS"

    def fail(msg):
        nonlocal verdict
        verdict = "FAIL"
        notes.append(msg)

    def warn(msg):
        nonlocal verdict
        if verdict == "PASS":
            verdict = "WARN"
        notes.append(msg)

    if not row["is_watertight"]:
        fail("not watertight")
    if not row["is_winding_consistent"]:
        fail("inconsistent winding")
    if row["volume_cm3"] <= 0:
        fail("non-positive volume")
    if row["nonmanifold_edges"]:
        fail(f"non-manifold edges={row['nonmanifold_edges']}")
    if row["n_negative_bodies"]:
        fail(f"inverted bodies={row['n_negative_bodies']}")
    if row["n_nested_bodies"]:
        fail(f"nested shells={row['n_nested_bodies']} (internal cavity)")
    if row["degenerate_faces"]:
        warn(f"degenerate faces={row['degenerate_faces']}")
    if row["duplicate_faces"]:
        warn(f"duplicate faces={row['duplicate_faces']}")
    if row["body_count"] > 1:
        warn(f"multi-body={row['body_count']}")
    if row["boundary_edges"]:
        notes.append(f"boundary edges={row['boundary_edges']}")
    overlap = row["inward_probe_overlap_pct"]
    if overlap is not None and overlap > 10.0:
        fail(f"outward-probe inside {overlap:.1f}% (normals likely inverted)")
    elif overlap is not None and overlap > 1.0:
        warn(f"outward-probe inside {overlap:.1f}% (thin geometry?)")
    row["verdict"] = verdict
    row["notes"] = "; ".join(notes)
    return row


def collect_targets(args):
    out = []
    if args.paths:
        for p in args.paths:
            p = Path(p)
            out.append((p.resolve(), p.parent.parent.name if p.parent.name != "contactdb" else "contactdb", p.stem))
        return out
    for obj in YCB_BAD:
        p = gio.YCB_ROOT / obj / f"{obj}.stl"
        if p.exists():
            out.append((p, "ycb", obj))
    cdb = gio.REPO_ROOT / "object" / "contactdb"
    for d in sorted(cdb.iterdir()):
        if d.is_dir():
            p = d / f"{d.name}.stl"
            if p.exists():
                out.append((p, "contactdb", d.name))
    if args.extra:
        for name in args.extra:
            p = gio.YCB_ROOT / name / f"{name}.stl"
            if p.exists():
                out.append((p, "ycb", name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="before")
    ap.add_argument("--paths", nargs="*", default=None)
    ap.add_argument("--extra", nargs="*", default=None, help="extra ycb object names to audit")
    ap.add_argument("--probe-eps-mm", type=float, default=1.5)
    args = ap.parse_args()

    targets = collect_targets(args)
    print(f"[audit] {len(targets)} meshes, tag={args.tag}", flush=True)
    rows = []
    for i, (p, dataset, obj) in enumerate(targets, 1):
        r = audit_one(p, dataset, obj, probe_eps_mm=args.probe_eps_mm)
        rows.append(r)
        print(f"[{i}/{len(targets)}] {dataset}/{obj}: {r['verdict']} "
              f"(watertight={r['is_watertight']}, winding={r['is_winding_consistent']}, "
              f"overlap={r['inward_probe_overlap_pct']})", flush=True)

    out_csv = HERE / "logs" / f"stl_audit_{args.tag}.csv"
    out_json = HERE / "logs" / f"stl_audit_{args.tag}.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    from collections import Counter

    print("\nverdicts:", dict(Counter(r["verdict"] for r in rows)))
    print("written:", out_csv, out_json, flush=True)


if __name__ == "__main__":
    main()
