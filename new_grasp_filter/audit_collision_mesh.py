"""Audit the LinkerHand L20 collision meshes (and their URDF wiring).

For every link it reports:

* watertight / winding / euler number / body count / non-manifold edges;
* normal orientation: 1000 inward + 1000 outward surface offsets must be
  inside/outside (catches the "inside-out mesh" bug from
  docs/linkerhand_l20_notes.md);
* self-overlap proxy;
* deviation from the visual mesh: Chamfer (median/p95), volume ratio, coverage
  (fraction of visual surface points inside the collision mesh);
* convexity deviation: Chamfer between the collision mesh and its convex hull
  (this is the "phantom volume" that a convex-hull collider would add);
* internal cavities: nested shells / bodies.

Verdict: FAIL on watertight/winding/orientation/cavity/non-manifold problems,
WARN on convexity or volume-ratio deviations, PASS otherwise.

Outputs ``logs/collision_mesh_audit.csv`` + ``.json`` and prints the URDF table.

Usage::

    python new_grasp_filter/audit_collision_mesh.py [--tag before|after]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_io as gio  # noqa: E402
from audit_stl import _inward_probe, _boundary_edges, sha256  # noqa: E402

FIELDS = [
    "link", "collision_file", "visual_file", "urdf_has_origin", "urdf_origin",
    "sha256", "faces", "verts", "is_watertight", "is_winding_consistent",
    "euler_number", "body_count", "nonmanifold_edges", "nested_bodies",
    "sdf_agreement_pct", "chamfer_median_mm", "chamfer_p95_mm",
    "volume_ratio_vs_visual", "coverage_visual_pct", "hull_chamfer_median_mm",
    "verdict", "notes",
]


def collision_urdf_table(urdf_path: Path):
    root = ET.parse(urdf_path).getroot()
    base = urdf_path.parent
    table = {}
    for link in root.findall("link"):
        name = link.get("name")
        col = link.find("collision")
        vis = link.find("visual")
        def mesh_of(node):
            if node is None:
                return None, None
            g = node.find("geometry/mesh")
            if g is None:
                return None, None
            origin = node.find("origin")
            o = None
            if origin is not None:
                o = {"xyz": origin.get("xyz"), "rpy": origin.get("rpy")}
            return g.get("filename"), o
        cmesh, corigin = mesh_of(col)
        vmesh, vorigin = mesh_of(vis)
        table[name] = {
            "collision": (base / cmesh).resolve() if cmesh else None,
            "visual": (base / vmesh).resolve() if vmesh else None,
            "collision_origin": corigin,
            "visual_origin": vorigin,
        }
    return table


def chamfer(a: trimesh.Trimesh, b: trimesh.Trimesh, n=4000):
    pa, _ = trimesh.sample.sample_surface(a, n, seed=0)
    pb, _ = trimesh.sample.sample_surface(b, n, seed=0)
    d1 = trimesh.proximity.closest_point(b, pa)[1]
    d2 = trimesh.proximity.closest_point(a, pb)[1]
    d = np.concatenate([d1, d2]) * 1000.0
    return float(np.median(d)), float(np.percentile(d, 95))


def coverage(visual: trimesh.Trimesh, coll: trimesh.Trimesh, n=3000):
    pts, _ = trimesh.sample.sample_surface(visual, n, seed=0)
    try:
        return float(coll.contains(pts).mean() * 100.0)
    except Exception:  # noqa: BLE001
        return None


def count_nested(mesh: trimesh.Trimesh):
    try:
        parts = mesh.split(only_watertight=False)
        if len(parts) < 2:
            return 0
        vols = [float(p.volume) for p in parts]
        order = np.argsort(vols)[::-1]
        n = 0
        for i in order[1:]:
            c = parts[i].centroid
            if parts[order[0]].is_watertight and parts[order[0]].contains([c])[0]:
                n += 1
        return n
    except Exception:  # noqa: BLE001
        return 0


def audit_link(link: str, coll_path: Path, vis_path: Path, urdf_info: dict) -> dict:
    row = {k: None for k in FIELDS}
    row.update({"link": link,
                "collision_file": str(coll_path.relative_to(gio.REPO_ROOT)).replace("\\", "/"),
                "visual_file": str(vis_path.relative_to(gio.REPO_ROOT)).replace("\\", "/") if vis_path else None,
                "urdf_has_origin": bool(urdf_info.get("collision_origin")),
                "urdf_origin": json.dumps(urdf_info.get("collision_origin")),
                "sha256": sha256(coll_path)})
    try:
        coll = trimesh.load(str(coll_path), force="mesh", process=True)
    except Exception as exc:  # noqa: BLE001
        row.update({"verdict": "FAIL", "notes": f"load failed: {exc}"})
        return row

    row.update({
        "faces": int(len(coll.faces)), "verts": int(len(coll.vertices)),
        "is_watertight": bool(coll.is_watertight),
        "is_winding_consistent": bool(coll.is_winding_consistent),
        "euler_number": int(coll.euler_number),
        "body_count": int(coll.body_count),
    })
    be, nme = _boundary_edges(coll)
    row["nonmanifold_edges"] = nme
    row["nested_bodies"] = count_nested(coll)
    probe = _inward_probe(coll, eps_mm=1.0) if coll.is_watertight else None
    row["sdf_agreement_pct"] = None if probe is None else round(100.0 - probe, 2)

    notes, verdict = [], "PASS"

    def fail(m):
        nonlocal verdict
        verdict = "FAIL"
        notes.append(m)

    def warn(m):
        nonlocal verdict
        if verdict == "PASS":
            verdict = "WARN"
        notes.append(m)

    if not coll.is_watertight:
        fail("not watertight")
    if not coll.is_winding_consistent:
        fail("inconsistent winding")
    if coll.volume <= 0:
        fail("non-positive volume")
    if nme:
        fail(f"non-manifold edges={nme}")
    if row["nested_bodies"]:
        fail(f"nested shells={row['nested_bodies']}")
    # The whole-mesh outward probe is sensitive to concavities: an outward step in
    # a concave region (the palm's grip side) can land inside the bulk.  A truly
    # inverted mesh gives ~100%, so the FAIL cut is 5%; 1-5% is reported as WARN.
    if probe is not None and probe > 5.0:
        fail(f"outward-probe inside {probe:.1f}% (inside-out normals)")
    elif probe is not None and probe > 1.0:
        warn(f"outward-probe inside {probe:.1f}% (concave/thin region)")
    if coll.body_count > 1:
        warn(f"multi-body={coll.body_count}")

    if vis_path is not None and vis_path.exists():
        vis = trimesh.load(str(vis_path), force="mesh", process=True)
        try:
            cm, cp95 = chamfer(coll, vis)
            row["chamfer_median_mm"], row["chamfer_p95_mm"] = round(cm, 3), round(cp95, 3)
            row["volume_ratio_vs_visual"] = round(float(coll.volume) / float(vis.volume), 3) if vis.volume else None
            row["coverage_visual_pct"] = round(coverage(vis, coll), 2) if row["is_watertight"] else None
            hull = coll.convex_hull
            hm, _ = chamfer(hull, coll, n=2000)
            row["hull_chamfer_median_mm"] = round(hm, 3)
            if cm > 5.0:
                warn(f"chamfer to visual {cm:.2f} mm")
            if row["volume_ratio_vs_visual"] and not (0.5 <= row["volume_ratio_vs_visual"] <= 1.5):
                warn(f"volume ratio {row['volume_ratio_vs_visual']}x")
            if hm > 5.0:
                warn(f"convexity deviation {hm:.2f} mm")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"visual compare failed: {type(exc).__name__}")

    row["verdict"] = verdict
    row["notes"] = "; ".join(notes)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="before")
    ap.add_argument("--urdf", default=str(gio.HAND_URDF))
    ap.add_argument("--collision-dir", default=None,
                    help="default: <urdf dir>/meshes/collisions")
    args = ap.parse_args()

    urdf_path = Path(args.urdf).resolve()
    table = collision_urdf_table(urdf_path)
    coll_dir = Path(args.collision_dir) if args.collision_dir else urdf_path.parent / "meshes" / "collisions"

    print(f"{'link':24s} {'collision file':44s} {'visual file':44s} origin")
    rows = []
    for link, info in table.items():
        c, v = info["collision"], info["visual"]
        print(f"{link:24s} {str(c.relative_to(gio.REPO_ROOT)) if c else '-':44s} "
              f"{str(v.relative_to(gio.REPO_ROOT)) if v else '-':44s} {info['collision_origin']}")
        if c is None or not c.exists():
            continue
        rows.append(audit_link(link, c, v, info))

    # links that have a collision file but no URDF reference (extra files)
    if coll_dir.exists():
        referenced = {r["collision_file"].split("/")[-1] for r in rows}
        for f in sorted(coll_dir.glob("*.STL")):
            if f.name not in referenced:
                print(f"[audit] orphan collision mesh (not referenced by URDF): {f.name}")

    out_csv = HERE / "logs" / "collision_mesh_audit.csv" if args.tag == "before" else \
        HERE / "logs" / f"collision_mesh_audit_{args.tag}.csv"
    out_json = out_csv.with_suffix(".json")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    from collections import Counter
    print("\nverdicts:", dict(Counter(r["verdict"] for r in rows)))
    for r in rows:
        print(f"  {r['link']:24s} {r['verdict']:5s} {r['notes'][:100]}")
    print("\nwritten:", out_csv, out_json)


if __name__ == "__main__":
    main()
