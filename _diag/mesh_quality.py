"""Mesh quality (watertightness / winding / volume) with vertex merging.

The HandModel loads STLs with ``process=False`` which yields a vertex-triangle
"soup" (every triangle owns 3 private vertices). That makes trimesh report
watertight=False and euler==n_faces for ANY STL, so it cannot be interpreted
directly. This script merges duplicate vertices first so the topology numbers
are meaningful, and separately reports whether the *soup* is what warp sees.
"""

import argparse
import json
import os

import numpy as np
import trimesh as tm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "graspqp", "assets")


def analyse(path, label):
    raw = tm.load_mesh(path, process=False)
    m = tm.Trimesh(vertices=raw.vertices, faces=raw.faces, process=True)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    info = {
        "label": label,
        "n_tri": int(len(raw.faces)),
        "n_vert_soup": int(len(raw.vertices)),
        "n_vert_merged": int(len(m.vertices)),
        "watertight": bool(m.is_watertight),
        "winding_consistent": bool(m.is_winding_consistent),
        "euler": int(m.euler_number),
        "components": int(len(m.split(only_watertight=False))),
        "area_cm2": round(float(m.area) * 1e4, 3),
        "vol_signed_cm3": round(float(m.volume) * 1e6, 3),
        "bbox_min_mm": (m.bounds[0] * 1000).round(2).tolist(),
        "bbox_max_mm": (m.bounds[1] * 1000).round(2).tolist(),
        # fraction of sampled points that trimesh's own (watertight) containment test says are inside
        "dup_vert_frac": round(1.0 - len(m.vertices) / max(len(raw.vertices), 1), 4),
    }
    return info, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="linkerhand_l20")
    ap.add_argument("--contacts", action="store_true")
    args = ap.parse_args()

    adir = os.path.join(ASSETS, args.hand)
    meshdir = os.path.join(adir, "meshes")
    rows = []
    for f in sorted(os.listdir(meshdir)):
        if not f.lower().endswith((".stl", ".obj")):
            continue
        info, _ = analyse(os.path.join(meshdir, f), f"meshes/{f}")
        rows.append(info)
    if args.contacts:
        cdir = os.path.join(meshdir, "contacts")
        if os.path.isdir(cdir):
            for f in sorted(os.listdir(cdir)):
                if not f.lower().endswith((".stl", ".obj")):
                    continue
                info, _ = analyse(os.path.join(cdir, f), f"meshes/contacts/{f}")
                rows.append(info)

    hdr = f"{'file':44s} {'#tri':>7s} {'#vsoup':>8s} {'#vmerge':>8s} {'wt':>3s} {'wind':>5s} {'euler':>7s} {'comp':>4s} {'area':>8s} {'vol':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['label']:44s} {r['n_tri']:7d} {r['n_vert_soup']:8d} {r['n_vert_merged']:8d} "
            f"{str(r['watertight'])[0]:>3s} {str(r['winding_consistent'])[0]:>5s} {r['euler']:7d} "
            f"{r['components']:4d} {r['area_cm2']:8.2f} {r['vol_signed_cm3']:9.2f}"
        )
        print(f"    bbox mm {r['bbox_min_mm']} .. {r['bbox_max_mm']}")


if __name__ == "__main__":
    main()
