"""Headless geometry report for a hand asset directory (CPU only, no plotly, no browser).

Usage:
    python _diag/geo_report.py --hand linkerhand_l20
    python _diag/geo_report.py --hand allegro
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh as tm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))

ASSETS = os.path.join(ROOT, "graspqp", "assets")


def mesh_stats(m, name):
    st = {
        "name": name,
        "n_verts": int(len(m.vertices)),
        "n_faces": int(len(m.faces)),
        "area_cm2": float(m.area) * 1e4,
        "volume_cm3": float(m.volume) * 1e6 if m.is_watertight else float("nan"),
        "watertight": bool(m.is_watertight),
        "winding_consistent": bool(m.is_winding_consistent),
        "euler": int(m.euler_number),
        "components": int(len(m.split(only_watertight=False))),
        "bbox_min_mm": (m.bounds[0] * 1000).round(2).tolist(),
        "bbox_max_mm": (m.bounds[1] * 1000).round(2).tolist(),
        "bbox_extent_mm": ((m.bounds[1] - m.bounds[0]) * 1000).round(2).tolist(),
    }
    # volume of convex hull as a proxy for "closed-ish"
    try:
        st["hull_volume_cm3"] = float(m.convex_hull.volume) * 1e6
    except Exception:
        st["hull_volume_cm3"] = float("nan")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="linkerhand_l20")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    adir = os.path.join(ASSETS, args.hand)
    meshdir = os.path.join(adir, "meshes")
    coldir = os.path.join(adir, "collisions")

    print(f"### hand = {args.hand}  ({adir})")
    print()

    # ---------------- urdf structure ----------------
    import xml.etree.ElementTree as ET

    urdf_path = None
    for f in os.listdir(adir):
        if f.endswith(".urdf"):
            urdf_path = os.path.join(adir, f)
    r = ET.parse(urdf_path).getroot()
    print(f"URDF: {os.path.basename(urdf_path)}")
    print()

    report = {}

    print("=== per-link geometry ===")
    hdr = f"{'link':26s} {'kind':8s} {'file':46s} {'#v':>7s} {'#f':>7s} {'area cm2':>9s} {'wt':>3s} {'euler':>6s} extent_mm"
    print(hdr)
    print("-" * len(hdr))
    for l in r.findall("link"):
        lname = l.get("name")
        report[lname] = {}
        for tag, kind in (("visual", "visual"), ("collision", "collision")):
            for g in l.findall(tag):
                gtag = g.find("geometry")[0].tag
                if gtag != "mesh":
                    print(f"{lname:26s} {kind:8s} {gtag}")
                    continue
                fn = g.find("geometry/mesh").get("filename")
                scale = g.find("geometry/mesh").get("scale")
                # resolve the way HandModel does: mesh_path / filename, else basename
                res = None
                for cand in (
                    os.path.join(meshdir, fn),
                    os.path.join(meshdir, os.path.basename(fn)),
                    os.path.join(adir, fn),
                ):
                    if os.path.exists(cand):
                        res = cand
                        break
                if res is None:
                    print(f"{lname:26s} {kind:8s} {fn:46s}  !! NOT FOUND")
                    continue
                m = tm.load_mesh(res, process=False)
                st = mesh_stats(m, lname)
                st["resolved"] = os.path.relpath(res, adir)
                st["urdf_name"] = fn
                st["scale"] = scale
                report[lname][kind] = st
                rel = os.path.relpath(res, adir)
                flag = "" if rel.startswith(("meshes/" + os.path.basename(fn), "collisions/")) else "  <-- FALLBACK"
                note = ""
                if kind == "collision" and fn.startswith("collisions/") and not os.path.exists(
                    os.path.join(meshdir, fn)
                ):
                    note = "  <== URDF points to collisions/, resolved to a different file"
                print(
                    f"{lname:26s} {kind:8s} {rel:46s} {st['n_verts']:7d} {st['n_faces']:7d} "
                    f"{st['area_cm2']:9.2f} {str(st['watertight'])[0]:>3s} {st['euler']:6d} "
                    f"{st['bbox_extent_mm']}{note}"
                )

    # ---------------- contact meshes ----------------
    cp = os.path.join(adir, "contact_points.json")
    print()
    print("=== contact_points.json ===")
    if os.path.exists(cp):
        cj = json.load(open(cp))
        print(f"{'link':26s} {'file':46s} {'n_req':>6s} {'#v':>7s} {'#f':>7s} {'area cm2':>9s} extent_mm")
        tot = 0
        for ln, entries in cj.items():
            for e in entries:
                f, n = e[0], e[1]
                tot += n
                p = os.path.join(meshdir, f)
                if not os.path.exists(p):
                    print(f"{ln:26s} {f:46s} {n:6d}  !! NOT FOUND")
                    continue
                m = tm.load_mesh(p, process=False)
                print(
                    f"{ln:26s} {f:46s} {n:6d} {len(m.vertices):7d} {len(m.faces):7d} "
                    f"{m.area*1e4:9.2f} {((m.bounds[1]-m.bounds[0])*1000).round(2).tolist()}"
                )
        print(f"total requested contact candidates = {tot}")
    else:
        print("MISSING")

    # ---------------- penetration points ----------------
    pp = os.path.join(adir, "penetration_points.json")
    print()
    print("=== penetration_points.json ===")
    if os.path.exists(pp):
        pj = json.load(open(pp))
        ntot = 0
        for ln, entries in pj.items():
            arr = np.array(entries, dtype=float)
            ntot += len(arr)
            r_ = arr[:, 3] if arr.shape[1] == 4 else np.full(len(arr), np.nan)
            print(
                f"{ln:26s} n={len(arr):3d} r=[{np.nanmin(r_)*1000:.2f},{np.nanmax(r_)*1000:.2f}]mm "
                f"span_cm={((arr[:, :3].max(0)-arr[:, :3].min(0))*100).round(2).tolist()}"
            )
        print(f"total penetration spheres = {ntot}")
    else:
        print("MISSING")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
