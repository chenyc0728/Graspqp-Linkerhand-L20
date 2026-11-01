"""Compare the warp hand-link SDF (exactly as HandModel builds it) against trimesh.

This is the headless version of `visualize_hand_model.py --show_occupancy_grid`:
it reports, per link, whether the SDF's inside/outside classification agrees with
trimesh's containment test, and where it does not.

Usage: CUDA_VISIBLE_DEVICES=0 python _diag/sdf_check.py --hand linkerhand_l20
"""

import argparse
import os
import sys

import numpy as np
import torch
import trimesh as tm
import warp as wp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))

from graspqp.utils import warp as wp_utils  # noqa: E402


def build_wp_mesh(verts_np, faces_np, device="cuda"):
    v = wp.from_numpy(np.ascontiguousarray(verts_np.astype(np.float32)), device=device, dtype=wp.vec3)
    f = wp.from_numpy(np.ascontiguousarray(faces_np.astype(np.int32).flatten()), device=device, dtype=wp.int32)
    return wp.Mesh(points=v, indices=f, support_winding_number=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="linkerhand_l20")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_query", type=int, default=20000)
    ap.add_argument("--links", default=None, help="comma separated subset")
    ap.add_argument("--wn", action="store_true", help="use the winding-number sign instead of the pseudo-normal")
    args = ap.parse_args()

    adir = os.path.join(ROOT, "graspqp", "assets", args.hand)
    meshdir = os.path.join(adir, "meshes")
    import xml.etree.ElementTree as ET

    urdf = [f for f in os.listdir(adir) if f.endswith(".urdf")][0]
    root = ET.parse(os.path.join(adir, urdf)).getroot()

    rng = np.random.default_rng(0)
    hdr = (f"{'link':26s} {'#f':>6s} {'method':>9s} {'agree':>7s} {'out_as_in':>9s} {'in_as_out':>9s} "
           f"{'max|d| err':>10s} {'worst dist':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for l in root.findall("link"):
        lname = l.get("name")
        if args.links and lname not in args.links.split(","):
            continue
        cols = l.findall("collision")
        if not cols:
            continue
        fn = cols[0].find("geometry/mesh").get("filename")
        p = os.path.join(meshdir, fn)
        if not os.path.exists(p):
            p = os.path.join(meshdir, os.path.basename(fn))
        raw = tm.load_mesh(p, process=False)
        verts, faces = np.array(raw.vertices, dtype=np.float32), np.array(raw.faces, dtype=np.int64)
        m_ref = tm.Trimesh(verts, faces, process=True)
        m_ref.merge_vertices()

        lo, hi = m_ref.bounds
        ext = hi - lo
        # query points: uniform in a box inflated by 60% of the extent, plus near-surface points
        pts = lo + (hi - lo + 0.6 * ext) * rng.random((args.n_query // 2, 3)) - 0.3 * ext
        near = m_ref.sample(args.n_query // 2)
        nrm = m_ref.face_normals[rng.integers(0, len(m_ref.faces), len(near))]
        near = near + nrm * rng.uniform(-0.06 * ext.max(), 0.06 * ext.max(), (len(near), 1))
        pts = np.concatenate([pts, near], 0).astype(np.float32)

        wp_mesh = build_wp_mesh(verts, faces, args.device)
        pts_wp = wp.from_torch(torch.tensor(pts, device=args.device), dtype=wp.vec3)
        d, n, g = wp_utils.calc_sdf_field(pts_wp, wp_mesh.id, 1e6, args.wn)
        d = wp.to_torch(d).cpu().numpy()

        _, tridist, _ = tm.proximity.closest_point(m_ref, pts)
        inside_ref = m_ref.contains(pts)
        inside_warp = d < 0

        agree = float((inside_ref == inside_warp).mean())
        out_as_in = float(((~inside_ref) & inside_warp).mean())
        in_as_out = float((inside_ref & (~inside_warp)).mean())
        derr = float(np.abs(np.abs(d) - tridist).max())
        # where does it break?
        bad = (~inside_ref) & inside_warp
        worst = float(np.abs(d[bad]).max()) if bad.any() else 0.0
        print(
            f"{lname:26s} {len(faces):6d} {('winding' if args.wn else 'pseudo'):>9s} {agree*100:6.2f}% "
            f"{out_as_in*100:8.2f}% {in_as_out*100:8.2f}% {derr*1000:10.2f} {worst*1000:10.2f}"
        )


if __name__ == "__main__":
    main()
