"""Re-check the repaired scratch outputs with a thinner SDF probe (0.5 mm) and
smaller sample counts.  Prints one row per mesh so we can separate real failures
from thin-wall artifacts of the 3 mm probe.
"""

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair_stls import object_path, load_mesh  # noqa: E402

FAILED = ["ycb/tomato_soup_can", "ycb/tuna_fish_can", "contactdb/bowl", "contactdb/flashlight",
          "contactdb/hand", "contactdb/knife", "contactdb/mouse", "contactdb/pan",
          "contactdb/stapler", "contactdb/toothpaste", "contactdb/train",
          "contactdb/utah_teapot", "contactdb/wristwatch"]


def sdf_agreement(m, eps_mm, n=300):
    pts, fid = trimesh.sample.sample_surface(m, n, seed=1)
    nrm = m.face_normals[fid]
    off = nrm * (eps_mm / 1000.0)
    a = float(np.mean(m.contains(pts - off)))
    b = float(np.mean(~m.contains(pts + off)))
    return (a + b) / 2 * 100


print(f"{'mesh':26s} {'wt':3s} {'wind':4s} {'vol%':>8s} {'chamfer':>8s} {'sdf0.5':>7s} {'eps3.0':>7s}")
for name in FAILED:
    ds, obj = name.split("/", 1)
    src = object_path(ds, obj)
    out = Path("new_grasp_filter/_blender_probe/repair_out") / f"{ds}_{obj}.stl"
    if not out.exists():
        print(f"{name:26s} missing scratch output")
        continue
    o, n = load_mesh(src), load_mesh(out)
    vol = (n.volume / o.volume - 1) * 100 if o.volume else float("nan")
    ch = trimesh.proximity.closest_point(n, trimesh.sample.sample_surface(o, 500, seed=0)[0])[1]
    print(f"{name:26s} {str(n.is_watertight):3s} {str(n.is_winding_consistent):4s} "
          f"{vol:8.2f} {np.median(ch)*1000:8.3f} {sdf_agreement(n, 0.5):7.1f} {sdf_agreement(n, 3.0):7.1f}",
          flush=True)
