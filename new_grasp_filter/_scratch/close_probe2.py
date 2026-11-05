"""Scratch: does applying the stored closing velocities improve hand-object contact?"""
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grasp_io as gio

obj_name = sys.argv[1] if len(sys.argv) > 1 else "apple"
f = gio.list_grasp_files(obj_name)[0]
g = gio.load_grasp_file(f)
raw = torch.load(f, map_location="cpu", weights_only=False)
vel = np.stack([raw["grasp_velocities"][n].numpy() for n in gio.DRIVE_NAMES], -1).astype(np.float64)
vel_full = np.stack([raw["full_grasp_velocities"][n].numpy() for n in gio.DRIVE_NAMES], -1).astype(np.float64)

urdf = gio.UrdfModel(gio.HAND_URDF)
links = gio.hand_link_meshes()
meshes = {k: trimesh.load(v, force="mesh") for k, v in links.items()}
sub = {k: m.vertices[:: max(1, len(m.vertices) // 200)] for k, m in meshes.items()}
obj = trimesh.load(gio.object_mesh_path(obj_name), force="mesh")

idx = np.argsort(g["values"])[:6]


def stats(joints, i):
    T = gio.fk_batch(urdf, gio.build_full_angles(joints[i : i + 1]))
    rp = g["root_pose"][i]
    Tr = np.eye(4)
    Tr[:3, :3] = gio.quat_xyzw_to_matrix(rp[3:7])
    Tr[:3, 3] = rp[:3]
    mins = {}
    for name, Tm in T.items():
        if name not in sub:
            continue
        W = Tr @ Tm[0]
        V = sub[name] @ W[:3, :3].T + W[:3, 3]
        d = trimesh.proximity.closest_point(obj, V)[1]
        mins[name] = float(d.min())
    near = [k for k, v in mins.items() if v < 0.003]
    return min(mins.values()) * 1000, len(near), near


for i in idx:
    a = stats(g["joints"], i)
    b = stats(g["joints"] + vel, i)
    c = stats(g["joints"] + vel_full, i)
    print(f"grasp {i:2d} E={g['values'][i]:.2f} | saved {a[0]:6.2f}mm near={a[1]:2d} {a[2]}")
    print(f"                       +vel  {b[0]:6.2f}mm near={b[1]:2d} {b[2]}")
    print(f"                       +full {c[0]:6.2f}mm near={c[1]:2d} {c[2]}")
