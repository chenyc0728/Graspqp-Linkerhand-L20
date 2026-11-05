"""Scratch: does applying grasp_velocities close the fingers onto the object?"""
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
vel_off = np.stack([raw["grasp_velocities_off"][n].numpy() for n in gio.DRIVE_NAMES], -1).astype(np.float64)
vel_full = np.stack([raw["full_grasp_velocities"][n].numpy() for n in gio.DRIVE_NAMES], -1).astype(np.float64)

urdf = gio.UrdfModel(gio.HAND_URDF)
links = gio.hand_link_meshes()
keep = ["index_distal", "middle_distal", "ring_distal", "pinky_distal", "thumb_distal", "hand_base_link"]
meshes = {k: trimesh.load(links[k], force="mesh") for k in keep}
sub = {k: m.vertices[:: max(1, len(m.vertices) // 200)] for k, m in meshes.items()}
obj = trimesh.load(gio.object_mesh_path(obj_name), force="mesh")

idx = np.argsort(g["values"])[:6]


def mindist(joints, i):
    T = gio.fk_batch(urdf, gio.build_full_angles(joints[i : i + 1]))
    rp = g["root_pose"][i]
    Tr = np.eye(4)
    Tr[:3, :3] = gio.quat_xyzw_to_matrix(rp[3:7])
    Tr[:3, 3] = rp[:3]
    best, who = 1e9, ""
    for name, Tm in T.items():
        if name not in sub:
            continue
        W = Tr @ Tm[0]
        V = sub[name] @ W[:3, :3].T + W[:3, 3]
        d, _, _ = trimesh.proximity.closest_point(obj, V)
        m = float(d.min())
        if m < best:
            best, who = m, name
    return best * 1000, who


for i in idx:
    a = mindist(g["joints"], i)
    b = mindist(g["joints"] + vel, i)
    c = mindist(g["joints"] + vel_full, i)
    d = mindist(g["joints"] + vel_off, i)
    print(
        f"grasp {i} E={g['values'][i]:.3f} | saved {a[0]:6.2f}mm | +grasp_vel {b[0]:6.2f}mm({b[1]}) "
        f"| +full {c[0]:6.2f}mm | +off {d[0]:6.2f}mm",
        flush=True,
    )
