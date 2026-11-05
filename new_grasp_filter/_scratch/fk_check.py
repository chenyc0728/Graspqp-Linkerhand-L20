"""Scratch: validate FK / pose conventions against the object surface (fast)."""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grasp_io as gio

obj_name = sys.argv[1] if len(sys.argv) > 1 else "apple"
f = gio.list_grasp_files(obj_name)[0]
print("file", f, flush=True)
g = gio.load_grasp_file(f)
print("joints", g["joints"].shape, "root", g["root_pose"].shape, "values", g["values"].shape, flush=True)
print("values sorted:", np.round(np.sort(g["values"])[:6], 3), flush=True)

urdf = gio.UrdfModel(gio.HAND_URDF)
links = gio.hand_link_meshes()
meshes = {k: trimesh.load(v, force="mesh") for k, v in links.items()}
obj = trimesh.load(gio.object_mesh_path(obj_name), force="mesh")
print("object", gio.object_mesh_path(obj_name), obj.extents, flush=True)

full = gio.build_full_angles(g["joints"])
T = gio.fk_batch(urdf, full)
print("fk links", len(T), flush=True)

# subsample hand mesh vertices once
sub = {k: m.vertices[:: max(1, len(m.vertices) // 300)] for k, m in meshes.items()}
query = trimesh.proximity.ProximityQuery(obj)


def eval_mode(mode, n=8):
    rows = []
    for i in range(min(n, len(g["values"]))):
        rp = g["root_pose"][i]
        qi = rp[3:7] if mode == "xyzw" else rp[[6, 3, 4, 5]]
        R = gio.quat_xyzw_to_matrix(qi)
        Troot = np.eye(4)
        Troot[:3, :3] = R
        Troot[:3, 3] = rp[:3]
        dmin = 1e9
        npen = 0
        for name, Tm in T.items():
            if name not in sub:
                continue
            W = Troot @ Tm[i]
            V = sub[name] @ W[:3, :3].T + W[:3, 3]
            d = query.signed_distance(V)  # positive inside
            dmin = min(dmin, float(-np.min(d)))
            if (d > 0.0005).mean() > 0.02:
                npen += 1
        rows.append((round(dmin * 1000, 2), npen))
    return rows


for mode in ["xyzw", "wxyz"]:
    print(mode, eval_mode(mode), flush=True)
