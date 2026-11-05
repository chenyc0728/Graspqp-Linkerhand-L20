"""Scratch: determine the root_pose quaternion/frame convention empirically."""
import itertools
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grasp_io as gio

obj_name = sys.argv[1] if len(sys.argv) > 1 else "apple"
g = gio.load_grasp_file(gio.list_grasp_files(obj_name)[0])
urdf = gio.UrdfModel(gio.HAND_URDF)
meshes = {k: trimesh.load(v, force="mesh") for k, v in gio.hand_link_meshes().items()}
sub = {k: m.vertices[:: max(1, len(m.vertices) // 250)] for k, m in meshes.items()}
obj = trimesh.load(gio.object_mesh_path(obj_name), force="mesh")
pq = trimesh.proximity.ProximityQuery(obj)
T = gio.fk_batch(urdf, gio.build_full_angles(g["joints"]))
idx = np.argsort(g["values"])[:6]
print("grasp idx", idx, "energies", np.round(g["values"][idx], 3), flush=True)

FIXED = [0, 1, 2, 3]  # identity permutation slot mapping


def score(perm, transpose=False):
    ds = []
    for i in idx:
        rp = g["root_pose"][i]
        q = rp[3:7][list(perm)]
        R = gio.quat_xyzw_to_matrix(q)
        if transpose:
            R = R.T
        Tr = np.eye(4)
        Tr[:3, :3] = R
        Tr[:3, 3] = rp[:3]
        best = 1e9
        for name, Tm in T.items():
            if name not in sub:
                continue
            W = Tr @ Tm[i]
            V = sub[name] @ W[:3, :3].T + W[:3, 3]
            d = pq.signed_distance(V)
            best = min(best, float(np.min(np.abs(d))))
        ds.append(best)
    return float(np.mean(ds)), float(np.max(ds))


results = []
for perm in itertools.permutations([0, 1, 2, 3]):
    for tr in (False, True):
        m, mx = score(perm, tr)
        results.append((m, mx, perm, tr))
results.sort()
for m, mx, perm, tr in results[:12]:
    print(f"mean|d|={m*1000:7.2f}mm max={mx*1000:7.2f}mm perm={perm} R_T={tr}", flush=True)
