"""Scratch: render a 3-view scatter of object + posed hand links to a PNG."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grasp_io as gio

obj_name = sys.argv[1] if len(sys.argv) > 1 else "apple"
grasp_i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
f = gio.list_grasp_files(obj_name)[0]
g = gio.load_grasp_file(f)
urdf = gio.UrdfModel(gio.HAND_URDF)
meshes = {k: trimesh.load(v, force="mesh") for k, v in gio.hand_link_meshes().items()}
obj = trimesh.load(gio.object_mesh_path(obj_name), force="mesh")
T = gio.fk_batch(urdf, gio.build_full_angles(g["joints"][grasp_i : grasp_i + 1]))
rp = g["root_pose"][grasp_i]
Tr = np.eye(4)
Tr[:3, :3] = gio.quat_xyzw_to_matrix(rp[3:7])
Tr[:3, 3] = rp[:3]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
views = [(0, 1), (0, 2), (1, 2)]
ov = obj.vertices[:: max(1, len(obj.vertices) // 3000)]
axes[0].scatter(ov[:, 0], ov[:, 1], s=1, c="k")
axes[1].scatter(ov[:, 0], ov[:, 2], s=1, c="k")
axes[2].scatter(ov[:, 1], ov[:, 2], s=1, c="k")
cmap = plt.get_cmap("tab20")
for j, (name, Tm) in enumerate(T.items()):
    if name not in meshes:
        continue
    W = Tr @ Tm[0]
    V = meshes[name].vertices[:: max(1, len(meshes[name].vertices) // 150)]
    V = V @ W[:3, :3].T + W[:3, 3]
    c = cmap(j % 20)
    for ax, (a, b) in zip(axes, views):
        ax.scatter(V[:, a], V[:, b], s=2, c=[c], alpha=0.6)
    axes[0].annotate(name, (W[0, 3], W[1, 3]), fontsize=5)
for ax, (a, b) in zip(axes, views):
    ax.set_aspect("equal")
    ax.set_xlabel("xyz"[a])
    ax.set_ylabel("xyz"[b])
    ax.set_title(f"{'xyz'[a]}-{'xyz'[b]}")
fig.suptitle(f"{obj_name} grasp {grasp_i} E={g['values'][grasp_i]:.3f} t={np.round(rp[:3],3)}")
out = Path(__file__).resolve().parent / f"pose_{obj_name}_{grasp_i}.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
print("hand pts range", np.round(ov.min(0), 3), np.round(ov.max(0), 3))
