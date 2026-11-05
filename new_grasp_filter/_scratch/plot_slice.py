"""Scratch: cross-section plot (|y|<4mm) of object surface + hand links."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grasp_io as gio
from sapien_grasp_checker import SapienGraspChecker, CheckerConfig

obj = sys.argv[1] if len(sys.argv) > 1 else "apple"
gi = int(sys.argv[2]) if len(sys.argv) > 2 else 15
chk = SapienGraspChecker(obj, CheckerConfig())
g = gio.load_grasp_file(gio.list_grasp_files(obj)[0])
M = chk._pose_matrices(g["root_pose"][gi], g["joints"][gi])

objv = chk.obj_mesh.vertices
sel = np.abs(objv[:, 1]) < 0.004
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (a, b, name) in zip(axes, [(0, 2, "x-z"), (0, 1, "x-y")]):
    ax.scatter(objv[sel, a], objv[sel, b], s=3, c="k", label="object (|y|<4mm)")
    cmap = plt.get_cmap("tab20")
    for j, (lname, W) in enumerate(M.items()):
        V = chk._link_vertices[lname] @ W[:3, :3].T + W[:3, 3]
        if name == "x-y":
            keep = np.abs(V[:, 2]) < 0.004
        else:
            keep = np.abs(V[:, 1]) < 0.004
        if keep.sum() == 0:
            continue
        ax.scatter(V[keep, a], V[keep, b], s=4, c=[cmap(j % 20)], alpha=0.8)
        ax.annotate(lname, (V[keep, a].mean(), V[keep, b].mean()), fontsize=5)
    ax.set_aspect("equal")
    ax.set_title(f"{name} cross-section")
    ax.grid(alpha=0.2)
fig.suptitle(f"{obj} grasp {gi} E={g['values'][gi]:.3f}")
out = Path(__file__).resolve().parent / f"slice_{obj}_{gi}.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
