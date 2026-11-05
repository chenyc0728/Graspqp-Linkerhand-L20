"""Inspect filtered grasps: object + posed left/right L20 hand + contacts + penetration.

Two backends (``--backend``):

* ``open3d`` (default): lightweight geometry only, no physics.  Interactive window
  with keys ``n`` (next), ``p`` (previous), ``s`` (save PNG), ``q`` (quit).
* ``sapien``: builds a SAPIEN scene with render meshes, same key bindings.

``--offscreen`` skips the window and writes one PNG per grasp (for headless /
Ubuntu), using a matplotlib 3D renderer that needs no GPU or display.

Single grasp::

    python new_grasp_filter/visualize_filtered.py \
        --grasp-file object1/ycb/apple/grasp_predictions/linkerhand_l20/12_contacts/graspqp/default/apple.dexgrasp.pt \
        --index 3 --backend open3d

Batch (all feasible grasps of one object from the merged index)::

    python new_grasp_filter/visualize_filtered.py --index-csv new_grasp_filter/filtered_grasps/index.csv \
        --object apple --source object1 --backend open3d

Keyboard: ``n``/``p`` browse, ``s`` screenshot, ``q`` quit.  Screenshots go to
``new_grasp_filter/vis_output/``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_io as gio  # noqa: E402
from object_sdf import VoxelSDF  # noqa: E402

VIS_DIR = HERE / "vis_output"
PEN_THRESHOLD_MM = 5.0


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass
class GraspItem:
    path: Path
    index: int
    object: str
    source: str = ""
    verdict: str = ""
    reason: str = ""
    metrics: Optional[dict] = None

    @property
    def label(self) -> str:
        return f"{self.object} | {self.path.name} | idx {self.index} | {self.verdict or 'n/a'}"


def items_from_index(index_csv: Path, object_name: str, source: str = "") -> List[GraspItem]:
    with open(index_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r["object"] != object_name:
            continue
        if source and r.get("source", "") != source:
            continue
        feas = [int(x) for x in (r.get("feasible_indices") or "").split()]
        path = index_csv.parent / r["grasp_file"]
        for i in feas:
            out.append(GraspItem(path=path, index=i, object=object_name,
                                 source=r.get("source", ""), verdict=r.get("verdict", ""),
                                 reason=(r.get("reason_counts") or "")[:80]))
    return out


def load_pose(item: GraspItem):
    g = gio.load_grasp_file(item.path)
    return g["root_pose"][item.index], g["joints"][item.index], float(g["values"][item.index])


def hand_mesh_paths(robot_name: str) -> Dict[str, Path]:
    if robot_name == "linkerhand_right":
        return gio.hand_link_meshes(gio.CMAP_RIGHT_URDF)
    return gio.hand_link_meshes(gio.HAND_URDF)


def compose_hand(root_pose: np.ndarray, joints: np.ndarray, robot_name: str = "linkerhand"):
    """World transforms for every hand link (numpy dict), in the GraspQP root frame."""
    urdf = gio.UrdfModel(gio.HAND_URDF)
    T = gio.fk_batch(urdf, gio.build_full_angles(joints[None, :]))
    R = gio.quat_xyzw_to_matrix(root_pose[3:7])
    Tr = np.eye(4)
    Tr[:3, :3] = R
    Tr[:3, 3] = root_pose[:3]
    return {k: Tr @ v[0] for k, v in T.items()}


# --------------------------------------------------------------------------
# overlays
# --------------------------------------------------------------------------
def compute_overlay(object_mesh: trimesh.Trimesh, link_meshes: Dict[str, trimesh.Trimesh],
                    matrices: Dict[str, np.ndarray], max_pts: int = 120):
    """Contact points (near surface) and penetration points (inside)."""
    sdf = VoxelSDF.__new__(VoxelSDF)  # not used; we query with trimesh signed distance
    obj_pts = np.asarray(object_mesh.vertices)
    contacts, penetrations = [], []
    for name, m in link_meshes.items():
        if name not in matrices:
            continue
        W = matrices[name]
        V = m.vertices[:: max(1, len(m.vertices) // max_pts)] @ W[:3, :3].T + W[:3, 3]
        d = trimesh.proximity.ProximityQuery(object_mesh).signed_distance(V)  # + inside
        near = V[np.abs(d) < 0.002]
        inside = V[d > PEN_THRESHOLD_MM / 1000.0]
        if len(near):
            contacts.append((name, near))
        if len(inside):
            penetrations.append((name, inside))
    return contacts, penetrations


# --------------------------------------------------------------------------
# matplotlib PNG (headless-safe)
# --------------------------------------------------------------------------
def render_png(item: GraspItem, obj_mesh, link_meshes, matrices, contacts, penetrations,
               energy: float, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 6))
    views = [(0, 1), (0, 2), (1, 2)]
    ov = obj_mesh.vertices[:: max(1, len(obj_mesh.vertices) // 4000)]
    colors = plt.get_cmap("tab20")
    for k, (a, b) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.scatter(ov[:, a], ov[:, b], s=1, c="k", alpha=0.25, label="object")
        for j, (name, W) in enumerate(matrices.items()):
            if name not in link_meshes:
                continue
            m = link_meshes[name]
            V = m.vertices[:: max(1, len(m.vertices) // 200)] @ W[:3, :3].T + W[:3, 3]
            ax.scatter(V[:, a], V[:, b], s=2, c=[colors(j % 20)], alpha=0.75)
        for name, pts in contacts:
            ax.scatter(pts[:, a], pts[:, b], s=26, c="red", marker="x", linewidths=1.2)
        for name, pts in penetrations:
            ax.scatter(pts[:, a], pts[:, b], s=22, c="orange", marker="o", alpha=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("xyz"[a])
        ax.set_ylabel("xyz"[b])
        ax.grid(alpha=0.2)
    pen_mm = max([0.0] + [float(np.max(trimesh.proximity.ProximityQuery(obj_mesh).signed_distance(p))) * 1000
                          for _, p in penetrations]) if penetrations else 0.0
    fig.suptitle(f"{item.label}\nE={energy:.3f}  contact links={len(contacts)}  "
                 f"penetration>5mm links={len(penetrations)}  max_pen={pen_mm:.1f}mm", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# open3d backend
# --------------------------------------------------------------------------
def o3d_geometries(obj_mesh, link_meshes, matrices, contacts, penetrations):
    import open3d as o3d

    geoms = []
    obj = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(obj_mesh.vertices),
        o3d.utility.Vector3iVector(obj_mesh.faces),
    )
    obj.compute_vertex_normals()
    obj.paint_uniform_color([0.75, 0.75, 0.75])
    geoms.append(obj)
    colors = [[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
              [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.5, 0.5, 0.5]]
    for j, (name, m) in enumerate(link_meshes.items()):
        if name not in matrices:
            continue
        W = matrices[name]
        V = m.vertices @ W[:3, :3].T + W[:3, 3]
        lm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(m.faces))
        lm.compute_vertex_normals()
        lm.paint_uniform_color(colors[j % len(colors)])
        geoms.append(lm)
    for name, pts in contacts:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc.paint_uniform_color([1.0, 0.0, 0.0])
        geoms.append(pc)
    for name, pts in penetrations:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc.paint_uniform_color([1.0, 0.6, 0.0])
        geoms.append(pc)
    return geoms


def run_open3d(items, args, object_name):
    import open3d as o3d

    obj_mesh = trimesh.load(gio.object_mesh_path(object_name, args.grasp_root), force="mesh")
    meshes = {k: trimesh.load(v, force="mesh") for k, v in hand_mesh_paths(args.robot).items()}
    state = {"i": 0}

    def build(i):
        it = items[i]
        rp, joints, energy = load_pose(it)
        mats = compose_hand(rp, joints, args.robot)
        contacts, pens = compute_overlay(obj_mesh, meshes, mats)
        if args.offscreen:
            out = VIS_DIR / f"{object_name}_{it.source}_{it.path.stem}_{it.index}.png"
            render_png(it, obj_mesh, meshes, mats, contacts, pens, energy, out)
            print(f"[{i+1}/{len(items)}] {out}", flush=True)
            return None
        return o3d_geometries(obj_mesh, meshes, mats, contacts, pens)

    if args.offscreen:
        for i in range(min(len(items), args.max)):
            build(i)
        return

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="grasp viewer (n/p/s/q)", width=1280, height=800)

    def show(i):
        state["i"] = max(0, min(len(items) - 1, i))
        vis.clear_geometries()
        for g in build(state["i"]):
            vis.add_geometry(g)
        print(f"[{state['i']+1}/{len(items)}] {items[state['i']].label}", flush=True)

    def save(_vis):
        it = items[state["i"]]
        rp, joints, energy = load_pose(it)
        mats = compose_hand(rp, joints, args.robot)
        contacts, pens = compute_overlay(obj_mesh, meshes, mats)
        out = VIS_DIR / f"{object_name}_{it.source}_{it.path.stem}_{it.index}.png"
        render_png(it, obj_mesh, meshes, mats, contacts, pens, energy, out)
        print(f"saved {out}", flush=True)
        return False

    vis.register_key_callback(ord("N"), lambda v: (show(state["i"] + 1), False)[1])
    vis.register_key_callback(ord("P"), lambda v: (show(state["i"] - 1), False)[1])
    vis.register_key_callback(ord("S"), save)
    show(0)
    vis.run()
    vis.destroy_window()


# --------------------------------------------------------------------------
# sapien backend
# --------------------------------------------------------------------------
def run_sapien(items, args, object_name):
    import sapien

    sapien.set_log_level("error")
    scene = sapien.Scene()
    scene.set_ambient_light([0.25, 0.25, 0.25])
    scene.add_directional_light([1, 1, -1], [0.45, 0.45, 0.45])
    scene.add_directional_light([-1, -0.5, -0.5], [0.2, 0.2, 0.2])
    obj_mesh = trimesh.load(gio.object_mesh_path(object_name, args.grasp_root), force="mesh")
    hand_meshes = {k: trimesh.load(v, force="mesh") for k, v in hand_mesh_paths(args.robot).items()}

    obj_actor = None
    hand_actors: Dict[str, object] = {}
    camera = scene.add_camera("cam", 1280, 800, 1.0, 0.01, 10.0)
    camera.set_pose(sapien.Pose(*_look_at_pose([0.32, -0.32, 0.24], [0.0, 0.0, 0.0])))
    state = {"i": 0}

    def build_hand(rp, joints):
        mats = compose_hand(rp, joints, args.robot)
        for name, W in mats.items():
            if name not in hand_meshes:
                continue
            q = gio.matrix_to_quat_xyzw(W[:3, :3])[[3, 0, 1, 2]]
            if name in hand_actors:
                hand_actors[name].set_pose(sapien.Pose(W[:3, 3].tolist(), q.tolist()))
                continue
            b = scene.create_actor_builder()
            b.add_visual_from_file(str(hand_mesh_paths(args.robot)[name]))
            b.set_initial_pose(sapien.Pose(W[:3, 3].tolist(), q.tolist()))
            hand_actors[name] = b.build_static(name)

    def show(i):
        nonlocal obj_actor
        state["i"] = max(0, min(len(items) - 1, i))
        it = items[state["i"]]
        rp, joints, energy = load_pose(it)
        b = scene.create_actor_builder()
        b.add_visual_from_file(str(gio.object_mesh_path(object_name, args.grasp_root)))
        obj_actor = b.build_static("object")
        build_hand(rp, joints)
        print(f"[{state['i']+1}/{len(items)}] {it.label}", flush=True)

    def capture(out: Path):
        scene.update_render()
        camera.take_picture()
        rgba = camera.get_picture("Color")
        from PIL import Image

        img = (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
        if float(img.std()) < 2.0:  # blank frame -> fall back to the headless renderer
            raise RuntimeError("blank render (no display/GPU render system)")
        Image.fromarray(img).save(out)

    show(0)
    if args.offscreen:
        for i in range(min(len(items), args.max)):
            show(i)
            scene.update_render()
            out = VIS_DIR / f"{object_name}_{items[i].source}_{items[i].path.stem}_{items[i].index}_sapien.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                capture(out)
                print(f"saved {out}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[sapien] offscreen capture failed ({exc}); use --backend open3d --offscreen", flush=True)
        return

    viewer = scene.create_viewer(camera)
    viewer.set_camera_xyz(0.35, -0.35, 0.25)
    viewer.set_camera_rpy(0, -0.6, 2.2)

    def on_key(key):
        if key in ("n", "N"):
            show(state["i"] + 1)
        elif key in ("p", "P"):
            show(state["i"] - 1)
        elif key in ("q", "Q", "\x1b"):
            viewer.close()
        elif key in ("s", "S"):
            it = items[state["i"]]
            out = VIS_DIR / f"{object_name}_{it.source}_{it.path.stem}_{it.index}_sapien.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            capture(out)
            print(f"saved {out}", flush=True)

    if hasattr(viewer.window, "key_press"):
        viewer.window.key_press = on_key
    while not viewer.closed:
        viewer.render()


def _look_at_pose(eye, target):
    """SAPIEN camera pose (x forward, y left, z up) looking from ``eye`` at ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    left = np.cross(world_up, fwd)
    if np.linalg.norm(left) < 1e-8:
        left = np.array([0.0, 1.0, 0.0])
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    R = np.stack([fwd, left, up], axis=1)
    q = gio.matrix_to_quat_xyzw(R)[[3, 0, 1, 2]]
    return eye.tolist(), q.tolist()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["open3d", "sapien"], default="open3d")
    ap.add_argument("--grasp-file", default=None, help="single .pt file")
    ap.add_argument("--index", type=int, default=0, help="grasp index inside --grasp-file")
    ap.add_argument("--index-csv", default=None, help="merged filtered_grasps/index.csv (batch mode)")
    ap.add_argument("--object", default=None, help="object name for batch mode")
    ap.add_argument("--source", default="", help="filter batch by source (object / object1)")
    ap.add_argument("--robot", default="linkerhand", choices=["linkerhand", "linkerhand_right"])
    ap.add_argument("--grasp-root", default=None, help="root holding <object>/<object>.stl (default: object/ycb)")
    ap.add_argument("--offscreen", action="store_true", help="no window; write PNGs to vis_output/")
    ap.add_argument("--max", type=int, default=20, help="max grasps in offscreen batch mode")
    args = ap.parse_args()

    if args.grasp_file:
        p = Path(args.grasp_file).resolve()
        obj = p.name.split(".")[0].replace("_step_", "|").split("|")[0]
        items = [GraspItem(path=p, index=args.index, object=obj, source="", verdict="manual")]
        object_name = obj
        args.grasp_root = p.parents[5] if args.grasp_root is None else args.grasp_root
    elif args.index_csv and args.object:
        items = items_from_index(Path(args.index_csv).resolve(), args.object, args.source)
        if not items:
            raise SystemExit("no feasible grasps found for that object/source")
        object_name = args.object
    else:
        raise SystemExit("use --grasp-file/--index or --index-csv/--object")

    print(f"{len(items)} grasp(s), backend={args.backend}, offscreen={args.offscreen}", flush=True)
    if args.backend == "open3d":
        run_open3d(items, args, object_name)
    else:
        run_sapien(items, args, object_name)


if __name__ == "__main__":
    main()
