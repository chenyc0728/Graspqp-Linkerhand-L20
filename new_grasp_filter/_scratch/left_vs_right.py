"""Scratch: compare left (repo asset) vs right (CMap) hand FK on the same joint angles."""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grasp_io as gio

CMAP_R = gio.REPO_ROOT / "Cmap_dataset/data_urdf/robot/linkerhand/right/linkerhand_l20_right_extended.urdf"
g = gio.load_grasp_file(gio.list_grasp_files("apple")[0])
obj = trimesh.load(gio.object_mesh_path("apple"), force="mesh")

keep = ["index_distal", "middle_distal", "ring_distal", "pinky_distal", "thumb_distal", "thumb_proximal", "hand_base_link", "base_link"]


def make_model(path):
    urdf = gio.UrdfModel(path)
    link_meshes = {}
    base = Path(path).parent
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    for link in root.findall("link"):
        vis = link.find("visual")
        if vis is None:
            continue
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        link_meshes[link.get("name")] = (base / mesh.get("filename")).resolve()
    return urdf, link_meshes


def run(tag, path, joint_names):
    urdf, link_meshes = make_model(path)
    meshes = {}
    for k in keep:
        if k in link_meshes and link_meshes[k].exists():
            meshes[k] = trimesh.load(link_meshes[k], force="mesh")
    sub = {k: m.vertices[:: max(1, len(m.vertices) // 400)] for k, m in meshes.items()}
    full = gio.build_full_angles(g["joints"])
    if "thumb_dip" in joint_names:
        full[:, 20] = full[:, 19] * 1.1619
    T = gio.fk_batch(urdf, full)
    print(f"== {tag}: links={len(T)}")
    for i in np.argsort(g["values"])[:3]:
        rp = g["root_pose"][i]
        Tr = np.eye(4)
        Tr[:3, :3] = gio.quat_xyzw_to_matrix(rp[3:7])
        Tr[:3, 3] = rp[:3]
        out = []
        for name in keep:
            if name not in sub:
                continue
            W = Tr @ T[name][i]
            V = sub[name] @ W[:3, :3].T + W[:3, 3]
            d = trimesh.proximity.closest_point(obj, V)[1]
            out.append(f"{name}:{d.min()*1000:.0f}")
        print(f"  grasp {i} E={g['values'][i]:.2f} " + " ".join(out))


run("LEFT asset", gio.HAND_URDF, gio.CHAIN_JOINT_NAMES)
run("RIGHT cmap", CMAP_R, [n.replace("thumb_ip", "thumb_dip") for n in gio.CHAIN_JOINT_NAMES])
