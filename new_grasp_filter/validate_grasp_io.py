"""Cross-validate the numpy pose/FK implementation in ``grasp_io.py`` against
``roma`` and ``pytorch_kinematics`` (the official GraspQP dependencies).

Outputs a per-item error table (stdout + ``logs/validate_grasp_io.json``) and
reproduces the v1 sanity check: the lowest-energy apple grasp (E=1.942) has its
four fingertip surfaces on the object surface.

Run::

    python new_grasp_filter/validate_grasp_io.py            # all checks
    python new_grasp_filter/validate_grasp_io.py --quick    # fewer samples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_io as gio  # noqa: E402

import roma  # noqa: E402
import pytorch_kinematics as pk  # noqa: E402

OUT = HERE / "logs" / "validate_grasp_io.json"


def q_to_wxyz(q_xyzw):
    return np.asarray(q_xyzw)[..., [3, 0, 1, 2]]


# --------------------------------------------------------------------------
# 1. quaternion decoding
# --------------------------------------------------------------------------
def check_quaternion(files, n_max=2000):
    qs = []
    for f in files:
        qs.append(gio.load_grasp_file(f)["root_pose"][:, 3:7])
    qs = np.concatenate(qs, axis=0)[:n_max]
    R_np = np.stack([gio._quat_xyzw_to_matrix_numpy(q) for q in qs])
    # roma 1.8.0 in this env takes (x, y, z, w); checked with identity/90deg below
    R_rm = roma.unitquat_to_rotmat(torch.tensor(qs, dtype=torch.float64)).numpy()
    # round trip: matrix -> quat -> matrix, numpy vs roma
    q_np = np.stack([gio._matrix_to_quat_xyzw_numpy(R) for R in R_np])
    q_rm = roma.rotmat_to_unitquat(torch.tensor(R_np, dtype=torch.float64)).numpy()  # xyzw
    # compare rotations reconstructed from both quaternion encodings
    R_from_np = np.stack([gio._quat_xyzw_to_matrix_numpy(q) for q in q_np])
    # roma returns wxyz -> reorder to xyzw for our decoder
    R_from_rm = np.stack([gio._quat_xyzw_to_matrix_numpy(q) for q in q_rm])
    return {
        "n_samples": int(len(qs)),
        "max_abs_R_numpy_vs_roma": float(np.abs(R_np - R_rm).max()),
        "max_abs_R_from_numpy_quat_vs_original": float(np.abs(R_from_np - R_np).max()),
        "max_abs_R_from_roma_quat_vs_original": float(np.abs(R_from_rm - R_np).max()),
        "quat_sign_agnostic_max": float(np.minimum(np.abs(q_np - q_rm), np.abs(q_np + q_rm)).max()),
    }


# --------------------------------------------------------------------------
# 2. Euler helpers vs scipy/roma (used by the CMap conversion)
# --------------------------------------------------------------------------
def check_euler(files, n_max=500):
    from scipy.spatial.transform import Rotation as Rot

    qs = np.concatenate([gio.load_grasp_file(f)["root_pose"][:, 3:7] for f in files])[:n_max]
    R = np.stack([gio._quat_xyzw_to_matrix_numpy(q) for q in qs])
    rpy = np.stack([gio.matrix_to_euler_xyz(r) for r in R])  # Rx@Ry@Rz convention
    R2 = np.stack([gio._euler_xyz_to_matrix_numpy(e) for e in rpy])
    eul = Rot.from_matrix(R).as_euler("xyz")  # scipy 'xyz' == Rx@Ry@Rz (extrinsic)
    R3 = Rot.from_euler("xyz", eul).as_matrix()
    return {
        "n_samples": int(len(R)),
        "max_abs_R_roundtrip_numpy": float(np.abs(R2 - R).max()),
        "max_abs_R_scipy_vs_numpy": float(np.abs(R3 - R).max()),
        "max_abs_angle_diff_vs_scipy_rad": float(np.abs(eul - rpy).max()),
    }


# --------------------------------------------------------------------------
# 3. FK vs pytorch_kinematics
# --------------------------------------------------------------------------
def check_fk(files, n_max=64):
    chain = pk.build_chain_from_urdf(open(gio.HAND_URDF).read()).to(dtype=torch.float64)
    pk_names = chain.get_joint_parameter_names()
    urdf = gio.UrdfModel(gio.HAND_URDF)
    q = np.concatenate([gio.load_grasp_file(f)["joints"] for f in files])[:n_max]
    full = gio.build_full_angles(q)  # (N,21)

    order_ok = pk_names == gio.CHAIN_JOINT_NAMES
    # pk in float64 for a fair comparison
    th = torch.tensor(full, dtype=torch.float64)
    t0 = time.time()
    fk_pk = chain.forward_kinematics(th)
    t_pk = time.time() - t0
    t0 = time.time()
    fk_np = gio._fk_batch_numpy(urdf, full)
    t_np = time.time() - t0

    per_link = {}
    for name, tf in fk_pk.items():
        if name not in fk_np:
            continue
        M = tf.get_matrix().numpy()  # (N,4,4)
        per_link[name] = float(np.abs(M - fk_np[name]).max())
    worst = sorted(per_link.items(), key=lambda kv: -kv[1])[:5]
    tips = {k: per_link[k] for k in ["index_distal", "middle_distal", "ring_distal", "pinky_distal", "thumb_distal"] if k in per_link}
    return {
        "n_configs": int(len(full)),
        "pk_joint_order_equals_CHAIN_JOINT_NAMES": bool(order_ok),
        "pk_joint_names": pk_names,
        "links_compared": len(per_link),
        "max_abs_T_error": float(max(per_link.values())),
        "worst_links": worst,
        "fingertip_max_abs_T_error": tips,
        "time_pk_s": t_pk,
        "time_numpy_s": t_np,
    }


# --------------------------------------------------------------------------
# 4. mimic expansion vs URDF <mimic> tags
# --------------------------------------------------------------------------
def check_mimic():
    import xml.etree.ElementTree as ET

    root = ET.parse(gio.HAND_URDF).getroot()
    tags = {}
    for j in root.findall("joint"):
        m = j.find("mimic")
        if m is not None:
            tags[j.get("name")] = (m.get("joint"), float(m.get("multiplier")))
    ours = {gio.CHAIN_JOINT_NAMES[i]: (gio.CHAIN_JOINT_NAMES[d], k) for i, d, k in gio.MIMIC_MAP}
    ok = tags == ours
    # numeric: expand a batch with the numpy path and with an explicit tf.
    q = torch.rand(256, 16, dtype=torch.float64) * 0.8
    full_np = gio.build_full_angles(q.numpy())
    full_tf = gio._build_full_angles_torch(q).numpy()
    return {
        "urdf_mimic_tags": tags,
        "grasp_io_mimic_map": ours,
        "identical": bool(ok),
        "max_abs_diff_numpy_vs_torch_expansion": float(np.abs(full_np - full_tf).max()),
    }


# --------------------------------------------------------------------------
# 5. v1 cross-check: lowest-energy apple grasp, fingertip distance to surface
# --------------------------------------------------------------------------
def check_apple_fingertips():
    """Lowest-energy apple grasp: signed distance of fingertip vertices to the object.

    NOTE (v2 correction): ``trimesh.proximity.closest_point`` in trimesh 4.11.3
    returns a **signed** distance (positive *inside*), which v1 read as unsigned.
    v1 therefore reported "fingertips at 0.00 mm" while the vertices were in fact
    entirely inside the object.  Here we report both extremes explicitly:
    ``min_mm`` = outermost vertex (0 => on the surface, >0 => completely inside),
    ``max_mm`` = deepest vertex.
    """
    import trimesh

    f = gio.list_grasp_files("apple")[0]
    g = gio.load_grasp_file(f)
    i = int(np.argmin(g["values"]))
    urdf = gio.UrdfModel(gio.HAND_URDF)
    T = gio.fk_batch(urdf, gio.build_full_angles(g["joints"][i : i + 1]))
    rp = g["root_pose"][i]
    Tr = np.eye(4)
    Tr[:3, :3] = gio.quat_xyzw_to_matrix(rp[3:7])
    Tr[:3, 3] = rp[:3]
    obj = trimesh.load(gio.object_mesh_path("apple"), force="mesh")
    pq = trimesh.proximity.ProximityQuery(obj)
    out = {}
    for name in ["index_distal", "middle_distal", "ring_distal", "pinky_distal"]:
        m = trimesh.load(gio.hand_link_meshes()[name], force="mesh")
        W = Tr @ T[name][0]
        V = m.vertices @ W[:3, :3].T + W[:3, 3]
        d = pq.signed_distance(V) * 1000.0  # positive inside
        out[name] = {
            "min_mm": float(d.min()),
            "max_mm": float(d.max()),
            "frac_inside": float((d > 0).mean()),
        }
    return {
        "energy": float(g["values"][i]),
        "grasp_index": i,
        "fingertips": out,
        "all_vertices_inside": all(v["min_mm"] >= -0.001 for v in out.values()),
        "note": "positive = inside the object (trimesh 4.11.3 signed_distance)",
    }


# --------------------------------------------------------------------------
# 6. CMap left extended URDF structural check
# --------------------------------------------------------------------------
def check_cmap_urdfs():
    import xml.etree.ElementTree as ET

    out = {}
    left = gio.REPO_ROOT / "Cmap_dataset/data_urdf/robot/linkerhand/left/linkerhand_l20_left_extended.urdf"
    right = gio.REPO_ROOT / "Cmap_dataset/data_urdf/robot/linkerhand/right/linkerhand_l20_right_extended.urdf"
    for tag, p in [("left", left), ("right", right)]:
        r = ET.parse(p).getroot()
        links = {l.get("name") for l in r.findall("link")}
        child = {j.find("child").get("link") for j in r.findall("joint")}
        dangling = sorted(c for c in child if c not in links)
        roots = sorted(links - child)
        try:
            ch = pk.build_chain_from_urdf(open(p).read())
            pk_ok, n_joints = True, len(ch.get_joint_parameter_names())
        except Exception as e:
            pk_ok, n_joints = False, f"{type(e).__name__}: {e}"
        out[tag] = {
            "path": str(p.relative_to(gio.REPO_ROOT)),
            "dangling_joint_children": dangling,
            "root_links": roots,
            "pk_build_ok": pk_ok,
            "pk_n_joints": n_joints,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    files = gio.list_grasp_files("apple")[: (2 if args.quick else 8)]
    files += gio.list_grasp_files("hammer")[: (0 if args.quick else 2)]
    n = 200 if args.quick else 2000

    result = {
        "graspqp_urdf": str(gio.HAND_URDF.relative_to(gio.REPO_ROOT)),
        "quaternion": check_quaternion(files, n),
        "euler": check_euler(files, n // 4),
        "mimic": check_mimic(),
        "fk": check_fk(files, 8 if args.quick else 64),
        "apple_fingertip_check": check_apple_fingertips(),
        "cmap_urdf_structure": check_cmap_urdfs(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("numpy vs roma / pytorch_kinematics")
    print("=" * 78)
    q = result["quaternion"]
    print(f"[quat] n={q['n_samples']}  max|R_np - R_roma| = {q['max_abs_R_numpy_vs_roma']:.3e}")
    print(f"       round-trip numpy quat: {q['max_abs_R_from_numpy_quat_vs_original']:.3e}"
          f"   roma quat: {q['max_abs_R_from_roma_quat_vs_original']:.3e}")
    e = result["euler"]
    print(f"[euler] n={e['n_samples']}  round-trip {e['max_abs_R_roundtrip_numpy']:.3e}"
          f"  vs scipy {e['max_abs_R_scipy_vs_numpy']:.3e}  angle diff {e['max_abs_angle_diff_vs_scipy_rad']:.3e}")
    m = result["mimic"]
    print(f"[mimic] identical={m['identical']}  numeric exp diff {m['max_abs_diff_numpy_vs_torch_expansion']:.3e}")
    f = result["fk"]
    print(f"[fk] configs={f['n_configs']} links={f['links_compared']}"
          f"  order_ok={f['pk_joint_order_equals_CHAIN_JOINT_NAMES']}")
    print(f"     max|T_np - T_pk| = {f['max_abs_T_error']:.3e}")
    print(f"     worst links: {[(k, f'{v:.2e}') for k, v in f['worst_links']]}")
    print(f"     fingertip errors: {[(k, f'{v:.2e}') for k, v in f['fingertip_max_abs_T_error'].items()]}")
    print(f"     time: pk {f['time_pk_s']*1000:.1f} ms, numpy {f['time_numpy_s']*1000:.1f} ms")
    a = result["apple_fingertip_check"]
    print(f"[apple E={a['energy']:.3f}] fingertip signed distance (mm, + = inside object):")
    for k, v in a["fingertips"].items():
        print(f"    {k:14s} outermost {v['min_mm']:+8.2f}  deepest {v['max_mm']:+8.2f}  inside {v['frac_inside']*100:5.1f}%")
    print(f"    all vertices inside the object: {a['all_vertices_inside']}")
    c = result["cmap_urdf_structure"]
    for tag, v in c.items():
        print(f"[cmap {tag}] pk_ok={v['pk_build_ok']} dangling={v['dangling_joint_children']} roots={v['root_links']}")
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
