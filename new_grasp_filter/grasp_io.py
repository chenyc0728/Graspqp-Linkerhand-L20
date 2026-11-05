"""Shared IO / kinematics helpers for the GraspQP -> Sapien -> CMapDataset pipeline.

Only additive: this module reads the existing GraspQP assets/data but never writes
to them. Everything here is dependency-light (numpy + torch + trimesh), so it can
run in the ``genh2r`` environment without the GraspQP stack (warp/pytorch3d/etc.).

Conventions (see REPORT.md, section "调研"):

* ``*.dexgrasp.pt`` maps joint name -> ``(G,)`` joint angle, plus
  ``parameters["root_pose"]`` of shape ``(G, 7)``.
* ``root_pose`` is ``[tx, ty, tz, qx, qy, qz, qw]`` (quaternion **xyzw**,
  verified from ``scripts/vis/visualize_result.py`` which reorders to wxyz before
  building the rotation matrix).
* The pose places the URDF root link (``root_link``) in the object/world frame;
  ``x_world = R @ x_link + t``.
* The 16 actuated joints are stored in ``DRIVE_NAMES`` order.  The 5 coupled DIP
  joints are expanded with the URDF mimic rules (reproduced from
  ``graspqp/hands/linkerhand_l20.py``).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------
# Repository layout (all relative to the repository root)
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

HAND_URDF = REPO_ROOT / "graspqp" / "assets" / "linkerhand_l20" / "linkerhand_l20.urdf"
HAND_MESH_DIR = REPO_ROOT / "graspqp" / "assets" / "linkerhand_l20" / "meshes"
YCB_ROOT = REPO_ROOT / "object" / "ycb"
GRASP_SUBDIR = Path("grasp_predictions/linkerhand_l20/12_contacts/graspqp/default")

#: 16 driven joint names, in the order used by GraspQP (chain order).
DRIVE_NAMES: List[str] = [
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
]

#: (index in the 21-dim URDF chain vector, driving index, multiplier)
MIMIC_MAP: List[Tuple[int, int, float]] = [
    (3, 2, 0.8917),    # pinky_dip  <- pinky_pip
    (7, 6, 0.8917),    # ring_dip   <- ring_pip
    (11, 10, 0.8917),  # middle_dip <- middle_pip
    (15, 14, 0.8917),  # index_dip  <- index_pip
    (20, 19, 1.1619),  # thumb_ip   <- thumb_mcp
]

#: indices of the 16 driven joints inside the 21-dim URDF chain vector
DRIVE_INDICES: List[int] = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19]


# --------------------------------------------------------------------------
# Small rotation helpers (no scipy / roma dependency)
# --------------------------------------------------------------------------
def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternion (x, y, z, w) -> (..., 3, 3) rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )
    return m


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrix -> (..., 4) quaternion (x, y, z, w)."""
    m = np.asarray(m, dtype=np.float64)
    batch = m.reshape(-1, 3, 3)
    out = np.zeros((batch.shape[0], 4))
    for i, r in enumerate(batch):
        tr = np.trace(r)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
        out[i] = [x, y, z, w]
    return out.reshape(*m.shape[:-2], 4)


def euler_xyz_to_matrix(e: np.ndarray) -> np.ndarray:
    """URDF/``rpy`` fixed-axis roll-pitch-yaw -> (..., 3, 3) = Rz @ Ry @ Rx."""
    e = np.asarray(e, dtype=np.float64)
    r, p, y = np.moveaxis(e, -1, 0)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    m = np.stack(
        [
            np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], axis=-1),
            np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], axis=-1),
            np.stack([-sp, cp * sr, cp * cr], axis=-1),
        ],
        axis=-2,
    )
    return m


def matrix_to_euler_xyz(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) roll-pitch-yaw, matching :func:`euler_xyz_to_matrix`."""
    m = np.asarray(m, dtype=np.float64)
    batch = m.reshape(-1, 3, 3)
    out = np.zeros((batch.shape[0], 3))
    for i, r in enumerate(batch):
        sp = -r[2, 0]
        sp = np.clip(sp, -1.0, 1.0)
        p = np.arcsin(sp)
        if abs(sp) < 1.0 - 1e-9:
            roll = np.arctan2(r[2, 1], r[2, 2])
            yaw = np.arctan2(r[1, 0], r[0, 0])
        else:  # gimbal lock: only roll-yaw sum/difference is defined -> set roll = 0
            roll = 0.0
            yaw = np.arctan2(-r[0, 1], r[1, 1]) if sp > 0 else np.arctan2(-r[0, 1], r[1, 1])
        out[i] = [roll, p, yaw]
    return out.reshape(*m.shape[:-2], 3)


def axis_angle_to_matrix(axis: Sequence[float], q: np.ndarray) -> np.ndarray:
    """Rodrigues rotation about ``axis`` for batched angles ``q`` -> (..., 3, 3)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.broadcast_to(np.eye(3), np.shape(q) + (3, 3)).copy()
    axis = axis / n
    q = np.asarray(q, dtype=np.float64)
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    s = np.sin(q)[..., None, None]
    c = np.cos(q)[..., None, None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


# --------------------------------------------------------------------------
# URDF kinematics
# --------------------------------------------------------------------------
@dataclass
class JointSpec:
    name: str
    jtype: str
    parent: str
    child: str
    origin_t: np.ndarray  # (3,)
    origin_R: np.ndarray  # (3, 3)
    axis: np.ndarray  # (3,)


class UrdfModel:
    """Minimal URDF tree + FK used to place the hand links in SAPIEN.

    The kinematics deliberately mirrors ``pytorch_kinematics``/GraspQP: chain
    order = document order, revolute transform = ``T_origin @ Rot(axis, q)``.
    """

    def __init__(self, urdf_path: Path):
        self.path = Path(urdf_path)
        root = ET.parse(self.path).getroot()
        self.name = root.get("name", "")
        self.links: List[str] = [link.get("name") for link in root.findall("link")]
        self.joints: List[JointSpec] = []
        for j in root.findall("joint"):
            origin = j.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                if origin.get("xyz"):
                    xyz = np.fromstring(origin.get("xyz"), sep=" ")
                if origin.get("rpy"):
                    rpy = np.fromstring(origin.get("rpy"), sep=" ")
            axis_el = j.find("axis")
            axis = np.array([0.0, 0.0, 1.0])
            if axis_el is not None and axis_el.get("xyz"):
                axis = np.fromstring(axis_el.get("xyz"), sep=" ")
            self.joints.append(
                JointSpec(
                    name=j.get("name"),
                    jtype=j.get("type"),
                    parent=j.find("parent").get("link"),
                    child=j.find("child").get("link"),
                    origin_t=xyz,
                    origin_R=euler_xyz_to_matrix(rpy),
                    axis=axis,
                )
            )
        self.parent_joint: Dict[str, JointSpec] = {j.child: j for j in self.joints}
        children = {j.parent for j in self.joints}
        roots = [l for l in self.links if l not in self.parent_joint]
        if len(roots) != 1:
            raise ValueError(f"expected a single URDF root, got {roots}")
        self.root_link = roots[0]
        self._order = self._topological_order()

    def _topological_order(self) -> List[str]:
        order = [self.root_link]
        changed = True
        while changed:
            changed = False
            for j in self.joints:
                if j.parent in order and j.child not in order:
                    order.append(j.child)
                    changed = True
        if len(order) != len(self.links):
            missing = set(self.links) - set(order)
            raise ValueError(f"disconnected URDF links: {missing}")
        return order

    def zero_angles(self) -> Dict[str, float]:
        return {j.name: 0.0 for j in self.joints if j.jtype in ("revolute", "prismatic", "continuous")}

    def fk(self, angles: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Forward kinematics at the given joint angles -> link name -> (4, 4)."""
        T: Dict[str, np.ndarray] = {self.root_link: np.eye(4)}
        for link in self._order[1:]:
            j = self.parent_joint[link]
            parent_T = T[j.parent]
            local = np.eye(4)
            local[:3, :3] = j.origin_R
            local[:3, 3] = j.origin_t
            q = angles.get(j.name, 0.0)
            if j.jtype in ("revolute", "continuous"):
                rot = axis_angle_to_matrix(j.axis, np.array(q))
                local = local @ np.block([[rot, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
            elif j.jtype == "prismatic":
                off = np.eye(4)
                off[:3, 3] = j.axis / (np.linalg.norm(j.axis) or 1.0) * q
                local = local @ off
            elif j.jtype == "floating":
                raise NotImplementedError("floating joints are not supported")
            T[link] = parent_T @ local
        return T


def build_full_angles(joint_angles: np.ndarray) -> np.ndarray:
    """(G, 16) driven angles -> (G, 21) URDF chain angles (mimic expansion)."""
    joint_angles = np.asarray(joint_angles, dtype=np.float64)
    if joint_angles.ndim == 1:
        joint_angles = joint_angles[None]
    if joint_angles.shape[-1] != 16:
        raise ValueError(f"expected 16 driven joint angles, got {joint_angles.shape}")
    out = np.zeros(joint_angles.shape[:-1] + (21,))
    mimic_lookup = {m: (DRIVE_INDICES.index(d), k) for m, d, k in MIMIC_MAP}
    for full_idx in range(21):
        if full_idx in mimic_lookup:
            pos, k = mimic_lookup[full_idx]
            out[..., full_idx] = joint_angles[..., pos] * k
        else:
            out[..., full_idx] = joint_angles[..., DRIVE_INDICES.index(full_idx)]
    return out


#: order of the 21 revolute joints inside the URDF chain vector (document order)
CHAIN_JOINT_NAMES: List[str] = [
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip",
]


def fk_batch(urdf: UrdfModel, full_angles: np.ndarray) -> Dict[str, np.ndarray]:
    """Batched FK: (G, 21) -> link name -> (G, 4, 4) homogeneous matrices."""
    full_angles = np.asarray(full_angles, dtype=np.float64)
    n = full_angles.shape[0]
    T: Dict[str, np.ndarray] = {urdf.root_link: np.broadcast_to(np.eye(4), (n, 4, 4)).copy()}
    for link in urdf._order[1:]:
        j = urdf.parent_joint[link]
        local = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        local[:, :3, :3] = j.origin_R
        local[:, :3, 3] = j.origin_t
        q = full_angles[:, CHAIN_JOINT_NAMES.index(j.name)] if j.name in CHAIN_JOINT_NAMES else np.zeros(n)
        if j.jtype in ("revolute", "continuous"):
            rot = axis_angle_to_matrix(j.axis, q)  # (n,3,3)
            step = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            step[:, :3, :3] = rot
            local = local @ step
        elif j.jtype == "prismatic":
            step = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            step[:, :3, 3] = j.axis / (np.linalg.norm(j.axis) or 1.0) * q[:, None]
            local = local @ step
        T[link] = T[j.parent] @ local
    return T


def hand_link_meshes(urdf_path: Path = HAND_URDF) -> Dict[str, Path]:
    """link name -> visual mesh path from the URDF (resolved relative to the file)."""
    root = ET.parse(urdf_path).getroot()
    base = Path(urdf_path).parent
    out: Dict[str, Path] = {}
    for link in root.findall("link"):
        vis = link.find("visual")
        if vis is None:
            continue
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        out[link.get("name")] = (base / mesh.get("filename")).resolve()
    return out


# --------------------------------------------------------------------------
# Grasp file IO
# --------------------------------------------------------------------------
def list_grasp_files(object_name: str) -> List[Path]:
    d = YCB_ROOT / object_name / GRASP_SUBDIR
    return sorted(d.glob("*.pt")) if d.is_dir() else []


def load_grasp_file(path: Path) -> Dict[str, np.ndarray]:
    """Load one ``*.dexgrasp.pt`` into plain numpy arrays (no GraspQP import)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    params = data["parameters"]
    joints = np.stack([np.asarray(params[n].detach().cpu(), dtype=np.float64).reshape(-1) for n in DRIVE_NAMES], axis=-1)
    root_pose = np.asarray(params["root_pose"].detach().cpu(), dtype=np.float64).reshape(-1, 7)
    values = np.asarray(data["values"].detach().cpu(), dtype=np.float64).reshape(-1)
    contact_idx = data.get("contact_idx")
    contact_idx = None if contact_idx is None else np.asarray(contact_idx.detach().cpu(), dtype=np.int64).reshape(-1, 12)
    return {
        "joints": joints,
        "root_pose": root_pose,
        "values": values,
        "contact_idx": contact_idx,
        "grasp_type": np.array(data.get("grasp_type", ""), dtype=object),
    }


def object_mesh_path(object_name: str) -> Path:
    """Watertight object mesh used for collision / signed distances.

    ``ObjectModel`` (GraspQP) loads ``coacd/remeshed.obj``, then
    ``coacd/decomposed.obj`` and finally globs ``*.obj`` -- for the prepared YCB
    folders that is ``coacd_allinone.obj``.  We verified that this file has the
    same bounding box and volume as ``<object>.stl`` (it is the CoACD remesh of
    the same surface), *but* it is a union of convex parts whose internal faces
    cross the interior, so point-in-mesh / signed distance queries on it are
    meaningless.  For collision and geometry we therefore prefer the original
    ``<object>.stl`` and only fall back to the CoACD mesh.
    """
    d = YCB_ROOT / object_name
    stl = d / f"{object_name}.stl"
    if stl.exists():
        return stl
    for cand in (d / "coacd" / "remeshed.obj", d / "coacd" / "decomposed.obj", d / "coacd_allinone.obj"):
        if cand.exists():
            return cand
    return stl


__all__ = [
    "REPO_ROOT", "HAND_URDF", "HAND_MESH_DIR", "YCB_ROOT", "GRASP_SUBDIR",
    "DRIVE_NAMES", "MIMIC_MAP", "DRIVE_INDICES", "CHAIN_JOINT_NAMES",
    "quat_xyzw_to_matrix", "matrix_to_quat_xyzw", "euler_xyz_to_matrix",
    "matrix_to_euler_xyz", "axis_angle_to_matrix",
    "UrdfModel", "build_full_angles", "fk_batch", "hand_link_meshes",
    "list_grasp_files", "load_grasp_file", "object_mesh_path",
]
