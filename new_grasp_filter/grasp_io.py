"""Shared IO / kinematics helpers for the GraspQP -> Sapien -> CMapDataset pipeline.

v2: the pose/FK maths are now backed by the official GraspQP dependencies
(``roma`` for quaternions, ``pytorch_kinematics`` for URDF FK), which are torch
based and can run on CUDA.  The original numpy implementations are kept as a
fallback and can be selected at runtime::

    import grasp_io as gio
    gio.set_backend("numpy")     # legacy path (v1 behaviour, bit-identical maths)
    gio.set_backend("torch")     # default: roma + pytorch_kinematics
    gio.set_device("cuda")       # torch backend on GPU

Public functions keep returning **numpy** so that every existing caller
(SAPIEN pose setting, trimesh, CSV) is unaffected; ``*_torch`` variants expose
the batched torch tensors for GPU work.

Conventions (verified in ``validate_grasp_io.py`` and REPORT_v2 §1):

* ``*.dexgrasp.pt`` maps joint name -> ``(G,)`` joint angle, plus
  ``parameters["root_pose"]`` of shape ``(G, 7)``.
* ``root_pose`` is ``[tx, ty, tz, qx, qy, qz, qw]`` (quaternion **xyzw**).
* The pose places the URDF root link (``root_link``) of the GraspQP hand in the
  object/world frame: ``x_world = R @ x_link + t``.
* The 16 actuated joints are stored in ``DRIVE_NAMES`` order; the 5 coupled DIP
  joints are expanded with the URDF ``<mimic>`` rules.
* CMap's ``q`` is ``[x, y, z, roll, pitch, yaw, 16 joints]`` and its virtual chain
  is ``T(x,y,z) @ Rx(roll) @ Ry(pitch) @ Rz(yaw)`` ending at the **hand base**
  (``hand_base_link`` in the left extended URDF).  GraspQP's root pose is for
  ``root_link``, which sits at ``(-0.03, 0, -0.10)`` in the hand-base frame, so a
  conversion must apply that fixed transform (see :func:`root_pose_to_cmap_q`).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------
# Repository layout (all relative to the repository root)
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

HAND_URDF = REPO_ROOT / "graspqp" / "assets" / "linkerhand_l20" / "linkerhand_l20.urdf"
HAND_MESH_DIR = REPO_ROOT / "graspqp" / "assets" / "linkerhand_l20" / "meshes"
YCB_ROOT = REPO_ROOT / "object" / "ycb"
YCB1_ROOT = REPO_ROOT / "object1" / "ycb"
GRASP_SUBDIR = Path("grasp_predictions/linkerhand_l20/12_contacts/graspqp/default")

CMAP_ROOT = REPO_ROOT / "Cmap_dataset"
CMAP_LEFT_URDF = CMAP_ROOT / "data_urdf/robot/linkerhand/left/linkerhand_l20_left_extended.urdf"
CMAP_RIGHT_URDF = CMAP_ROOT / "data_urdf/robot/linkerhand/right/linkerhand_l20_right_extended.urdf"
CMAP_ASSETS_META = CMAP_ROOT / "data_urdf/robot/urdf_assets_meta.json"

#: fixed transform of ``root_link`` -> ``hand_base_link`` in the GraspQP L20 URDF
ROOT_TO_HAND_BASE = np.array([-0.03, 0.0, -0.10])

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

#: order of the 21 revolute joints inside the URDF chain vector (document order)
CHAIN_JOINT_NAMES: List[str] = [
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "pinky_dip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip", "ring_dip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "middle_dip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip", "index_dip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip",
]

# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------
_BACKENDS = ("torch", "numpy")
_BACKEND = "torch"
_DEVICE = "cpu"


def set_backend(name: str, device: Optional[str] = None) -> None:
    """Select the maths backend: ``"torch"`` (roma + pytorch_kinematics) or ``"numpy"``."""
    global _BACKEND, _DEVICE
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend {name!r}, expected one of {_BACKENDS}")
    if name == "torch":
        import roma  # noqa: F401
        import pytorch_kinematics  # noqa: F401
    _BACKEND = name
    if device is not None:
        _DEVICE = device
    if _BACKEND == "torch":
        torch.set_grad_enabled(False)


def set_device(device: str) -> None:
    global _DEVICE
    _DEVICE = device


def get_backend() -> str:
    return _BACKEND


def get_device() -> str:
    return _DEVICE


# --------------------------------------------------------------------------
# numpy implementations (v1, kept as fallback / reference)
# --------------------------------------------------------------------------
def _quat_xyzw_to_matrix_numpy(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )


def _matrix_to_quat_xyzw_numpy(m: np.ndarray) -> np.ndarray:
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


def _euler_xyz_to_matrix_numpy(e: np.ndarray) -> np.ndarray:
    """``rpy`` fixed-axis roll-pitch-yaw -> Rz @ Ry @ Rx (URDF ``rpy``)."""
    e = np.asarray(e, dtype=np.float64)
    r, p, y = np.moveaxis(e, -1, 0)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.stack(
        [
            np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], axis=-1),
            np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], axis=-1),
            np.stack([-sp, cp * sr, cp * cr], axis=-1),
        ],
        axis=-2,
    )


def _matrix_to_euler_xyz_numpy(m: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,3) roll/pitch/yaw with R = Rz @ Ry @ Rx."""
    m = np.asarray(m, dtype=np.float64)
    batch = m.reshape(-1, 3, 3)
    out = np.zeros((batch.shape[0], 3))
    for i, r in enumerate(batch):
        sp = np.clip(-r[2, 0], -1.0, 1.0)
        p = np.arcsin(sp)
        if abs(sp) < 1.0 - 1e-9:
            roll = np.arctan2(r[2, 1], r[2, 2])
            yaw = np.arctan2(r[1, 0], r[0, 0])
        else:
            roll = 0.0
            yaw = np.arctan2(-r[0, 1], r[1, 1])
        out[i] = [roll, p, yaw]
    return out.reshape(*m.shape[:-2], 3)


def _axis_angle_to_matrix_numpy(axis: Sequence[float], q: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.broadcast_to(np.eye(3), np.shape(q) + (3, 3)).copy()
    axis = axis / n
    q = np.asarray(q, dtype=np.float64)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    s = np.sin(q)[..., None, None]
    c = np.cos(q)[..., None, None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _build_full_angles_numpy(joint_angles: np.ndarray) -> np.ndarray:
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


def _build_full_angles_torch(joint_angles: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    """Torch mimic expansion (same table as :func:`_build_full_angles_numpy`)."""
    t = torch.as_tensor(joint_angles, dtype=torch.float64, device=device or _DEVICE)
    if t.ndim == 1:
        t = t[None]
    out = torch.zeros(t.shape[:-1] + (21,), dtype=t.dtype, device=t.device)
    mimic_lookup = {m: (DRIVE_INDICES.index(d), k) for m, d, k in MIMIC_MAP}
    for full_idx in range(21):
        if full_idx in mimic_lookup:
            pos, k = mimic_lookup[full_idx]
            out[..., full_idx] = t[..., pos] * k
        else:
            out[..., full_idx] = t[..., DRIVE_INDICES.index(full_idx)]
    return out


def _fk_batch_numpy(urdf: "UrdfModel", full_angles: np.ndarray) -> Dict[str, np.ndarray]:
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
            step = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            step[:, :3, :3] = _axis_angle_to_matrix_numpy(j.axis, q)
            local = local @ step
        elif j.jtype == "prismatic":
            step = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            step[:, :3, 3] = j.axis / (np.linalg.norm(j.axis) or 1.0) * q[:, None]
            local = local @ step
        T[link] = T[j.parent] @ local
    return T


# --------------------------------------------------------------------------
# public dispatchers
# --------------------------------------------------------------------------
def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """``(..., 4)`` xyzw quaternion -> ``(..., 3, 3)`` rotation matrix."""
    if _BACKEND == "torch":
        import roma

        # NOTE: the installed roma 1.8.0 interprets unit quaternions as (x, y, z, w),
        # verified in validate_grasp_io.py (identity [0,0,0,1] -> I, [1,0,0,0] -> Rx(pi)).
        t = torch.as_tensor(np.asarray(q, dtype=np.float64), dtype=torch.float64, device="cpu")
        return roma.unitquat_to_rotmat(t).cpu().numpy()
    return _quat_xyzw_to_matrix_numpy(q)


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrix -> ``(..., 4)`` xyzw quaternion."""
    if _BACKEND == "torch":
        import roma

        t = torch.as_tensor(np.asarray(m, dtype=np.float64), dtype=torch.float64,
                            device=_DEVICE if _DEVICE != "numpy" else "cpu")
        # inverse of the above: roma returns (x, y, z, w)
        return roma.rotmat_to_unitquat(t).cpu().numpy()
    return _matrix_to_quat_xyzw_numpy(m)


def euler_xyz_to_matrix(e: np.ndarray) -> np.ndarray:
    """URDF ``rpy`` (R = Rz @ Ry @ Rx) -> rotation matrix."""
    return _euler_xyz_to_matrix_numpy(e)


def matrix_to_euler_xyz(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> URDF ``rpy`` (R = Rz @ Ry @ Rx)."""
    return _matrix_to_euler_xyz_numpy(m)


def axis_angle_to_matrix(axis: Sequence[float], q: np.ndarray) -> np.ndarray:
    return _axis_angle_to_matrix_numpy(axis, q)


def build_full_angles(joint_angles) -> np.ndarray:
    """(G, 16) driven angles -> (G, 21) URDF chain angles (mimic expansion)."""
    if isinstance(joint_angles, torch.Tensor):
        return _build_full_angles_torch(joint_angles).cpu().numpy()
    if _BACKEND == "torch":
        t = torch.as_tensor(np.asarray(joint_angles, dtype=np.float64), dtype=torch.float64, device=_DEVICE)
        return _build_full_angles_torch(t).cpu().numpy()
    return _build_full_angles_numpy(joint_angles)


def fk_batch(urdf: "UrdfModel", full_angles: np.ndarray) -> Dict[str, np.ndarray]:
    """Batched FK -> link name -> ``(G, 4, 4)`` numpy, in the URDF root frame."""
    if _BACKEND == "torch":
        T = fk_batch_torch(urdf, full_angles, device=_DEVICE)
        return {k: v.cpu().numpy() for k, v in T.items()}
    return _fk_batch_numpy(urdf, full_angles)


def fk_batch_torch(urdf: "UrdfModel", full_angles, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Torch FK (``pytorch_kinematics``) -> link name -> ``(G, 4, 4)`` tensors."""
    chain = urdf.pk_chain_on(device)
    th = torch.as_tensor(np.asarray(full_angles, dtype=np.float64), dtype=torch.float64, device=device)
    if th.shape[-1] != len(chain.get_joint_parameter_names()):
        raise ValueError(
            f"URDF {urdf.path.name} has {len(chain.get_joint_parameter_names())} chain joints, "
            f"got {th.shape[-1]} angles; pass the full chain vector (see CHAIN_JOINT_NAMES)"
        )
    fk = chain.forward_kinematics(th)
    return {k: v.get_matrix() for k, v in fk.items()}


# --------------------------------------------------------------------------
# CMap helpers
# --------------------------------------------------------------------------
def cmap_left_urdf(repaired: bool = True) -> Path:
    """Path of CMap's left extended URDF (optionally a repaired copy).

    CMap's left extended URDF references a ``base_link`` that does not exist
    (``virtual_robot``'s child), while the finger chain hangs off a disconnected
    ``hand_base_link`` root.  ``pytorch_kinematics`` therefore fails with
    ``KeyError: 'base_link'``.  The repaired copy (written under
    ``new_grasp_filter/assets/``) only redirects that fixed joint's child to
    ``hand_base_link``; nothing else is changed.
    """
    if not repaired:
        return CMAP_LEFT_URDF
    dst = HERE / "assets" / "cmap_left_repaired.urdf"
    if not dst.exists():
        import re

        txt = CMAP_LEFT_URDF.read_text(encoding="utf-8")
        if '<child link="base_link"' in txt:
            txt = txt.replace('<child link="base_link"', '<child link="hand_base_link"')
        # the copy lives elsewhere: make mesh paths absolute so both
        # pytorch_kinematics and trimesh can resolve them
        base = CMAP_LEFT_URDF.parent

        def _abs(m):
            fn = m.group(1)
            p = Path(fn)
            if not p.is_absolute():
                p = (base / p).resolve()
            return f'filename="{p.as_posix()}"'

        txt = re.sub(r'filename="([^"]+)"', _abs, txt)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(txt, encoding="utf-8")
    return dst


def root_pose_to_cmap_q(root_pose: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """GraspQP ``root_pose`` (root_link) -> CMap ``q`` (hand-base frame, 22 dims).

    The CMap ``q`` translation/rotation describe the pose of the hand base
    (``hand_base_link`` in the left extended URDF, ``base_link`` in the right
    one).  GraspQP's ``root_link`` is offset by ``ROOT_TO_HAND_BASE`` in that
    frame, so::

        t_base = t_root + R_root @ (-0.03, 0, -0.10)
        R_base = R_root

    The euler angles use ``R = Rx(roll) @ Ry(pitch) @ Rz(yaw)`` (CMap virtual
    chain order, verified against the URDF).
    """
    R = quat_xyzw_to_matrix(np.asarray(root_pose[3:7], dtype=np.float64))
    t = np.asarray(root_pose[:3], dtype=np.float64) + R @ ROOT_TO_HAND_BASE
    rpy = _matrix_to_euler_cmap(R)
    return np.concatenate([t, rpy, np.asarray(joints, dtype=np.float64)]).astype(np.float32)


def matrix_to_cmap_rpy(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (roll, pitch, yaw) with ``R = Rx @ Ry @ Rz``."""
    return _matrix_to_euler_cmap(m)


def _matrix_to_euler_cmap(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    flat = m.reshape(-1, 3, 3)
    out = np.zeros((flat.shape[0], 3))
    for i, r in enumerate(flat):
        sp = np.clip(r[0, 2], -1.0, 1.0)
        pitch = np.arcsin(sp)
        if abs(sp) < 1.0 - 1e-9:
            roll = np.arctan2(-r[1, 2], r[2, 2])
            yaw = np.arctan2(-r[0, 1], r[0, 0])
        else:
            roll = 0.0
            yaw = np.arctan2(r[1, 0], r[1, 1])
        out[i] = [roll, pitch, yaw]
    return out.reshape(*m.shape[:-2], 3)


# --------------------------------------------------------------------------
# URDF kinematics (parser shared by both backends)
# --------------------------------------------------------------------------
@dataclass
class JointSpec:
    name: str
    jtype: str
    parent: str
    child: str
    origin_t: np.ndarray
    origin_R: np.ndarray
    axis: np.ndarray


class UrdfModel:
    """Minimal URDF tree used to place hand links, backed by pk when available."""

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
            raise ValueError(
                f"{self.path.name}: expected a single URDF root, got {roots}. "
                "For CMap's left extended URDF use grasp_io.cmap_left_urdf() (repaired copy)."
            )
        self.root_link = roots[0]
        self._order = self._topological_order()
        self._pk_chains: Dict[str, object] = {}
        self._pk_names: Optional[List[str]] = None

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

    def pk_chain_on(self, device: str = "cpu"):
        """``pytorch_kinematics`` chain cached per device (CPU and CUDA copies)."""
        key = str(device)
        if key not in self._pk_chains:
            import pytorch_kinematics as pk

            chain = pk.build_chain_from_urdf(self.path.read_text(encoding="utf-8"))
            chain = chain.to(dtype=torch.float64, device=key)
            self._pk_chains[key] = chain
            self._pk_names = list(chain.get_joint_parameter_names())
        return self._pk_chains[key]

    @property
    def pk_chain(self):
        return self.pk_chain_on("cpu")

    def zero_angles(self) -> Dict[str, float]:
        return {j.name: 0.0 for j in self.joints if j.jtype in ("revolute", "prismatic", "continuous")}

    def fk(self, angles: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Single-config FK -> link name -> 4x4 matrix (relative to the URDF root)."""
        if _BACKEND == "torch":
            chain = self.pk_chain_on(_DEVICE)
            names = chain.get_joint_parameter_names()
            vec = torch.tensor([[float(angles.get(n, 0.0)) for n in names]],
                               dtype=torch.float64, device=_DEVICE)
            fk = chain.forward_kinematics(vec)
            return {k: v.get_matrix()[0].cpu().numpy() for k, v in fk.items()}
        T: Dict[str, np.ndarray] = {self.root_link: np.eye(4)}
        for link in self._order[1:]:
            j = self.parent_joint[link]
            local = np.eye(4)
            local[:3, :3] = j.origin_R
            local[:3, 3] = j.origin_t
            q = angles.get(j.name, 0.0)
            if j.jtype in ("revolute", "continuous"):
                step = np.eye(4)
                step[:3, :3] = _axis_angle_to_matrix_numpy(j.axis, np.array(q))
                local = local @ step
            elif j.jtype == "prismatic":
                step = np.eye(4)
                step[:3, 3] = j.axis / (np.linalg.norm(j.axis) or 1.0) * q
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
def list_grasp_files(object_name: str, root: Optional[Path] = None) -> List[Path]:
    d = Path(root or YCB_ROOT) / object_name / GRASP_SUBDIR
    return sorted(d.glob("*.pt")) if d.is_dir() else []


def list_objects(root: Optional[Path] = None) -> List[str]:
    r = Path(root or YCB_ROOT)
    if not r.is_dir():
        return []
    return sorted(p.name for p in r.iterdir() if p.is_dir() and list_grasp_files(p.name, r))


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


def object_mesh_path(object_name: str, root: Optional[Path] = None) -> Path:
    """Watertight object mesh used for collision / signed distances.

    Prefers ``<object>.stl`` (the CoACD ``coacd_allinone.obj`` is a union of
    convex parts whose internal faces make point-in-mesh queries meaningless).
    """
    d = Path(root or YCB_ROOT) / object_name
    stl = d / f"{object_name}.stl"
    if stl.exists():
        return stl
    # fall back to the canonical mesh of the same object (object1 may only ship grasps)
    alt = REPO_ROOT / "object" / "ycb" / object_name / f"{object_name}.stl"
    if alt.exists():
        return alt
    for cand in (d / "coacd" / "remeshed.obj", d / "coacd" / "decomposed.obj", d / "coacd_allinone.obj"):
        if cand.exists():
            return cand
    return stl


__all__ = [
    "REPO_ROOT", "HERE", "HAND_URDF", "HAND_MESH_DIR", "YCB_ROOT", "YCB1_ROOT", "GRASP_SUBDIR",
    "CMAP_LEFT_URDF", "CMAP_RIGHT_URDF", "ROOT_TO_HAND_BASE",
    "DRIVE_NAMES", "MIMIC_MAP", "DRIVE_INDICES", "CHAIN_JOINT_NAMES",
    "set_backend", "set_device", "get_backend", "get_device",
    "quat_xyzw_to_matrix", "matrix_to_quat_xyzw", "euler_xyz_to_matrix",
    "matrix_to_euler_xyz", "axis_angle_to_matrix",
    "UrdfModel", "build_full_angles", "fk_batch", "fk_batch_torch", "hand_link_meshes",
    "list_grasp_files", "list_objects", "load_grasp_file", "object_mesh_path",
    "cmap_left_urdf", "root_pose_to_cmap_q", "matrix_to_cmap_rpy",
]
