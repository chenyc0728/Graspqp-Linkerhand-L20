"""Verify the custom FK / Jacobian of `hands/linkerhand_l20.py` (CPU only).

Checks, and reports numbers for:
 1. joint ordering assumed by `_build_full_angles` vs `chain.get_joint_parameter_names()`
 2. the mimic coupling values vs the URDF `<mimic>` tags
 3. custom FK vs `chain.forward_kinematics` on the expanded 21 angles (must be identical)
 4. custom Jacobian vs a finite-difference Jacobian of the same FK
    (report both the raw pk `chain.jacobian` and the mimic-folded `calculate_jacobian`)
 5. behaviour exactly at the singular zero configuration

Usage: python _diag/fk_jac_check.py
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src", "graspqp", "hands"))

import pytorch_kinematics as pk  # noqa: E402

from linkerhand_l20 import DRIVE_INDICES, MIMIC_MAP, _build_full_angles, calculate_jacobian, calculate_joints  # noqa: E402

URDF = f"{ROOT}/graspqp/assets/linkerhand_l20/linkerhand_l20.urdf"


class Shim:
    """Minimal stand-in for HandModel (the callbacks only use `chain`)."""

    def __init__(self, chain):
        self.chain = chain


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def main():
    chain = pk.build_chain_from_urdf(open(URDF).read())
    names = chain.get_joint_parameter_names()
    shim = Shim(chain)

    print("### 1. joint ordering")
    print("chain joint order           :", names)
    print("DRIVE_INDICES               :", DRIVE_INDICES)
    print("DRIVE names                 :", [names[i] for i in DRIVE_INDICES])
    mimic_names = [names[m] for m, _, _ in MIMIC_MAP]
    print("MIMIC names                 :", mimic_names)
    print("mimic targets (drivers)     :", [names[d] for _, d, _ in MIMIC_MAP])
    ok = all(names[m] in ("pinky_dip", "ring_dip", "middle_dip", "index_dip", "thumb_ip") for m in [x[0] for x in MIMIC_MAP])
    print("  ordering assumptions hold :", ok)
    print("  drive set == 21 - mimic   :", sorted(DRIVE_INDICES + [m for m, _, _ in MIMIC_MAP]) == list(range(21)))

    print()
    print("### 2. mimic multipliers vs URDF")
    import xml.etree.ElementTree as ET

    urdf_mimic = {}
    for j in ET.parse(URDF).getroot().findall("joint"):
        mim = j.find("mimic")
        if mim is not None:
            urdf_mimic[j.get("name")] = (mim.get("joint"), float(mim.get("multiplier")), float(mim.get("offset", 0.0)))
    print("  URDF:", urdf_mimic)
    for m, d, mult in MIMIC_MAP:
        mn = names[m]
        src = urdf_mimic.get(mn)
        good = src is not None and src[0] == names[d] and abs(src[1] - mult) < 1e-9 and abs(src[2]) < 1e-12
        print(f"  {mn:12s} <- {names[d]:12s} mult={mult:<8} urdf={src} -> {'OK' if good else 'MISMATCH'}")

    print()
    print("### 3. custom FK vs chain.forward_kinematics")
    rng = np.random.default_rng(0)
    lo = np.array([-0.17, 0.0, 0.0] * 4 + [0.0, 0.0, 0.0, 0.0])
    hi = np.array([0.17, 1.4, 1.57] * 4 + [1.4, 1.22, 0.79, 1.05])
    for label, q in (
        ("zeros", np.zeros(16)),
        ("random x4", lo + (hi - lo) * rng.random(16)),
    ):
        q_t = torch.tensor(q, dtype=torch.float32).reshape(1, -1)
        res = calculate_joints(q_t, shim)
        ref = chain.forward_kinematics(_build_full_angles(q_t))
        err = max(float((res[k].get_matrix() - ref[k].get_matrix()).abs().max()) for k in ref)
        print(f"  {label:10s} max |dM| = {err:.3e}")

    print()
    print("### 4. Jacobian: pk analytical vs finite differences of the same FK (float64 FD)")
    chain64 = pk.build_chain_from_urdf(open(URDF).read()).to(dtype=torch.float64)
    shim64 = Shim(chain64)
    eps = 1e-6

    def fk_mat64(q16):
        return chain64.forward_kinematics(_build_full_angles(torch.tensor(q16, dtype=torch.float64).reshape(1, -1)))
    links = list(fk_mat64(np.zeros(16)).keys())

    rng2 = np.random.default_rng(1)
    for label, q in (("zeros", np.zeros(16)), ("mid", np.full(16, 0.4)), ("random", lo + (hi - lo) * rng2.random(16))):
        J_num = np.zeros((len(links), 6, 16))
        for i in range(16):
            qp, qm = q.copy(), q.copy()
            qp[i] += eps
            qm[i] -= eps
            fp, fm = fk_mat64(qp), fk_mat64(qm)
            for li, ln in enumerate(links):
                Mp = fp[ln].get_matrix()[0].numpy()
                Mm = fm[ln].get_matrix()[0].numpy()
                J_num[li, :3, i] = (Mp[:3, 3] - Mm[:3, 3]) / (2 * eps)
                J_num[li, 3:, i] = rotvec(Mp[:3, :3] @ Mm[:3, :3].T) / (2 * eps)
        q16_32 = torch.tensor(q, dtype=torch.float32).reshape(1, -1)
        J_custom = calculate_jacobian(q16_32, shim)[0].numpy()
        scale = max(np.abs(J_num).max(), 1e-9)
        d_custom = np.abs(J_custom - J_num)
        J_pk_full = chain.jacobian(_build_full_angles(q16_32))[0].numpy()
        J_eff = J_pk_full[..., DRIVE_INDICES].copy()
        for m, d, mult in MIMIC_MAP:
            pos = DRIVE_INDICES.index(d)
            J_eff[..., pos] = J_eff[..., pos] + J_pk_full[..., m] * mult
        d_pk = np.abs(J_eff - J_num)
        print(f"  {label:8s} |J_numeric|max = {scale:8.4f}")
        print(f"           calculate_jacobian() vs FD : max abs err {d_custom.max():.4e}  rel {d_custom.max()/scale:.3e}")
        print(f"           chain.jacobian()+fold  vs FD: max abs err {d_pk.max():.4e}  rel {d_pk.max()/scale:.3e}")
        per_link = d_pk.reshape(len(links), -1).max(1)
        worst = np.argsort(-per_link)[:3]
        print("           worst links (pk):", [(links[i], round(float(per_link[i]), 4)) for i in worst])
        # split linear vs angular
        print(f"           pk linear-part err {np.abs(J_eff[:, :3] - J_num[:, :3]).max():.4e}"
              f"   angular-part err {np.abs(J_eff[:, 3:] - J_num[:, 3:]).max():.4e}")

    print()
    print("### 5. Jacobian at the exact zero configuration (singularity check)")
    q0 = torch.zeros(1, 16)
    J0 = calculate_jacobian(q0, shim)
    print("  calculate_jacobian at q=0: finite =", bool(torch.isfinite(J0).all()),
          " shape =", tuple(J0.shape), " |J|max =", float(J0.abs().max()))
    Jp = chain.jacobian(torch.zeros(1, 21))
    print("  chain.jacobian at theta=0 : finite =", bool(torch.isfinite(Jp).all()),
          " |J|max =", float(Jp.abs().max()))
    for eps_ in (1e-4, 1e-6, 1e-8):
        try:
            Jq = calculate_jacobian(torch.zeros(1, 16) + eps_, shim)
            print(f"  q=+{eps_:g}: finite={bool(torch.isfinite(Jq).all())} |J|max={float(Jq.abs().max()):.4f}"
                  f"  d(|J|max)/d q = {float(Jq.abs().max()-J0.abs().max()):+.3e}")
        except Exception as e:
            print(f"  q=+{eps_:g}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
