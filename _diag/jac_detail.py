"""Per-link / per-column detail of the Jacobian error (pk analytical vs FD)."""
import os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src", "graspqp", "hands"))
import pytorch_kinematics as pk
from linkerhand_l20 import DRIVE_INDICES, MIMIC_MAP, _build_full_angles

URDF = f"{ROOT}/graspqp/assets/linkerhand_l20/linkerhand_l20.urdf"
chain64 = pk.build_chain_from_urdf(open(URDF).read()).to(dtype=torch.float64)

def fk(q16):
    return chain64.forward_kinematics(_build_full_angles(torch.tensor(q16, dtype=torch.float64).reshape(1,-1)))

def rotvec_stable(R):
    v = 0.5*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    s = np.linalg.norm(v)
    c = np.clip((np.trace(R)-1)/2, -1, 1)
    th = np.arctan2(s, c)
    return v if s < 1e-12 else v*(th/s)

q = np.full(16, 0.4)
eps = 1e-6
links = list(fk(q).keys())
J_ang = np.zeros((len(links), 3, 16)); J_lin = np.zeros((len(links), 3, 16))
for i in range(16):
    qp, qm = q.copy(), q.copy(); qp[i]+=eps; qm[i]-=eps
    fp, fm = fk(qp), fk(qm)
    for li, ln in enumerate(links):
        Mp, Mm = fp[ln].get_matrix()[0].numpy(), fm[ln].get_matrix()[0].numpy()
        J_lin[li,:,i] = (Mp[:3,3]-Mm[:3,3])/(2*eps)
        J_ang[li,:,i] = rotvec_stable(Mp[:3,:3] @ Mm[:3,:3].T)/(2*eps)

chain32 = pk.build_chain_from_urdf(open(URDF).read())
Jpk = chain32.jacobian(_build_full_angles(torch.tensor(q, dtype=torch.float32).reshape(1,-1)))[0].numpy()
J_eff = Jpk[..., DRIVE_INDICES].copy()
for m,d,mult in MIMIC_MAP:
    pos = DRIVE_INDICES.index(d); J_eff[...,pos] = J_eff[...,pos] + Jpk[...,m]*mult

print(f"{'link':26s} {'lin err':>10s} {'ang err':>10s}   per-column ang err (16 drive cols)")
for li, ln in enumerate(links):
    e_lin = np.abs(J_eff[li,:3]-J_lin[li]).max()
    e_ang = np.abs(J_eff[li,3:]-J_ang[li]).max()
    cols = np.abs(J_eff[li,3:]-J_ang[li]).max(0)
    print(f"{ln:26s} {e_lin:10.2e} {e_ang:10.2e}   " + " ".join(f"{c:4.2f}" for c in cols))
