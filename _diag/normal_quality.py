"""Are the cached hand contact normals actually outward?

For every contact candidate c with cached normal n, query the (robust) winding-number
SDF at c + eps*n and c - eps*n. Outward => outside at +eps, inside at -eps.
Also reports E_dis's normal weight exp(1 - cos) and the same test for allegro.
"""
import argparse, sys, os
import numpy as np, torch, warp as wp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))
from graspqp.hands import get_hand_model
from graspqp.utils import warp as wp_utils

ap = argparse.ArgumentParser()
ap.add_argument("--hand", default="linkerhand_l20")
ap.add_argument("--device", default="cuda")
ap.add_argument("--eps", type=float, default=25e-5)
a = ap.parse_args()
hm = get_hand_model(a.hand, a.device, grasp_type="all")
hm.use_winding_number = True

print(f"hand={a.hand}   eps={a.eps*1000:.2f} mm")
print(f"{'link':26s} {'#cand':>6s} {'outward':>8s} {'inward':>7s} {'ambiguous':>10s}")
tot = ok = bad = 0
for ln in hm.mesh:
    cand = hm.mesh[ln]["contact_candidates"]
    nrm = hm.mesh[ln]["normal_candidates"]
    if len(cand) == 0:
        continue
    fv = hm.mesh[ln]["face_verts"]
    q = torch.cat([cand + a.eps * nrm, cand - a.eps * nrm], 0).contiguous()
    d, _ = wp_utils.CalcSdfField.apply(q, fv.id, 1e6, True)  # exterior positive
    dp, dm = d[: len(cand)], d[len(cand):]
    good = (dp > 0) & (dm < 0)
    amb = (dp.abs() < a.eps * 0.5) | (dm.abs() < a.eps * 0.5)
    tot += len(cand); ok += int(good.sum()); bad += int((~good).sum())
    print(f"{ln:26s} {len(cand):6d} {int(good.sum()):8d} {int((~good).sum()):7d} {int(amb.sum()):10d}")
print(f"{'TOTAL':26s} {tot:6d} {ok:8d} {bad:7d}  -> outward fraction {ok/max(tot,1):.3f}")
