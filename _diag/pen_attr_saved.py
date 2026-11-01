"""Attribute the hand<->object penetration of SAVED grasps to individual links,
and check the cached hand contact normals for sanity."""
import argparse, glob, os, sys
import numpy as np, roma, torch, trimesh as tm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))
from graspqp.core import ObjectModel
from graspqp.hands import get_hand_model

ap = argparse.ArgumentParser()
ap.add_argument("--hand", default="linkerhand_l20")
ap.add_argument("--data_root", default="min_dataset/release2")
ap.add_argument("--obj", default="apple")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

hm = get_hand_model(a.hand, a.device, grasp_type="all")
path = os.path.join(os.path.abspath(a.data_root), a.obj, "grasp_predictions", a.hand,
                    "12_contacts", "graspqp", "default", f"{a.obj}.dexgrasp.pt")
if not os.path.exists(path):
    files = glob.glob(os.path.join(os.path.dirname(path), "*.dexgrasp.pt"))
    path = sorted(files)[-1]
ck = torch.load(path, map_location="cpu", weights_only=False)
params = {k: v.detach().cpu() for k, v in ck["parameters"].items()}
root_pose = params["root_pose"].to(a.device).float()
J = torch.stack([params[n] for n in hm._actuated_joints_names], dim=-1).to(a.device)
order = torch.argsort(ck["values"])[: a.n]
root_pose, J = root_pose[order], J[order]
cidx = ck["contact_idx"].to(a.device)[order]
om = ObjectModel(data_root_path=os.path.abspath(a.data_root), batch_size_each=len(J),
                 num_samples=2500, device=a.device, scale=1.0)
om.initialize([os.path.join(os.path.abspath(a.data_root), a.obj)])
R6 = roma.unitquat_to_rotmat(root_pose[..., [4, 5, 6, 3]]).mT.flatten(1, 2)
hm.set_parameters(torch.cat([root_pose[..., :3], R6[..., :6], J], -1).to(a.device), cidx)

with torch.no_grad():
    scale = om.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    osp = om.surface_points_tensor * scale
    hsd, lidx = hm.cal_distance(osp, return_link_index=True)
    names = hm.link_names
print(f"obj={a.obj}  n={len(J)}  saved E mean={ck['values'][order].mean():.3f}")
print(f"{'link':26s} {'pts inside':>11s} {'frac':>7s} {'max mm':>8s} {'mean mm':>8s}")
tot = hsd.numel()
for i, ln in enumerate(names):
    sel = (lidx == i) & (hsd > 0)
    if sel.sum() == 0:
        continue
    d = hsd[sel]
    print(f"{ln:26s} {int(sel.sum()):11d} {100*sel.sum().item()/tot:6.2f}% {d.max()*1000:8.2f} {d.mean()*1000:8.2f}")
allpos = hsd > 0
print(f"{'TOTAL':26s} {int(allpos.sum()):11d} {100*allpos.sum().item()/tot:6.2f}% {hsd[allpos].max()*1000:8.2f} {hsd[allpos].mean()*1000:8.2f}")

# ---- cached hand contact normals sanity: should point away from the link centroid ----
print()
print("cached hand contact normal vs 'link centroid -> contact point' direction")
print(f"{'link':26s} {'#cand':>6s} {'mean cos':>9s} {'frac cos>0':>11s}")
for ln in hm.mesh:
    cand = hm.mesh[ln]["contact_candidates"]
    nrm = hm.mesh[ln]["normal_candidates"]
    if len(cand) == 0:
        continue
    v = hm.mesh[ln]["vertices"]
    c = v.mean(0, keepdim=True)
    outdir = torch.nn.functional.normalize(cand - c, dim=-1)
    cos = (outdir * nrm).sum(-1)
    print(f"{ln:26s} {len(cand):6d} {float(cos.mean()):9.3f} {float((cos>0).float().mean()):11.2f}")
