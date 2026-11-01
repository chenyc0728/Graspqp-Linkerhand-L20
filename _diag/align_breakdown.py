"""Break the contact-normal alignment down by link, for the selected contacts."""
import argparse, glob, os, sys
from collections import defaultdict
import numpy as np, roma, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))
from graspqp.core import ObjectModel
from graspqp.hands import get_hand_model

ap = argparse.ArgumentParser()
ap.add_argument("--hand", default="linkerhand_l20")
ap.add_argument("--data_root", default="min_dataset/release2")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
root = os.path.abspath(a.data_root)
hm = get_hand_model(a.hand, a.device, grasp_type="all")
stats = defaultdict(list)
allcos = []
for obj in sorted(os.listdir(root)):
    p = os.path.join(root, obj, "grasp_predictions", a.hand, "12_contacts", "graspqp", "default", f"{obj}.dexgrasp.pt")
    if not os.path.exists(p):
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    params = {k: v.detach().cpu() for k, v in ck["parameters"].items()}
    rp = params["root_pose"].to(a.device).float()
    J = torch.stack([params[n] for n in hm._actuated_joints_names], -1).to(a.device)
    order = torch.argsort(ck["values"])[: a.n]
    rp, J = rp[order], J[order]
    cidx = ck["contact_idx"].to(a.device)[order]
    om = ObjectModel(data_root_path=root, batch_size_each=len(J), num_samples=1000, device=a.device, scale=1.0)
    om.initialize([os.path.join(root, obj)])
    R6 = roma.unitquat_to_rotmat(rp[..., [4, 5, 6, 3]]).mT.flatten(1, 2)
    hm.set_parameters(torch.cat([rp[..., :3], R6[..., :6], J], -1).to(a.device), cidx)
    with torch.no_grad():
        _, cnorm = om.cal_distance(hm.contact_points)
        cos = torch.sum(-cnorm * hm.contact_normals, -1)  # (B, 12)
        links = hm.global_index_to_link_index[cidx]        # (B, 12)
    allcos.append(cos.cpu().numpy().ravel())
    for b in range(len(J)):
        for k in range(cos.shape[1]):
            stats[hm.link_names[int(links[b, k])]].append(float(cos[b, k]))
allcos = np.concatenate(allcos)
print(f"hand={a.hand}   contacts={len(allcos)}  mean cos={allcos.mean():.3f}  frac cos<0={np.mean(allcos<0):.3f}  frac cos>0.5={np.mean(allcos>0.5):.3f}")
print(f"{'link':26s} {'n':>6s} {'mean cos':>9s} {'frac<0':>7s}")
for k, v in sorted(stats.items(), key=lambda kv: -len(kv[1])):
    v = np.array(v)
    print(f"{k:26s} {len(v):6d} {v.mean():9.3f} {np.mean(v<0):7.2f}")
