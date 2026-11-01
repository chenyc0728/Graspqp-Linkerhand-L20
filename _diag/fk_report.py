"""Headless hand-model report: FK layout, contact/penetration placement, SDF sanity.

Runs on CPU (uses trimesh for the SDF-free parts) and optionally on GPU for the
warp SDF occupancy test.

Usage:
    python _diag/fk_report.py --hand linkerhand_l20
    python _diag/fk_report.py --hand allegro
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh as tm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))

from graspqp.hands import get_hand_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="linkerhand_l20")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_model", action="store_true", help="skip building HandModel (pk only)")
    args = ap.parse_args()
    dev = args.device

    if args.no_model:
        import pytorch_kinematics as pk

        urdf = f"{ROOT}/graspqp/assets/{args.hand}/" + [f for f in os.listdir(f"{ROOT}/graspqp/assets/{args.hand}") if f.endswith(".urdf")][0]
        chain = pk.build_chain_from_urdf(open(urdf).read())
        names = chain.get_joint_parameter_names()
        fk = chain.forward_kinematics(torch.zeros(1, len(names)))
        print(f"{'link':28s} {'origin xyz (mm)':>26s}  axis dirs (cols of R)")
        for k, v in fk.items():
            M = v.get_matrix()[0].numpy()
            t = (M[:3, 3] * 1000).round(2)
            print(f"{k:28s} [{t[0]:8.2f} {t[1]:8.2f} {t[2]:8.2f}]")
        return

    hm = get_hand_model(args.hand, dev, grasp_type="all")
    print()
    print("=" * 100)
    print(f"hand={args.hand}  n_dofs={hm.n_dofs}  n_contact_candidates={hm.n_contact_candidates}")
    print("actuated joints :", hm._actuated_joints_names)
    print("default_state   :", np.round(hm.default_state.detach().cpu().numpy(), 4).tolist())
    print("joints_lower    :", np.round(hm.joints_lower.detach().cpu().numpy(), 3).tolist())
    print("joints_upper    :", np.round(hm.joints_upper.detach().cpu().numpy(), 3).tolist())
    print("forward_axis    :", hm.forward_axis.tolist(), " up_axis:", hm.up_axis.tolist(), " grasp_axis:", hm.grasp_axis.tolist())
    print("=" * 100)

    # ---------- zero pose layout ----------
    for label, joint_angles in (
        ("zero", torch.zeros(hm.n_dofs, device=dev)),
        ("default", hm.default_state.to(dev)),
    ):
        pose = torch.cat(
            [torch.zeros(9, device=dev), joint_angles]
        ).reshape(1, -1).detach().requires_grad_(True)
        hm.set_parameters(pose, "all")
        print()
        print(f"##### FK layout @ {label} pose #####")
        hdr = f"{'link':26s} {'origin mm':>24s} {'mesh bbox mm (world)':>40s} {'#cand':>6s} {'#pen':>5s}"
        print(hdr)
        # overall extents of the hand geometry
        allv = []
        for lname in hm.mesh:
            M = hm.current_status[lname].get_matrix()[0].detach().cpu().numpy()
            v = hm.mesh[lname]["vertices"].detach().cpu().numpy()
            n = len(v)
            if n:
                vw = v @ M[:3, :3].T + M[:3, 3]
                allv.append(vw)
                bmin, bmax = vw.min(0) * 1000, vw.max(0) * 1000
                bb = f"[{bmin[0]:7.1f} {bmin[1]:7.1f} {bmin[2]:7.1f}]..[{bmax[0]:7.1f} {bmax[1]:7.1f} {bmax[2]:7.1f}]"
            else:
                bb = "-"
            t = M[:3, 3] * 1000
            print(
                f"{lname:26s} [{t[0]:7.1f} {t[1]:7.1f} {t[2]:7.1f}] {bb:>40s} "
                f"{len(hm.mesh[lname]['contact_candidates']):6d} {len(hm.mesh[lname]['penetration_keypoints']):5d}"
            )
        if allv:
            allv = np.concatenate(allv, 0)
            bmin, bmax = allv.min(0) * 1000, allv.max(0) * 1000
            print(f"{'WHOLE HAND':26s} bbox mm [{bmin[0]:.1f} {bmin[1]:.1f} {bmin[2]:.1f}] .. [{bmax[0]:.1f} {bmax[1]:.1f} {bmax[2]:.1f}]  extent {np.round(bmax-bmin,1)}")

    # ---------- contact candidate placement ----------
    print()
    print("##### contact candidates (link frame) sanity #####")
    print("For each link: how far are the sampled contact candidates from the link surface, what's their")
    print("bbox, and what fraction is INSIDE the same link's own collision mesh (should be ~0).")
    hdr = f"{'link':26s} {'#cand':>6s} {'mean|d| mm':>10s} {'max|d| mm':>10s} {'frac_inside':>11s}"
    print(hdr)
    for lname in hm.mesh:
        cand = hm.mesh[lname]["contact_candidates"].detach().cpu().numpy()
        if len(cand) == 0:
            continue
        v = hm.mesh[lname]["vertices"].detach().cpu().numpy()
        f = hm.mesh[lname]["faces"].detach().cpu().numpy()
        m = tm.Trimesh(v, f, process=False)
        pq, pd, _ = tm.proximity.closest_point(m, cand)
        inside = m.contains(cand) if len(cand) else np.zeros(0, bool)
        print(
            f"{lname:26s} {len(cand):6d} {pd.mean()*1000:10.3f} {pd.max()*1000:10.3f} {inside.mean():11.3f}"
        )

    # ---------- penetration spheres ----------
    print()
    print("##### penetration spheres (link frame) vs own collision mesh #####")
    print("r_mm / distance from the link surface: a sphere centered on the surface with radius r is")
    print("expected; |d| should be small relative to r and r should be ~ local thickness/2.")
    hdr = f"{'link':26s} {'#sph':>5s} {'r mm':>9s} {'mean|d| mm':>10s} {'max|d| mm':>10s} {'r/|d|':>8s}"
    print(hdr)
    for lname in hm.mesh:
        kp = hm.mesh[lname]["penetration_keypoints"].detach().cpu().numpy()
        if len(kp) == 0:
            continue
        # scales were kept as an attribute on the model
        idx0 = 0
        v = hm.mesh[lname]["vertices"].detach().cpu().numpy()
        f = hm.mesh[lname]["faces"].detach().cpu().numpy()
        m = tm.Trimesh(v, f, process=False)
        _, pd, _ = tm.proximity.closest_point(m, kp)
        sc = hm.mesh[lname]["scales"].detach().cpu().numpy() * 1000
        print(
            f"{lname:26s} {len(kp):5d} {sc.mean():9.2f} {pd.mean()*1000:10.3f} {pd.max()*1000:10.3f} "
            f"{(sc.mean()/max(pd.mean()*1000,1e-6)):8.2f}"
        )


if __name__ == "__main__":
    main()
