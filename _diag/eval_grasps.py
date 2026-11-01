"""Headless (text/JSON only) evaluation of saved grasps — no plotly, no browser.

Replaces `scripts/vis/visualize_result.py` for analysis: loads the `.dexgrasp.pt`
checkpoints written by `scripts/fit.py`, rebuilds the hand + object models, and
prints per-grasp diagnostics as tables:

  * hand root pose and joint angles (vs. default_state)
  * contact point signed distance to the object surface (positive = inside)
  * contact-normal consistency with the object outward normal
  * hand->object penetration (object surface points inside the hand)
  * self-penetration
  * force-closure energy (GraspQP, same solver as fit.py)
  * a simple, explicit feasibility criterion

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python _diag/eval_grasps.py \
        --hand linkerhand_l20 --data_root min_dataset/release2 \
        --suffix "" --max_grasps 32

    # write machine-readable output too
    ... --json _diag/out/eval_l20.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import roma
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))

from graspqp.core import ObjectModel  # noqa: E402
from graspqp.core.energy import calculate_energy  # noqa: E402
from graspqp.hands import get_hand_model  # noqa: E402
from graspqp.metrics import GraspSpanMetricFactory  # noqa: E402


def pct(x):
    return f"{100.0*x:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="linkerhand_l20")
    ap.add_argument("--data_root", default="min_dataset/release2")
    ap.add_argument("--num_contacts", default="12_contacts")
    ap.add_argument("--energy", default="graspqp")
    ap.add_argument("--grasp_type", default="default")
    ap.add_argument("--suffix", default="", help="'' = final checkpoint, or e.g. '_step_2000'")
    ap.add_argument("--max_grasps", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--friction", type=float, default=0.2)
    ap.add_argument("--max_lambda_limit", type=float, default=20.0)
    ap.add_argument("--n_friction_cone", type=int, default=4)
    ap.add_argument("--contact_tol", type=float, default=0.005, help="|contact distance| tolerance [m]")
    ap.add_argument("--pen_tol", type=float, default=0.005, help="per-point hand-object penetration tolerance [m]")
    ap.add_argument("--fc_tol", type=float, default=1.0, help="E_fc threshold")
    args = ap.parse_args()
    dev = args.device

    hand_model = get_hand_model(args.hand, dev, use_collision_if_possible=True, grasp_type="all")
    energy_fnc = GraspSpanMetricFactory.create(
        GraspSpanMetricFactory.MetricType.GRASPQP,
        solver_kwargs={
            "friction": args.friction,
            "max_limit": args.max_lambda_limit,
            "n_cone_vecs": args.n_friction_cone,
        },
    )

    objects = sorted(
        [
            o
            for o in os.listdir(os.path.join(ROOT, args.data_root))
            if os.path.isdir(os.path.join(ROOT, args.data_root, o))
        ]
    )

    rows = []
    per_object = {}

    for obj in objects:
        pat = os.path.join(
            ROOT, args.data_root, obj, "grasp_predictions", args.hand, args.num_contacts, args.energy,
            args.grasp_type, f"*{args.suffix}.dexgrasp.pt",
        )
        files = sorted(glob.glob(pat))
        if not files:
            print(f"[skip] no checkpoint for {obj}: {pat}")
            continue
        # prefer the exact '<obj><suffix>.dexgrasp.pt'
        exact = [f for f in files if os.path.basename(f) == f"{obj}{args.suffix}.dexgrasp.pt"]
        path = exact[0] if exact else files[-1]

        ck = torch.load(path, map_location=dev, weights_only=False)
        params = {k: v.detach().cpu() for k, v in ck["parameters"].items()}
        root_pose = params["root_pose"].to(dev).float()
        joint_states = torch.stack(
            [params[jn] for jn in hand_model._actuated_joints_names], dim=-1
        ).to(dev)
        energies_saved = ck.get("values")
        contact_idxs = ck.get("contact_idx")

        n = min(args.max_grasps, len(root_pose))
        order = torch.argsort(energies_saved)[:n] if energies_saved is not None else torch.arange(n)
        root_pose, joint_states = root_pose[order], joint_states[order]
        if contact_idxs is not None:
            contact_idxs = contact_idxs.to(dev)[order]

        budget = latest = None
        object_model = ObjectModel(
            data_root_path=os.path.join(ROOT, args.data_root),
            batch_size_each=n,
            num_samples=2500,
            device=dev,
            scale=1.0,
        )
        object_model.initialize([os.path.join(ROOT, args.data_root, obj)])

        if "offset" in ck and ck["offset"] is not None:
            off = ck["offset"]
            if off.ndim == 1:
                off = off.unsqueeze(0)
            if off.shape[0] != n:
                off = off.expand(n, -1)
            R_off = roma.unitquat_to_rotmat(off[..., [4, 5, 6, 3]])
            t_off = off[:, :3]
            R_w = roma.unitquat_to_rotmat(root_pose[..., [4, 5, 6, 3]])
            t_w = root_pose[:, :3]
            R_new = R_w @ R_off
            t_new = torch.einsum("nij,nj->ni", R_w, t_off) + t_w
            quat = roma.rotmat_to_unitquat(R_new)[..., [3, 0, 1, 2]]
            root_pose = torch.cat([t_new, quat], dim=-1)

        R = roma.unitquat_to_rotmat(root_pose[..., [4, 5, 6, 3]]).mT.flatten(1, 2)
        hand_params = torch.cat([root_pose[..., :3], R[..., :6], joint_states], dim=-1).to(dev)
        hand_model.set_parameters(
            hand_params, contact_point_indices=contact_idxs if contact_idxs is not None else "all"
        )

        with torch.no_grad():
            losses = calculate_energy(
                hand_model, object_model, energy_fnc=energy_fnc,
                energy_names=["E_dis", "E_fc", "E_pen", "E_spen", "E_joints"],
                method="gendexgrasp", svd_gain=0.1,
            )
            # contact signed distance to the object surface (positive => inside the object)
            cdist, cnorm = object_model.cal_distance(hand_model.contact_points)
            # hand->object penetration: object surface points vs. hand SDF (positive => inside hand)
            scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
            osp = object_model.surface_points_tensor * scale
            hsd = hand_model.cal_distance(osp)
            pen_pts = (hsd > 0).float()
            pen_max = hsd.clamp(min=0).max(-1).values
            pen_sum = hsd.clamp(min=0).sum(-1)
            # contact normals vs object outward normal (-cnorm is the outward normal)
            nH = hand_model.contact_normals
            align = torch.sum(-cnorm * nH, dim=-1).mean(-1)
            joint_dev = (joint_states - hand_model.default_state.to(dev).unsqueeze(0)).abs()
            joint_max_dev = joint_dev.max(-1).values
            joint_mean_dev = joint_dev.mean(-1)

        for i in range(n):
            rows.append(
                dict(
                    object=obj,
                    idx=int(order[i]) if order is not None else i,
                    E_total=float(
                        energies_saved[order[i]] if energies_saved is not None else float("nan")
                    ),
                    E_dis=float(losses["E_dis"][i]),
                    E_fc=float(losses["E_fc"][i]),
                    E_pen=float(losses["E_pen"][i]),
                    E_spen=float(losses["E_spen"][i]),
                    E_joints=float(losses["E_joints"][i]),
                    contact_abs_mean=float(cdist.abs().mean(-1)[i]),
                    contact_abs_max=float(cdist.abs().max(-1).values[i]),
                    contact_pen_depth_max=float(cdist.max(-1).values[i]),
                    normal_align=float(align[i]),
                    hand_pen_frac=float(pen_pts.mean(-1)[i]),
                    hand_pen_max=float(pen_max[i]),
                    hand_pen_sum=float(pen_sum[i]),
                    joint_mean_abs_dev=float(joint_mean_dev[i]),
                    joint_max_abs_dev=float(joint_max_dev[i]),
                    joints=np.round(joint_states[i].detach().cpu().numpy(), 3).tolist(),
                )
            )
        per_object[obj] = len(rows)

    if not rows:
        print("no checkpoints found")
        return

    # ---------------- print ----------------
    print()
    print("=" * 150)
    print(f"grasp evaluation: hand={args.hand}  data_root={args.data_root}  suffix='{args.suffix}'  n={len(rows)}")
    print("=" * 150)
    hdr = (
        f"{'object':16s} {'#':>3s} {'E_tot':>8s} {'E_dis':>8s} {'E_fc':>7s} {'E_pen':>8s} {'E_spen':>7s} "
        f"{'|cd|mean':>8s} {'|cd|max':>8s} {'penMax':>8s} {'penFrac':>7s} {'align':>6s} {'dJoint':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['object'][:16]:16s} {r['idx']:3d} {r['E_total']:8.3f} {r['E_dis']:8.3f} {r['E_fc']:7.4f} "
            f"{r['E_pen']:8.4f} {r['E_spen']:7.4f} {r['contact_abs_mean']*1000:8.2f} "
            f"{r['contact_abs_max']*1000:8.2f} {r['hand_pen_max']*1000:8.2f} {pct(r['hand_pen_frac']):>7s} "
            f"{r['normal_align']:6.3f} {r['joint_mean_abs_dev']:7.3f}"
        )

    # ---------------- aggregate ----------------
    def agg(key, f=np.mean):
        return f([r[key] for r in rows])

    feasible = [
        r
        for r in rows
        if abs(r["contact_abs_max"]) <= args.contact_tol
        and r["hand_pen_max"] <= args.pen_tol
        and r["E_fc"] <= args.fc_tol
        and r["normal_align"] >= 0.5
    ]
    print()
    print("--- aggregate ---")
    print(f"  E_dis            mean={agg('E_dis'):9.4f}  median={np.median([r['E_dis'] for r in rows]):9.4f}")
    print(f"  E_fc             mean={agg('E_fc'):9.4f}  median={np.median([r['E_fc'] for r in rows]):9.4f}")
    print(f"  E_pen            mean={agg('E_pen'):9.4f}  fraction>1e-3 = {np.mean([r['E_pen']>1e-3 for r in rows]):.2f}")
    print(f"  E_spen           mean={agg('E_spen'):9.4f}")
    print(f"  contact |d| mm   mean={agg('contact_abs_mean')*1000:7.2f}  max={agg('contact_abs_max', np.max)*1000:7.2f}")
    print(f"  contact pen mm   mean={agg('contact_pen_depth_max')*1000:7.2f}  frac>3mm = {np.mean([r['contact_pen_depth_max']>0.003 for r in rows]):.2f}")
    print(f"  hand pen  mm     max-of-max={agg('hand_pen_max', np.max)*1000:7.2f}  mean-of-max={agg('hand_pen_max')*1000:7.2f}")
    print(f"  hand pen  frac   mean={agg('hand_pen_frac'):.4f}")
    print(f"  normal align     mean={agg('normal_align'):.3f}")
    print(f"  joint  |q-q0|    mean={agg('joint_mean_abs_dev'):.3f} rad  max={agg('joint_max_abs_dev', np.max):.3f} rad")
    print(f"  joint  range     per-joint min/max over grasps:")
    J = np.array([r["joints"] for r in rows])
    for j, jn in enumerate(hand_model._actuated_joints_names):
        print(
            f"      {jn:18s} min={J[:, j].min():+.3f} max={J[:, j].max():+.3f} "
            f"std={J[:, j].std():.3f}  (default {float(hand_model.default_state[j]):+.3f})"
        )
    print()
    print(
        f"  FEASIBLE (|cd|max<={args.contact_tol*1000:.0f}mm, penMax<={args.pen_tol*1000:.0f}mm, "
        f"E_fc<={args.fc_tol}, align>=0.5): {len(feasible)}/{len(rows)} = {pct(len(feasible)/len(rows))}"
    )
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump(
            dict(
                hand=args.hand,
                suffix=args.suffix,
                n=len(rows),
                feasible=len(feasible),
                rows=rows,
                joint_names=hand_model._actuated_joints_names,
            ),
            open(args.json, "w"),
            indent=1,
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
