# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Headless (text / JSON / PNG) evaluation of the grasps written by ``scripts/fit.py``.

Drop-in analysis companion to ``visualize_result.py`` that never opens a browser.
For every saved ``*.dexgrasp.pt`` checkpoint it reloads the hand and the object,
re-scores the grasps and prints a table with:

  * the saved total energy and the individual energy terms
    (``E_dis``, ``E_fc``, ``E_pen``, ``E_spen``, ``E_joints``) recomputed with the
    current hand/object model
  * contact point signed distance to the object surface (positive = inside the
    object, i.e. the contact point has sunk into the object)
  * the contact-normal vs. object-outward-normal alignment (1 = perfect)
  * hand -> object penetration: how many object surface points lie inside the
    hand and how deep (this is what "the hand clearly penetrates the object"
    looks like numerically)
  * self penetration
  * the joint angles, their deviation from ``default_state`` and their spread over
    the batch (a hand whose fingers "never bend" shows a tiny spread here)
  * an explicit, fully configurable feasibility criterion

Examples
--------
    CUDA_VISIBLE_DEVICES=0 python scripts/vis/visualize_result_headless.py \
        --hand_name linkerhand_l20 --data_root min_dataset/release2

    CUDA_VISIBLE_DEVICES=0 python scripts/vis/visualize_result_headless.py \
        --hand_name allegro --data_root min_dataset/release2 --suffix _step_2000 \
        --json out/allegro.json --png_dir out/png
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import roma
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "graspqp", "src"))

from graspqp.core import ObjectModel  # noqa: E402
from graspqp.core.energy import calculate_energy  # noqa: E402
from graspqp.hands import AVAILABLE_HANDS, get_hand_model  # noqa: E402
from graspqp.metrics import GraspSpanMetricFactory  # noqa: E402

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


def _pct(x):
    return f"{100.0 * x:5.1f}%"


def fix_contact_normal_signs(hand_model, eps=25e-5):
    """Flip cached contact normals that point into their own link.

    ``HandModel`` derives the contact normal from the closest *triangle* of the collision
    mesh. For a link made of several shells the closest triangle can belong to an inner
    part, so the normal points inwards and the contact-distance weighting
    ``exp(1 - <object normal, hand normal>)`` — and the exported ``closing_force_des`` —
    are wrong. This probes the SDF a quarter millimetre either side of every candidate
    along its cached normal and flips the ones that are inconsistent.

    Returns the number of flipped normals. The change is in-memory only.

    Note: on the LinkerHand L20 this *lowered* the measured ``normal_align`` (24 % -> 13 %
    feasible) because the probe is ambiguous within a fraction of a millimetre of a soup
    mesh, so treat it as a diagnostic, not as a recommended fix.
    """
    from graspqp.utils import warp as wp_utils

    flipped = 0
    for lname in hand_model.mesh:
        cand = hand_model.mesh[lname]["contact_candidates"]
        nrm = hand_model.mesh[lname]["normal_candidates"]
        if len(cand) == 0:
            continue
        fv = hand_model.mesh[lname]["face_verts"]
        q = torch.cat([cand + eps * nrm, cand - eps * nrm], 0).contiguous()
        d, _ = wp_utils.CalcSdfField.apply(q, fv.id, 1e6, True)  # exterior positive
        outward = (d[: len(cand)] > 0) & (d[len(cand):] < 0)
        if (~outward).any():
            nrm[~outward] = -nrm[~outward]
            hand_model.mesh[lname]["normal_candidates"] = nrm
            flipped += int((~outward).sum())
    # rebuild the flattened candidate tensors the energies actually read
    hand_model.contact_normals_candidates = torch.cat(
        [hand_model.mesh[ln]["normal_candidates"] for ln in hand_model.mesh], dim=0
    )
    return flipped


def evaluate_object(hand_model, energy_fnc, args, obj, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    params = {k: v.detach().cpu() for k, v in ck["parameters"].items()}
    root_pose = params["root_pose"].to(args.device).float()
    joint_states = torch.stack([params[j] for j in hand_model._actuated_joints_names], dim=-1).to(args.device)
    if root_pose.ndim == 1 or len(root_pose) == 0:
        return []

    n = min(args.max_grasps, len(root_pose))
    saved = ck.get("values")
    order = torch.argsort(saved)[:n] if saved is not None else torch.arange(n)
    root_pose, joint_states = root_pose[order], joint_states[order]
    contact_idxs = ck.get("contact_idx")
    if contact_idxs is not None:
        contact_idxs = contact_idxs.to(args.device)[order]

    object_model = ObjectModel(
        data_root_path=os.path.join(args.data_root), batch_size_each=n,
        num_samples=args.num_samples, device=args.device, scale=1.0,
    )
    object_model.initialize([os.path.join(args.data_root, obj)])

    if ck.get("offset") is not None:
        off = ck["offset"].to(args.device)
        if off.ndim == 1:
            off = off.unsqueeze(0)
        if off.shape[0] != n:
            off = off.expand(n, -1)
        R_off, t_off = roma.unitquat_to_rotmat(off[..., [4, 5, 6, 3]]), off[:, :3]
        R_w, t_w = roma.unitquat_to_rotmat(root_pose[..., [4, 5, 6, 3]]), root_pose[:, :3]
        R_new = R_w @ R_off
        t_new = torch.einsum("nij,nj->ni", R_w, t_off) + t_w
        root_pose = torch.cat([t_new, roma.rotmat_to_unitquat(R_new)[..., [3, 0, 1, 2]]], dim=-1)

    R6 = roma.unitquat_to_rotmat(root_pose[..., [4, 5, 6, 3]]).mT.flatten(1, 2)
    hand_params = torch.cat([root_pose[..., :3], R6[..., :6], joint_states], dim=-1).to(args.device)
    hand_model.set_parameters(
        hand_params, contact_point_indices=contact_idxs if contact_idxs is not None else "all"
    )

    with torch.no_grad():
        losses = calculate_energy(
            hand_model, object_model, energy_fnc=energy_fnc,
            energy_names=["E_dis", "E_fc", "E_pen", "E_spen", "E_joints"],
            method="gendexgrasp", svd_gain=0.1,
        )
        cdist, cnorm = object_model.cal_distance(hand_model.contact_points)
        scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
        osp = object_model.surface_points_tensor * scale
        hsd = hand_model.cal_distance(osp)
        pen_frac = (hsd > 0).float().mean(-1)
        pen_max = hsd.clamp(min=0).max(-1).values
        align = torch.sum(-cnorm * hand_model.contact_normals, dim=-1).mean(-1)
        jdev = (joint_states - hand_model.default_state.unsqueeze(0)).abs()

    rows = []
    for i in range(n):
        rows.append(dict(
            object=obj, slot=int(order[i]),
            E_saved=float(saved[order[i]]) if saved is not None else float("nan"),
            E_dis=float(losses["E_dis"][i]), E_fc=float(losses["E_fc"][i]),
            E_pen=float(losses["E_pen"][i]), E_spen=float(losses["E_spen"][i]),
            E_joints=float(losses["E_joints"][i]),
            contact_abs_mean=float(cdist.abs().mean(-1)[i]),
            contact_abs_max=float(cdist.abs().max(-1).values[i]),
            contact_pen_max=float(cdist.max(-1).values[i]),
            normal_align=float(align[i]),
            hand_pen_frac=float(pen_frac[i]), hand_pen_max=float(pen_max[i]),
            joint_mean_abs_dev=float(jdev.mean(-1)[i]), joint_max_abs_dev=float(jdev.max(-1).values[i]),
            joints=np.round(joint_states[i].cpu().numpy(), 4).tolist(),
            root_t=np.round(root_pose[i, :3].cpu().numpy(), 5).tolist(),
        ))

    if args.png_dir:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(args.png_dir, exist_ok=True)
            for i in range(min(n, args.png_max)):
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111, projection="3d")
                p = osp[i].cpu().numpy()
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c="tab:green", label="object")
                cp = hand_model.contact_points[i].cpu().numpy()
                ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], s=25, c="tab:red", label="contacts")
                ax.set_title(f"{obj} #{rows[i]['slot']}  E={rows[i]['E_saved']:.2f}")
                fig.savefig(os.path.join(args.png_dir, f"{obj}_{rows[i]['slot']}.png"), dpi=110)
                plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"  could not write PNGs ({e})")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Headless evaluation of fit.py outputs.")
    ap.add_argument("--hand_name", default="linkerhand_l20", choices=AVAILABLE_HANDS)
    ap.add_argument("--data_root", default="min_dataset/release2")
    ap.add_argument("--num_contacts", default="12_contacts")
    ap.add_argument("--energy", default="graspqp")
    ap.add_argument("--grasp_type", default="default")
    ap.add_argument("--suffix", default="", help="'' = final checkpoint, or e.g. _step_2000")
    ap.add_argument("--max_grasps", type=int, default=32)
    ap.add_argument("--num_samples", type=int, default=2500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--friction", type=float, default=0.2)
    ap.add_argument("--max_lambda_limit", type=float, default=20.0)
    ap.add_argument("--n_friction_cone", type=int, default=4)
    # feasibility criterion
    ap.add_argument("--contact_tol", type=float, default=0.005)
    ap.add_argument("--pen_tol", type=float, default=0.005)
    ap.add_argument("--fc_tol", type=float, default=1.0)
    ap.add_argument("--align_tol", type=float, default=0.5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--png_dir", default=None)
    ap.add_argument("--png_max", type=int, default=8)
    ap.add_argument("--fix_contact_normals", action="store_true",
                    help="flip cached contact normals that point into their own link (in memory)")
    args = ap.parse_args()

    args.data_root = os.path.abspath(args.data_root)
    hand_model = get_hand_model(args.hand_name, args.device, use_collision_if_possible=True, grasp_type="all")
    if args.fix_contact_normals:
        n = fix_contact_normal_signs(hand_model)
        print(f"[fix] flipped {n} inward-pointing contact normals")
    energy_fnc = GraspSpanMetricFactory.create(
        GraspSpanMetricFactory.MetricType.GRASPQP,
        solver_kwargs={"friction": args.friction, "max_limit": args.max_lambda_limit,
                       "n_cone_vecs": args.n_friction_cone},
    )

    objects = sorted(o for o in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, o)))
    rows = []
    for obj in objects:
        d = os.path.join(args.data_root, obj, "grasp_predictions", args.hand_name,
                         args.num_contacts, args.energy, args.grasp_type)
        pat = os.path.join(d, f"*{args.suffix}.dexgrasp.pt")
        files = sorted(glob.glob(pat))
        if not files:
            print(f"[skip] no checkpoint for {obj} ({pat})")
            continue
        exact = [f for f in files if os.path.basename(f) == f"{obj}{args.suffix}.dexgrasp.pt"]
        path = exact[0] if exact else files[-1]
        print(f"[load] {os.path.relpath(path)}")
        rows += evaluate_object(hand_model, energy_fnc, args, obj, path)

    if not rows:
        print("nothing to evaluate")
        return

    print()
    print("=" * 150)
    print(f"grasp evaluation  hand={args.hand_name}  suffix='{args.suffix}'  n={len(rows)}")
    print("=" * 150)
    hdr = (f"{'object':18s} {'#':>3s} {'E_saved':>9s} {'E_dis':>8s} {'E_fc':>7s} {'E_pen':>8s} {'E_spen':>7s} "
           f"{'|cd|me':>7s} {'|cd|mx':>7s} {'penMax':>7s} {'penFr':>7s} {'align':>6s} {'dJnt':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['object'][:18]:18s} {r['slot']:3d} {r['E_saved']:9.3f} {r['E_dis']:8.3f} {r['E_fc']:7.4f} "
              f"{r['E_pen']:8.4f} {r['E_spen']:7.4f} {r['contact_abs_mean']*1000:7.2f} "
              f"{r['contact_abs_max']*1000:7.2f} {r['hand_pen_max']*1000:7.2f} {_pct(r['hand_pen_frac']):>7s} "
              f"{r['normal_align']:6.3f} {r['joint_mean_abs_dev']:6.3f}")

    def m(k, f=np.mean):
        return f([r[k] for r in rows])

    feas = [r for r in rows
            if r["contact_abs_max"] <= args.contact_tol
            and r["hand_pen_max"] <= args.pen_tol
            and r["E_fc"] <= args.fc_tol
            and r["normal_align"] >= args.align_tol]
    print()
    print("--- aggregate ---")
    print(f"  E_saved                 mean={m('E_saved'):10.3f}")
    print(f"  E_dis                   mean={m('E_dis'):10.4f}   median={np.median([r['E_dis'] for r in rows]):10.4f}")
    print(f"  E_fc                    mean={m('E_fc'):10.4f}   median={np.median([r['E_fc'] for r in rows]):10.4f}")
    print(f"  E_pen                   mean={m('E_pen'):10.4f}   frac>1e-3 {np.mean([r['E_pen'] > 1e-3 for r in rows]):.2f}")
    print(f"  E_spen                  mean={m('E_spen'):10.4f}")
    print(f"  contact |d| [mm]        mean={m('contact_abs_mean')*1000:7.2f}  max={m('contact_abs_max', np.max)*1000:7.2f}")
    print(f"  contact penetration     mean={m('contact_pen_max')*1000:7.2f} mm  frac>3mm "
          f"{np.mean([r['contact_pen_max'] > 0.003 for r in rows]):.2f}")
    print(f"  hand->object penetration max-of-max={m('hand_pen_max', np.max)*1000:7.2f} mm  "
          f"mean-of-max={m('hand_pen_max')*1000:7.2f} mm  mean point fraction={m('hand_pen_frac'):.4f}")
    print(f"  normal alignment        mean={m('normal_align'):7.3f}")
    print(f"  joint deviation         mean={m('joint_mean_abs_dev'):7.3f} rad   max={m('joint_max_abs_dev', np.max):7.3f} rad")
    J = np.array([r["joints"] for r in rows])
    print("  per-joint range over the batch:")
    for j, jn in enumerate(hand_model._actuated_joints_names):
        print(f"      {jn:20s} min={J[:, j].min():+.3f} max={J[:, j].max():+.3f} std={J[:, j].std():.3f} "
              f"default={float(hand_model.default_state[j]):+.3f}")
    print()
    print(f"  FEASIBLE  (|cd|max<={args.contact_tol*1000:.0f}mm, penMax<={args.pen_tol*1000:.0f}mm, "
          f"E_fc<={args.fc_tol}, align>={args.align_tol}): {len(feas)}/{len(rows)} = "
          f"{_pct(len(feas) / len(rows))}")
    # per-object summary
    print()
    print("  per object:")
    for obj in sorted({r["object"] for r in rows}):
        sub = [r for r in rows if r["object"] == obj]
        f = [r for r in feas if r["object"] == obj]
        print(f"      {obj[:34]:34s} n={len(sub):3d}  feasible={len(f):3d}  "
              f"E_dis={np.mean([r['E_dis'] for r in sub]):8.3f}  penMax={np.mean([r['hand_pen_max'] for r in sub])*1000:7.2f} mm")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        json.dump(dict(hand=args.hand_name, suffix=args.suffix, n=len(rows), feasible=len(feas), rows=rows,
                       joint_names=list(hand_model._actuated_joints_names)), open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
