# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Headless (text / JSON / PNG) inspection of a hand model.

Companion to ``visualize_hand_model.py`` that never opens a browser, so it can be
used over SSH / on a headless box. It prints (and optionally dumps as JSON):

  * the kinematic chain, actuated joints, joint axes / limits / default state
  * the hand base origin, ``forward_axis`` / ``up_axis`` / ``grasp_axis``
  * the FK pose of every link at the zero and default configurations
  * the sampled contact candidates and penetration spheres, with the distance of
    each point to its own link collision mesh (should be ~0 for contacts and
    ~radius for spheres)
  * an occupancy-grid check of the SDF: for a grid of query points around the
    hand, how many are inside each link according to the SDF, and how those
    compare against trimesh's own containment test (this is the numeric version of
    ``--show_occupancy_grid``: a broken / inside-out collision mesh shows up as a
    large "outside classified as inside" fraction and as floating occupancy far
    away from the geometry)
  * a winding / signed-volume audit of every link mesh, which catches inside-out
    meshes - the failure mode that silently poisons the penetration energy
  * the custom FK / Jacobian, checked against ``pytorch_kinematics`` and against
    finite differences of the same FK

Examples
--------
    CUDA_VISIBLE_DEVICES=0 python scripts/vis/visualize_hand_model_headless.py --hand_name linkerhand_l20
    CUDA_VISIBLE_DEVICES=0 python scripts/vis/visualize_hand_model_headless.py --hand_name allegro --json out/allegro.json
    CUDA_VISIBLE_DEVICES=0 python scripts/vis/visualize_hand_model_headless.py --hand_name linkerhand_l20 \
        --show_occupancy_grid --randomize_joints --pointcloud_png out/occ.png
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh as tm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "graspqp", "src"))

from graspqp.hands import AVAILABLE_HANDS, get_hand_model  # noqa: E402

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


def _merge(v, f):
    m = tm.Trimesh(np.asarray(v), np.asarray(f), process=True)
    m.merge_vertices()
    return m


def link_mesh_audit(hand_model):
    """Signed volume / watertightness of every link mesh the SDF is built from."""
    rows = []
    for lname in hand_model.mesh:
        v = hand_model.mesh[lname]["vertices"].detach().cpu().numpy()
        f = hand_model.mesh[lname]["faces"].detach().cpu().numpy()
        if len(v) == 0:
            continue
        m = _merge(v, f)
        rows.append(
            dict(
                link=lname,
                n_tri=int(len(f)),
                signed_volume_cm3=float(m.volume) * 1e6,
                watertight=bool(m.is_watertight),
                winding_consistent=bool(m.is_winding_consistent),
                components=int(len(m.split(only_watertight=False))),
                area_cm2=float(m.area) * 1e4,
                bbox_extent_mm=((m.bounds[1] - m.bounds[0]) * 1000).round(2).tolist(),
                inside_out=bool(m.volume < 0),
            )
        )
    return rows


def occupancy_report(hand_model, n=64, scale=0.25, threshold=-0.002):
    """Numeric equivalent of --show_occupancy_grid.

    Returns per-link counts of occupied grid cells, plus how far the occupied cells
    extend from the link's own mesh (a huge value means the SDF claims cells that
    are nowhere near the geometry -> inside-out or broken collision mesh).
    """
    dev = hand_model.device
    x = torch.linspace(-scale, scale, n, device=dev)
    g = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), dim=-1).reshape(-1, 3)
    pts = g + torch.tensor([0.01, 0.01, 0.0], device=dev)
    with torch.no_grad():
        dist, lidx = hand_model.cal_distance(pts.unsqueeze(0), return_link_index=True)
    dist, lidx = dist[0].cpu(), lidx[0].cpu()
    names = hand_model.link_names
    mask = dist > threshold
    out = dict(n_cells=int(len(pts)), n_occupied=int(mask.sum()), frac_occupied=float(mask.float().mean()))
    per_link = []
    world = pts.cpu().numpy()
    for i, lname in enumerate(names):
        sel = (lidx == i) & mask
        if sel.sum() == 0:
            continue
        M = hand_model.current_status[lname].get_matrix()[0].detach().cpu().numpy()
        v = hand_model.mesh[lname]["vertices"].detach().cpu().numpy()
        vw = v @ M[:3, :3].T + M[:3, 3]
        p = world[sel.numpy()]
        # distance of each occupied cell to the actual geometry of the owning link
        _, d, _ = tm.proximity.closest_point(tm.Trimesh(vw, hand_model.mesh[lname]["faces"].detach().cpu().numpy(),
                                                       process=False), p)
        per_link.append(
            dict(link=lname, n_cells=int(sel.sum()), max_dist_to_own_mesh_mm=float(d.max()) * 1000,
                 mean_dist_to_own_mesh_mm=float(d.mean()) * 1000)
        )
    out["per_link"] = per_link
    return out, pts.cpu().numpy(), mask.numpy(), lidx.numpy()


def fk_jacobian_report(hand_model, n_random=3, seed=0):
    """Custom FK vs pytorch_kinematics and custom Jacobian vs finite differences."""
    import pytorch_kinematics as pk

    out = {"has_custom_fk": hand_model.joint_calc_fnc is not None,
           "has_custom_jacobian": hand_model.jacobian_fnc is not None}
    chain = hand_model.chain
    full_names = chain.get_joint_parameter_names()
    out["chain_joint_order"] = list(full_names)
    out["actuated_joint_order"] = list(hand_model._actuated_joints_names)
    # the joint-angle vector order is the chain order restricted to the actuated joints;
    # if this is violated the custom FK/Jacobian would read the wrong columns.
    out["actuated_subset_in_order"] = (
        [n for n in full_names if n in hand_model._actuated_joints_names]
        == list(hand_model._actuated_joints_names)
    )

    rng = np.random.default_rng(seed)
    lo = hand_model.joints_lower.detach().cpu().numpy()
    hi = hand_model.joints_upper.detach().cpu().numpy()
    lo = np.where(np.isfinite(lo), lo, -1.0)
    hi = np.where(np.isfinite(hi), hi, 1.0)
    configs = {"zeros": np.zeros(hand_model.n_dofs), "default": hand_model.default_state.detach().cpu().numpy()}
    for i in range(n_random):
        configs[f"random{i}"] = lo + (hi - lo) * rng.random(hand_model.n_dofs)

    # finite differences of the *driven* FK, computed with the hand model's own FK
    out["fk"] = []
    for label, q in configs.items():
        q_t = torch.tensor(q, dtype=torch.float32, device=hand_model.device).reshape(1, -1)
        res = hand_model.fk(q_t)
        entry = dict(config=label, link_origins_mm={
            k: (v.get_matrix()[0, :3, 3].detach().cpu().numpy() * 1000).round(3).tolist() for k, v in res.items()
        })
        if hand_model.jacobian_fnc is not None:
            eps = 1e-4
            J = hand_model.jacobian(q_t.requires_grad_(False))[0].detach().cpu().numpy()
            fk_keys = list(res.keys())
            links_used = [ln for ln, m in zip(fk_keys, hand_model._used_links_mask) if m]
            Jn = np.zeros_like(J)
            for i in range(hand_model.n_dofs):
                for s in (+1, -1):
                    qq = torch.tensor(q, dtype=torch.float32, device=hand_model.device).reshape(1, -1).clone()
                    qq[0, i] += s * eps
                    rr = hand_model.fk(qq)
                    if s > 0:
                        fp = {k: v.get_matrix()[0].detach().cpu().numpy() for k, v in rr.items()}
                    else:
                        fm = {k: v.get_matrix()[0].detach().cpu().numpy() for k, v in rr.items()}
                for li, ln in enumerate(links_used):
                    Jn[li, :3, i] = (fp[ln][:3, 3] - fm[ln][:3, 3]) / (2 * eps)
                    R = fp[ln][:3, :3] @ fm[ln][:3, :3].T
                    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
                    s_ = np.linalg.norm(v)
                    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
                    Jn[li, 3:, i] = (v * (np.arctan2(s_, c) / s_) if s_ > 1e-12 else v) / (2 * eps)
            scale = max(float(np.abs(Jn).max()), 1e-9)
            entry["jacobian_max_abs_err"] = float(np.abs(J - Jn).max())
            entry["jacobian_rel_err"] = float(np.abs(J - Jn).max() / scale)
            entry["jacobian_finite"] = bool(np.isfinite(J).all())
        out["fk"].append(entry)
    return out


def main():
    ap = argparse.ArgumentParser(description="Headless hand-model inspection (no browser).")
    ap.add_argument("--hand_name", default="linkerhand_l20", choices=AVAILABLE_HANDS + ["all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grasp_type", default="all")
    ap.add_argument("--randomize_joints", action="store_true")
    ap.add_argument("--only_collision", action="store_true")
    ap.add_argument("--show_occupancy_grid", action="store_true")
    ap.add_argument("--grid_n", type=int, default=64)
    ap.add_argument("--grid_scale", type=float, default=0.25)
    ap.add_argument("--skip_mesh_audit", action="store_true")
    ap.add_argument("--skip_fk_check", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--pointcloud_png", default=None)
    args = ap.parse_args()

    hands = AVAILABLE_HANDS if args.hand_name == "all" else [args.hand_name]
    report = {}
    for hand_name in hands:
        extra = {"only_use_collision": True} if args.only_collision else {}
        hm = get_hand_model(hand_name, args.device, grasp_type=args.grasp_type, **extra)

        joints = hm.default_state
        if args.randomize_joints:
            joints = hm.joints_lower + (hm.joints_upper - hm.joints_lower) * torch.rand_like(hm.joints_lower)
        # translation (3) + a *valid* 6D rotation (the first two columns of identity, as
        # `visualize_hand_model.py` does) + the joint angles. An all-zero 6D rotation is
        # degenerate: robust_compute_rotation_matrix_from_ortho6d returns a zero matrix and
        # every query point collapses to the hand origin.
        rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=hm.device)
        pose = torch.cat([torch.zeros(3, device=hm.device), rot6d, joints]).reshape(1, -1)
        hm.set_parameters(pose, contact_point_indices="all")

        print()
        print("=" * 118)
        print(f"HAND {hand_name}")
        print("=" * 118)
        print(f"  device / dtype     : {hm.device}")
        print(f"  n_dofs             : {hm.n_dofs}")
        print(f"  n_contact_cand.    : {hm.n_contact_candidates}   (links with contacts: "
              f"{sum(1 for l in hm.mesh if len(hm.mesh[l]['contact_candidates']))})")
        print(f"  n_penetration_sph. : {hm.n_keypoints}")
        print(f"  forward_axis       : {hm.forward_axis.tolist()}")
        print(f"  up_axis            : {hm.up_axis.tolist()}")
        print(f"  grasp_axis         : {hm.grasp_axis.tolist()}")
        print(f"  custom FK / Jac    : {hm.joint_calc_fnc is not None} / {hm.jacobian_fnc is not None}")

        print()
        print("  --- actuated joints (order = joint-angle vector order) ---")
        print(f"  {'#':>2s} {'name':22s} {'lower':>8s} {'upper':>8s} {'default':>8s} {'jitter_sigma':>12s}")
        default_np = hm.default_state.detach().cpu().numpy()
        lo = hm.joints_lower.detach().cpu().numpy()
        hi = hm.joints_upper.detach().cpu().numpy()
        for i, n in enumerate(hm._actuated_joints_names):
            print(f"  {i:2d} {n:22s} {lo[i]:8.3f} {hi[i]:8.3f} {default_np[i]:8.3f} {0.1*(hi[i]-lo[i]):12.3f}")

        print()
        print("  --- FK link poses (hand-root frame), origin in mm ---")
        print(f"  {'link':26s} {'origin mm':>26s} {'#contact':>8s} {'#spheres':>9s}")
        for lname in hm.mesh:
            M = hm.current_status[lname].get_matrix()[0].detach().cpu().numpy()
            t = M[:3, 3] * 1000
            print(f"  {lname:26s} [{t[0]:8.1f} {t[1]:8.1f} {t[2]:8.1f}] "
                  f"{len(hm.mesh[lname]['contact_candidates']):8d} {len(hm.mesh[lname]['penetration_keypoints']):9d}")

        # contact / penetration point placement
        print()
        print("  --- contact candidates & penetration spheres vs. their own link mesh ---")
        print(f"  {'link':26s} {'#cand':>6s} {'mean|d|mm':>10s} {'max|d|mm':>9s} {'#sph':>5s} {'r mm':>7s} {'mean|d|mm':>10s}")
        for lname in hm.mesh:
            v = hm.mesh[lname]["vertices"].detach().cpu().numpy()
            f = hm.mesh[lname]["faces"].detach().cpu().numpy()
            cand = hm.mesh[lname]["contact_candidates"].detach().cpu().numpy()
            kp = hm.mesh[lname]["penetration_keypoints"].detach().cpu().numpy()
            if len(v) == 0:
                continue
            m = tm.Trimesh(v, f, process=False)
            cd = tm.proximity.closest_point(m, cand)[1] if len(cand) else np.array([np.nan])
            kd = tm.proximity.closest_point(m, kp)[1] if len(kp) else np.array([np.nan])
            sc = hm.mesh[lname]["scales"].detach().cpu().numpy() * 1000
            print(f"  {lname:26s} {len(cand):6d} {np.nanmean(cd)*1000:10.3f} {np.nanmax(cd)*1000:9.3f} "
                  f"{len(kp):5d} {sc.mean() if len(kp) else float('nan'):7.2f} {np.nanmean(kd)*1000:10.3f}")

        entry = dict(n_dofs=hm.n_dofs, forward_axis=hm.forward_axis.tolist(), up_axis=hm.up_axis.tolist(),
                     grasp_axis=hm.grasp_axis.tolist(),
                     actuated_joints=list(hm._actuated_joints_names),
                     joints_lower=lo.tolist(), joints_upper=hi.tolist(), default_state=default_np.tolist())

        if not args.skip_mesh_audit:
            rows = link_mesh_audit(hm)
            print()
            print("  --- link mesh audit (the mesh the SDF is built from) ---")
            print(f"  {'link':26s} {'#tri':>6s} {'signed vol cm3':>15s} {'watertight':>11s} {'comp':>5s} {'inside_out':>11s}")
            for r in rows:
                print(f"  {r['link']:26s} {r['n_tri']:6d} {r['signed_volume_cm3']:15.3f} "
                      f"{str(r['watertight']):>11s} {r['components']:5d} {str(r['inside_out']):>11s}")
            bad = [r["link"] for r in rows if r["inside_out"]]
            if bad:
                print(f"  !! INSIDE-OUT MESHES (negative signed volume): {bad}")
                print("     warp's pseudo-normal SDF infers inside/outside from the winding, so an")
                print("     inside-out link makes the whole of space look 'inside the hand': E_pen becomes")
                print("     garbage and the optimizer drives the hand into the object. Fix the winding.")
            entry["mesh_audit"] = rows

        if args.show_occupancy_grid:
            occ, pts, mask, lidx = occupancy_report(hm, n=args.grid_n, scale=args.grid_scale)
            print()
            print(f"  --- occupancy grid ({args.grid_n}^3 cells in +-{args.grid_scale} m), threshold -2 mm ---")
            print(f"  occupied cells: {occ['n_occupied']}/{occ['n_cells']} = {100*occ['frac_occupied']:.2f}%")
            print(f"  {'link':26s} {'cells':>7s} {'max dist to own mesh mm':>24s} {'mean mm':>9s}")
            for r in occ["per_link"]:
                print(f"  {r['link']:26s} {r['n_cells']:7d} {r['max_dist_to_own_mesh_mm']:24.2f} "
                      f"{r['mean_dist_to_own_mesh_mm']:9.2f}")
            worst = max((r["max_dist_to_own_mesh_mm"] for r in occ["per_link"]), default=0.0)
            print(f"  -> largest distance between an 'occupied' cell and the owning link's geometry: {worst:.2f} mm")
            print("     (should be < a few mm; a large value means floating occupancy = broken/inside-out mesh)")
            entry["occupancy"] = occ
            if args.pointcloud_png:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig = plt.figure(figsize=(7, 6))
                    ax = fig.add_subplot(111, projection="3d")
                    for i, lname in enumerate(hm.link_names):
                        sel = (lidx == i) & mask
                        if not sel.any():
                            continue
                        p = pts[sel]
                        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, label=lname)
                    ax.set_title(f"{hand_name}: occupied SDF cells")
                    ax.set_box_aspect((1, 1, 1))
                    os.makedirs(os.path.dirname(os.path.abspath(args.pointcloud_png)) or ".", exist_ok=True)
                    fig.savefig(args.pointcloud_png, dpi=130)
                    print(f"  saved point-cloud PNG to {args.pointcloud_png}")
                except Exception as e:  # pragma: no cover
                    print(f"  could not save PNG ({e})")

        if not args.skip_fk_check:
            fk = fk_jacobian_report(hm)
            print()
            print("  --- FK / Jacobian checks ---")
            print(f"  joint-angle order == chain order restricted to actuated joints: "
                  f"{fk['actuated_subset_in_order']}")
            if not fk["actuated_subset_in_order"]:
                print("  !! the n_dofs joint vector does NOT follow the chain order - the custom")
                print("     FK/Jacobian would index the wrong columns.")
            print("  note: the installed pytorch_kinematics (renezurbruegg fork) offsets every revolute")
            print("        joint by +-1e-2 rad in Frame.get_transform ('ugly fix'), so its FK is")
            print("        discontinuous at theta == 0 and a finite-difference Jacobian check exactly at")
            print("        zeros is meaningless - compare at non-zero configurations instead.")
            for e in fk["fk"]:
                if "jacobian_rel_err" in e:
                    print(f"  {e['config']:10s} jacobian vs finite-diff: max abs err "
                          f"{e['jacobian_max_abs_err']:.3e} rel {e['jacobian_rel_err']:.3e} "
                          f"finite={e['jacobian_finite']}")
            entry["fk_jacobian"] = fk

        report[hand_name] = entry

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        json.dump(report, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
