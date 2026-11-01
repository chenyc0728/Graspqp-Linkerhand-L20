"""Attribute hand<->object penetration to individual hand links, at the pose produced
by initialize_convex_hull (or at a pose saved in a .pt checkpoint).

Also reports signed volume / winding sanity for every link mesh, because the warp
pseudo-normal SDF infers inside/outside from the winding - an inside-out mesh makes
every far-away point look "inside" and poisons E_pen globally.

Usage:
  CUDA_VISIBLE_DEVICES=0 python _diag/pen_breakdown.py --hand linkerhand_l20 --object apple --urdf ...
"""

import argparse
import os
import sys
import types

import numpy as np
import torch
import trimesh as tm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))

from graspqp.core import ObjectModel  # noqa: E402
from graspqp.core.initializations import initialize_convex_hull  # noqa: E402
from graspqp.hands import get_hand_model  # noqa: E402


def make_args(batch_size, n_contact, jitter=0.1):
    return types.SimpleNamespace(
        distance_lower=0.05, distance_upper=0.1,
        rotate_lower=-np.pi, rotate_upper=np.pi,
        pitch_lower=-15 * np.pi / 180, pitch_upper=15 * np.pi / 180,
        tilt_lower=-45 * np.pi / 180, tilt_upper=45 * np.pi / 180,
        jitter_strength=jitter, n_contact=n_contact, batch_size=batch_size,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="linkerhand_l20")
    ap.add_argument("--object", default="apple")
    ap.add_argument("--data_root", default="min_dataset/release2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_contact", type=int, default=12)
    ap.add_argument("--urdf", default=None)
    args = ap.parse_args()
    dev = args.device

    kw = {"mjcf_path": os.path.abspath(args.urdf)} if args.urdf else {}
    hand_model = get_hand_model(args.hand, dev, grasp_type="all", **kw)
    object_model = ObjectModel(
        data_root_path=os.path.join(ROOT, args.data_root), batch_size_each=args.batch_size,
        num_samples=1000, device=dev, scale=1.0,
    )
    object_model.initialize([os.path.join(ROOT, args.data_root, args.object)])
    torch.manual_seed(1)
    np.random.seed(1)
    initialize_convex_hull(hand_model, object_model, make_args(args.batch_size, args.n_contact))

    # ---- mesh winding sanity ----
    print("=== link mesh winding / volume (signed volume of the raw mesh as HandModel loads it) ===")
    print(f"{'link':26s} {'signed vol cm3':>15s} {'abs vol cm3':>12s} {'winding_ok':>11s}")
    for lname in hand_model.mesh:
        v = hand_model.mesh[lname]["vertices"].cpu().numpy()
        f = hand_model.mesh[lname]["faces"].cpu().numpy()
        m = tm.Trimesh(v, f, process=True)
        m.merge_vertices()
        print(f"{lname:26s} {m.volume*1e6:15.3f} {abs(m.volume)*1e6:12.3f} {str(m.is_winding_consistent):>11s}")

    # ---- per-link penetration attribution ----
    with torch.no_grad():
        scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
        osp = object_model.surface_points_tensor * scale
        hsd, lidx = hand_model.cal_distance(osp, return_link_index=True)
        names = hand_model.link_names
        print()
        print("=== per-link penetration attribution (object surface points inside the hand) ===")
        print(f"{'link':26s} {'n_pts':>8s} {'frac':>7s} {'max mm':>8s} {'mean mm':>8s}")
        total = hsd.numel()
        for i, lname in enumerate(names):
            sel = (lidx == i) & (hsd > 0)
            if sel.sum() == 0:
                continue
            d = hsd[sel]
            print(f"{lname:26s} {int(sel.sum()):8d} {100*sel.sum().item()/total:6.2f}% {d.max()*1000:8.2f} {d.mean()*1000:8.2f}")
        allpos = hsd > 0
        print(f"{'TOTAL':26s} {int(allpos.sum()):8d} {100*allpos.sum().item()/total:6.2f}% {hsd[allpos].max()*1000:8.2f} {hsd[allpos].mean()*1000:8.2f}")

        # same, excluding any provided links
        for excl in ap.parse_args().hand and [["thumb_metacarpals_base2"]]:
            keep = [i for i, n in enumerate(names) if n not in excl]
            print()
            print(f"--- excluding {excl} ---")
            # recompute using only the kept links
            dis = []
            x = (osp - hand_model.global_translation.unsqueeze(1)) @ hand_model.global_rotation
            from graspqp.utils import warp as wp_utils
            import warp as wp

            link_poses = torch.stack(
                [hand_model.current_status[n].get_matrix().expand(x.shape[0], -1, -1) for n in names], dim=1
            )
            R = link_poses[:, :, :3, :3]
            t = link_poses[:, :, :3, 3]
            x_local = torch.matmul(x.unsqueeze(1) - t.unsqueeze(2), R)
            for i, n in enumerate(names):
                if n in excl:
                    continue
                d, _ = wp_utils.CalcSdfField.apply(
                    x_local[:, i].contiguous(), hand_model.mesh[n]["face_verts"].id, 1e6, False
                )
                dis.append(d)
            st = torch.stack(dis, 0)
            best = st.max(0).values
            print(f"{'TOTAL(excl)':26s} {int((best>0).sum()):8d} {100*(best>0).float().mean().item():6.2f}% "
                  f"{best.max()*1000:8.2f} {best[best>0].mean()*1000 if (best>0).any() else 0:8.2f}")


if __name__ == "__main__":
    main()
