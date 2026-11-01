"""Probe `initialize_convex_hull` for different hand axis conventions / presets.

Reports the *initial* grasp statistics: E_dis, E_fc, E_pen, the fraction of object
surface points that fall inside the hand (hand->object penetration), contact
distances, normal alignment and the joint spread. Cheap (no optimization), so it
isolates the initialization/axes question from the optimizer.

Usage:
  CUDA_VISIBLE_DEVICES=0 python _diag/init_probe.py --hand linkerhand_l20 --object apple
  CUDA_VISIBLE_DEVICES=0 python _diag/init_probe.py --hand allegro --object apple
  CUDA_VISIBLE_DEVICES=0 python _diag/init_probe.py --axis_sets "x,z;x,y" --urdf _diag/x.urdf
"""

import argparse
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))

from graspqp.core import ObjectModel  # noqa: E402
from graspqp.core.energy import calculate_energy  # noqa: E402
from graspqp.core.initializations import initialize_convex_hull  # noqa: E402
from graspqp.hands import get_hand_model  # noqa: E402
from graspqp.metrics import GraspSpanMetricFactory  # noqa: E402

PRESETS = {
    "flat": None,  # whatever the hand module defines
    "pregrasp": [0.0, 0.30, 0.40, 0.0, 0.30, 0.40, 0.0, 0.30, 0.40, 0.0, 0.30, 0.40, 0.50, 0.30, 0.20, 0.30],
}


def make_args(batch_size, n_contact, jitter=0.1):
    return types.SimpleNamespace(
        distance_lower=0.05, distance_upper=0.1,
        rotate_lower=-np.pi, rotate_upper=np.pi,
        pitch_lower=-15 * np.pi / 180, pitch_upper=15 * np.pi / 180,
        tilt_lower=-45 * np.pi / 180, tilt_upper=45 * np.pi / 180,
        jitter_strength=jitter, n_contact=n_contact, batch_size=batch_size,
    )


def stats(hand_model, object_model, energy_fnc):
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
        frac = (hsd > 0).float().mean(-1)
        pmax = hsd.clamp(min=0).max(-1).values
        align = torch.sum(-cnorm * hand_model.contact_normals, dim=-1).mean(-1)
        jd = (hand_model.hand_pose[:, 9:] - hand_model.default_state.unsqueeze(0)).abs()
    return dict(
        E_dis=float(losses["E_dis"].mean()), E_fc=float(losses["E_fc"].mean()),
        E_pen=float(losses["E_pen"].mean()), E_spen=float(losses["E_spen"].mean()),
        pen_frac=float(frac.mean()), pen_max_mm=float(pmax.mean()) * 1000,
        cd_mm=float(cdist.abs().mean()) * 1000, align=float(align.mean()),
        jdev=float(jd.mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="linkerhand_l20")
    ap.add_argument("--object", default="apple")
    ap.add_argument("--data_root", default="min_dataset/release2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_contact", type=int, default=12)
    ap.add_argument("--axis_sets", default=None)
    ap.add_argument("--urdf", default=None, help="override the URDF path")
    ap.add_argument("--presets", default="flat", help="comma list of default_state presets")
    ap.add_argument("--seeds", default="1")
    args = ap.parse_args()
    dev = args.device

    if args.axis_sets:
        axis_sets = [tuple(s.split(",")) for s in args.axis_sets.split(";")]
    elif args.hand == "linkerhand_l20":
        axis_sets = [("z", "x")]
    else:
        axis_sets = [(None, None)]

    energy_fnc = GraspSpanMetricFactory.create(
        GraspSpanMetricFactory.MetricType.GRASPQP,
        solver_kwargs={"friction": 0.2, "max_limit": 20.0, "n_cone_vecs": 4},
    )

    hdr = (
        f"{'config':26s} {'urdf':>6s} {'preset':9s} {'E_dis':>8s} {'E_fc':>8s} {'E_pen':>9s} "
        f"{'penFrac':>8s} {'penMax':>8s} {'|cd| mm':>8s} {'align':>6s} {'|q-q0|':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for fwd, up in axis_sets:
        for preset in [p.strip() for p in args.presets.split(",")]:
            for seed in [int(s) for s in args.seeds.split(",")]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                kwargs = {}
                if fwd is not None:
                    kwargs.update(forward_axis=fwd, up_axis=up)
                if args.urdf:
                    kwargs["mjcf_path"] = os.path.abspath(args.urdf)
                if PRESETS[preset] is not None:
                    kwargs["default_state"] = torch.tensor(PRESETS[preset], dtype=torch.float, device=dev)
                hand_model = get_hand_model(args.hand, dev, grasp_type="all", **kwargs)
                object_model = ObjectModel(
                    data_root_path=os.path.join(ROOT, args.data_root),
                    batch_size_each=args.batch_size, num_samples=2500, device=dev, scale=1.0,
                )
                object_model.initialize([os.path.join(ROOT, args.data_root, args.object)])
                initialize_convex_hull(hand_model, object_model, make_args(args.batch_size, args.n_contact))
                s = stats(hand_model, object_model, energy_fnc)
                tag = f"fwd={fwd} up={up}" if fwd else "default"
                print(
                    f"{tag:26s} {('new' if args.urdf else 'orig'):>6s} {preset:9s} {s['E_dis']:8.4f} "
                    f"{s['E_fc']:8.3f} {s['E_pen']:9.4f} {100*s['pen_frac']:7.1f}% {s['pen_max_mm']:8.2f} "
                    f"{s['cd_mm']:8.2f} {s['align']:6.3f} {s['jdev']:7.3f}"
                )
                del hand_model, object_model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
