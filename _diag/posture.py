"""Posture analysis of saved grasps: do the fingers curl into a usable shape?

Loads the joint angles from saved .pt checkpoints and runs FK (CPU) to report, per
grasp, the spread of the five fingertips and the thumb-vs-fingertips distance.
A "flat hand" has fingertips ~15-20 cm apart; a closing grasp brings the thumb tip
within ~3-5 cm of the finger tips.

Usage:
  python _diag/posture.py --hand linkerhand_l20 --root min_dataset/release2
  python _diag/posture.py --hand allegro --root _diag/backup/grasp_predictions_before
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src", "graspqp", "hands"))

TIP_LINKS = {
    "linkerhand_l20": ["index_distal", "middle_distal", "ring_distal", "pinky_distal", "thumb_distal"],
    "allegro": ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="linkerhand_l20")
    ap.add_argument("--root", default="min_dataset/release2")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    if args.hand == "linkerhand_l20":
        from linkerhand_l20 import _build_full_angles
        import pytorch_kinematics as pk

        chain = pk.build_chain_from_urdf(open(f"{ROOT}/graspqp/assets/linkerhand_l20/linkerhand_l20.urdf").read())
        names = list(chain.get_joint_parameter_names())
        actuated = ["pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip", "ring_mcp_roll", "ring_mcp_pitch",
                    "ring_pip", "middle_mcp_roll", "middle_mcp_pitch", "middle_pip", "index_mcp_roll",
                    "index_mcp_pitch", "index_pip", "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"]

        def tip_positions(q16):
            fk = chain.forward_kinematics(_build_full_angles(torch.tensor(q16, dtype=torch.float32).reshape(1, -1)))
            return np.stack([fk[k].get_matrix()[0, :3, 3].numpy() for k in TIP_LINKS["linkerhand_l20"]])
    else:
        import pytorch_kinematics as pk

        chain = pk.build_chain_from_urdf(open(f"{ROOT}/graspqp/assets/allegro/allegro_hand.urdf").read())
        names = list(chain.get_joint_parameter_names())
        actuated = names

        def tip_positions(q16):
            fk = chain.forward_kinematics(torch.tensor(q16, dtype=torch.float32).reshape(1, -1))
            return np.stack([fk[k].get_matrix()[0, :3, 3].numpy() for k in TIP_LINKS["allegro"]])

    print(f"hand={args.hand}  root={args.root}")
    print(f"{'object':18s} {'i':>3s}  {'tip spread mm':>13s} {'thumb-finger mm':>16s} {'min pairwise mm':>16s}")
    rows = []
    for obj in sorted(os.listdir(args.root)):
        p = os.path.join(args.root, obj, "grasp_predictions", args.hand, "12_contacts", "graspqp", "default")
        if not os.path.isdir(p):  # flat backup layout: <root>/<obj>/<hand>/...
            p = os.path.join(args.root, obj, args.hand, "12_contacts", "graspqp", "default")
        if not os.path.isdir(p):
            continue
        exact = os.path.join(p, f"{obj}{args.suffix}.dexgrasp.pt")
        files = [exact] if os.path.exists(exact) else sorted(glob.glob(os.path.join(p, f"*{args.suffix}.dexgrasp.pt")))
        if not files:
            continue
        ck = torch.load(files[0], map_location="cpu", weights_only=False)
        J = torch.stack([ck["parameters"][n] for n in actuated], dim=-1).numpy()
        for i in range(min(args.n, len(J))):
            P = tip_positions(J[i]) * 1000
            spread = np.linalg.norm(P - P.mean(0), axis=1).max()
            if len(P) == 5:
                thumb_to_fingers = np.linalg.norm(P[4] - P[:4], axis=1).mean()
            else:  # allegro: thumb is the last one
                thumb_to_fingers = np.linalg.norm(P[3] - P[:3], axis=1).mean()
            pd = [np.linalg.norm(P[a] - P[b])
                  for a in range(len(P)) for b in range(a + 1, len(P))]
            rows.append((spread, thumb_to_fingers, min(pd)))
            print(f"{obj[:18]:18s} {i:3d}  {spread:13.1f} {thumb_to_fingers:16.1f} {min(pd):16.1f}")
    if rows:
        R = np.array(rows)
        print()
        print(f"  mean tip spread       = {R[:,0].mean():.1f} mm   (flat hand ~ 150-200 mm, closed fist ~ 30-50 mm)")
        print(f"  mean thumb-to-fingers = {R[:,1].mean():.1f} mm")
        print(f"  mean min pairwise     = {R[:,2].mean():.1f} mm")


if __name__ == "__main__":
    main()
