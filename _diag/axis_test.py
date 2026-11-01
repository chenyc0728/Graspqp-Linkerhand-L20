"""Decide the palm-normal / finger-axis directions of the L20 from FK, and test
whether a plausible closed-fist (power grasp) pose exists.

Usage: python _diag/axis_test.py
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))

import pytorch_kinematics as pk  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "graspqp", "src", "graspqp", "hands"))
from linkerhand_l20 import _build_full_angles  # noqa: E402

URDF = f"{ROOT}/graspqp/assets/linkerhand_l20/linkerhand_l20.urdf"
chain = pk.build_chain_from_urdf(open(URDF).read())
NAMES = chain.get_joint_parameter_names()
DRIVE = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19]


def fk(drive_angles):
    full = _build_full_angles(torch.tensor(drive_angles, dtype=torch.float32).reshape(1, -1))
    return chain.forward_kinematics(full)


def tip(fkres, name):
    return fkres[name].get_matrix()[0, :3, 3].numpy() * 1000


def main():
    z = np.zeros(16)
    base = fk(z)
    print("zero-pose link origins (mm):")
    for k, v in base.items():
        t = tip(base, k)
        print(f"  {k:26s} [{t[0]:8.2f} {t[1]:8.2f} {t[2]:8.2f}]")

    print()
    print("### flexion direction: move index joints one by one and see where index_distal tip goes")
    for j, jn in enumerate(["index_mcp_roll", "index_mcp_pitch", "index_pip"]):
        idx = DRIVE.index([12, 13, 14][j])
        a = z.copy()
        a[idx] = 0.7
        d = tip(fk(a), "index_distal") - tip(base, "index_distal")
        d /= np.linalg.norm(d)
        print(f"  {jn:18s} +0.7 -> tip displacement direction {np.round(d,3)}")

    print()
    print("### thumb joints")
    for j, jn in enumerate(["thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"]):
        idx = DRIVE.index([16, 17, 18, 19][j])
        a = z.copy()
        a[idx] = 0.7
        d = tip(fk(a), "thumb_distal") - tip(base, "thumb_distal")
        d /= np.linalg.norm(d)
        print(f"  {jn:18s} +0.7 -> tip displacement direction {np.round(d,3)}")

    print()
    print("### fist search: minimise the spread of the 5 fingertips (index/middle/ring/pinky distal + thumb distal)")
    tips = ["index_distal", "middle_distal", "ring_distal", "pinky_distal", "thumb_distal"]
    best = None
    rng = np.random.default_rng(0)
    lo = np.array([-0.17, 0.0, 0.0] * 4 + [0.0, 0.0, 0.0, 0.0])
    hi = np.array([0.17, 1.4, 1.57] * 4 + [1.4, 1.22, 0.79, 1.05])
    for it in range(4000):
        a = lo + (hi - lo) * rng.random(16) if it else np.concatenate([np.zeros(14), [1.4, 1.22, 1.05]] * 1)[:16]
        if it == 0:
            a = np.array([0.0, 1.2, 1.2] * 4 + [1.2, 1.0, 0.6, 0.9])
        f = fk(a)
        pts = np.stack([tip(f, t) for t in tips])
        spread = np.linalg.norm(pts - pts.mean(0), axis=1).max()
        if best is None or spread < best[0]:
            best = (spread, a.copy(), pts)
    print(f"best max-fingertip-radius = {best[0]:.1f} mm")
    print("  joints:", np.round(best[1], 3).tolist())
    for t, p in zip(tips, best[2]):
        print(f"    {t:16s} [{p[0]:8.2f} {p[1]:8.2f} {p[2]:8.2f}]")
    c = best[2].mean(0)
    print(f"  centroid [{c[0]:.2f} {c[1]:.2f} {c[2]:.2f}]")
    print("  palm front (hand_base_link max x) ~ +2 mm ; knuckles z ~ 59-64 mm")


if __name__ == "__main__":
    main()
