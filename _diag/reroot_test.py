"""Compute + validate the re-rooting transform for a hand whose URDF frame does not
follow the graspqp convention (+Z = approach / palm normal, +X = "up").

Raw LinkerHand L20 URDF frame (from FK, see _diag/axis_test.py):
    +Z = finger direction  (fingertip links at z ~ +0.136 m)
    +X = palm normal       (flexing any mcp_pitch/pip moves the tips towards +X)
    +Y = finger spread     (index -27 mm ... pinky +38 mm)

Target (same convention as allegro / shadow / schunk / robotiq):
    new +Z (forward / approach) = old +X  (palm normal)
    new +X (up)                 = old +Z  (finger direction)
    new +Y                      = old -Y  (right-handed: new_Y = new_Z x new_X)

The new origin is placed at the centroid of the five fingertip links in a
closed-fist pose ("centered between the finger contact points",
docs/adding_hand.md step 5).

Usage: python _diag/reroot_test.py [--write]
"""

import argparse
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src"))
sys.path.insert(0, os.path.join(ROOT, "graspqp", "src", "graspqp", "hands"))

import pytorch_kinematics as pk  # noqa: E402

URDF = f"{ROOT}/graspqp/assets/linkerhand_l20/linkerhand_l20.urdf"
OUT_URDF = f"{ROOT}/_diag/linkerhand_l20_rh.urdf"


def rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_rpy(R):
    p = np.arcsin(np.clip(-R[2, 0], -1, 1))
    if abs(np.cos(p)) > 1e-8:
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])
    elif p < 0:  # pitch = -pi/2 : R = Rz(y+r) Ry(-pi/2)
        r = 0.0
        y = np.arctan2(-R[0, 1], -R[0, 2])
    else:  # pitch = +pi/2 : R = Rz(y-r) Ry(+pi/2)
        r = 0.0
        y = np.arctan2(-R[0, 1], R[0, 2])
    return np.array([r, p, y])


ORIGIN_RE = re.compile(
    r'(<joint\s+name="fixed_to_root"\s+type="fixed">\s*<origin\s+rpy=")([^"]*)("\s+xyz=")([^"]*)(")'
)


def read_root_origin(txt):
    m = ORIGIN_RE.search(txt)
    return np.array([float(v) for v in m.group(2).split()]), np.array([float(v) for v in m.group(4).split()])


def rewrite_root_origin(txt, rpy, xyz):
    return ORIGIN_RE.sub(
        lambda m: f"{m.group(1)}{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}{m.group(3)}"
        f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}{m.group(5)}",
        txt,
    )


def fist_centroid(chain, fist):
    from linkerhand_l20 import _build_full_angles

    fk = chain.forward_kinematics(_build_full_angles(torch.tensor(fist, dtype=torch.float32).reshape(1, -1)))
    tips = {
        k: fk[k].get_matrix()[0, :3, 3].numpy()
        for k in ("index_distal", "middle_distal", "ring_distal", "pinky_distal", "thumb_distal")
    }
    P = np.stack(list(tips.values()))
    return P.mean(0), float(np.linalg.norm(P - P.mean(0), axis=1).max()), tips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from axis_test_data import FIST_JOINTS

    fist = np.array(FIST_JOINTS)
    txt = open(URDF).read()
    chain = pk.build_chain_from_urdf(txt)
    old_rpy, old_xyz = read_root_origin(txt)
    print("current root joint origin: rpy =", old_rpy.tolist(), " xyz =", (old_xyz * 1000).round(2).tolist(), "mm")

    c_old, radius, tips = fist_centroid(chain, fist)
    print("closed-fist fingertip centroid (old root frame, mm):", (c_old * 1000).round(2).tolist())
    print("  max tip radius around it:", round(radius * 1000, 1), "mm")

    new_x = np.array([0.0, 0.0, 1.0])   # up      <- old +Z (finger direction)
    new_y = np.array([0.0, -1.0, 0.0])  # lateral <- old -Y
    new_z = np.array([1.0, 0.0, 0.0])   # forward <- old +X (palm normal)
    R = np.stack([new_x, new_y, new_z], axis=1)
    print()
    print("rotation R (new axes as columns, expressed in old coords):")
    print(np.round(R, 4))
    print("  det =", round(float(np.linalg.det(R)), 6))

    rpy = R_to_rpy(R)
    err = float(np.abs(rpy_to_R(*rpy) - R).max())
    assert err < 1e-9, f"rpy round-trip failed ({err})"
    print("  rpy =", np.round(rpy, 6).tolist(), " (round-trip err", err, ")")

    xyz = R @ (old_xyz - c_old)
    print("  xyz =", (xyz * 1000).round(3).tolist(), "mm")

    new_txt = rewrite_root_origin(txt, rpy, xyz)
    chain2 = pk.build_chain_from_urdf(new_txt)

    c2, radius2, tips2 = fist_centroid(chain2, fist)
    print()
    print("VERIFY (new frame):")
    print("  fist centroid (mm):", (c2 * 1000).round(3).tolist(), " (expect ~0,0,0)")
    print("  max tip radius:", round(radius2 * 1000, 1), "mm")
    zfk = chain2.forward_kinematics(torch.zeros(1, 21))
    o = {k: zfk[k].get_matrix()[0, :3, 3].numpy() * 1000 for k in zfk.keys()}
    print("  zero pose, hand_base_link origin (mm):", o["hand_base_link"].round(1).tolist())
    print("  zero pose, index_distal   origin (mm):", o["index_distal"].round(1).tolist())
    print("  zero pose, thumb_distal   origin (mm):", o["thumb_distal"].round(1).tolist())
    allx = np.stack(list(o.values()))
    print("  zero pose, all link-origin bbox (mm):", allx.min(0).round(1).tolist(), "..", allx.max(0).round(1).tolist())

    from linkerhand_l20 import _build_full_angles

    def tipof(ch, a, name="index_distal"):
        fk = ch.forward_kinematics(_build_full_angles(torch.tensor(a, dtype=torch.float32).reshape(1, -1)))
        return fk[name].get_matrix()[0, :3, 3].numpy()

    a = np.zeros(16)
    a1 = a.copy()
    a1[10] = 0.7  # index_mcp_pitch
    d = tipof(chain2, a1) - tipof(chain2, a)
    d /= np.linalg.norm(d)
    print("  index_mcp_pitch +0.7 moves the tip along (new frame):", d.round(3).tolist(), " (expect ~ +Z)")

    if args.write:
        open(OUT_URDF, "w").write(new_txt)
        print("\nwrote", OUT_URDF)


if __name__ == "__main__":
    main()
