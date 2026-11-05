"""Phase-0 research: compare the GraspQP L20 URDF with CMap's left extended URDF."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grasp_io as gio

GP = gio.HAND_URDF
CM = gio.REPO_ROOT / "Cmap_dataset/data_urdf/robot/linkerhand/left/linkerhand_l20_left_extended.urdf"
RIGHT = gio.REPO_ROOT / "Cmap_dataset/data_urdf/robot/linkerhand/right/linkerhand_l20_right_extended.urdf"


def joint_table(path):
    import xml.etree.ElementTree as ET

    r = ET.parse(path).getroot()
    out = {}
    for j in r.findall("joint"):
        o = j.find("origin")
        ax = j.find("axis")
        lim = j.find("limit")
        mim = j.find("mimic")
        out[j.get("name")] = {
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": np.fromstring(o.get("xyz"), sep=" ") if o is not None and o.get("xyz") else np.zeros(3),
            "rpy": np.fromstring(o.get("rpy"), sep=" ") if o is not None and o.get("rpy") else np.zeros(3),
            "axis": np.fromstring(ax.get("xyz"), sep=" ") if ax is not None and ax.get("xyz") else None,
            "limit": (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None and lim.get("lower") else None,
            "mimic": (mim.get("joint"), float(mim.get("multiplier"))) if mim is not None else None,
        }
    return out


gp = joint_table(GP)
cm = joint_table(CM)
print("graspqp joints:", len(gp), "cmap-left joints:", len(cm))
print("\n-- joints present in one but not the other --")
print("only graspqp:", sorted(set(gp) - set(cm)))
print("only cmap   :", sorted(set(cm) - set(gp)))

DRIVES = gio.DRIVE_NAMES
print("\n-- 16 driven joints: name order --")
print("graspqp chain order:", [n for n in gp if n in DRIVES])
print("cmap    chain order:", [n for n in cm if n in DRIVES])
print("same order:", [n for n in gp if n in DRIVES] == [n for n in cm if n in DRIVES])

print("\n-- per-joint geometry deltas (origin xyz/rpy, axis) --")
worst = 0.0
for n in [x for x in gp if x in cm]:
    a, b = gp[n], cm[n]
    dxyz = np.abs(a["xyz"] - b["xyz"]).max()
    drpy = np.abs(a["rpy"] - b["rpy"]).max()
    daxis = np.abs(a["axis"] - b["axis"]).max() if a["axis"] is not None and b["axis"] is not None else float("nan")
    worst = max(worst, dxyz, drpy, daxis)
    if max(dxyz, drpy, 0 if np.isnan(daxis) else daxis) > 1e-6:
        print(f"  {n:24s} dxyz={dxyz:.2e} drpy={drpy:.2e} daxis={daxis:.2e}")
print("worst delta over all shared joints:", worst)

print("\n-- mimic --")
for n, j in gp.items():
    if j["mimic"]:
        print("  graspqp", n, j["mimic"], " cmap:", cm[n]["mimic"])
for n, j in cm.items():
    if j["mimic"] and n not in gp:
        print("  cmap", n, j["mimic"])

print("\n-- virtual joint chain (CMap extended) --")
for n, j in cm.items():
    if n.startswith("virtual"):
        print(f"  {n:22s} {j['type']:9s} parent={j['parent']:16s} child={j['child']:16s} xyz={j['xyz']} rpy={j['rpy']} axis={j['axis']}")

print("\n-- root/fixed joints (graspqp) --")
for n, j in gp.items():
    if j["type"] == "fixed":
        print(f"  {n} parent={j['parent']} child={j['child']} xyz={j['xyz']} rpy={j['rpy']}")
print("-- fixed joints (cmap) --")
for n, j in cm.items():
    if j["type"] == "fixed":
        print(f"  {n} parent={j['parent']} child={j['child']} xyz={j['xyz']} rpy={j['rpy']}")

# geometric comparison of the 16-joint FK (base link frame)
u_gp = gio.UrdfModel(GP)
u_cm = gio.UrdfModel(CM)
rng = np.random.default_rng(0)
max_err = 0.0
for trial in range(20):
    q = rng.uniform(0.0, 0.8, 16)
    full = gio.build_full_angles(q[None, :])[0]
    ang_gp = {n: float(full[gio.CHAIN_JOINT_NAMES.index(n)]) for n in gio.CHAIN_JOINT_NAMES}
    ang_cm = dict(ang_gp)
    T_gp = u_gp.fk(ang_gp)
    T_cm = u_cm.fk(ang_cm)
    base_gp = "hand_base_link"
    base_cm = "base_link"
    for link in ["index_distal", "middle_distal", "ring_distal", "pinky_distal", "thumb_distal"]:
        A = np.linalg.inv(T_gp[base_gp]) @ T_gp[link]
        B = np.linalg.inv(T_cm[base_cm]) @ T_cm[link]
        max_err = max(max_err, float(np.abs(A - B).max()))
print("\nmax |T_gp - T_cmap| over 5 fingertip links, 20 random configs (base-relative):", f"{max_err:.3e}")
