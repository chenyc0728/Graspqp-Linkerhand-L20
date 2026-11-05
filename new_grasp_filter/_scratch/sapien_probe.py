"""Scratch: Sapien scene with static hand (exact triangle meshes) + dynamic object."""
import sys
import time
from pathlib import Path

import numpy as np
import sapien

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grasp_io as gio

sapien.set_log_level("error")
obj_name = sys.argv[1] if len(sys.argv) > 1 else "apple"
grasp_i = int(sys.argv[2]) if len(sys.argv) > 2 else 15

g = gio.load_grasp_file(gio.list_grasp_files(obj_name)[0])
urdf = gio.UrdfModel(gio.HAND_URDF)
links = gio.hand_link_meshes()

t0 = time.time()
scene = sapien.Scene()
scene.set_timestep(1 / 240)

Troot = np.eye(4)
rp = g["root_pose"][grasp_i]
Troot[:3, :3] = gio.quat_xyzw_to_matrix(rp[3:7])
Troot[:3, 3] = rp[:3]
T = gio.fk_batch(urdf, gio.build_full_angles(g["joints"][grasp_i : grasp_i + 1]))

hand = {}
for name, path in links.items():
    W = Troot @ T[name][0]
    b = scene.create_actor_builder()
    b.set_name(name)
    b.add_nonconvex_collision_from_file(str(path), density=1000)
    b.set_initial_pose(sapien.Pose(W[:3, 3].tolist(), gio.matrix_to_quat_xyzw(W[:3, :3]).tolist()))
    hand[name] = b.build_static(name)
print(f"hand built {len(hand)} in {time.time()-t0:.2f}s", flush=True)

b = scene.create_actor_builder()
b.set_name("object")
b.add_nonconvex_collision_from_file(str(gio.object_mesh_path(obj_name)))
b.set_initial_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
obj = b.build(name="object")
comp = obj.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
comp.disable_gravity = True
comp.set_linear_velocity(np.zeros(3))
comp.set_angular_velocity(np.zeros(3))
print("object mass", comp.mass, "shapes", len(comp.get_collision_shapes()), flush=True)
for a in list(hand.values()) + [obj]:
    cc = a.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
    for sh in cc.get_collision_shapes():
        sh.set_contact_offset(0.01)
        sh.set_rest_offset(0.0)

t = time.time()
scene.step()
print(f"step {time.time()-t:.3f}s", flush=True)
contacts = scene.get_contacts()
print("pairs", len(contacts))
tot = 0
minsep = 1e9
per = {}
for c in contacts:
    names = (c.bodies[0].get_name(), c.bodies[1].get_name())
    per[names] = per.get(names, 0) + len(c.points)
    for p in c.points:
        tot += 1
        minsep = min(minsep, p.separation)
print("contact points", tot, "min separation (m)", minsep)
for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
    print("  ", k, v)

t = time.time()
for _ in range(20):
    scene.step()
print(f"20 steps {time.time()-t:.3f}s -> {np.round(comp.get_pose().p, 5)}")
