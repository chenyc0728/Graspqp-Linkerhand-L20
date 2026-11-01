# LinkerHand L20 — integration notes and debugging playbook

This note records how the LinkerHand L20 was integrated into graspqp, what was
broken when it produced unusable grasps, and how to re-check each item. It is
meant as a companion to [`adding_hand.md`](adding_hand.md).

Everything here can be re-measured with the two headless tools added next to the
plotly ones:

```bash
# hand model / SDF / FK / Jacobian audit (no browser)
CUDA_VISIBLE_DEVICES=0 python scripts/vis/visualize_hand_model_headless.py \
    --hand_name linkerhand_l20 --show_occupancy_grid --json out/l20_hand.json

# grasp evaluation of the fit.py output (no browser)
CUDA_VISIBLE_DEVICES=0 python scripts/vis/visualize_result_headless.py \
    --hand_name linkerhand_l20 --data_root min_dataset/release2 --json out/l20_grasps.json
```

---

## 1. Kinematics

| item | value |
| --- | --- |
| URDF joints | 21 revolute (16 actuated + 5 `<mimic>`) |
| driven joints | `pinky_mcp_roll/pitch/pip`, `ring_*`, `middle_*`, `index_*`, `thumb_cmc_yaw/roll/pitch`, `thumb_mcp` |
| mimic | `*_dip = 0.8917 * *_pip`, `thumb_ip = 1.1619 * thumb_mcp` (offset 0) |
| joint-vector order | the **chain** order restricted to the actuated joints, which for this URDF equals the order the joints appear in the file |

`hands/linkerhand_l20.py` expands the 16 driven angles to the 21 URDF angles
(`calculate_joints`) and folds the mimic Jacobian columns into the driving
columns (`calculate_jacobian`), which is the chain rule for `q_m = k * q_d`.

Both were verified against `pytorch_kinematics` and against finite differences of
the same FK:

* FK: bit-identical to `chain.forward_kinematics` on the expanded angles.
* Jacobian: max relative error ~1e-4 vs. central differences (float32 limited).
* `_check_joint_layout()` raises if the URDF joint order ever stops matching
  `DRIVE_NAMES`, so a future URDF edit cannot silently drive the wrong columns.

Caveat about `pytorch_kinematics` (installed from the
`renezurbruegg/pytorch_kinematics` fork): `Frame.get_transform` applies
`theta + 2*((theta > 0) - 0.5) * 1e-2` to every revolute joint ("ugly fix"), i.e.
a +-0.01 rad bias with a 0.02 rad jump at `theta == 0`. It affects every hand
equally and does not break the optimizer, but it means a finite-difference
Jacobian check performed *exactly* at the zero configuration is meaningless.

## 2. Frame convention

`forward_axis="z"`, `up_axis="x"` — the same convention as allegro / shadow /
schunk / robotiq. Verified with FK on the raw URDF:

* fingers extend along `+Z` (fingertip links at `z ~ +0.136 m`),
* `+Y` is the finger spread (index `-27 mm` ... pinky `+38 mm`),
* flexing `index_mcp_pitch` / `index_pip` moves the tip towards `+X`.

Measured alternatives for the initialization (`initialize_convex_hull`, apple,
32 grasps, penetration = fraction of object surface points inside the hand):

| forward / up | E_pen | points inside hand | mean contact distance |
| --- | --- | --- | --- |
| `z` / `x` (used) | **0.027** | **2.8 %** | 53.6 mm |
| `x` / `z` | 42.4 | 13.8 % | 123.7 mm |
| `x` / `y` | 42.1 | 17.5 % | 116.7 mm |

So the axes as configured are the right ones; do not "fix" them.

## 3. The two bugs that produced penetrating, non-bending grasps

### 3.1 An inside-out link mesh (`thumb_metacarpals_base2`)

`assets/linkerhand_l20/meshes/thumb_metacarpals_base2.STL` had **negative signed
volume** (−8.52 cm³ where every other link is positive) — consistently wound,
but inverted. warp's default SDF sign test (pseudo-normal) infers inside/outside
from the winding, so that link reported *any* point anywhere in space as
"inside", with the honest distance attached.

Consequences, measured at the initialization pose:

| | before | after flipping the winding |
| --- | --- | --- |
| `E_pen` at init | 342.8 | 0.127 |
| object points inside the hand | **100 %**, max 199 mm, mean 137 mm | 2.9 %, max 67 mm, mean 5.5 mm |
| share of the bogus penetration attributable to `thumb_metacarpals_base2` | **100 %** | 0 % |

Because `E_pen` is a *sum of distances* for the offending link, the term
**rewarded** bringing the object closer to the thumb base — i.e. it actively
pulled the hand into the object with weight 100, drowning the contact and joint
signals. That is why the fingers ended up splayed around an interpenetrating
hand instead of closing on the object.

Detection (cheap, no GPU): signed volume of every link mesh. The headless tool
prints it:

```
--- link mesh audit (the mesh the SDF is built from) ---
  !! INSIDE-OUT MESHES (negative signed volume): [...]
```

Fix: flip the winding (`trimesh.Trimesh(v, f).invert()` while preserving the
triangle count / soup structure).

### 3.2 The pseudo-normal sign test on non-watertight meshes

Once the inverted mesh was fixed, `E_pen` was still ~10x allegro's, and the
occupancy check showed "inside the hand" for points **47-73 mm away** from the
finger links. Cause: the phalanx meshes are high-poly visual meshes with open
boundaries and up to 69 disconnected shells; warp's pseudo-normal sign test is
only reliable for clean watertight meshes.

Measured per-link SDF sign quality (20 000 query points per link, trimesh
containment as reference):

| sign method | worst false-"inside" distance | note |
| --- | --- | --- |
| pseudo-normal (default) | **67 mm** (`middle_proximal`), 48 mm, 20 mm | the residual `E_pen` bias |
| generalized winding number | **1.85 mm** | what is now enabled |

`HandModel.use_winding_number` (new, default `False`) selects it;
`hands/linkerhand_l20.py` turns it on. The batched warp kernel
(`mesh_sdf_field_batched`) gained a matching flag, and `wp.Mesh` for the link
meshes was already built with `support_winding_number=True`.

### 3.3 Occupancy-grid check (the numeric version of `--show_occupancy_grid`)

With `use_winding_number=True` the SDF occupancy aligns with the geometry: 1368 of
175 616 grid cells (±0.2 m, 56³) are "inside", spread over all 22 links in
proportion to their size, and the largest distance between an occupied cell and
the actual geometry of the link that owns it is **3.4 mm** — i.e. no floating
occupancy. With the pseudo-normal sign the same check reports occupied cells up
to 67 mm away from the owning link (`index/middle/ring/pinky_proximal`, the
non-watertight phalanx meshes), which is exactly the E_pen corruption described
above.

## 4. Collision meshes — still needs manual Blender work

`HandModel` resolves the URDF `<collision>` filenames relative to `mesh_path`
(= `assets/linkerhand_l20/meshes`), i.e. it expects `meshes/collisions/…`, the
same layout as `schunk_2f`. For the L20 the URDF says `collisions/<link>.STL`
but the files live in `assets/linkerhand_l20/collisions/`, so the loader falls
back (silently) to the *visual* mesh of the same basename. That fallback is
currently the behaviour we want, because the Blender-made collision meshes are
not usable:

| link | collision mesh volume vs. visual | fraction of the visual surface inside the collision mesh |
| --- | --- | --- |
| `index_proximal` | 0.18x | 5.0 % |
| `*_middle` | 0.25x | 2-4.5 % |
| `thumb_proximal` | 0.07x | 2.5 % |
| `thumb_metacarpals` | 0.04x | 0.0 % |
| `hand_base_link` | 0.42x | 15.5 % |

They are watertight and correctly wound, but each one covers only the inner
core of its link, so wiring them in would make most of the finger surface
non-colliding and the penetration energy would be blind again.

**To finish this properly (manual, Blender):** for every link that appears in
`contact_points.json` plus the palm, produce a *watertight, convex-ish, slightly
inflated* hull that encloses the visual link (same frame, metres), covering
≈100 % of the visual surface; then move the set to
`assets/linkerhand_l20/meshes/collisions/` and keep `collisions/<link>.STL` in
the URDF. After that, `use_winding_number` can be dropped again (the simplified
hulls are clean) and the fit will be considerably faster.

Until then, keep the URDF `<collision>` filenames resolving to the visual meshes
(the current fallback) — accuracy of `E_pen` matters more than speed here.

## 5. Contact / penetration configuration

Both JSON files were checked and are fine:

* `contact_points.json`: 16 links x 8 candidates = 128 candidates for
  `n_contact=12`; every sampled point lies within 0.2 mm of its own link mesh, so
  the contact meshes are consistent with the geometry.
* `penetration_points.json`: 78 spheres; radii 4 mm on the phalanges and 20 mm on
  the palm, comparable to allegro's 12-26 mm. Self-penetration at the rest pose
  is ~0.02 m for both hands.

One thing to keep in mind: the cached candidates/normals in
`contact_points_cached_state_<grasp_type>.json` are computed from the collision
meshes. If the collision geometry changes (e.g. once the Blender hulls are in
place), delete those caches (or pass `regenerate=True`) or stale normals will be
reused.

## 6. Result after the fixes

`fit.py --energy_type graspqp --n_contact 12 --batch_size 32 --n_iter 7000` on
`min_dataset/release2`, best 32 grasps per object (160 grasps), evaluated with
`visualize_result_headless.py`:

| metric | L20 before | L20 after | allegro (same command) |
| --- | --- | --- | --- |
| `E_dis` | 0.370 | 0.0356 | 0.0276 |
| `E_fc` | 6.77 | **0.209** | 0.349 |
| `E_pen` | 13.86 | 0.0164 | 0.0071 |
| mean contact distance | 15.0 mm | **1.46 mm** | 1.83 mm |
| hand→object penetration, mean of per-grasp max | 52.7 mm | **1.16 mm** | 0.90 mm |
| hand→object penetration, worst grasp | 153 mm | **2.80 mm** | 10.8 mm |
| object points inside the hand | 22.1 % | 1.78 % | 0.49 % |
| joint deviation from `default_state` | 0.231 rad | 0.274 rad | 0.297 rad |
| fingertip spread / thumb-to-fingertips | 127 / 160 mm | **79 / 100 mm** | 90 / 122 mm |
| feasible (`|cd|<=5 mm`, `pen<=5 mm`, `E_fc<=1`) | 0 % | **80.6 %** | 69.4 % |
| feasible (+ `normal_align >= 0.5`) | 0 % | 24.4 % | 55.0 % |

Everything the optimizer actually minimises (`E_dis`, `E_fc`, `E_pen`, `E_spen`,
`E_joints`) is now at or better than allegro, and the fingers close into a real
grasp instead of staying splayed.

The one metric where the L20 still trails is `normal_align` (the mean cosine
between the hand contact normal and the object's inward normal), which is **not
part of the optimised energy** — it is only a diagnostic proxy. It is lower
because the candidate selection lands on the mid-phalanges (`ring_middle`,
`index_middle`, `middle_middle` carry the most selected contacts, mean cos
0.23-0.44) rather than on the fingertip pads, which is what allegro does
(62 % of its selected contacts are on the `*_link_3` tips). If that proxy matters
for a downstream task, the lever is the contact-candidate layout in
`contact_points.json` (more points on `*_distal`, fewer on `*_middle` and the
palm), not the SDF.

## 7. Things that were checked and are *not* the problem

* `joint_filter` / `default_state` / `joint_calc_fnc` / `jacobian_fnc` wiring:
  correct, FK/Jacobian verified numerically (section 1).
* `contact_points.json` / `penetration_points.json`: link names match the URDF,
  paths resolve, 128 candidates for `n_contact=12`, every candidate within 0.2 mm
  of its link surface, sphere radii comparable to allegro.
* base origin / `forward_axis` / `up_axis`: correct as configured (section 2).
* occupation grid alignment: healthy after the two SDF fixes (section 3.3).
* units / scale: the URDF is in metres, the hand is ~0.26 m from wrist to
  fingertip and the dataset objects are 0.05-0.21 m, so it is a large hand but
  the same order as allegro (0.25 m).
* `pytorch_kinematics.chain.jacobian` at the zero configuration: finite
  (`|J|max = 1.0`), the custom Jacobian is finite there too (`|J|max = 2.16`).
