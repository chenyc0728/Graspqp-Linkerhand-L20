# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""LinkerHand L20 (16 DoF, 5 fingers with coupled DIP joints) hand model.

The URDF exposes 21 revolute joints: 16 are actuated and 5 are coupled by URDF
``<mimic>`` tags (DIP follows PIP, thumb IP follows the thumb MCP). The graspqp
core has no notion of mimic joints, so this module

* declares the 16 driven joints via ``joint_filter``,
* expands the 16-dimensional joint vector to the 21 URDF joint angles
  (``calculate_joints``), and
* folds the mimic column derivatives back into the driving columns
  (``calculate_jacobian``),

which is exactly the chain rule for a coupled joint,
``dq_mimic/dq_drive = multiplier``.

URDF mimic relations (verified against the file, see
``scripts/vis/visualize_hand_model_headless.py``)::

    pinky_dip  = 0.8917 * pinky_pip
    ring_dip   = 0.8917 * ring_pip
    middle_dip = 0.8917 * middle_pip
    index_dip  = 0.8917 * index_pip
    thumb_ip   = 1.1619 * thumb_mcp        (offset 0 for all of them)

Note on joint ordering
----------------------
``HandModel`` builds the joint-angle vector from
``chain.get_joint_parameter_names()`` restricted to ``joint_filter`` -- i.e. the
**chain order**, not the order in which the names are written in
``joint_filter``. For this URDF the two coincide (pytorch_kinematics walks the
tree depth-first in document order and the filter lists the joints in that same
order); ``_check_joint_layout`` re-verifies this at model-construction time so
that a future edit cannot silently desynchronise ``_build_full_angles`` from the
URDF.

Note on the collision geometry
------------------------------
``only_use_collision=True`` + ``use_collision_if_possible=True`` makes the SDF use
the per-link ``<collision>`` geometry, whose URDF filenames are
``collisions/<link>.STL``. ``HandModel`` resolves those relative to ``mesh_path``
(= ``assets/linkerhand_l20/meshes``), i.e. it expects ``meshes/collisions/`` like
every other hand in this repo; since that directory does not exist, the loader
falls back to the *visual* mesh of the same basename. That fallback is currently
what makes the penetration energy meaningful: the Blender-made meshes in
``assets/linkerhand_l20/collisions/`` only cover 0-26 % of each link's surface
(they are much smaller than the link they are supposed to bound) and must not be
wired in as-is. See ``docs/adding_hand.md`` step 6 for the expected standard.
"""

import torch

from graspqp.core import HandModel

# ---- URDF mimic joints ------------------------------------------------
# (index of the mimic joint in the 21-dim chain vector,
#  index of the joint it mimics in the 21-dim chain vector,
#  multiplier)
MIMIC_MAP = [
    (3, 2, 0.8917),  # pinky_dip  <- pinky_pip
    (7, 6, 0.8917),  # ring_dip   <- ring_pip
    (11, 10, 0.8917),  # middle_dip <- middle_pip
    (15, 14, 0.8917),  # index_dip  <- index_pip
    (20, 19, 1.1619),  # thumb_ip   <- thumb_mcp
]

# The 16 driven joints, as indices into the 21-dim chain vector.
DRIVE_INDICES = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19]

# Names of the driven joints, in joint-vector order (= chain order).
DRIVE_NAMES = [
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
]


def _build_full_angles(joint_angles: torch.Tensor) -> torch.Tensor:
    """Expand the 16 driven joint angles into the 21 URDF joint angles.

    Args:
        joint_angles: ``(..., 16)`` driven joint angles (radians), in
            ``DRIVE_NAMES`` / chain order.

    Returns:
        ``(..., 21)`` joint angles in URDF chain order, with the mimic joints
        filled in from their driving joint and the URDF multiplier.
    """
    if joint_angles.shape[-1] != len(DRIVE_INDICES):
        raise ValueError(
            f"expected {len(DRIVE_INDICES)} driven joint angles, got {joint_angles.shape[-1]}"
        )
    cols = []
    mimic_lookup = {m: (DRIVE_INDICES.index(d), mult) for m, d, mult in MIMIC_MAP}
    for full_idx in range(len(DRIVE_INDICES) + len(MIMIC_MAP)):
        if full_idx in mimic_lookup:
            drive_pos, mult = mimic_lookup[full_idx]
            cols.append(joint_angles[..., drive_pos] * mult)
        else:
            cols.append(joint_angles[..., DRIVE_INDICES.index(full_idx)])
    return torch.stack(cols, dim=-1)


def calculate_joints(joint_angles: torch.Tensor, hand_model):
    """Forward kinematics for the L20, with the mimic joints expanded.

    Args:
        joint_angles: ``(B, 16)`` driven joint angles (radians).
        hand_model: :class:`graspqp.core.HandModel` providing ``chain``.

    Returns:
        dict[str, pytorch_kinematics.Transform3d]: batched link poses.
    """
    return hand_model.chain.forward_kinematics(_build_full_angles(joint_angles))


def calculate_jacobian(joint_angles: torch.Tensor, hand_model):
    """Geometric Jacobian of the L20, with the mimic joints folded into the drivers.

    For a coupled joint ``q_m = k * q_d`` the chain rule gives
    ``d/dq_d = d/dq_d + k * d/dq_m``, which is exactly what the fold below does.

    Args:
        joint_angles: ``(B, 16)`` driven joint angles (radians).
        hand_model: :class:`graspqp.core.HandModel` providing ``chain``.

    Returns:
        ``(B, n_links, 6, 16)`` Jacobian, columns ordered like ``DRIVE_NAMES``.
    """
    full_angles = _build_full_angles(joint_angles)
    J_full = hand_model.chain.jacobian(full_angles)  # (..., n_links, 6, 21)
    J_eff = J_full[..., DRIVE_INDICES].clone()  # (..., n_links, 6, 16)
    for mimic_idx, drive_idx_in_full, mult in MIMIC_MAP:
        pos = DRIVE_INDICES.index(drive_idx_in_full)
        J_eff[..., pos] = J_eff[..., pos] + J_full[..., mimic_idx] * mult
    return J_eff


def _check_joint_layout(hand_model) -> None:
    """Fail loudly if the joint-vector order no longer matches ``DRIVE_NAMES``.

    ``_build_full_angles`` and ``calculate_jacobian`` address the joint vector by
    fixed positions, and ``pytorch_kinematics`` orders the parameters by its own
    depth-first walk of the URDF. If those ever disagree every joint would be
    driven by the wrong column (and the grasp would look "stuck" in the default
    pose), so this is checked once at model construction instead of failing
    silently.
    """
    chain_names = list(hand_model.chain.get_joint_parameter_names())
    actuated = [n for n in chain_names if n in DRIVE_NAMES]
    if actuated != DRIVE_NAMES:
        raise ValueError(
            "LinkerHand L20 joint ordering mismatch.\n"
            f"  chain order (actuated only): {actuated}\n"
            f"  DRIVE_NAMES expected     : {DRIVE_NAMES}\n"
            "Update DRIVE_INDICES / _build_full_angles / DRIVE_NAMES to the new URDF order."
        )
    for mimic_idx, drive_idx_in_full, _mult in MIMIC_MAP:
        if mimic_idx >= len(chain_names) or drive_idx_in_full >= len(chain_names):
            raise ValueError("MIMIC_MAP index out of range for the parsed chain; URDF changed?")
    # every non-actuated chain joint must be a mimic joint we expand ourselves
    extra = [n for n in chain_names if n not in DRIVE_NAMES]
    expected = [chain_names[m] for m, _, _ in MIMIC_MAP]
    if sorted(extra) != sorted(expected):
        raise ValueError(f"unexpected non-driven joints {extra}; expected the mimic joints {expected}")


def getHandModel(device: str, asset_dir: str, **kwargs) -> HandModel:
    """Build the 16-DoF LinkerHand L20 model.

    Args:
        device: Torch device for the model tensors (e.g. ``"cuda"``).
        asset_dir: Root assets directory; hand files are read from
            ``{asset_dir}/linkerhand_l20``.
        **kwargs: Additional :class:`~graspqp.core.HandModel` arguments that
            override the defaults set here.

    Returns:
        HandModel: the configured LinkerHand L20 model.
    """
    params = dict(
        mjcf_path=f"{asset_dir}/linkerhand_l20/linkerhand_l20.urdf",
        mesh_path=f"{asset_dir}/linkerhand_l20/meshes",
        contact_points_path=f"{asset_dir}/linkerhand_l20/contact_points.json",
        penetration_points_path=f"{asset_dir}/linkerhand_l20/penetration_points.json",
        device=device,
        n_surface_points=512,
        # Every other hand in this repo authors its URDF so that +Z is the approach
        # direction (the palm normal) and +X is "up". Verified for the L20 with FK:
        # flexing index_mcp_pitch / index_pip moves the fingertip towards +X while the
        # fingers themselves extend along +Z, i.e. +X is the palm normal and the
        # fingertips sit at z ~ +0.136 m. With the base origin at the URDF root the
        # hand therefore approaches palm-first along +Z, exactly like allegro.
        forward_axis="z",
        up_axis="x",
        use_collision_if_possible=True,
        only_use_collision=True,
        # The link geometry the SDF is built from is the high-poly *visual* mesh (see the
        # module docstring: the Blender ``collisions/`` meshes are unusable, so the loader
        # falls back to ``meshes/``). Those meshes are not watertight (the phalanx meshes
        # have up to 69 disconnected shells), and warp's default pseudo-normal sign test
        # then reports points up to 60 mm away as "inside the hand". That inflates E_pen
        # with a bogus term that grows with the distance to the finger links and hides real
        # penetration. The generalized winding number classifies the same meshes with a
        # worst-case sign error of ~1 mm, so use it.
        use_winding_number=True,
        # A small non-zero push-off from the singular all-zero configuration. With mimic
        # joints the chain has no reason to be singular at zero, but pytorch_kinematics'
        # Frame.get_transform adds a +-1e-2 rad bias and jumps by 2e-2 rad at theta == 0,
        # so staying off the exact zero pose keeps the FK smooth. All values stay well
        # inside the joint limits and initialize_convex_hull jitters around this with
        # ``jitter_strength * (upper - lower)`` regardless.
        default_state=torch.full((len(DRIVE_NAMES),), 0.01, device=device),
        joint_filter=list(DRIVE_NAMES),
        joint_calc_fnc=calculate_joints,
        jacobian_fnc=calculate_jacobian,
    )
    params.update(kwargs)
    hand_model = HandModel(**params)
    _check_joint_layout(hand_model)
    return hand_model
