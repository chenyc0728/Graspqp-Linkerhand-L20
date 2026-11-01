import torch
from graspqp.core import HandModel

# ---- 常量：URDF 中 mimic 关节的信息 -------------------------------
# (mimic 关节在 chain 完整 21 维中的索引,
#  被模仿的驱动关节在 chain 完整 21 维中的索引,
#  multiplier)
MIMIC_MAP = [
    (3,  2,  0.8917),   # pinky_dip  <- pinky_pip
    (7,  6,  0.8917),   # ring_dip   <- ring_pip
    (11, 10, 0.8917),   # middle_dip <- middle_pip
    (15, 14, 0.8917),   # index_dip  <- index_pip
    (20, 19, 1.1619),   # thumb_ip   <- thumb_mcp
]

# 16 个驱动关节在 chain 完整 21 维中的索引
DRIVE_INDICES = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19]


def _build_full_angles(joint_angles: torch.Tensor) -> torch.Tensor:
    """把 16 维驱动关节角展开成 chain 所需的 21 维完整关节角。

    输入  joint_angles: (..., 16)，顺序与 joint_filter 一致
    输出  full_angles:  (..., 21)，顺序与 URDF joint 定义顺序一致
    """
    # 按 joint_filter 顺序取出每个驱动关节角
    pinky_roll   = joint_angles[..., 0]
    pinky_pitch  = joint_angles[..., 1]
    pinky_pip    = joint_angles[..., 2]
    ring_roll    = joint_angles[..., 3]
    ring_pitch   = joint_angles[..., 4]
    ring_pip     = joint_angles[..., 5]
    middle_roll  = joint_angles[..., 6]
    middle_pitch = joint_angles[..., 7]
    middle_pip   = joint_angles[..., 8]
    index_roll   = joint_angles[..., 9]
    index_pitch  = joint_angles[..., 10]
    index_pip    = joint_angles[..., 11]
    thumb_yaw    = joint_angles[..., 12]
    thumb_roll   = joint_angles[..., 13]
    thumb_pitch  = joint_angles[..., 14]
    thumb_mcp    = joint_angles[..., 15]

    full_angles = torch.stack(
        [
            pinky_roll,                          # 0  pinky_mcp_roll
            pinky_pitch,                         # 1  pinky_mcp_pitch
            pinky_pip,                           # 2  pinky_pip
            pinky_pip * 0.8917,                  # 3  pinky_dip     (mimic)
            ring_roll,                           # 4  ring_mcp_roll
            ring_pitch,                          # 5  ring_mcp_pitch
            ring_pip,                            # 6  ring_pip
            ring_pip * 0.8917,                   # 7  ring_dip      (mimic)
            middle_roll,                         # 8  middle_mcp_roll
            middle_pitch,                        # 9  middle_mcp_pitch
            middle_pip,                          # 10 middle_pip
            middle_pip * 0.8917,                 # 11 middle_dip    (mimic)
            index_roll,                          # 12 index_mcp_roll
            index_pitch,                         # 13 index_mcp_pitch
            index_pip,                           # 14 index_pip
            index_pip * 0.8917,                  # 15 index_dip     (mimic)
            thumb_yaw,                           # 16 thumb_cmc_yaw
            thumb_roll,                          # 17 thumb_cmc_roll
            thumb_pitch,                         # 18 thumb_cmc_pitch
            thumb_mcp,                           # 19 thumb_mcp
            thumb_mcp * 1.1619,                  # 20 thumb_ip      (mimic)
        ],
        dim=-1,
    )
    return full_angles


def calculate_joints(joint_angles: torch.Tensor, hand_model):
    """LinkerHand L20 的自定义前向运动学（处理 mimic 关节）。

    输入  joint_angles: (..., 16)  16 个驱动关节角（顺序与 joint_filter 一致）
    输出  forward_kinematics 结果，包含全部 21 个关节 link 的位姿
    """
    full_angles = _build_full_angles(joint_angles)
    return hand_model.chain.forward_kinematics(full_angles)


def calculate_jacobian(joint_angles: torch.Tensor, hand_model):
    """LinkerHand L20 的自定义 jacobian（把 mimic 关节合并到驱动关节列）。

    输入  joint_angles: (..., 16)
    输出  jacobian:     (..., num_links, 6, 16)
    """
    full_angles = _build_full_angles(joint_angles)
    J_full = hand_model.chain.jacobian(full_angles)   # (..., num_links, 6, 21)

    # 取出 16 个驱动关节对应的列
    J_eff = J_full[..., DRIVE_INDICES].clone()        # (..., num_links, 6, 16)

    # 每个 mimic 关节的 jacobian 按其 multiplier 累加到对应的驱动关节列上
    for mimic_idx, drive_idx_in_full, mult in MIMIC_MAP:
        pos = DRIVE_INDICES.index(drive_idx_in_full)
        J_eff[..., pos] = J_eff[..., pos] + J_full[..., mimic_idx] * mult

    return J_eff


def getHandModel(device: str, asset_dir: str, **kwargs) -> HandModel:
    params = dict(
        mjcf_path=f"{asset_dir}/linkerhand_l20/linkerhand_l20.urdf",
        mesh_path=f"{asset_dir}/linkerhand_l20/meshes",
        contact_points_path=f"{asset_dir}/linkerhand_l20/contact_points.json",
        penetration_points_path=f"{asset_dir}/linkerhand_l20/penetration_points.json",
        device=device,
        n_surface_points=512,
        # ---- 坐标系约定：从 URDF 看，手指沿 +Z 伸展，需在 step 5 验证 up_axis ----
        forward_axis="z",
        up_axis="x",                      # 若 step 5 可视化发现朝向不对，改为 "y" 或 "-x"
        use_collision_if_possible=True,
        only_use_collision=True,
        # 避免零配置奇异点：所有驱动关节给一个小的非零初值
        default_state=torch.full((16,), 0.01, device=device),   # 16 个驱动关节的初始角
        joint_filter=[
            "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
            "ring_mcp_roll",  "ring_mcp_pitch",  "ring_pip",
            "middle_mcp_roll","middle_mcp_pitch","middle_pip",
            "index_mcp_roll", "index_mcp_pitch", "index_pip",
            "thumb_cmc_yaw",  "thumb_cmc_roll",  "thumb_cmc_pitch",
            "thumb_mcp",
        ],
        joint_calc_fnc=calculate_joints,
        jacobian_fnc=calculate_jacobian,
    )
    params.update(kwargs)
    return HandModel(**params)
