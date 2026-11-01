#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET
import torch


def parse_urdf_nonfixed_joints(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = []
    for j in root.findall("joint"):
        if j.get("type") == "fixed":
            continue
        m = j.find("mimic")
        mimic_info = None
        if m is not None:
            mimic_info = {
                "joint": m.get("joint"),
                "multiplier": float(m.get("multiplier", 1.0)),
                "offset": float(m.get("offset", 0.0)),
            }
        joints.append({"name": j.get("name"), "type": j.get("type"), "mimic": mimic_info})
    return joints


def build_full_angles_by_name(q16, urdf_joints, drive_names):
    drive = {name: q16[..., i] for i, name in enumerate(drive_names)}
    full = dict(drive)
    for j in urdf_joints:
        if j["mimic"]:
            full[j["name"]] = full[j["mimic"]["joint"]] * j["mimic"]["multiplier"] + j["mimic"]["offset"]
    return torch.stack([full[j["name"]] for j in urdf_joints], dim=-1)


def get_link_matrix(fk_result, link_name):
    t = fk_result[link_name]
    M = t.get_matrix() if hasattr(t, "get_matrix") else t
    return M.unsqueeze(0) if M.dim() == 2 else M


# ============================================================
# 验证 1: 关节顺序
# ============================================================
def verify_structure(hand_model, urdf_joints):
    print("=" * 78)
    print("验证 1: chain 的关节顺序 vs URDF 顺序")
    print("=" * 78)
    chain_names = list(hand_model.chain.get_joint_parameter_names())
    urdf_names = [j["name"] for j in urdf_joints]
    print(f"chain 关节数: {len(chain_names)}, URDF 非 fixed 关节数: {len(urdf_names)}\n")
    print(f"{'idx':>4} | {'chain':30s} | {'URDF':30s} | match")
    print("-" * 78)
    ok = True
    for i in range(max(len(chain_names), len(urdf_names))):
        c = chain_names[i] if i < len(chain_names) else "---"
        u = urdf_names[i] if i < len(urdf_names) else "---"
        m = "OK" if c == u else "MISMATCH"
        if c != u:
            ok = False
        print(f"{i:>4} | {c:30s} | {u:30s} | {m}")
    print("\n" + ("✅ 顺序一致" if ok else "⚠️ 顺序不一致"))
    return ok


# ============================================================
# 验证 2: FK
# ============================================================
def verify_fk(hand_model, urdf_joints, drive_names, test_qs):
    print("\n" + "=" * 78)
    print("验证 2: FK 自洽性 (A: calculate_joints  vs  B: forward_kinematics_by_name)")
    print("=" * 78)
    from graspqp.hands.linkerhand_l20 import calculate_joints
    link_names = list(hand_model.link_names)          # ← 修补点

    all_ok = True
    for qi, q16 in enumerate(test_qs):
        with torch.no_grad():
            fk_a = calculate_joints(q16.unsqueeze(0), hand_model)
            full_b = build_full_angles_by_name(q16, urdf_joints, drive_names).unsqueeze(0)
            fk_b = hand_model.chain.forward_kinematics(full_b)
        per_link = {}
        for lk in link_names:
            Ma = get_link_matrix(fk_a, lk)[0]
            Mb = get_link_matrix(fk_b, lk)[0]
            per_link[lk] = (Ma - Mb).abs().max().item()
        max_d = max(per_link.values())
        worst = max(per_link, key=per_link.get)
        s = "✅" if max_d < 1e-5 else "❌"
        print(f"  q[{qi}]: max diff = {max_d:.3e}  {s}  (最差 link: {worst})")
        if max_d >= 1e-5:
            all_ok = False
            for lk, d in sorted(per_link.items(), key=lambda kv: -kv[1])[:5]:
                print(f"     {lk:32s}  {d:.3e}")
    return all_ok


# ============================================================
# 验证 3: 雅可比合并逻辑
# ============================================================
def verify_jacobian_logic(hand_model, drive_names, test_qs):
    print("\n" + "=" * 78)
    print("验证 3: calculate_jacobian 的 mimic 合并逻辑")
    print("=" * 78)
    from graspqp.hands.linkerhand_l20 import (
        calculate_jacobian, DRIVE_INDICES, MIMIC_MAP, _build_full_angles,
    )

    all_ok = True
    for qi, q16 in enumerate(test_qs):
        with torch.no_grad():
            full = _build_full_angles(q16.unsqueeze(0))
            J_full = hand_model.chain.jacobian(full)
            if qi == 0:  # 只打印一次 shape 诊断
                print(f"  chain.jacobian shape = {tuple(J_full.shape)}  "
                      f"(期望 (1, {len(hand_model.link_names)}, 6, 21))")
            J_manual = J_full[..., DRIVE_INDICES].clone()
            for mimic_idx, drive_idx_in_full, mult in MIMIC_MAP:
                pos = DRIVE_INDICES.index(drive_idx_in_full)
                J_manual[..., pos] = J_manual[..., pos] + J_full[..., mimic_idx] * mult
            J_user = calculate_jacobian(q16.unsqueeze(0), hand_model)
        d = (J_manual - J_user).abs().max().item()
        s = "✅" if d < 1e-5 else "❌"
        print(f"  q[{qi}]: max |J_manual - J_user| = {d:.3e}  {s}   shape={tuple(J_user.shape)}")
        if d >= 1e-5:
            all_ok = False
    return all_ok


# ============================================================
# 验证 4: 数值微分
# ============================================================
def get_chain_link_names(chain):
    """兼容不同 pytorch_kinematics 版本：拿到 chain 的 link 名字（含 root_link）。"""
    try:
        return list(chain.get_link_names())
    except AttributeError:
        pass
    if hasattr(chain, "links"):
        return [l.name for l in chain.links]
    raise RuntimeError("无法获取 chain 的 link 名字，请检查 pytorch_kinematics 版本")


def verify_jacobian_numeric(hand_model, urdf_joints, drive_names, test_qs, eps=1e-4):
    print("\n" + "=" * 78)
    print(f"验证 4: 数值微分  eps = {eps}")
    print("=" * 78)
    from graspqp.hands.linkerhand_l20 import calculate_jacobian, calculate_joints

    # ===== 关键修复：用 FK 返回字典的 key 拿 chain 的 link 顺序 =====
    dummy_q = torch.zeros(16)
    with torch.no_grad():
        fk_probe = calculate_joints(dummy_q.unsqueeze(0), hand_model)
    if isinstance(fk_probe, dict):
        chain_names = list(fk_probe.keys())
    else:
        raise RuntimeError(
            f"calculate_joints 返回类型不是 dict，而是 {type(fk_probe)}，"
            "请检查 chain.forward_kinematics 的输出类型"
        )
    print(f"chain link 数: {len(chain_names)}")
    print(f"hand_model.link_names 数: {len(hand_model.link_names)}")
    print(f"chain_names[:3] = {chain_names[:3]}")
    print(f"hand_model.link_names[:3] = {list(hand_model.link_names)[:3]}")

    n_drive = len(drive_names)

    for qi, q16 in enumerate(test_qs):
        q16 = q16.clone().float()
        with torch.no_grad():
            J_a = calculate_jacobian(q16.unsqueeze(0), hand_model).squeeze(0)
            # 诊断：J_a 的第 0 维应等于 chain_names 长度
            assert J_a.shape[0] == len(chain_names), (
                f"J_a 第 0 维 {J_a.shape[0]} != chain link 数 {len(chain_names)}"
            )

            fk0 = calculate_joints(q16.unsqueeze(0), hand_model)
            J_n = torch.zeros_like(J_a)
            for i in range(n_drive):
                qp = q16.clone(); qp[i] += eps
                qm = q16.clone(); qm[i] -= eps
                fkp = calculate_joints(qp.unsqueeze(0), hand_model)
                fkm = calculate_joints(qm.unsqueeze(0), hand_model)
                for li, lk in enumerate(chain_names):
                    Mp = get_link_matrix(fkp, lk)[0]
                    Mm = get_link_matrix(fkm, lk)[0]
                    M0 = get_link_matrix(fk0, lk)[0]
                    # 线速度
                    J_n[li, :3, i] = (Mp[:3, 3] - Mm[:3, 3]) / (2 * eps)
                    # 空间角速度: [ω]× = dR/dt @ R^T
                    dR = (Mp[:3, :3] - Mm[:3, :3]) / (2 * eps)
                    Om = dR @ M0[:3, :3].T
                    J_n[li, 3:, i] = torch.stack([
                        Om[2, 1] - Om[1, 2],
                        Om[0, 2] - Om[2, 0],
                        Om[1, 0] - Om[0, 1],
                    ]) * 0.5

        d_lin = (J_a[..., :3, :] - J_n[..., :3, :]).abs()
        d_ang = (J_a[..., 3:, :] - J_n[..., 3:, :]).abs()
        ml, ma = d_lin.max().item(), d_ang.max().item()
        sl = "✅" if ml < 1e-2 else ("⚠️" if ml < 1e-1 else "❌")
        sa = "✅" if ma < 1e-2 else ("⚠️" if ma < 1e-1 else "❌")
        print(f"  q[{qi}]: linear max={ml:.3e} {sl}   angular max={ma:.3e} {sa}")

        if ml > 1e-2 or ma > 1e-2:
            score = d_lin.amax(dim=(1, 2)) + d_ang.amax(dim=(1, 2))
            print("    差异最大的 link:")
            for idx in score.argsort(descending=True)[:5]:
                idx_i = int(idx)  # ← 转 Python int，避免 tensor 索引
                print(f"      {chain_names[idx_i]:32s} "
                      f"lin={d_lin.amax(dim=(1,2))[idx_i]:.2e} "
                      f"ang={d_ang.amax(dim=(1,2))[idx_i]:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=str,
                    default="graspqp/assets/linkerhand_l20/linkerhand_l20.urdf")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    from graspqp.hands import get_hand_model
    hand_model = get_hand_model("linkerhand_l20", args.device)
    urdf_joints = parse_urdf_nonfixed_joints(args.urdf)

    drive_names = [
        "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
        "ring_mcp_roll",  "ring_mcp_pitch",  "ring_pip",
        "middle_mcp_roll","middle_mcp_pitch","middle_pip",
        "index_mcp_roll", "index_mcp_pitch", "index_pip",
        "thumb_cmc_yaw",  "thumb_cmc_roll",  "thumb_cmc_pitch",
        "thumb_mcp",
    ]

    torch.manual_seed(0)
    test_qs = [
        torch.zeros(16),
        torch.full((16,), 0.3),
        torch.tensor([0.1, 0.5, 0.4, 0.1, 0.5, 0.4, 0.1, 0.5, 0.4,
                      0.1, 0.5, 0.4, 0.3, 0.5, 0.4, 0.5]),
        torch.rand(16) * 0.7,
        torch.rand(16) * 1.0,
    ]

    ok1 = verify_structure(hand_model, urdf_joints)
    ok2 = verify_fk(hand_model, urdf_joints, drive_names, test_qs)
    ok3 = verify_jacobian_logic(hand_model, drive_names, test_qs)
    verify_jacobian_numeric(hand_model, urdf_joints, drive_names, test_qs)

    print("\n" + "=" * 78)
    print("总结")
    print("=" * 78)
    print(f"  结构顺序       : {'✅' if ok1 else '❌'}")
    print(f"  FK 自洽        : {'✅' if ok2 else '❌'}")
    print(f"  雅可比合并逻辑 : {'✅' if ok3 else '❌'}")
    print("  数值微分       : 见上面输出")


if __name__ == "__main__":
    main()