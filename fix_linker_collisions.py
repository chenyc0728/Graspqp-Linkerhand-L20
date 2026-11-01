#!/usr/bin/env python3
"""
批量修复 LinkerHand L20 的碰撞网格。
用法:
    python fix_linker_collisions.py --collisions_dir /path/to/linkerhand_l20/collisions

依赖:
    pip install trimesh pymeshfix numpy
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
import pymeshfix


def diagnose(mesh: trimesh.Trimesh) -> dict:
    """返回网格的水密性诊断信息。"""
    boundary_edges = trimesh.grouping.group_rows(
        mesh.edges_sorted, require_count=1
    )
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": mesh.is_watertight,
        "winding_consistent": mesh.is_winding_consistent,
        "is_volume": mesh.is_volume,
        "boundary_edges": len(boundary_edges),
        "components": len(mesh.split(only_watertight=False)),
    }


def repair_single(in_path: Path, out_path: Path) -> bool:
    """修复单个 STL 文件，成功返回 True。"""
    try:
        mesh = trimesh.load(in_path, force="mesh", process=False)
    except Exception as e:
        print(f"  [跳过] 无法加载 {in_path.name}: {e}")
        return False

    before = diagnose(mesh)
    print(f"  修复前: watertight={before['watertight']}, "
          f"boundary_edges={before['boundary_edges']}, "
          f"components={before['components']}")

    # 如果已经完美，跳过
    if before["watertight"] and before["winding_consistent"] and before["is_volume"]:
        print(f"  [跳过] {in_path.name} 已是水密流形，无需修复")
        # 仍然复制到输出，方便统一替换
        mesh.export(out_path)
        return True

    # ---- 阶段 1: 基础清理 ----
    mesh.merge_vertices(digits_vertex=5)          # 合并近重复顶点
    mesh.remove_degenerate_faces()                # 移除退化面
    mesh.remove_duplicate_faces()                 # 移除重复面
    trimesh.repair.fill_holes(mesh)               # 填充简单孔洞[reference:3]
    mesh.fix_normals()                            # 统一法线朝向

    mid = diagnose(mesh)
    print(f"  基础清理后: watertight={mid['watertight']}, "
          f"boundary_edges={mid['boundary_edges']}, "
          f"components={mid['components']}")

    # ---- 阶段 2: 强力水密修复 (pymeshfix) ----
    if not mid["watertight"]:
        try:
            v, f = pymeshfix.clean_from_arrays(
                mesh.vertices.astype(np.float64),
                mesh.faces.astype(np.int32),
            )
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
            mesh.fix_normals()
        except Exception as e:
            print(f"  [警告] pymeshfix 失败: {e}，将保留基础清理结果")

    after = diagnose(mesh)
    print(f"  修复后: watertight={after['watertight']}, "
          f"boundary_edges={after['boundary_edges']}, "
          f"components={after['components']}")

    if not after["watertight"]:
        print(f"  [失败] {in_path.name} 仍未水密，需手动处理")
        return False

    mesh.export(out_path)
    print(f"  [完成] 已保存到 {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="批量修复 LinkerHand L20 碰撞网格"
    )
    parser.add_argument(
        "--collisions_dir", required=True,
        help="collisions/ 目录路径"
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="修复后输出目录（默认在 collisions/ 旁新建 collisions_fixed/）"
    )
    parser.add_argument(
        "--in_place", action="store_true",
        help="直接覆盖原文件（会先备份为 *_backup.stl）"
    )
    args = parser.parse_args()

    src_dir = Path(args.collisions_dir)
    if not src_dir.is_dir():
        raise SystemExit(f"目录不存在: {src_dir}")

    if args.in_place:
        out_dir = src_dir
    else:
        out_dir = Path(args.out_dir) if args.out_dir else src_dir.parent / "collisions_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    stl_files = sorted(src_dir.glob("*.stl")) + sorted(src_dir.glob("*.STL"))
    if not stl_files:
        raise SystemExit(f"{src_dir} 下未找到 STL 文件")

    print(f"找到 {len(stl_files)} 个碰撞网格，输出到: {out_dir}\n")

    ok, fail = 0, 0
    for f in stl_files:
        print(f"[处理] {f.name}")
        out_path = out_dir / f.name

        if args.in_place:
            backup = f.with_name(f.stem + "_backup" + f.suffix)
            if not backup.exists():
                f.rename(backup)
            in_path = backup
        else:
            in_path = f

        if repair_single(in_path, out_path):
            ok += 1
        else:
            fail += 1
        print()

    print(f"完成: 成功 {ok}, 失败 {fail}")
    if fail > 0:
        print("失败的网格需要手动用 MeshLab / Blender 处理，或改用 CoACD 凸分解作为替代碰撞体。")


if __name__ == "__main__":
    main()