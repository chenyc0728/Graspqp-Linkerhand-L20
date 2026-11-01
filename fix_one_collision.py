#!/usr/bin/env python3
"""
诊断 + 修复单个碰撞 STL：
  1. 合并顶点（消除 STL 面独立假象）
  2. 删除漂浮组件（小碎片）
  3. pymeshfix 水密化（填补孔洞）
  4. 验证 watertight / volume / components

用法:
    python fix_one_collision.py --file pinky_middle.STL --dry_run
    python fix_one_collision.py --file pinky_middle.STL --apply
"""

import argparse
from pathlib import Path
import numpy as np
import trimesh
import pymeshfix


def analyze(mesh, tag=""):
    comps = mesh.split(only_watertight=False)
    print(f"  [{tag}] faces={len(mesh.faces)} verts={len(mesh.vertices)} "
          f"watertight={mesh.is_watertight} components={len(comps)}")
    if len(comps) > 1:
        comps_sorted = sorted(comps, key=lambda c: -len(c.faces))
        for i, c in enumerate(comps_sorted[:6]):
            diag = float((c.bounding_box.extents ** 2).sum() ** 0.5)
            print(f"      comp[{i}] faces={len(c.faces):5d} "
                  f"volume={abs(c.volume):.3e} bbox_diag={diag:.6f}")
        if len(comps_sorted) > 6:
            print(f"      ... 还有 {len(comps_sorted)-6} 个小组件")
    return comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry_run", action="store_true", default=False)
    ap.add_argument("--min_bbox_diag_ratio", type=float, default=1e-2)
    ap.add_argument("--abs_min_bbox_diag", type=float, default=0.005)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.file)
    print(f"\n=== {path.name} ===")

    # ---- 加载 + 合并顶点（关键第一步）----
    mesh = trimesh.load(path, force="mesh", process=False)
    print("修复前:")
    analyze(mesh, "raw")

    mesh.merge_vertices(digits_vertex=5)
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    print("\n合并顶点后:")
    comps = analyze(mesh, "merged")

    # ---- 删除漂浮组件 ----
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        diags = np.array([
            float((c.bounding_box.extents ** 2).sum() ** 0.5) for c in comps
        ])
        main_idx = int(np.argmax(diags))
        main_diag = diags[main_idx]
        keep = [
            i for i in range(len(comps))
            if i == main_idx
            or (diags[i] >= main_diag * args.min_bbox_diag_ratio
                and diags[i] >= args.abs_min_bbox_diag)
        ]
        removed = [i for i in range(len(comps)) if i not in keep]
        print(f"\n删除 {len(removed)} 个漂浮组件 (保留 {len(keep)} 个)")
        mesh = trimesh.util.concatenate([comps[i] for i in keep])
        mesh.merge_vertices(digits_vertex=5)
        mesh.remove_degenerate_faces()
        mesh.fix_normals()
    print("\n删碎片后:")
    analyze(mesh, "cleaned")

    # ---- pymeshfix 水密化 ----
    if not mesh.is_watertight:
        print("\n执行 pymeshfix 水密化...")
        try:
            v, f = pymeshfix.clean_from_arrays(
                mesh.vertices.astype(np.float64),
                mesh.faces.astype(np.int32),
            )
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
            mesh.merge_vertices(digits_vertex=5)
            mesh.fix_normals()
            print("pymeshfix 完成")
        except Exception as e:
            print(f"pymeshfix 失败: {e}")
        print("\n水密化后:")
        analyze(mesh, "fixed")

    # ---- 最终判定 ----
    ok = (mesh.is_watertight and mesh.is_volume
          and mesh.is_winding_consistent and len(mesh.split(only_watertight=False)) == 1)
    if ok:
        print("\n✅ 网格已完全修复：watertight + volume + 单组件")
    else:
        print("\n⚠️  仍未完全修复，需手动处理或改用 CoACD 凸分解")

    if args.dry_run or not args.apply:
        print("\n[DRY RUN] 未写文件。确认后加 --apply")
        return
    if not ok:
        print("拒绝写入未完全修复的网格。加 --apply 也没用，请先解决上面的 warning。")
        return

    out = Path(args.out) if args.out else path
    if out == path:
        backup = path.with_name(path.stem + "_pre_fix" + path.suffix)
        if not backup.exists():
            path.rename(backup)
            print(f"原文件备份到: {backup}")
    mesh.export(out)
    print(f"已写入: {out}")


if __name__ == "__main__":
    main()