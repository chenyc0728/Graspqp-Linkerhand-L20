#!/usr/bin/env python3
"""
清除碰撞网格中的漂浮组件（小碎片 / 孤立连通体）。

用法:
    # 1. 先诊断，不改文件
    python clean_floating_components.py --file pinky_middle.STL --dry_run
    # 2. 确认索引后应用
    python clean_floating_components.py --file pinky_middle.STL --apply
    # 3. 如想先导出每个组件单独检查
    python clean_floating_components.py --file pinky_middle.STL --dry_run --export_components
"""

import argparse
from pathlib import Path
import numpy as np
import trimesh


def collect_components(mesh):
    """返回 (comps, stats)，stats 按体积降序排列。"""
    comps = mesh.split(only_watertight=False)
    stats = []
    for i, c in enumerate(comps):
        extents = c.bounding_box.extents
        diag = float(np.linalg.norm(extents))
        # 非水密组件用凸包体积做代理，避免 volume 为 0/负
        vol = abs(float(c.volume)) if c.is_watertight else float(c.convex_hull.volume)
        stats.append({
            "idx": i,
            "faces": len(c.faces),
            "verts": len(c.vertices),
            "volume": vol,
            "bbox_diag": diag,
            "centroid": c.centroid,
        })
    stats.sort(key=lambda s: s["bbox_diag"], reverse=True)
    return comps, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="要清理的 STL 路径")
    ap.add_argument("--dry_run", action="store_true", default=False)
    ap.add_argument("--apply", action="store_true", default=False)
    ap.add_argument(
        "--keep", type=int, nargs="*", default=None,
        help="强制保留的组件索引（先 dry_run 查看）",
    )
    ap.add_argument(
        "--min_bbox_diag_ratio", type=float, default=1e-2,
        help="保留组件的 bbox 对角线至少为主体的这个比例（默认 1%%）",
    )
    ap.add_argument(
        "--abs_min_bbox_diag", type=float, default=0.005,
        help="绝对最小 bbox 对角线（米），低于此值一律删除",
    )
    ap.add_argument(
        "--export_components", action="store_true",
        help="把每个组件导出到 <stem>_components/ 供可视化检查",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.file)
    mesh = trimesh.load(path, force="mesh", process=False)

    # 先清掉退化面 / 重复面，避免退化三角形被误当成独立组件
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.merge_vertices(digits_vertex=5)

    comps, stats = collect_components(mesh)
    main_diag = stats[0]["bbox_diag"]
    main_vol = stats[0]["volume"]

    print(f"文件: {path}")
    print(f"连通组件数: {len(comps)}\n")
    header = f"{'idx':>4} {'faces':>7} {'volume':>13} {'bbox_diag':>11} {'diag_ratio':>11}  centroid"
    print(header)
    print("-" * len(header))
    for s in stats:
        ratio = s["bbox_diag"] / main_diag if main_diag > 0 else 0
        print(f"{s['idx']:>4} {s['faces']:>7} {s['volume']:>13.4e} "
              f"{s['bbox_diag']:>11.6f} {ratio:>11.3e}  "
              f"{np.round(s['centroid'], 4).tolist()}")

    # 判定待删除
    keep_set = set(args.keep or [])
    to_remove = []
    for s in stats:
        if s["idx"] in keep_set:
            continue
        if s["idx"] == stats[0]["idx"]:  # 主体永远保留
            continue
        small_rel = s["bbox_diag"] < main_diag * args.min_bbox_diag_ratio
        tiny_abs = s["bbox_diag"] < args.abs_min_bbox_diag
        if small_rel or tiny_abs:
            to_remove.append(s["idx"])

    print(f"\n主体 bbox_diag = {main_diag:.6f} m, 主体体积 ≈ {main_vol:.4e}")
    print(f"建议删除的组件索引: {to_remove if to_remove else '（无）'}")

    if args.export_components:
        comp_dir = path.parent / f"{path.stem}_components"
        comp_dir.mkdir(exist_ok=True)
        for s in stats:
            comps[s["idx"]].export(comp_dir / f"comp_{s['idx']:03d}.stl")
        print(f"已导出 {len(stats)} 个组件到: {comp_dir}")

    if not to_remove:
        print("未检测到需要删除的漂浮组件。")
        return

    if args.dry_run or not args.apply:
        print("\n[DRY RUN] 未修改文件。确认无误后加 --apply 执行。")
        return

    # 重建网格：保留未删除的组件
    removed_set = set(to_remove)
    kept = [comps[i] for i in range(len(comps)) if i not in removed_set]
    new_mesh = trimesh.util.concatenate(kept)
    new_mesh.merge_vertices(digits_vertex=5)
    new_mesh.remove_degenerate_faces()
    new_mesh.fix_normals()

    out = Path(args.out) if args.out else path
    if out == path:
        backup = path.with_name(path.stem + "_pre_clean" + path.suffix)
        if not backup.exists():
            path.rename(backup)
            print(f"原文件已备份为: {backup}")
    new_mesh.export(out)
    print(f"已写入: {out}")
    print(f"清理后 watertight={new_mesh.is_watertight}, "
          f"is_volume={new_mesh.is_volume}, "
          f"is_winding_consistent={new_mesh.is_winding_consistent}")

    if not new_mesh.is_watertight:
        print("⚠️  整体仍不水密。建议对清理后的文件再跑一次 pymeshfix，")
        print("   或回到 fix_linker_collisions.py 处理。")


if __name__ == "__main__":
    main()