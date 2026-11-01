#!/usr/bin/env python3
"""
从 URDF 的 collision mesh 自动生成 penetration_points.json。

每个 link 的 collision mesh 沿 OBB 最长轴分层，每层 1 个球，
半径 = min(短轴半宽) * overshoot，裁剪到 [min_radius, max_radius]。
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    m.merge_vertices(digits_vertex=5)
    m.remove_degenerate_faces()
    m.remove_duplicate_faces()
    return m


def obb_spheres(
    mesh: trimesh.Trimesh,
    overshoot: float = 1.0,
    min_radius: float = 0.004,
    max_radius: float = 0.02,
    max_spheres: int = 4,
) -> list:
    obb = mesh.bounding_box_oriented
    center = np.asarray(obb.centroid, dtype=float)
    extents = np.asarray(obb.primitive.extents, dtype=float)  # 全尺寸
    axes = obb.primitive.transform[:3, :3]
    axes = axes / np.linalg.norm(axes, axis=0, keepdims=True)

    order = np.argsort(-extents)
    long_idx = order[0]
    long_axis = axes[:, long_idx]
    long_len = extents[long_idx]

    short_half = extents[order[1:]] / 2.0
    radius = float(np.clip(min(short_half) * overshoot, min_radius, max_radius))

    n = max(1, int(np.ceil(long_len / (2.0 * radius * 0.85))))
    n = min(n, max_spheres)

    spheres = []
    for i in range(n):
        t = 0.0 if n == 1 else (i / (n - 1) - 0.5)
        c = center + long_axis * long_len * t
        spheres.append([round(float(c[0]), 5), round(float(c[1]), 5),
                        round(float(c[2]), 5), round(radius, 5)])
    return spheres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--mesh_root", required=True, help="URDF 中相对 mesh 路径的根目录")
    ap.add_argument("--out", default="penetration_points.json")
    ap.add_argument("--overshoot", type=float, default=1.0,
                    help="半径相对 OBB 半宽的倍数（>1 略外溢，<1 更保守）")
    ap.add_argument("--min_radius", type=float, default=0.004)
    ap.add_argument("--max_radius", type=float, default=0.02)
    ap.add_argument("--max_spheres", type=int, default=4)
    ap.add_argument("--link_filter", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=None)
    args = ap.parse_args()

    root = ET.parse(args.urdf).getroot()
    mesh_root = Path(args.mesh_root)

    result = {}
    for link in root.findall("link"):
        name = link.get("name")
        if args.link_filter and name not in args.link_filter:
            continue
        if args.exclude and name in args.exclude:
            continue
        col = link.find("collision")
        if col is None:
            continue
        mesh_elem = col.find("geometry/mesh")
        if mesh_elem is None:
            continue
        rel = mesh_elem.get("filename")
        if rel.startswith("package://") or rel.startswith("file://"):
            print(f"[跳过] {name}: 非相对路径 {rel}")
            continue
        path = mesh_root / rel
        if not path.exists():
            print(f"[跳过] {name}: {path} 不存在")
            continue

        mesh = load_mesh(path)
        ext = np.round(mesh.bounding_box.extents, 4).tolist()
        spheres = obb_spheres(
            mesh,
            overshoot=args.overshoot,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            max_spheres=args.max_spheres,
        )
        result[name] = spheres
        print(f"[{name}] extents={ext}  ->  {len(spheres)} 球")
        for s in spheres:
            print(f"    c=({s[0]:+.4f}, {s[1]:+.4f}, {s[2]:+.4f})  r={s[3]:.4f}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n写入 {args.out}，覆盖 {len(result)} 个 link")


if __name__ == "__main__":
    main()