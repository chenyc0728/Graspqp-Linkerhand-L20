#!/usr/bin/env python
"""P1 任务 1：把 ``object/ycb`` 里 v3 修好的 mesh 同步到 ``object1/ycb``。

为什么要做：``object1/ycb`` 的 mesh 没有同步 v3 修复（HANDOFF §7-9），其中 8 个物体仍是
**非水密**的原始 STL。非水密网格会让 ``object_sdf.VoxelSDF`` 的符号/内外判定不可靠，
因此"重筛 object1"必须先把 mesh 补齐，否则等于用一个已知有问题的基础再判一遍。

安全策略（不会误伤）：

1. 先把 ``object1/ycb`` 下**所有**几何文件（``*.stl/*.obj/*.ply/*.urdf``，含子目录）备份到
   ``object1/ycb/_mesh_backup_<ts>/``，保持相对路径；
2. 逐物体判定 `object/<obj>.stl` / `object1/<obj>.stl` / `object/_stl_backup_*/<obj>.stl` 的关系：

   * ``identical``                —— 两边已经一样，跳过（绝大多数物体属于此类）；
   * ``object1_is_pristine_backup`` —— object1 还是修复前的原件，**安全替换**；
   * ``new_geometry``            —— 三方都不同（object1 不是修复前原件），**默认跳过**，
     只有 ``FORCE_REPLACE_NEW_GEOMETRY=1`` 才替换（并在 JSON 里标 ``forced``）；
   * ``no_backup_to_compare``    —— 没有备份可比对，同样默认跳过（同上受 FORCE 控制）；
3. object1 独有的几何文件（object 里没有同名文件）**不删不改**，只在 JSON 里列出；
4. URDF：只有当 object 与 object1 的 ``<obj>.urdf`` 内容不同才一并复制（本仓库实测 28 个全部相同，
   且引用都是相对路径 ``<obj>.stl``，所以替换 STL 就够）；
5. 替换后逐文件核对 sha256（必须与 object 侧一致），并做几何对比：
   体积、水密性、欧拉数、Chamfer 距离（双向最近点）、以及 checker 自己的 ``VoxelSDF``
   在随机查询点上的取值差。

用法::

    python new_grasp_filter/rescreen_mesh_replace.py            # 全部 28 个物体
    python new_grasp_filter/rescreen_mesh_replace.py --objects apple sponge
    python new_grasp_filter/rescreen_mesh_replace.py --dry-run  # 只出方案，不动文件
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OBJ = REPO / "object" / "ycb"
OBJ1 = REPO / "object1" / "ycb"
LOGS = HERE / "logs"
MESH_SUFFIXES = (".stl", ".obj", ".ply")

sys.path.insert(0, str(HERE))


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def objects_of(root: Path) -> list:
    return sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))


def geometry_files(root: Path, obj: str) -> dict:
    """{相对物体目录的路径: 绝对路径}，含 coacd/ 子目录。"""
    out = {}
    d = root / obj
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*")):
        if p.is_file() and (p.suffix.lower() in MESH_SUFFIXES or p.suffix.lower() == ".urdf"):
            out[str(p.relative_to(d))] = p
    return out


def backup_hashes() -> dict:
    idx: dict = {}
    for d in sorted(REPO.glob("object/_stl_backup_*/ycb")):
        if not d.is_dir():
            continue
        for od in sorted(x for x in d.iterdir() if x.is_dir()):
            stl = od / f"{od.name}.stl"
            if stl.exists():
                idx.setdefault(od.name, set()).add(sha256(stl))
    return idx


def pristine_backup_path(obj: str, want_hash: str | None = None) -> Path | None:
    """在 ``object/_stl_backup_*/ycb/<obj>/<obj>.stl`` 里找修复前的原始 mesh。"""
    for d in sorted(REPO.glob("object/_stl_backup_*/ycb")):
        cand = d / obj / f"{obj}.stl"
        if cand.exists() and (want_hash is None or sha256(cand) == want_hash):
            return cand
    return None


def classify(obj: str, bk: dict) -> tuple:
    po, p1 = OBJ / obj / f"{obj}.stl", OBJ1 / obj / f"{obj}.stl"
    ho = sha256(po) if po.exists() else None
    h1 = sha256(p1) if p1.exists() else None
    hb = bk.get(obj, set())
    if h1 is not None and h1 == ho:
        return "identical", ho, h1, sorted(hb)
    if hb and h1 in hb:
        return "object1_is_pristine_backup", ho, h1, sorted(hb)
    if hb and h1 not in hb and ho not in hb:
        return "new_geometry", ho, h1, sorted(hb)
    if not hb:
        return "no_backup_to_compare", ho, h1, sorted(hb)
    return "other", ho, h1, sorted(hb)


def mesh_metrics(path: Path) -> dict:
    import trimesh

    m = trimesh.load(str(path), force="mesh", process=True)
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "vertices": int(len(m.vertices)), "faces": int(len(m.faces)),
            "volume_mm3": float(abs(m.volume) * 1e9), "watertight": bool(m.is_watertight),
            "euler_number": int(m.euler_number),
            "extents_mm": [round(float(x) * 1000, 3) for x in m.extents]}, m


def chamfer_mm(a, b, n: int = 20000, seed: int = 0) -> dict:
    """双向最近点距离（Chamfer）统计，mm。"""
    import numpy as np
    import trimesh

    pa, _ = trimesh.sample.sample_surface(a, n, seed=seed)
    pb, _ = trimesh.sample.sample_surface(b, n, seed=seed)
    da = trimesh.proximity.closest_point(b, pa)[1]
    db = trimesh.proximity.closest_point(a, pb)[1]
    d = np.concatenate([da, db]) * 1000.0
    return {"n_samples": n, "mean_mm": round(float(d.mean()), 6),
            "p95_mm": round(float(np.percentile(d, 95)), 6),
            "max_mm": round(float(d.max()), 6),
            "hausdorff_mm": round(float(max(da.max(), db.max()) * 1000.0), 6)}


def sdf_difference(path_a: Path, path_b: Path, n: int = 2000, seed: int = 0) -> dict:
    """用 checker 自己的 VoxelSDF（object_sdf，边界修正版，见 HANDOFF §7-2）
    在两个网格上各建一次 SDF，比较随机查询点的取值。VoxelSDF 接受**路径**。"""
    import numpy as np
    from object_sdf import VoxelSDF

    sa, sb = VoxelSDF(path_a, pitch=0.002), VoxelSDF(path_b, pitch=0.002)
    a, b = sa.mesh, sb.mesh
    lo = np.minimum(a.bounds[0], b.bounds[0])
    hi = np.maximum(a.bounds[1], b.bounds[1])
    rng = np.random.default_rng(seed)
    pts = rng.uniform(lo, hi, size=(n, 3))
    va, vb = sa.query(pts), sb.query(pts)
    va, vb = np.asarray(va, dtype=float), np.asarray(vb, dtype=float)
    d = np.abs(va - vb) * 1000.0
    return {"n_points": n, "pitch_mm": 2.0,
            "mean_abs_diff_mm": round(float(d.mean()), 6),
            "p99_abs_diff_mm": round(float(np.percentile(d, 99)), 6),
            "max_abs_diff_mm": round(float(d.max()), 6),
            "max_abs_diff_um": round(float(d.max()) * 1000.0, 3),
            "sign_disagreement_frac": round(float(np.mean(np.sign(va) != np.sign(vb))), 6)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--objects", nargs="+", default=None, help="只处理这些物体（默认全部 28 个）")
    ap.add_argument("--ts", default=os.environ.get("TS") or time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--out", default=str(LOGS / "rescreen_mesh_replace.json"))
    ap.add_argument("--dry-run", action="store_true", help="只输出计划，不复制任何文件")
    ap.add_argument("--force-new-geometry", action="store_true",
                    default=os.environ.get("FORCE_REPLACE_NEW_GEOMETRY", "") not in ("", "0", "false", "False"),
                    help="允许替换 new_geometry / no_backup_to_compare 的物体（默认关闭）")
    ap.add_argument("--no-geom", action="store_true", help="跳过 Chamfer / SDF 对比（只做 sha256）")
    args = ap.parse_args()

    t0 = time.time()
    ts = args.ts
    bk = backup_hashes()
    objects = args.objects or objects_of(OBJ)
    missing = [o for o in objects if not (OBJ / o).is_dir()]
    if missing:
        raise SystemExit(f"[error] object/ycb 下不存在: {missing}")

    backup_root = OBJ1 / f"_mesh_backup_{ts}"
    rep = {"task": "P1-task1-mesh-replace", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "ts": ts, "dry_run": bool(args.dry_run), "backup_root": rel(backup_root),
           "force_new_geometry": bool(args.force_new_geometry),
           "objects_requested": objects, "per_object": {}}

    # ------------------------------------------------ 1) 备份 object1 全部几何
    n_backed = 0
    if not args.dry_run:
        backup_root.mkdir(parents=True, exist_ok=True)
        for obj in objects_of(OBJ1):
            for name, src in geometry_files(OBJ1, obj).items():
                dst = backup_root / obj / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                if sha256(dst) != sha256(src):
                    raise SystemExit(f"[error] 备份校验失败: {dst}")
                n_backed += 1
    rep["backup"] = {"root": rel(backup_root), "n_files": n_backed,
                     "note": "备份的是 object1/ycb 下所有 *.stl/*.obj/*.ply/*.urdf（含子目录）"}
    print(f"[backup] {n_backed} 个几何文件 -> {rel(backup_root)}", flush=True)

    summary = Counter()
    for obj in objects:
        kind, ho, h1, hb = classify(obj, bk)
        action = "skip"
        reason = ""
        if kind == "identical":
            reason = "object1 与 object 已经逐字节相同"
        elif kind == "object1_is_pristine_backup":
            action, reason = "replace", "object1 仍是修复前原件（与 _stl_backup 相同），安全替换"
        elif kind in ("new_geometry", "no_backup_to_compare"):
            if args.force_new_geometry:
                action, reason = "replace_forced", f"{kind} + FORCE_REPLACE_NEW_GEOMETRY=1"
            else:
                reason = f"{kind}：object1 不是修复前原件，默认不动（需要 FORCE_REPLACE_NEW_GEOMETRY=1）"
        else:
            reason = f"未分类关系 {kind}，默认不动（需要 FORCE_REPLACE_NEW_GEOMETRY=1）"
            if args.force_new_geometry:
                action = "replace_forced"

        row = {"relation": kind, "action": action, "reason": reason,
               "object_sha256": ho, "object1_sha256_before": h1, "backup_sha256": hb,
               "files_replaced": [], "files_skipped_object1_only": [], "urdf_copied": False}
        print(f"[{obj:22s}] {kind:26s} -> {action:16s} {reason}", flush=True)

        # object1 独有的几何文件（不删不改，只报告）
        go, g1 = geometry_files(OBJ, obj), geometry_files(OBJ1, obj)
        row["files_skipped_object1_only"] = sorted(set(g1) - set(go))
        if row["files_skipped_object1_only"]:
            print(f"    [keep] object1 独有（不动）: {row['files_skipped_object1_only']}", flush=True)

        if action != "skip":
            # 4) URDF：只有内容不同才复制
            uo, u1 = OBJ / obj / f"{obj}.urdf", OBJ1 / obj / f"{obj}.urdf"
            need_urdf = uo.exists() and (not u1.exists() or sha256(uo) != sha256(u1))
            for name, src in sorted(go.items()):
                if src.suffix.lower() not in MESH_SUFFIXES:
                    continue
                dst = OBJ1 / obj / name
                if args.dry_run:
                    row["files_replaced"].append({"file": name, "dry_run": True,
                                                  "would_copy_sha256": sha256(src)})
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                h_new = sha256(dst)
                if h_new != sha256(src):
                    raise SystemExit(f"[error] 替换后 sha256 不一致: {dst}")
                row["files_replaced"].append({"file": name, "sha256_after": h_new,
                                              "matches_object": True})
            if need_urdf:
                if not args.dry_run:
                    dst = OBJ1 / obj / f"{obj}.urdf"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(uo, dst)
                row["urdf_copied"] = True
            if not args.dry_run:
                row["object1_sha256_after"] = sha256(OBJ1 / obj / f"{obj}.stl")
                row["sha256_now_matches_object"] = row["object1_sha256_after"] == ho
                if not row["sha256_now_matches_object"]:
                    raise SystemExit(f"[error] {obj}: 替换后 object1 与 object 仍不同")

        # 5) 几何对比（before = 替换前的原 mesh / after = object 的修复版）
        before_path = None
        for cand in ((backup_root / obj / f"{obj}.stl"), pristine_backup_path(obj)):
            if cand is not None and Path(cand).exists():
                before_path = Path(cand)
                break
        if not args.no_geom and before_path is not None and before_path.exists():
            try:
                after_path = OBJ / obj / f"{obj}.stl"
                mb, m_before = mesh_metrics(before_path)
                ma, m_after = mesh_metrics(after_path)
                row["geometry"] = {
                    "before_source": rel(before_path),
                    "before": {k: v for k, v in mb.items() if k not in ("path",)},
                    "after": {k: v for k, v in ma.items() if k not in ("path",)},
                    "volume_ratio_after_over_before": round(ma["volume_mm3"] / mb["volume_mm3"], 6)
                    if mb["volume_mm3"] else None,
                    "watertight_before": mb["watertight"], "watertight_after": ma["watertight"],
                    "chamfer": chamfer_mm(m_before, m_after),
                    "sdf": sdf_difference(before_path, after_path),
                }
                g = row["geometry"]
                print(f"    [geom] watertight {g['watertight_before']} -> {g['watertight_after']}  "
                      f"vol_ratio={g['volume_ratio_after_over_before']}  "
                      f"chamfer_mean={g['chamfer']['mean_mm']}mm max={g['chamfer']['max_mm']}mm  "
                      f"sdf_max={g['sdf']['max_abs_diff_mm']}mm "
                      f"sign_disagree={g['sdf']['sign_disagreement_frac']}", flush=True)
            except Exception as exc:  # noqa: BLE001
                row["geometry"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"    [geom] 失败: {type(exc).__name__}: {exc}", flush=True)

        rep["per_object"][obj] = row
        summary[f"{kind}->{action}"] += 1

    # Chamfer 有数值底噪（相同 mesh 互比也不为 0，见 REPORT_v6 §mesh 一节），实测一次记录下来
    if not args.no_geom:
        try:
            ref = OBJ / objects[0] / f"{objects[0]}.stl"
            _m, mesh_ref = mesh_metrics(ref)
            rep["chamfer_self_noise_floor"] = {"object": objects[0], **chamfer_mm(mesh_ref, mesh_ref)}
            print(f"[noise] 相同 mesh 自比对（Chamfer 底噪）: "
                  f"mean={rep['chamfer_self_noise_floor']['mean_mm']}mm "
                  f"max={rep['chamfer_self_noise_floor']['max_mm']}mm", flush=True)
        except Exception as exc:  # noqa: BLE001
            rep["chamfer_self_noise_floor"] = {"error": f"{type(exc).__name__}: {exc}"}
    rep["summary"] = dict(summary)
    rep["n_objects"] = len(objects)
    rep["elapsed_s"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[summary] {dict(summary)}", flush=True)
    print(f"[write] {out} ({out.stat().st_size} bytes, {rep['elapsed_s']}s)", flush=True)
    if any(v["action"] == "skip" and v["relation"] in ("new_geometry", "no_backup_to_compare")
           for v in rep["per_object"].values()):
        print("[warn] 有物体因为 new_geometry/no_backup_to_compare 被跳过；"
              "确认要替换请设 FORCE_REPLACE_NEW_GEOMETRY=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
