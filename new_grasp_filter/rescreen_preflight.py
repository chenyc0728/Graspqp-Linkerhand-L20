#!/usr/bin/env python
"""P1 任务 0：现状勘察（只读）。

回答四个必须先弄清的问题，并把**原始输出**写进 JSON，供 ``REPORT_v6_rescreen.md`` 引用：

a) ``object/ycb`` 与 ``object1/ycb`` 的物体列表是否一一对应；
b) 每个物体的 ``object/<obj>.stl`` / ``object1/<obj>.stl`` / ``object/_stl_backup_*`` 三者 sha256 关系
   —— 决定「object1 是修复前的原件」还是「全新的几何」；
c) v3 到底修了哪 8 个 ycb 网格（用 backup 目录反查，不靠记忆）；
d) object1 的 URDF 是否存在、mesh 引用是相对路径还是绝对路径（决定"只换 stl 够不够"）；
e) 两边的 grasp_predictions 是否齐全、内容是否一致；
f) 当前 ``index.csv`` / ``index_grasps.csv`` 的 source 分布（确认是 contactdb），
   ``index_ycb.csv`` 的 ``merged_from`` 取值分布（v1 污染的证据），并给出 apple 第 3 号抓取的例子。

用法::

    python new_grasp_filter/rescreen_preflight.py            # 写 logs/rescreen_preflight.json
    python new_grasp_filter/rescreen_preflight.py --no-geom  # 跳过 trimesh 几何（更快）

本脚本**不写任何数据**，只读 + 写自己的日志 JSON。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OBJ = REPO / "object" / "ycb"
OBJ1 = REPO / "object1" / "ycb"
FG = HERE / "filtered_grasps"
LOGS = HERE / "logs"
DEFAULT_OUT = LOGS / "rescreen_preflight.json"

MESH_SUFFIXES = (".stl", ".obj", ".ply")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def object_names(root: Path) -> list:
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))


def stl_of(root: Path, obj: str) -> Path:
    return root / obj / f"{obj}.stl"


def all_meshes(root: Path, obj: str) -> dict:
    """该物体目录下所有几何文件（含 coacd/ 子目录）。"""
    out = {}
    d = root / obj
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in MESH_SUFFIXES:
            out[rel(p)] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return out


def backup_index() -> dict:
    """{obj: [(backup_stl_path, sha256), ...]}，扫所有 object/_stl_backup_*/ycb/。"""
    idx: dict = {}
    for d in sorted(REPO.glob("object/_stl_backup_*/ycb")):
        if not d.is_dir():
            continue
        for obj_dir in sorted(x for x in d.iterdir() if x.is_dir()):
            stl = stl_of(d, obj_dir.name)
            if stl.exists():
                idx.setdefault(obj_dir.name, []).append({"path": rel(stl), "sha256": sha256(stl),
                                                         "bytes": stl.stat().st_size,
                                                         "backup_dir": rel(d)})
    return idx


def urdf_info(root: Path, obj: str) -> dict:
    p = root / obj / f"{obj}.urdf"
    info = {"path": rel(p), "exists": p.exists()}
    if not p.exists():
        return info
    info["sha256"] = sha256(p)
    info["bytes"] = p.stat().st_size
    refs, kinds = [], Counter()
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        stripped = text.lstrip()
        info["had_leading_whitespace"] = len(stripped) != len(text)
        tree = ET.ElementTree(ET.fromstring(stripped))
    except ET.ParseError as exc:
        info["parse_error"] = f"{type(exc).__name__}: {exc}"
        return info
    for mesh in tree.iter("mesh"):
        fn = mesh.get("filename")
        if fn is None:
            continue
        refs.append(fn)
        if os.path.isabs(fn):
            kinds["absolute"] += 1
            kinds["absolute_points_into_object1"] += int("object1" in fn)
        else:
            kinds["relative"] += 1
            kinds["relative_missing_file"] += int(not (p.parent / fn).exists())
    info["mesh_refs"] = sorted(set(refs))
    info["mesh_ref_kinds"] = dict(kinds)
    info["all_relative"] = kinds["absolute"] == 0 and len(refs) > 0
    info["n_refs"] = len(refs)
    return info


def compare_grasp_pt(root_o: Path, root_1: Path, names) -> dict:
    """逐张量比较两边的 *.dexgrasp.pt（**不比 md5 / sha256**，zip 容器不可逐字节复现）。"""
    import torch

    out = {}
    for name in names:
        po = next(iter(sorted(root_o.rglob(name))), None)
        p1 = next(iter(sorted(root_1.rglob(name))), None)
        if po is None or p1 is None:
            out[name] = {"error": "file missing on one side"}
            continue
        ao = torch.load(po, map_location="cpu", weights_only=False)
        a1 = torch.load(p1, map_location="cpu", weights_only=False)
        keys_equal = sorted(ao.keys()) == sorted(a1.keys())
        item = {"keys_equal": keys_equal, "torch_equal": {}, "max_abs_diff": {}}
        # 顶层扁平 key + ``parameters`` 里嵌套的 root_pose / 关节角都要比
        pairs = {k: (ao.get(k), a1.get(k)) for k in
                 ("values", "contact_idx", "grasp_velocities")}
        pa, pb = ao.get("parameters") or {}, a1.get("parameters") or {}
        for k in sorted(set(pa) | set(pb)):
            pairs[f"parameters.{k}"] = (pa.get(k), pb.get(k))
        for k, (x, y) in pairs.items():
            if x is None or y is None:
                item["torch_equal"][k] = False
                continue
            if not (torch.is_tensor(x) and torch.is_tensor(y)) or x.shape != y.shape:
                item["torch_equal"][k] = False
                continue
            item["torch_equal"][k] = bool(torch.equal(x, y))
            if x.is_floating_point():
                item["max_abs_diff"][k] = float((x - y).abs().max())
        item["identical_content"] = bool(item["keys_equal"] and item["torch_equal"]
                                         and all(item["torch_equal"].values()))
        out[name] = item
    return out


def grasp_files(root: Path, obj: str) -> dict:
    d = root / obj / "grasp_predictions"
    if not d.is_dir():
        return {"dir_exists": False, "files": {}}
    files = {}
    for p in sorted(d.rglob("*.dexgrasp.pt")):
        files[rel(p.relative_to(d))] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return {"dir_exists": True, "n_files": len(files), "files": files}


def csv_rows(path: Path) -> list:
    import csv
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def geom_stats(path: Path) -> dict:
    """体积 / 是否水密 / 欧拉数（trimesh，process=True 见 HANDOFF §7-3）。"""
    import trimesh
    m = trimesh.load(str(path), force="mesh", process=True)
    return {"vertices": int(len(m.vertices)), "faces": int(len(m.faces)),
            "volume_mm3": float(abs(m.volume) * 1e9),
            "watertight": bool(m.is_watertight),
            "euler_number": int(m.euler_number),
            "extents_mm": [round(float(x) * 1000, 3) for x in m.extents],
            "bounds_mm": [[round(float(v) * 1000, 3) for v in row] for row in m.bounds]}


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "cwd": os.getcwd()}
    for key in ("CONDA_DEFAULT_ENV", "CUDA_VISIBLE_DEVICES"):
        info[key] = os.environ.get(key)
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                            capture_output=True, text=True, check=False).stdout.strip()
        info["git_dirty_files"] = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                                 capture_output=True, text=True,
                                                 check=False).stdout.strip().splitlines()
    except OSError as exc:  # pragma: no cover
        info["git_error"] = str(exc)
    for mod in ("torch", "trimesh", "numpy", "sapien"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            info[mod] = f"unavailable: {type(exc).__name__}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-geom", action="store_true", help="跳过 trimesh 体积/水密检查")
    ap.add_argument("--json-indent", type=int, default=2)
    args = ap.parse_args()

    t_start = time.time()
    names_o, names_1 = object_names(OBJ), object_names(OBJ1)
    rep: dict = {"task": "P1-task0-preflight", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "env": env_info(),
                 "paths": {"object_ycb": rel(OBJ), "object1_ycb": rel(OBJ1),
                           "filtered_grasps": rel(FG)}}

    # ---------------------------------------------------------------- a) 列表
    so, s1 = set(names_o), set(names_1)
    rep["a_object_lists"] = {
        "n_object": len(names_o), "n_object1": len(names_1),
        "identical_sets": so == s1,
        "only_in_object": sorted(so - s1), "only_in_object1": sorted(s1 - so),
        "objects": names_o,
    }
    print(f"[a] object={len(names_o)} object1={len(names_1)} 集合相同={so == s1} "
          f"只在一侧={sorted(so ^ s1)}")

    # ---------------------------------------------------- b/c) sha256 + 备份
    backups = backup_index()
    rep["b_per_object"] = {}
    relation = Counter()
    fixed8 = []
    for obj in names_o:
        po, p1 = stl_of(OBJ, obj), stl_of(OBJ1, obj)
        row = {"object_stl": rel(po), "object_exists": po.exists(),
               "object_sha256": sha256(po) if po.exists() else None,
               "object1_stl": rel(p1), "object1_exists": p1.exists(),
               "object1_sha256": sha256(p1) if p1.exists() else None,
               "backups": backups.get(obj, []),
               "backup_sha256": [b["sha256"] for b in backups.get(obj, [])]}
        ho, h1 = row["object_sha256"], row["object1_sha256"]
        hb = set(row["backup_sha256"])
        if h1 is not None and ho is not None and h1 == ho:
            kind = "identical"
        elif hb and h1 in hb:
            kind = "object1_is_pristine_backup"      # object1 还是修复前的原件 -> 可安全替换
        elif hb and h1 not in hb and ho not in hb:
            kind = "new_geometry"                    # 三方都不同 -> 必须 FORCE 才替换
        elif not hb:
            kind = "no_backup_to_compare"
        else:
            kind = "other"
        row["relation"] = kind
        relation[kind] += 1
        if hb and ho not in hb:
            fixed8.append(obj)
        rep["b_per_object"][obj] = row
        print(f"[b] {obj:22s} {kind:26s} obj={str(ho)[:8]} obj1={str(h1)[:8]} "
              f"backup={[b[:8] for b in row['backup_sha256']]}")

    rep["b_relation_counts"] = dict(relation)
    rep["c_v3_fixed_meshes"] = {
        "objects_with_backup": sorted(backups),
        "n_with_backup": len(backups),
        "objects_whose_object_stl_differs_from_backup": sorted(fixed8),
        "n_fixed": len(fixed8),
    }
    print(f"[c] 有 backup 的物体 {len(backups)} 个: {sorted(backups)}")
    print(f"[c] object 与 backup 不同（= v3 修过）的物体 {len(fixed8)} 个: {sorted(fixed8)}")

    # ---------------------------------------------------------------- d) URDF
    rep["d_urdf"] = {}
    same_urdf = 0
    for obj in names_o:
        uo, u1 = urdf_info(OBJ, obj), urdf_info(OBJ1, obj)
        rep["d_urdf"][obj] = {"object": uo, "object1": u1,
                              "urdf_identical": uo.get("sha256") == u1.get("sha256")}
        same_urdf += rep["d_urdf"][obj]["urdf_identical"]
    all_rel = all(v["object1"].get("all_relative") for v in rep["d_urdf"].values())
    rep["d_summary"] = {"n_urdf_identical": same_urdf, "n_objects": len(names_o),
                        "object1_all_urdf_relative": all_rel,
                        "conclusion": ("URDF 全部是相对路径且两边相同 -> 只替换 .stl/.obj 就够，不需要动 URDF"
                                       if all_rel and same_urdf == len(names_o) else
                                       "存在绝对路径或 URDF 不一致 -> 需要把 object 的 URDF 一并复制（见 d_urdf）")}
    print(f"[d] URDF 两边相同 {same_urdf}/{len(names_o)}；object1 全部相对路径={all_rel}")

    # ------------------------------------------------------- e) grasp_predictions
    #    sha256 全都不一样是预期内的（torch.save 的 zip 容器不逐字节可复现），
    #    所以对前 3 个物体额外做语义比较：root_pose / values / contact_idx 逐张量 torch.equal
    semantic_objects = names_o[:3]
    rep["e_semantic_objects"] = semantic_objects
    rep["e_grasp_predictions"] = {}
    for obj in names_o:
        go, g1 = grasp_files(OBJ, obj), grasp_files(OBJ1, obj)
        so_, s1_ = go.get("files", {}), g1.get("files", {})
        diff = sorted(k for k in set(so_) | set(s1_)
                      if so_.get(k, {}).get("sha256") != s1_.get(k, {}).get("sha256"))
        entry = {
            "object": {"dir_exists": go["dir_exists"], "n_files": go.get("n_files", 0)},
            "object1": {"dir_exists": g1["dir_exists"], "n_files": g1.get("n_files", 0)},
            "same_file_names": set(so_) == set(s1_),
            "n_differing_files": len(diff), "differing_files": diff[:10],
        }
        if obj in semantic_objects and diff:
            try:
                entry["semantic"] = compare_grasp_pt(OBJ / obj, OBJ1 / obj, diff[:3])
            except Exception as exc:  # noqa: BLE001
                entry["semantic"] = {"error": f"{type(exc).__name__}: {exc}"}
        rep["e_grasp_predictions"][obj] = entry
    counts_o = Counter(v["object"]["n_files"] for v in rep["e_grasp_predictions"].values())
    counts_1 = Counter(v["object1"]["n_files"] for v in rep["e_grasp_predictions"].values())
    rep["e_summary"] = {"n_files_hist_object": dict(counts_o),
                        "n_files_hist_object1": dict(counts_1),
                        "any_differing_content": any(v["n_differing_files"] for v in rep["e_grasp_predictions"].values())}
    print(f"[e] dexgrasp 文件数分布 object={dict(counts_o)} object1={dict(counts_1)} "
          f"内容有差异={rep['e_summary']['any_differing_content']}")

    # ---------------------------------------------------------------- f) 索引
    def describe_index(path: Path) -> dict:
        rows = csv_rows(path)
        src = Counter(r.get("source", "") for r in rows)
        mf = Counter(r.get("merged_from", "") for r in rows)
        return {"path": rel(path), "exists": path.exists(), "n_rows": len(rows),
                "columns": list(rows[0].keys()) if rows else [],
                "source_counts": dict(src),
                "merged_from_counts": dict(mf),
                "n_feasible_sum": sum(int(r.get("n_feasible") or 0) for r in rows),
                "objects": sorted({r.get("object", "") for r in rows})}

    rep["f_indexes"] = {
        "index.csv": describe_index(FG / "index.csv"),
        "index_grasps.csv": describe_index(FG / "index_grasps.csv"),
        "index_ycb.csv": describe_index(FG / "index_ycb.csv"),
        "index_ycb_grasps.csv": describe_index(FG / "index_ycb_grasps.csv"),
    }
    apple_rows = [r for r in csv_rows(FG / "index_ycb.csv") if r.get("object") == "apple"]
    rep["f_apple_example"] = [{
        "object": r.get("object"), "source": r.get("source"),
        "merged_from": r.get("merged_from"), "n_feasible": r.get("n_feasible"),
        "feasible_indices": r.get("feasible_indices"), "reason_counts": r.get("reason_counts"),
        "grasp_file": r.get("grasp_file"), "sha256": r.get("sha256"),
        "contains_index_3": "3" in (r.get("feasible_indices") or "").split(),
    } for r in apple_rows]
    for name, d in rep["f_indexes"].items():
        print(f"[f] {name:22s} rows={d['n_rows']:5d} source={d['source_counts']} "
              f"merged_from={ {k or '(空)': v for k, v in d['merged_from_counts'].items()} }")
    print(f"[f] index_ycb.csv 的 apple 行: {json.dumps(rep['f_apple_example'], ensure_ascii=False)[:400]}")

    # ------------------------------------------------------ 可选：几何体积/水密
    if not args.no_geom:
        rep["geom"] = {}
        for obj in names_o:
            row = {}
            for tag, p in (("object", stl_of(OBJ, obj)), ("object1", stl_of(OBJ1, obj))):
                if p.exists():
                    try:
                        row[tag] = geom_stats(p)
                    except Exception as exc:  # noqa: BLE001
                        row[tag] = {"error": f"{type(exc).__name__}: {exc}"}
            if "object" in row and "object1" in row and "volume_mm3" in row["object"] and "volume_mm3" in row["object1"]:
                v0, v1 = row["object"]["volume_mm3"], row["object1"]["volume_mm3"]
                row["volume_ratio_object1_over_object"] = round(v1 / v0, 6) if v0 else None
                row["watertight_changed"] = (row["object"]["watertight"] != row["object1"]["watertight"])
            rep["geom"][obj] = row
            w_o = row.get("object", {}).get("watertight")
            w_1 = row.get("object1", {}).get("watertight")
            print(f"[geom] {obj:22s} watertight object={w_o} object1={w_1} "
                  f"ratio={row.get('volume_ratio_object1_over_object')}")
        rep["geom_summary"] = {
            "n_object1_not_watertight": sum(1 for v in rep["geom"].values()
                                            if v.get("object1", {}).get("watertight") is False),
            "n_object_watertight": sum(1 for v in rep["geom"].values()
                                       if v.get("object", {}).get("watertight") is True),
        }

    # ------------------------------------------------------- i) 任务 1 的结果（若有）
    mr_json = LOGS / "rescreen_mesh_replace.json"
    mr_prev = sorted(LOGS.glob("rescreen_mesh_replace_replace_run.json"))
    if mr_json.exists():
        try:
            rep["i_mesh_replace"] = json.loads(mr_json.read_text())
            if mr_prev:
                rep["i_mesh_replace_replace_run"] = json.loads(mr_prev[0].read_text())
        except Exception as exc:  # noqa: BLE001
            rep["i_mesh_replace"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        rep["i_mesh_replace"] = {"note": f"{mr_json} 不存在（任务 1 还没跑）"}

    rep["elapsed_s"] = round(time.time() - t_start, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=args.json_indent, ensure_ascii=False) + "\n")
    print(f"\n[write] {out}  ({out.stat().st_size} bytes, {rep['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
