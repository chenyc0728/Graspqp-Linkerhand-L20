#!/usr/bin/env python
"""P1 任务 6：合并前的三份索引校验（``convert_to_cmap.py`` 之前的守门人）。

校验内容（任何一项失败 -> 退出码 1，``rescreen_all.sh`` 立刻停）：

1. 每份索引的 ``source`` 列是否与期望一致（``object_rescreen`` / ``object1_rescreen`` /
   ``object_contactdb``），且每份都非空；
2. 每份索引**每一行**的 ``grasp_file`` 副本是否存在、``sha256`` 是否与行内记录一致
   （副本就是 ``convert_to_cmap`` 将要读的文件，这一项过了才谈得上"索引可信"）；
3. 每份的 ``sum(n_feasible)`` 是否等于它对应的**逐抓取**索引
   （``<prefix>_index_grasps.csv``）的行数；并检查 ``len(feasible_indices) == n_feasible``、
   ``max(feasible_indices) < n_grasps``；
4. 三份之间 ``(sha256, grasp_index)`` 是否重复（跨来源去重预检）——重复意味着同一条抓取会被
   转换两次、CMap 里去重后会少样本，需要提前发现；
5. 打印每份的物体覆盖列表与 ``n_feasible`` 分布。

另外会顺手报告 ``merged_from`` 列是否全空（重筛必须是全新筛选，见 P1 任务 2）。

用法::

    python new_grasp_filter/rescreen_merge_check.py \\
        --indexes new_grasp_filter/filtered_grasps/rescreen/ycb_object_index.csv \\
                  new_grasp_filter/filtered_grasps/rescreen/ycb_object1_index.csv \\
                  new_grasp_filter/filtered_grasps/index_contactdb_<ts>.csv \\
        --expected-sources object_rescreen object1_rescreen object_contactdb \\
        --out new_grasp_filter/logs/rescreen_merge_check.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_BASE = HERE / "filtered_grasps"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def resolve_grasp(relpath: str, bases) -> Path | None:
    rel = relpath.replace("\\", "/")
    for base in bases:
        cand = Path(base) / rel
        if cand.exists():
            return cand
    return None


def check_index(path: Path, expected_source: str, bases, max_examples: int = 5) -> dict:
    res: dict = {"path": str(path), "expected_source": expected_source, "ok": True, "errors": []}
    if not path.exists():
        res["ok"] = False
        res["errors"].append(f"索引不存在: {path}")
        return res
    rows = read_csv(path)
    res["n_rows"] = len(rows)
    if not rows:
        res["ok"] = False
        res["errors"].append("索引为空")
        return res
    sources = Counter(r.get("source", "") for r in rows)
    res["source_counts"] = dict(sources)
    if set(sources) != {expected_source}:
        res["ok"] = False
        res["errors"].append(f"source 列应为 {expected_source}，实际 {dict(sources)}")
    merged = Counter(r.get("merged_from", "") for r in rows)
    res["merged_from_counts"] = {k or "(空)": v for k, v in merged.items()}
    if any(k for k in merged):
        res["ok"] = False
        res["errors"].append(f"merged_from 非空（重筛必须全新筛选）: {res['merged_from_counts']}")

    # ---- 逐行：副本存在 + sha256 一致 + feasible_indices 自洽
    missing, hash_bad, idx_bad = [], [], []
    n_zero_feasible = 0
    feas_sum = 0
    per_object = Counter()
    per_object_feas = Counter()
    for r in rows:
        obj = r.get("object", "")
        per_object[obj] += 1
        n_feas = int(r.get("n_feasible") or 0)
        feas_sum += n_feas
        per_object_feas[obj] += n_feas
        fi = [int(x) for x in (r.get("feasible_indices") or "").split()]
        n_grasps = int(r.get("n_grasps") or 0)
        if len(fi) != n_feas:
            idx_bad.append({"grasp_file": r.get("grasp_file"), "n_feasible": n_feas,
                            "len_feasible_indices": len(fi)})
        if fi and max(fi) >= n_grasps:
            idx_bad.append({"grasp_file": r.get("grasp_file"), "n_grasps": n_grasps, "max_index": max(fi)})
        if n_feas == 0:
            # run_filter 只复制"至少 1 条可行抓取"的文件，0 可行的行本来就没有副本
            n_zero_feasible += 1
            continue
        p = resolve_grasp(r.get("grasp_file", ""), bases)
        if p is None:
            missing.append(r.get("grasp_file"))
            continue
        h = sha256(p)
        if h != r.get("sha256"):
            hash_bad.append({"file": str(p), "index_sha256": r.get("sha256"), "file_sha256": h})
    res["n_rows_zero_feasible_not_copied"] = n_zero_feasible
    res["n_rows_requiring_copy"] = len(rows) - n_zero_feasible
    res["n_missing_copies"] = len(missing)
    res["missing_copies_examples"] = missing[:max_examples]
    res["n_hash_mismatch"] = len(hash_bad)
    res["hash_mismatch_examples"] = hash_bad[:max_examples]
    res["n_feasible_index_inconsistencies"] = len(idx_bad)
    res["feasible_index_examples"] = idx_bad[:max_examples]
    if missing:
        res["ok"] = False
        res["errors"].append(f"{len(missing)} 行的 grasp_file 副本找不到")
    if hash_bad:
        res["ok"] = False
        res["errors"].append(f"{len(hash_bad)} 行的 sha256 与副本不一致")
    if idx_bad:
        res["ok"] = False
        res["errors"].append(f"{len(idx_bad)} 行的 feasible_indices 不自洽")

    # ---- sum(n_feasible) vs 逐抓取索引行数
    stem = path.name[: -len("_index.csv")] if path.name.endswith("_index.csv") else path.stem
    # 逐抓取索引的候选名：普通命名 <stem>_index_grasps.csv；备份命名 index_contactdb_<ts>.csv
    # 对应 index_grasps_contactdb_<ts>.csv；原始命名 index.csv 对应同目录 index_grasps.csv
    cands = [path.with_name(f"{stem}_index_grasps.csv")]
    m = re.match(r"^index_([A-Za-z0-9]+)_(.+)\.csv$", path.name)
    if m:
        cands.append(path.with_name(f"index_grasps_{m.group(1)}_{m.group(2)}.csv"))
    if path.name == "index.csv":
        cands.append(path.with_name("index_grasps.csv"))
    grasp_index = next((c for c in cands if c.exists()), cands[0])
    res["grasp_index_candidates"] = [str(c) for c in cands]
    res["grasp_index"] = str(grasp_index)
    res["sum_n_feasible"] = feas_sum
    if grasp_index.exists():
        grows = read_csv(grasp_index)
        res["n_grasp_rows"] = len(grows)
        res["n_feasible_sum_matches_grasp_rows"] = (len(grows) == feas_sum)
        if len(grows) != feas_sum:
            res["ok"] = False
            res["errors"].append(f"sum(n_feasible)={feas_sum} != 逐抓取索引行数={len(grows)}")
    else:
        res["n_grasp_rows"] = None
        res["n_feasible_sum_matches_grasp_rows"] = None
        res["errors"].append(f"[warn] 缺少逐抓取索引 {grasp_index.name}，跳过该项")
        res["warn_no_grasp_index"] = True

    res["objects"] = sorted(per_object)
    res["n_objects"] = len(per_object)
    res["per_object_rows"] = dict(sorted(per_object.items()))
    res["per_object_feasible"] = dict(sorted(per_object_feas.items()))
    res["feasible_histogram"] = dict(sorted(Counter(per_object_feas.values()).items()))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--indexes", nargs="+", required=True)
    ap.add_argument("--expected-sources", nargs="+", required=True)
    ap.add_argument("--grasp-base", default=str(DEFAULT_BASE),
                    help="grasp_file 相对路径的解析基准（默认 new_grasp_filter/filtered_grasps）")
    ap.add_argument("--required-objects", nargs="+", default=None,
                    help="要求这些物体在三份索引里合起来都出现（pilot 用）")
    ap.add_argument("--out", default=str(HERE / "logs" / "rescreen_merge_check.json"))
    ap.add_argument("--json-indent", type=int, default=2)
    args = ap.parse_args()

    if len(args.indexes) != len(args.expected_sources):
        raise SystemExit("--indexes 与 --expected-sources 数量必须一致（一一对应）")

    t0 = time.time()
    bases = [Path(args.grasp_base)]
    for idx in args.indexes:
        p = Path(idx)
        bases += [p.parent, p.parent.parent]
    bases.append(REPO)
    rep = {"task": "P1-task6-merge-check", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "grasp_base": str(bases[0]), "indexes": []}

    for idx, src in zip(args.indexes, args.expected_sources):
        one = check_index(Path(idx), src, bases)
        rep["indexes"].append(one)
        flag = "OK" if one.get("ok") else "FAIL"
        print(f"[{flag}] {Path(idx).name} source={src} rows={one.get('n_rows')} "
              f"sum_feasible={one.get('sum_n_feasible')} grasp_rows={one.get('n_grasp_rows')} "
              f"objects={one.get('n_objects')}", flush=True)
        print(f"       merged_from={one.get('merged_from_counts')} "
              f"missing={one.get('n_missing_copies')} hash_bad={one.get('n_hash_mismatch')} "
              f"idx_bad={one.get('n_feasible_index_inconsistencies')}", flush=True)
        for e in one.get("errors", []):
            print(f"       ! {e}", flush=True)

    # ---- 跨来源 (sha256, grasp_index) 去重预检
    seen: dict = {}
    dups = []
    unresolved: list = []
    n_checked = 0
    per_source_sha = {}
    for one in rep["indexes"]:
        rows = read_csv(Path(one["path"])) if Path(one["path"]).exists() else []
        # 注意：逐抓取索引里的 grasp_file 是 **basename**（v1 以来的约定），
        # 文件级索引里是完整相对路径，两边只能按 basename 对齐。
        sha_by_file = {Path(r["grasp_file"]).name: r.get("sha256") for r in rows}
        src = one["expected_source"]
        gpath = one.get("grasp_index")
        grows = read_csv(Path(gpath)) if gpath and Path(gpath).exists() else []
        if not grows:  # 没有逐抓取索引时，退回用 feasible_indices 展开
            grows = [{"grasp_file": r["grasp_file"], "grasp_index": i, "object": r["object"]}
                     for r in rows for i in (r.get("feasible_indices") or "").split()]
        for g in grows:
            n_checked += 1
            gf = g["grasp_file"]
            sha = sha_by_file.get(Path(gf).name) or sha_by_file.get(gf)
            if sha is None:
                unresolved.append({"grasp_file": gf, "source": src})
            key = (sha or f"<unresolved:{gf}>", int(g["grasp_index"]))
            per_source_sha.setdefault(src, set()).add(key)
            if key in seen:
                if len(dups) < 20:
                    dups.append({"sha256": key[0], "grasp_index": key[1],
                                 "object": g.get("object"),
                                 "sources": [seen[key], src]})
            else:
                seen[key] = src
    if unresolved:
        rep["cross_source_dedup_unresolved"] = unresolved[:10]
        print(f"[warn] {len(unresolved)} 条抓取无法对齐到 sha256（见 JSON）", flush=True)
    rep["cross_source_dedup"] = {
        "n_grasps_checked": n_checked,
        "n_unique": len(seen),
        "n_duplicates": len(dups),
        "duplicate_examples": dups,
        "per_source_unique": {k: len(v) for k, v in per_source_sha.items()},
    }
    overlap = {}
    srcs = sorted(per_source_sha)
    for i, a in enumerate(srcs):
        for b in srcs[i + 1:]:
            inter = per_source_sha[a] & per_source_sha[b]
            if inter:
                overlap[f"{a}|{b}"] = len(inter)
    rep["cross_source_overlap_counts"] = overlap
    ok_dedup = not dups
    print(f"[{'OK' if ok_dedup else 'FAIL'}] 跨来源去重预检: 检查 {n_checked} 条抓取，"
          f"唯一 {len(seen)}，重复 {len(dups)}", flush=True)
    if overlap:
        print(f"       来源两两重叠: {overlap}", flush=True)

    # ---- 物体覆盖
    covered = set()
    for one in rep["indexes"]:
        covered |= set(one.get("objects") or [])
    rep["objects_covered"] = sorted(covered)
    rep["n_objects_covered"] = len(covered)
    if args.required_objects:
        miss = [o for o in args.required_objects if o not in covered]
        rep["required_objects_missing"] = miss
        if miss:
            print(f"[FAIL] --required-objects 缺失: {miss}", flush=True)
    else:
        rep["required_objects_missing"] = []

    rep["ok"] = (all(one.get("ok") for one in rep["indexes"]) and ok_dedup
                 and not rep["required_objects_missing"])
    rep["elapsed_s"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=args.json_indent, ensure_ascii=False) + "\n")
    print(f"[write] {out} ({out.stat().st_size} bytes, {rep['elapsed_s']}s)", flush=True)
    print(f"RESULT={'OK' if rep['ok'] else 'FAIL'}",
          flush=True)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
