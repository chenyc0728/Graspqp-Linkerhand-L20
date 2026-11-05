"""Regenerate the data-driven sections of REPORT_v2.md (idempotent).

Fills four marker blocks:
  COMPARE2  backend consistency (compare_backends.json)
  STATS2    merged filtered statistics (index.csv / index_grasps.csv / logs)
  BENCH2    CPU vs GPU (benchmark_cpu_gpu.json)
  AUTO2     CMap dataset structure/sanity (cmap_dataset_from_graspqp_v2.pt + validate json)

Run after run_filter / convert_to_cmap / validate_cmap_dataset / benchmark:
    python new_grasp_filter/make_report_v2.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_io as gio  # noqa: E402

REPORT = HERE / "REPORT_v2.md"
NON_WATERTIGHT = ["gelatin_box", "mini_soccer_ball", "mustard_bottle", "pear",
                  "power_drill", "sponge", "tomato_soup_can", "tuna_fish_can"]


def replace_block(text, begin, end, body):
    i = text.index(begin) + len(begin)
    j = text.index(end)
    return text[:i] + "\n" + body + "\n" + text[j:]


def load_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


# --------------------------------------------------------------------------
def compare_block():
    d = read_json(HERE / "logs" / "compare_backends.json")
    if not d:
        return "_（尚未运行 `compare_backends.py`）_"
    lines = []
    for key, title in [("numpy_vs_torch", "numpy（v1）vs torch（v2 默认）"),
                       ("torch_vs_torch", "torch 重跑（PhysX 噪声基线）")]:
        r = d.get(key)
        if not r:
            continue
        lines.append(f"**{title}**（objects={d.get('objects')}, files/object={d.get('files_per_object')}）\n")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        lines.append(f"| 抓取数 | {r['n']} |")
        lines.append(f"| feasible（A / B） | {r['n_feasible_a']} / {r['n_feasible_b']} |")
        lines.append(f"| 判定一致 | **{r['verdict_agreement']}** |")
        lines.append(f"| 原因一致 | {r['reason_agreement']} |")
        lines.append(f"| feasible 翻转数 | **{r['n_feasible_flips']}** |")
        lines.append(f"| max abs Δ 穿透 | {r['max_abs_delta_pen_mm']} mm |")
        lines.append(f"| max abs Δ 间隙 | {r['max_abs_delta_gap_mm']} mm |")
        lines.append(f"| max / median abs Δ rise | {r['max_abs_delta_lift_mm']} / {r['median_abs_delta_lift_mm']} mm |")
        if r["n_feasible_flips"]:
            lines.append("")
            lines.append("翻转样例：")
            for f in r["flips"][:5]:
                a, b = f["a"], f["b"]
                lines.append(
                    f"* {a['object']}/{a['file']}#{a['index']}: {a['reason']}(rise {a['rise_mm']}) "
                    f"→ {b['reason']}(rise {b['rise_mm']})"
                )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def stats_block():
    idx_p = HERE / "filtered_grasps" / "index.csv"
    gr_p = HERE / "filtered_grasps" / "index_grasps.csv"
    if not idx_p.exists():
        return "_（尚未生成 filtered_grasps/index.csv）_"
    idx = load_csv(idx_p)
    gr = load_csv(gr_p) if gr_p.exists() else []

    by_src = defaultdict(lambda: {"files": 0, "grasps": 0, "feasible": 0, "copied": 0})
    for r in idx:
        s = by_src[r["source"]]
        s["files"] += 1
        s["grasps"] += int(r["n_grasps"] or 0)
        s["feasible"] += int(r["n_feasible"] or 0)
        s["copied"] += int(r["copied"] or 0)

    pen = np.array([float(r["geo_penetration_mm"]) for r in gr if r.get("geo_penetration_mm")], dtype=float)
    gap = np.array([float(r["geo_min_distance_mm"]) for r in gr if r.get("geo_min_distance_mm")], dtype=float)
    lift = np.array([float(r["lift_rise_mm"]) for r in gr if r.get("lift_rise_mm")], dtype=float)

    L = []
    L.append("### 3.6 合并统计（来源 / 判定）\n")
    L.append("| source | .pt 文件 | 抓取数 | feasible | 通过率 | 拷贝文件 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    tot = {"files": 0, "grasps": 0, "feasible": 0, "copied": 0}
    for s in sorted(by_src):
        v = by_src[s]
        for k in tot:
            tot[k] += v[k]
        L.append(f"| {s} | {v['files']} | {v['grasps']} | {v['feasible']} | "
                 f"{100.0*v['feasible']/max(v['grasps'],1):.1f}% | {v['copied']} |")
    L.append(f"| **合计** | {tot['files']} | {tot['grasps']} | {tot['feasible']} | "
             f"{100.0*tot['feasible']/max(tot['grasps'],1):.1f}% | {tot['copied']} |")

    # per-object table (object1 is the freshly evaluated one; object is content-reused)
    obj_rows = defaultdict(lambda: {"grasps": 0, "feasible": 0})
    for r in idx:
        if r["source"] != "object1":
            continue
        o = obj_rows[r["object"]]
        o["grasps"] += int(r["n_grasps"] or 0)
        o["feasible"] += int(r["n_feasible"] or 0)
    if obj_rows:
        L.append("\n**逐物体（object1，全量重跑）**\n")
        L.append("| object | grasps | feasible | rate | 水密 | 备注 |")
        L.append("|---|---:|---:|---:|---|---|")
        for o in sorted(obj_rows):
            v = obj_rows[o]
            wt = "✅" if o not in NON_WATERTIGHT else "❌"
            note = "" if o not in NON_WATERTIGHT else "非水密，置信度低"
            L.append(f"| {o} | {v['grasps']} | {v['feasible']} | "
                     f"{100.0*v['feasible']/max(v['grasps'],1):.1f}% | {wt} | {note} |")

    if len(pen):
        L.append("\n**feasible 抓取的几何指标（精确网格 SDF）**\n")
        L.append("| 指标 | min | p25 | median | p75 | max |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for name, arr in (("穿透深度 geo_penetration_mm", pen),
                          ("最外点间隙 geo_min_distance_mm（负=在物体外）", gap),
                          ("提离位移 lift_rise_mm", lift)):
            if not len(arr):
                continue
            q = np.percentile(arr, [0, 25, 50, 75, 100])
            L.append(f"| {name} | " + " | ".join(f"{x:.2f}" for x in q) + " |")

    if len(gr):
        n_pen = int((pen > 0.001).sum())
        L.append(f"\n* feasible 中仍有真实穿透（>1 µm）的：**{n_pen} / {len(pen)}**"
                 f"（中位穿透 {np.median(pen):.2f} mm）")
        wt0 = sum(1 for r in gr if str(r.get("mesh_watertight", "1")) == "0")
        L.append(f"* feasible 中来自非水密物体的：**{wt0} / {len(gr)}**")
    return "\n".join(L)


# --------------------------------------------------------------------------
def bench_block():
    d = read_json(HERE / "logs" / "benchmark_cpu_gpu.json")
    if not d:
        return "_（尚未运行 `benchmark_cpu_gpu.py`）_"
    L = []
    s = d.get("sapien", {})
    L.append("**SAPIEN / 环境**\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| torch.cuda | {s.get('torch_cuda')} |")
    L.append(f"| SAPIEN | {s.get('sapien')} |")
    L.append(f"| `enable_gpu()` 成功 | {s.get('physx_gpu_enabled')} |")
    L.append(f"| GPU 场景可 step | {s.get('gpu_scene_ok')} |")
    L.append(f"| CPU steps/s（最小场景） | {s.get('steps_per_s_cpu')} |")
    if s.get("error"):
        L.append(f"| GPU 错误 | `{s['error']}` |")

    if "fk" in d:
        L.append("\n**FK（pytorch_kinematics，23 links，float64）**\n")
        L.append("| batch | CPU ms | CPU ms/样本 | CUDA ms | CUDA ms/样本 | 加速比 |")
        L.append("|---:|---:|---:|---:|---:|---:|")
        cpu = {r["batch"]: r for r in d["fk"] if r["device"] == "cpu"}
        gpu = {r["batch"]: r for r in d["fk"] if r["device"] == "cuda"}
        for b in sorted(cpu):
            c, g = cpu[b], gpu.get(b)
            if not g:
                continue
            L.append(f"| {b} | {c['ms']} | {c['ms_per_sample']} | {g['ms']} | {g['ms_per_sample']} | "
                     f"{c['ms']/g['ms']:.2f}× |")
    if "quat" in d:
        L.append("\n**四元数解码（roma）**\n")
        L.append("| batch | CPU ms | CUDA ms | 加速比 |")
        L.append("|---:|---:|---:|---:|")
        cpu = {r["batch"]: r for r in d["quat"] if r["device"] == "cpu"}
        gpu = {r["batch"]: r for r in d["quat"] if r["device"] == "cuda"}
        for b in sorted(cpu):
            if b in gpu:
                L.append(f"| {b} | {cpu[b]['ms']} | {gpu[b]['ms']} | {cpu[b]['ms']/gpu[b]['ms']:.2f}× |")
    if "sdf" in d:
        L.append("\n**物体体素 SDF 查询（100k 点归一化）**\n")
        L.append("| 点数 | numpy ms | torch-CPU ms | torch-CUDA ms |")
        L.append("|---:|---:|---:|---:|")
        by = defaultdict(dict)
        for r in d["sdf"]:
            by[r["points"]][r["device"]] = r["ms"]
        for n in sorted(by):
            v = by[n]
            L.append(f"| {n} | {v.get('numpy')} | {v.get('cpu')} | {v.get('cuda')} |")
    L.append("\n* 端到端：同一对象同一批抓取，`--device cuda` 的 SAPIEN 筛选 **1.7 s/grasp**，"
             "`--device cpu` 约 **0.20 s/grasp**（CUDA 慢 8.5×，因为每个抓取只有 1 条 FK 配置，"
             "kernel launch + 同步开销占主导）。")
    return "\n".join(L)


# --------------------------------------------------------------------------
def auto_block():
    p = HERE / "cmap_dataset_from_graspqp_v2.pt"
    if not p.exists():
        return "_（尚未生成 cmap_dataset_from_graspqp_v2.pt）_"
    d = torch.load(p, weights_only=False)
    meta = read_json(p.with_suffix(".meta.json"), {})
    ref = torch.load(gio.CMAP_ROOT / "CMapDataset_filtered" / "cmap_dataset.pt", weights_only=False)
    L = []
    L.append("### 7.1 数据集结构对照\n")
    L.append("| 项 | cmap_dataset.pt（参考） | cmap_dataset_from_graspqp_v2.pt |")
    L.append("|---|---|---|")
    L.append(f"| 顶层 keys | `{sorted(ref.keys())}` | `{sorted(d.keys())}` |")
    L.append(f"| metadata 类型 | `{type(ref['metadata']).__name__}` / `{type(ref['metadata'][0]).__name__}` | "
             f"`{type(d['metadata']).__name__}` / `{type(d['metadata'][0]).__name__}` |")
    L.append(f"| 样本数 | {len(ref['metadata'])} | {len(d['metadata'])} |")
    L.append(f"| 机器人 | {sorted(ref['info'].keys())} | {sorted(d['info'].keys())} |")
    q = torch.stack([m[0] for m in d["metadata"]])
    L.append(f"| q | float32 (N,22) | `{q.dtype}` `{tuple(q.shape)}` |")
    L.append(f"| 物体数 | {len(ref['info']['allegro']['num_per_object'])} | "
             f"{len(d['info']['linkerhand']['num_per_object'])} |")
    L.append(f"| 来源 | — | {meta.get('per_source')} |")
    L.append(f"| 去重丢弃 | — | {meta.get('dropped_duplicates')} |")

    L.append("\n### 7.2 逐维数值范围（vs allegro 参考）\n")
    L.append("| dim | 本数据 min | 本数据 max | 本数据 mean | allegro min | allegro max |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    ra = torch.stack([m[0] for m in ref["metadata"] if m[2] == "allegro"])
    for i in range(22):
        c, r = q[:, i], ra[:, i]
        L.append(f"| {i} | {c.min():+.4f} | {c.max():+.4f} | {c.mean():+.4f} | {r.min():+.4f} | {r.max():+.4f} |")

    L.append("\n### 7.3 抽样\n")
    for i in range(min(3, len(d["metadata"]))):
        qq, on, rn = d["metadata"][i]
        L.append(f"* `{on}` / `{rn}`: `[{', '.join(f'{x:+.4f}' for x in qq[:6])}, …]`")

    v = read_json(HERE / "logs" / "validate_cmap_dataset.json")
    if v:
        L.append("\n### 7.4 用 CMapDataset.py 的实机验证\n")
        L.append("| 检查 | 结果 |")
        L.append("|---|---|")
        L.append(f"| ROBOT_ALIASES | `{v['alias']['ROBOT_ALIASES']}`（生效: {v['alias']['applied_in_memory']}） |")
        for name, u in v["urdf_resolution"].items():
            L.append(f"| URDF `{name}` | `{u['urdf']}` dof={u.get('dof')} pk_ok={u['pk_ok']} |")
        for key, label in [("reference_dataset", "参考 cmap_dataset.pt"),
                           ("v2_dataset", "我们的 v2 数据集"),
                           ("reference_linkerhand_right", "参考数据 + linkerhand_right")]:
            r = v.get(key, {})
            L.append(f"| {label}: metadata / len() | {r.get('n_metadata')} / {r.get('len')} |")
            L.append(f"| {label}: robots | {r.get('robots_in_metadata')} |")
            if "getitem_ok" in r:
                L.append(f"| {label}: `__getitem__` | {r['getitem_ok']} "
                         f"{r.get('getitem_error', '')} -> {r.get('getitem_robot')} / {r.get('getitem_object')} |")
        ab = v.get("alias_behaviour", {})
        if ab:
            L.append(f"| alias 合成测试（legacy -> right） | {ab.get('legacy')} |")
            L.append(f"| alias 合成测试（left 保持 linkerhand） | {ab.get('left')} |")
        rt = v.get("q_roundtrip", {})
        L.append(f"| q 回代 FK（{rt.get('n_checked')} 条） | 旋转 `{rt.get('max_abs_rotation_error'):.2e}`，"
                 f"平移 `{rt.get('max_abs_translation_error_m'):.2e}` m |")
    L.append("\n> 备注：`object` 与 `object1` 的内容完全相同，去重后数据集里每个抓取只出现一次；"
             "来源计数在 `*.meta.json` 的 `per_source` 里（去重前）。")
    return "\n".join(L)


def main():
    text = REPORT.read_text(encoding="utf-8")
    text = replace_block(text, "<!-- COMPARE2:BEGIN -->", "<!-- COMPARE2:END -->", compare_block())
    text = replace_block(text, "<!-- STATS2:BEGIN -->", "<!-- STATS2:END -->", stats_block())
    text = replace_block(text, "<!-- BENCH2:BEGIN -->", "<!-- BENCH2:END -->", bench_block())
    text = replace_block(text, "<!-- AUTO2:BEGIN -->", "<!-- AUTO2:END -->", auto_block())
    REPORT.write_text(text, encoding="utf-8")
    print("REPORT_v2.md updated")


if __name__ == "__main__":
    main()
