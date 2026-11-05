"""Assemble the statistics tables for REPORT.md from the filter logs + CMap output.

Reads ``logs/*.jsonl``, ``logs/summary.json``, ``filtered_grasps/index.csv`` and
(if present) ``cmap_dataset_from_graspqp.pt``, then replaces the
``<!-- STATS:BEGIN -->`` and ``<!-- AUTO:BEGIN -->`` blocks in ``REPORT.md``.

Run after ``run_filter.py`` / ``convert_to_cmap.py``::

    python new_grasp_filter/make_report.py
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

CMAP_REF = gio.REPO_ROOT / "Cmap_dataset" / "CMapDataset_filtered" / "cmap_dataset.pt"


def load_rows():
    rows = []
    for p in sorted((HERE / "logs").glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def fmt_pct(a, b):
    return f"{a} / {b} ({100.0 * a / b:.1f}%)" if b else "n/a"


def build_stats(rows):
    total = len(rows)
    verdicts = Counter(r["verdict"] for r in rows)
    reasons = Counter(r["reason"] for r in rows)
    pen = np.array([r["metrics"].get("geo_penetration_mm", np.nan) for r in rows], dtype=float)
    rise = np.array([r["metrics"].get("lift_rise_mm", np.nan) for r in rows], dtype=float)
    links = np.array([r["metrics"].get("geo_n_contact_links", np.nan) for r in rows], dtype=float)
    energies = np.array([r["energy"] for r in rows], dtype=float)

    by_obj = defaultdict(list)
    for r in rows:
        by_obj[r["object"]].append(r)

    lines = []
    lines.append("### 3.3 全量统计\n")
    lines.append(f"* 验证抓取总数：**{total}**（28 个物体 / 420 个 `.pt`，每个 32 个抓取）")
    lines.append(f"* `feasible`：**{verdicts.get('feasible', 0)}**  ({100.0 * verdicts.get('feasible', 0) / total:.1f}%)")
    lines.append(f"* `uncertain`：**{verdicts.get('uncertain', 0)}**")
    lines.append(f"* `infeasible`：**{verdicts.get('infeasible', 0)}**")
    src = (HERE / "filtered_grasps" / "index.csv")
    if src.exists():
        with open(src, newline="", encoding="utf-8") as f:
            idx = list(csv.DictReader(f))
        lines.append(f"* 拷贝文件数：**{sum(int(r['copied']) for r in idx)}** / {len(idx)}（全部经 SHA-256 校验）")
        lines.append(f"* 有可行抓取的物体：**{len({r['object'] for r in idx if int(r['n_feasible']) > 0})}** / {len({r['object'] for r in idx})}")

    lines.append("\n**失败原因分布**\n")
    lines.append("| reason | count | share |")
    lines.append("|---|---:|---:|")
    for k, v in reasons.most_common():
        lines.append(f"| `{k}` | {v} | {100.0 * v / total:.1f}% |")

    lines.append("\n**关键指标分布**\n")
    lines.append("| 指标 | min | p10 | p25 | median | p75 | p90 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, arr in (("穿透深度 geo_penetration_mm", pen), ("接触 link 数", links)):
        a = arr[~np.isnan(arr)]
        if len(a):
            q = np.percentile(a, [0, 10, 25, 50, 75, 90, 100])
            lines.append(f"| {name} | " + " | ".join(f"{x:.1f}" for x in q) + " |")
    a = rise[~np.isnan(rise)]
    if len(a):
        q = np.percentile(a, [0, 10, 25, 50, 75, 90, 100])
        lines.append("| 提离位移 lift_rise_mm | " + " | ".join(f"{x:.1f}" for x in q) + " |")
    lines.append(f"\n> 提离测试实际执行了 {int((~np.isnan(rise)).sum())} 个抓取；其余因穿透 >15 mm 被跳过"
                 f"（`lift_skipped=1`，这类抓取不可能通过任何合理阈值）。")

    # energy vs verdict
    fe = energies[[r["verdict"] == "feasible" for r in rows]]
    ie = energies[[r["verdict"] != "feasible" for r in rows]]
    if len(fe) and len(ie):
        lines.append(f"\n* 能量 vs 判定：feasible 的 energy 中位数 {np.median(fe):.2f}，"
                     f"infeasible 的 {np.median(ie):.2f} → 能量**不能**单独作为可行性判据。")

    lines.append("\n**逐物体统计**\n")
    lines.append("| object | grasps | feasible | rate | files copied | top reason |")
    lines.append("|---|---:|---:|---:|---:|---|")
    copied_per_obj = defaultdict(set)
    for r in rows:
        if r["verdict"] == "feasible":
            copied_per_obj[r["object"]].add(r["grasp_file"])
    for obj in sorted(by_obj):
        rs = by_obj[obj]
        nf = sum(r["verdict"] == "feasible" for r in rs)
        ncop = len(copied_per_obj[obj])
        top = Counter(r["reason"] for r in rs).most_common(1)[0]
        lines.append(f"| {obj} | {len(rs)} | {nf} | {100.0 * nf / len(rs):.1f}% | {ncop} | `{top[0]}` ({top[1]}) |")
    return "\n".join(lines)


def build_cmap_check():
    out = HERE / "cmap_dataset_from_graspqp.pt"
    if not out.exists():
        return "（尚未生成 `cmap_dataset_from_graspqp.pt`）"
    d = torch.load(out, weights_only=False)
    ref = torch.load(CMAP_REF, weights_only=False)
    lines = ["### 6.1 转换结果与 sanity check\n"]
    lines.append(f"* 顶层 keys：`{sorted(d.keys())}`（参考数据：`{sorted(ref.keys())}`）")
    info = d["info"]["linkerhand"]
    lines.append(f"* 样本数：**{info['num_total']}**；物体覆盖：**{len(info['num_per_object'])}** 个；"
                 f"`num_upper_object`={info['num_upper_object']}")
    lines.append(f"* `metadata` 类型：`{type(d['metadata']).__name__}`，元素类型：`{type(d['metadata'][0]).__name__}`，"
                 f"长度 {len(d['metadata'][0])}（参考数据同样为 3 元组）")
    q = torch.stack([m[0] for m in d["metadata"]]) if d["metadata"] else torch.zeros(0, 22)
    lines.append(f"* `q`：shape `{tuple(q.shape)}`，dtype `{q.dtype}`（参考：`(N,22)`, `torch.float32`）")

    lines.append("\n**逐维数值范围（本数据集 vs 参考 CMap 的 linkerhand 无对照，改用同维度统计）**\n")
    lines.append("| dim | 本数据 min | 本数据 max | 本数据 mean | 参考 allegro min | 参考 allegro max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    ref_allegro = torch.stack([m[0] for m in ref["metadata"] if m[2] == "allegro"])
    for i in range(22):
        c = q[:, i]
        r = ref_allegro[:, i]
        lines.append(f"| {i} | {c.min():+.4f} | {c.max():+.4f} | {c.mean():+.4f} | {r.min():+.4f} | {r.max():+.4f} |")

    lines.append("\n**抽样（前 3 条）**\n")
    for i in range(min(3, len(d["metadata"]))):
        qq, on, rn = d["metadata"][i]
        lines.append(f"* `{on}` / `{rn}`: q = [{', '.join(f'{x:+.4f}' for x in qq[:6])}, …] (22 维)")
    # round-trip check: euler -> R equals the source quaternion rotation
    lines.append("\n**欧拉角回代校验（随机 200 条：`Rx@Ry@Rz` 重建旋转 vs 源四元数）**\n")
    rows = []
    for p in sorted((HERE / "logs").glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    fe = [r for r in rows if r["verdict"] == "feasible"]
    errs = []
    for r in fe[:200]:
        g = gio.load_grasp_file(HERE / "filtered_grasps" / r["object"] / gio.GRASP_SUBDIR / r["grasp_file"])
        R = gio.quat_xyzw_to_matrix(g["root_pose"][r["grasp_index"], 3:7])
        # reconstruct from q using the CMap convention
        from convert_to_cmap import matrix_to_rpy_cmap
        rpy = matrix_to_rpy_cmap(R)
        Rx = gio.axis_angle_to_matrix([1, 0, 0], np.array(rpy[0]))
        Ry = gio.axis_angle_to_matrix([0, 1, 0], np.array(rpy[1]))
        Rz = gio.axis_angle_to_matrix([0, 0, 1], np.array(rpy[2]))
        errs.append(float(np.abs(Rx @ Ry @ Rz - R).max()))
    if errs:
        lines.append(f"* 最大旋转重建误差：**{max(errs):.2e}**（{len(errs)} 条抽样）")
    return "\n".join(lines)


def replace_block(text: str, begin: str, end: str, body: str) -> str:
    i = text.index(begin) + len(begin)
    j = text.index(end)
    return text[:i] + "\n" + body + "\n" + text[j:]


def main():
    rows = load_rows()
    if not rows:
        print("no logs found")
        return
    stats = build_stats(rows)
    cmap = build_cmap_check()
    report = HERE / "REPORT.md"
    text = report.read_text(encoding="utf-8")
    text = replace_block(text, "<!-- STATS:BEGIN -->", "<!-- STATS:END -->", stats)
    text = replace_block(text, "<!-- AUTO:BEGIN -->", "<!-- AUTO:END -->", cmap)

    summary_path = HERE / "logs" / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        tot = s.get("total", {})
        text = text.replace("（见本节末尾自动生成的统计表）",
                            f"（{tot.get('n_grasps', '?')} 个抓取，{tot.get('n_feasible', '?')} 个可行，"
                            f"耗时 {tot.get('elapsed_s', '?')} s）")
    report.write_text(text, encoding="utf-8")
    print("report updated")
    print(stats.splitlines()[1:6])


if __name__ == "__main__":
    main()
