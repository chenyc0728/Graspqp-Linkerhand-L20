"""Feasible-set consistency: v1 numpy maths vs the new roma/pytorch_kinematics backend.

Runs the *same* grasps through ``SapienGraspChecker`` three times:

1. ``numpy``  -- ``gio.set_backend("numpy")`` (v1 behaviour);
2. ``torch``  -- ``gio.set_backend("torch")`` (default);
3. ``torch2`` -- the torch backend again, to measure PhysX run-to-run noise.

Reports per-grasp verdict agreement and metric deltas.  Note that the physics is
contact-rich and chaotic: identical code re-run can change ``lift_rise_mm`` by
centimetres, so verdict agreement (not raw metric equality) is the criterion.

Run::

    python new_grasp_filter/compare_backends.py --object apple --files 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_io as gio  # noqa: E402


def run_pass(objects, files_per_object, limit, backend, device="cpu"):
    from sapien_grasp_checker import CheckerConfig, SapienGraspChecker

    gio.set_backend(backend)
    if backend == "torch":
        gio.set_device(device)
    out = []
    cfg = CheckerConfig()
    for obj in objects:
        checker = SapienGraspChecker(obj, cfg)
        for f in gio.list_grasp_files(obj)[:files_per_object]:
            g = gio.load_grasp_file(f)
            n = len(g["values"]) if limit is None else min(limit, len(g["values"]))
            for i in range(n):
                res = checker.check(g["root_pose"][i], g["joints"][i])
                out.append({
                    "object": obj, "file": f.name, "index": i,
                    "feasible": bool(res.feasible), "reason": res.reason,
                    "pen_mm": res.metrics.get("geo_penetration_mm"),
                    "gap_mm": res.metrics.get("geo_closest_gap_mm"),
                    "rise_mm": res.metrics.get("lift_rise_mm"),
                    "links": res.metrics.get("geo_n_contact_links"),
                })
    return out


def compare(a, b, label):
    assert len(a) == len(b)
    same_verdict = sum(x["feasible"] == y["feasible"] for x, y in zip(a, b))
    same_reason = sum(x["reason"] == y["reason"] for x, y in zip(a, b))
    flips = [(x, y) for x, y in zip(a, b) if x["feasible"] != y["feasible"]]
    dp = [abs((x["pen_mm"] or 0) - (y["pen_mm"] or 0)) for x, y in zip(a, b)]
    dg = [abs((x["gap_mm"] or 0) - (y["gap_mm"] or 0)) for x, y in zip(a, b)]
    dr = [abs((x["rise_mm"] or 0) - (y["rise_mm"] or 0)) for x, y in zip(a, b)
          if np.isfinite(x["rise_mm"] or np.nan) and np.isfinite(y["rise_mm"] or np.nan)]
    return {
        "label": label,
        "n": len(a),
        "n_feasible_a": sum(x["feasible"] for x in a),
        "n_feasible_b": sum(x["feasible"] for x in b),
        "verdict_agreement": f"{same_verdict}/{len(a)}",
        "reason_agreement": f"{same_reason}/{len(a)}",
        "n_feasible_flips": len(flips),
        "flips": [{"a": x, "b": y} for x, y in flips],
        "max_abs_delta_pen_mm": round(float(max(dp)), 6) if dp else None,
        "max_abs_delta_gap_mm": round(float(max(dg)), 6) if dg else None,
        "max_abs_delta_lift_mm": round(float(max(dr)), 3) if dr else None,
        "median_abs_delta_lift_mm": round(float(np.median(dr)), 3) if dr else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="apple")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--files", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "logs" / "compare_backends.json"))
    args = ap.parse_args()
    objects = args.objects or [args.object]

    print(f"running numpy pass on {objects} ...", flush=True)
    a = run_pass(objects, args.files, args.limit, "numpy")
    print(f"running torch pass ...", flush=True)
    b = run_pass(objects, args.files, args.limit, "torch")
    print(f"running torch pass #2 (determinism) ...", flush=True)
    c = run_pass(objects, args.files, args.limit, "torch")

    result = {
        "objects": objects,
        "files_per_object": args.files,
        "numpy_vs_torch": compare(a, b, "numpy vs torch"),
        "torch_vs_torch": compare(b, c, "torch vs torch (same code re-run)"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for k in ("numpy_vs_torch", "torch_vs_torch"):
        r = result[k]
        print(f"\n[{r['label']}] n={r['n']} feasible {r['n_feasible_a']} vs {r['n_feasible_b']}"
              f"  verdict {r['verdict_agreement']}  reason {r['reason_agreement']}"
              f"  flips={r['n_feasible_flips']}")
        print(f"   max|dpen|={r['max_abs_delta_pen_mm']}mm max|dgap|={r['max_abs_delta_gap_mm']}mm"
              f" max|dlift|={r['max_abs_delta_lift_mm']}mm median|dlift|={r['median_abs_delta_lift_mm']}mm")
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
