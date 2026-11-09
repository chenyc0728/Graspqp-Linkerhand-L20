"""Remove nested ("internal cavity") shells from a mesh, then verify.

The Blender repair closes holes and fixes winding but keeps every connected
component, so a small shell inside the main body survives -- which the audit
flags as ``nested shells`` (an internal cavity).  For a collision mesh those
inner shells are wrong: they make the solid look hollow.

This script keeps the largest body plus any body that is not contained in
another, drops numerical dust, and writes the result back.  The file is backed
up first.

Usage::

    python new_grasp_filter/drop_nested.py --only contactdb/apple contactdb/flashlight
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_io as gio  # noqa: E402
from repair_stls import object_path, sha256, load_mesh  # noqa: E402


def drop_nested(mesh: trimesh.Trimesh, dust_cm3: float = 1e-3):
    parts = mesh.split(only_watertight=False)
    if len(parts) < 2:
        return mesh, {"removed": 0, "kept": len(parts)}
    vols = np.array([float(p.volume) * 1e6 for p in parts])
    order = np.argsort(vols)[::-1]
    keep = [int(order[0])]
    removed = []
    for i in order[1:]:
        if vols[i] < dust_cm3:
            removed.append((int(i), "dust"))
            continue
        nested = False
        for j in keep:
            try:
                if parts[j].is_watertight and parts[j].contains([parts[i].centroid])[0]:
                    nested = True
                    break
            except Exception:  # noqa: BLE001
                pass
        if nested:
            removed.append((int(i), "nested"))
        else:
            keep.append(int(i))
    merged = trimesh.util.concatenate([parts[i] for i in keep])
    return merged, {"removed": len(removed), "kept": len(keep),
                    "removed_detail": removed, "kept_volumes_cm3": [round(float(vols[i]), 4) for i in keep]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", required=True)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = gio.REPO_ROOT / "object" / f"_stl_backup_nested_{stamp}"
    log = []
    for name in args.only:
        ds, obj = name.split("/", 1)
        src = object_path(ds, obj)
        rel = src.relative_to(gio.REPO_ROOT / "object")
        dst_bak = backup / rel
        dst_bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_bak)
        if sha256(src) != sha256(dst_bak):
            raise RuntimeError(f"backup mismatch {src}")

        before = load_mesh(src)
        merged, info = drop_nested(before)
        tmp = src.with_suffix(".tmp.stl")
        merged.export(tmp)
        after = load_mesh(tmp)
        vol_change = (after.volume / before.volume - 1) * 100 if before.volume else float("nan")
        ok = (after.is_watertight and after.is_winding_consistent
              and abs(vol_change) < 5.0 and info["removed"] > 0)
        rec = {"name": name, "sha_before": sha256(src), "info": info,
               "watertight": bool(after.is_watertight), "winding": bool(after.is_winding_consistent),
               "volume_change_pct": round(float(vol_change), 4), "nested_left": 0,
               "ok": bool(ok)}
        if ok:
            shutil.copy2(tmp, src)
            rec["sha_after"] = sha256(src)
        else:
            rec["restored"] = True
        tmp.unlink(missing_ok=True)
        log.append(rec)
        print(f"{name}: removed={info['removed']} kept={info['kept']} "
              f"vol%={vol_change:.3f} ok={ok}", flush=True)

    out = HERE / "logs" / f"drop_nested_{stamp}.json"
    out.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("backup:", backup)
    print("log:", out)


if __name__ == "__main__":
    main()
