"""Rebuild a missing/broken L20 collision mesh from its visual mesh.

The existing collision meshes are **slightly inflated convex hulls** of the visual
links (measured: hull->collision Chamfer median ≈ 0.8 mm, collision volume ≈ 1.05×
the hull), so the rebuild follows the same convention:

1. convex hull of the visual mesh;
2. uniform inflation by ``--margin-mm`` (default 0.8) about the centroid;
3. export to ``meshes/collisions/<link>.STL`` (backup first, if a file exists);
4. verify watertight / winding / volume / hull agreement.

Run::

    python new_grasp_filter/rebuild_collision_mesh.py --links thumb_metacarpals
"""

from __future__ import annotations

import argparse
import hashlib
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


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inflate(mesh: trimesh.Trimesh, margin_mm: float) -> trimesh.Trimesh:
    """Push every vertex outward along its direction from the centroid by margin."""
    m = mesh.copy()
    c = m.centroid
    v = m.vertices - c
    r = np.linalg.norm(v, axis=1, keepdims=True)
    r[r < 1e-9] = 1e-9
    m.vertices = m.vertices + v / r * (margin_mm / 1000.0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", nargs="+", required=True)
    ap.add_argument("--margin-mm", type=float, default=0.8)
    ap.add_argument("--mesh-root", default=str(gio.HAND_URDF.parent / "meshes"))
    args = ap.parse_args()

    root = Path(args.mesh_root).resolve()
    coll_dir = root / "collisions"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = gio.HAND_URDF.parent / f"collisions_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": stamp, "margin_mm": args.margin_mm, "backup": str(backup), "links": []}

    for link in args.links:
        vis_path = root / f"{link}.STL"
        out_path = coll_dir / f"{link}.STL"
        rec = {"link": link, "visual": str(vis_path.relative_to(gio.REPO_ROOT)),
               "output": str(out_path.relative_to(gio.REPO_ROOT))}
        if out_path.exists():
            shutil.copy2(out_path, backup / out_path.name)
            rec["backup"] = str((backup / out_path.name).relative_to(gio.REPO_ROOT))
            rec["backup_sha256"] = sha256(backup / out_path.name)
        else:
            rec["backup"] = None
            rec["note"] = "file was missing; nothing to back up"

        vis = trimesh.load(str(vis_path), force="mesh", process=True)
        hull = vis.convex_hull
        out = inflate(hull, args.margin_mm)
        out.export(out_path)
        got = trimesh.load(str(out_path), force="mesh", process=True)
        ch = trimesh.proximity.closest_point(got, trimesh.sample.sample_surface(hull, 2000, seed=0)[0])[1]
        rec.update({
            "visual_faces": int(len(vis.faces)),
            "faces": int(len(got.faces)),
            "watertight": bool(got.is_watertight),
            "winding": bool(got.is_winding_consistent),
            "volume_cm3": round(float(got.volume) * 1e6, 3),
            "visual_volume_cm3": round(float(vis.volume) * 1e6, 3),
            "volume_ratio": round(float(got.volume) / float(vis.volume), 3),
            "hull_to_out_chamfer_median_mm": round(float(np.median(ch)) * 1000, 3),
            "hull_to_out_chamfer_p95_mm": round(float(np.percentile(ch, 95)) * 1000, 3),
            "sha256": sha256(out_path),
            "ok": bool(got.is_watertight and got.is_winding_consistent and got.volume > 0),
        })
        report["links"].append(rec)
        print(f"[rebuild] {link}: ok={rec['ok']} faces={rec['faces']} "
              f"vol={rec['volume_cm3']}cm3 ratio={rec['volume_ratio']}x "
              f"hull_chamfer={rec['hull_to_out_chamfer_median_mm']}mm", flush=True)

    out_json = HERE / "logs" / f"collision_rebuild_{stamp}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("backup:", backup)
    print("log:", out_json)


if __name__ == "__main__":
    main()
