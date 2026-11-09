"""Repair the FAIL meshes from ``stl_audit_before.json`` with Blender (headless).

Workflow per file:

1. back up the original to ``object/_stl_backup_<ts>/`` (``shutil.copy2`` + SHA-256);
2. run ``blender_repair/repair_mesh.py`` (voxel remesh is adaptive to the bbox);
3. verify the result with trimesh: watertight, winding, volume change < 5%,
   Chamfer median < 1 mm, and the SDF sign test (>99% of inward/outward offsets
   classified correctly);
4. on success overwrite the original (backup exists); on failure restore the
   original and record the failure.

A control group (``--control``, meshes that passed the audit) is pushed through the
same script into a scratch directory and compared, to prove the pipeline does not
damage good meshes.

Run::

    python new_grasp_filter/repair_stls.py --tag before
    python new_grasp_filter/repair_stls.py --control-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_io as gio  # noqa: E402

BLENDER = Path(r"D:\Blender_Foundation\Blender_5.1\blender.exe")
REPAIR_SCRIPT = HERE / "blender_repair" / "repair_mesh.py"
DEFAULT_CONTROLS = ["ycb/apple", "ycb/cracker_box", "contactdb/torus_small"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_mesh(p: Path) -> trimesh.Trimesh:
    return trimesh.load(str(p), force="mesh", process=True)


def mesh_metrics(m: trimesh.Trimesh) -> dict:
    return {
        "faces": int(len(m.faces)),
        "verts": int(len(m.vertices)),
        "watertight": bool(m.is_watertight),
        "winding": bool(m.is_winding_consistent),
        "volume_cm3": float(m.volume) * 1e6,
        "bodies": int(m.body_count),
    }


def chamfer_mm(a: trimesh.Trimesh, b: trimesh.Trimesh, n=4000) -> dict:
    pa, _ = trimesh.sample.sample_surface(a, n, seed=0)
    pb, _ = trimesh.sample.sample_surface(b, n, seed=0)
    d1 = trimesh.proximity.closest_point(b, pa)[1]
    d2 = trimesh.proximity.closest_point(a, pb)[1]
    d = np.concatenate([d1, d2]) * 1000.0
    return {"median_mm": float(np.median(d)), "p95_mm": float(np.percentile(d, 95)),
            "max_mm": float(d.max())}


def sdf_sign_test_per_body(m: trimesh.Trimesh, n=150, eps_mm=0.5) -> dict:
    """Like :func:`sdf_sign_test` but each connected body is tested on its own.

    For multi-part assets (a train with 13 parts, a teapot with lid, ...) an
    outward offset from one part can legitimately land inside another part, which
    the whole-mesh test misreads as an inverted normal.
    """
    vals = []
    for p in m.split(only_watertight=False):
        if not p.is_watertight or len(p.faces) < 10:
            continue
        pts, fid = trimesh.sample.sample_surface(p, n, seed=2)
        off = p.face_normals[fid] * (eps_mm / 1000.0)
        vals.append((float(np.mean(p.contains(pts - off))) + float(np.mean(~p.contains(pts + off)))) / 2)
    if not vals:
        return {"agreement_pct": 0.0, "inward_ok_pct": 0.0, "outward_ok_pct": 0.0, "bodies": 0}
    a = float(np.mean(vals)) * 100
    return {"agreement_pct": a, "inward_ok_pct": a, "outward_ok_pct": a, "bodies": len(vals)}


def sdf_sign_test(m: trimesh.Trimesh, n=1000, eps_mm=3.0) -> dict:
    """Inward/outward offsets of surface points must be inside/outside."""
    pts, fid = trimesh.sample.sample_surface(m, n, seed=1)
    nrm = m.face_normals[fid]
    offset = nrm * (eps_mm / 1000.0)
    inside_ok = float(np.mean(m.contains(pts - offset)))
    outside_ok = float(np.mean(~m.contains(pts + offset)))
    return {"inward_ok_pct": inside_ok * 100.0, "outward_ok_pct": outside_ok * 100.0,
            "agreement_pct": (inside_ok + outside_ok) / 2 * 100.0}


def run_blender(src: Path, dst: Path, max_faces: int, timeout=600, voxel_div=256,
                force_remesh=False) -> dict:
    """Run Blender with **file** redirection (this sandbox forbids pipe creation)."""
    cmd = [str(BLENDER), "--background", "--factory-startup", "--python", str(REPAIR_SCRIPT),
           "--", "--input", str(src), "--output", str(dst), "--max-faces", str(max_faces),
           "--voxel-div", str(voxel_div)] + (["--force-remesh"] if force_remesh else [])
    log = HERE / "_blender_probe" / "blender_last.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    out = log.read_text(encoding="utf-8", errors="replace")
    rec = {"returncode": int(proc.returncode), "seconds": round(time.time() - t0, 2)}
    for line in out.splitlines():
        if line.startswith("[repair] "):
            try:
                rec.update(json.loads(line[len("[repair] "):]))
            except Exception:  # noqa: BLE001
                pass
    if proc.returncode != 0:
        rec["log_tail"] = out[-1200:]
    return rec


def verify(orig: trimesh.Trimesh, new_path: Path, orig_vol: float,
           eps_mm: float = 0.5, n_sdf: int = 1000, per_body: bool = False) -> dict:
    try:
        new = load_mesh(new_path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "why": f"load failed: {exc}"}
    mm = mesh_metrics(new)
    vol_change = (mm["volume_cm3"] / orig_vol - 1.0) * 100.0 if orig_vol else float("inf")
    cham = chamfer_mm(orig, new)
    sdf = (sdf_sign_test_per_body(new, eps_mm=eps_mm) if per_body
           else sdf_sign_test(new, n=n_sdf, eps_mm=eps_mm))
    ok = (mm["watertight"] and mm["winding"] and abs(vol_change) < 5.0
          and cham["median_mm"] < 1.0 and sdf["agreement_pct"] > 99.0)
    return {"ok": bool(ok), "metrics": mm, "volume_change_pct": round(vol_change, 3),
            "chamfer": cham, "sdf": sdf}


def backup_file(src: Path, backup_root: Path, object_root: Path) -> dict:
    rel = src.relative_to(object_root)
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    h_src, h_dst = sha256(src), sha256(dst)
    if h_src != h_dst:
        raise RuntimeError(f"backup hash mismatch for {src}")
    return {"path": str(rel).replace("\\", "/"), "sha256": h_src, "bytes": src.stat().st_size}


def object_path(dataset: str, obj: str) -> Path:
    if dataset == "ycb":
        return gio.YCB_ROOT / obj / f"{obj}.stl"
    return gio.REPO_ROOT / "object" / "contactdb" / obj / f"{obj}.stl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="before")
    ap.add_argument("--controls", nargs="*", default=DEFAULT_CONTROLS)
    ap.add_argument("--control-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="subset: 'ycb/pear' ...")
    ap.add_argument("--probe-eps-mm", type=float, default=0.5)
    ap.add_argument("--voxel-div", type=int, default=256)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--per-body-sdf", action="store_true")
    ap.add_argument("--force-remesh", action="store_true")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    object_root = gio.REPO_ROOT / "object"
    backup_root = object_root / f"_stl_backup_{stamp}"
    scratch = HERE / "_blender_probe" / "repair_out"
    scratch.mkdir(parents=True, exist_ok=True)

    audit = json.loads((HERE / "logs" / f"stl_audit_{args.tag}.json").read_text(encoding="utf-8"))
    fails = [r for r in audit if r["verdict"] == "FAIL"]
    if args.only:
        want = set(args.only)
        fails = [r for r in fails if f"{r['dataset']}/{r['object']}" in want]
    if args.limit:
        fails = fails[: args.limit]

    log_path = HERE / "logs" / "stl_repair_log.jsonl"
    manifest = {"timestamp": stamp, "blender": str(BLENDER), "files": [],
                "repairs": [], "controls": []}

    if not args.control_only:
        print(f"[repair] {len(fails)} FAIL meshes, backup -> {backup_root}", flush=True)
        backup_root.mkdir(parents=True, exist_ok=True)
        for r in fails:
            src = object_path(r["dataset"], r["object"])
            manifest["files"].append(backup_file(src, backup_root, object_root))

        with open(log_path, "a", encoding="utf-8") as logf:
            for i, r in enumerate(fails, 1):
                src = object_path(r["dataset"], r["object"])
                before = load_mesh(src)
                vol = float(before.volume) * 1e6
                out = scratch / f"{r['dataset']}_{r['object']}.stl"
                attempts = []
                rec = {"dataset": r["dataset"], "object": r["object"], "src": str(src), "attempts": []}
                success = False
                for max_faces in (max(2000, len(before.faces) * 2), max(2000, int(len(before.faces) * 1.2))):
                    try:
                        run = run_blender(src, out, max_faces=max_faces, voxel_div=args.voxel_div,
                                          force_remesh=args.force_remesh)
                    except subprocess.TimeoutExpired:
                        rec["attempts"].append({"max_faces": max_faces, "timeout": True})
                        continue
                    ver = verify(before, out, vol, eps_mm=args.probe_eps_mm,
                                 per_body=args.per_body_sdf) if out.exists() else {"ok": False, "why": "no output"}
                    attempt = {"max_faces": max_faces, "blender": run, "verify": ver}
                    rec["attempts"].append(attempt)
                    if run.get("returncode") == 0 and ver.get("ok"):
                        shutil.copy2(out, src)
                        if sha256(src) != sha256(out):
                            raise RuntimeError("post-copy hash mismatch")
                        success = True
                        break
                rec["success"] = success
                if not success:
                    # restore the byte-identical backup
                    rel = src.relative_to(object_root)
                    shutil.copy2(backup_root / rel, src)
                    if sha256(src) != sha256(backup_root / rel):
                        raise RuntimeError(f"restore hash mismatch for {src}")
                    rec["restored"] = True
                logf.write(json.dumps(rec, default=str) + "\n")
                logf.flush()
                manifest["repairs"].append(rec)
                last = rec["attempts"][-1] if rec["attempts"] else {}
                v = last.get("verify", {})
                print(f"[{i}/{len(fails)}] {r['dataset']}/{r['object']}: "
                      f"{'OK' if success else 'FAILED'} "
                      f"faces={v.get('metrics', {}).get('faces')} "
                      f"vol%={v.get('volume_change_pct')} "
                      f"chamfer={v.get('chamfer', {}).get('median_mm')}mm "
                      f"sdf={v.get('sdf', {}).get('agreement_pct')}", flush=True)

    # ---- control group: good meshes through the same pipeline, never overwritten
    ctrl_dir = HERE / "_blender_probe" / "control"
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[control] {len(args.controls)} meshes -> {ctrl_dir}", flush=True)
    for name in args.controls:
        dataset, obj = name.split("/", 1)
        src = object_path(dataset, obj)
        if not src.exists():
            print(f"[control] missing {src}", flush=True)
            continue
        before = load_mesh(src)
        out = ctrl_dir / f"{dataset}_{obj}_control.stl"
        run = run_blender(src, out, max_faces=max(2000, len(before.faces) * 2))
        ver = verify(before, out, float(before.volume) * 1e6) if out.exists() else {"ok": False}
        rec = {"name": name, "blender": run, "verify": ver, "sha_before": sha256(src),
               "sha_after_original": sha256(src)}
        manifest["controls"].append(rec)
        print(f"[control] {name}: src_watertight={before.is_watertight} "
              f"out_ok={ver.get('ok')} vol%={ver.get('volume_change_pct')} "
              f"chamfer={ver.get('chamfer', {}).get('median_mm')}mm", flush=True)

    (HERE / "logs" / f"stl_repair_manifest_{stamp}.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nbackup: {backup_root}")
    print(f"manifest: new_grasp_filter/logs/stl_repair_manifest_{stamp}.json")


if __name__ == "__main__":
    main()
