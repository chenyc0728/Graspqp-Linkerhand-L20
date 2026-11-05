"""CPU vs GPU benchmarks for the v2 pipeline (FK, pose decode, SDF query, SAPIEN).

Runs three independent micro-benchmarks and prints a markdown table + JSON:

* ``fk``     -- ``pytorch_kinematics`` forward kinematics for the L20, CPU vs CUDA;
* ``quat``   -- ``roma`` xyzw quaternion -> rotation matrix, CPU vs CUDA;
* ``sdf``    -- object voxel-SDF query (torch trilinear implementation), CPU vs CUDA;
* ``sapien`` -- reports whether the PhysX GPU backend could be used at all in this
  environment (the full screening comparison lives in REPORT_v2 §4).

Run::

    python new_grasp_filter/benchmark_cpu_gpu.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_io as gio  # noqa: E402


def timeit(fn, repeat=5, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench_fk():
    urdf = gio.UrdfModel(gio.HAND_URDF)
    rows = []
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        chain = urdf.pk_chain_on(device)
        for n in [1, 8, 64, 512, 4096]:
            q = torch.rand(n, 21, dtype=torch.float64, device=device) * 0.5
            f = lambda: {k: v.get_matrix() for k, v in chain.forward_kinematics(q).items()}
            dt = timeit(f, repeat=3)
            rows.append({"device": device, "batch": n, "ms": round(dt * 1000, 3),
                         "ms_per_sample": round(dt * 1000 / n, 4)})
    return rows


def bench_quat():
    import roma

    rows = []
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        for n in [1, 64, 4096, 100000]:
            q = torch.randn(n, 4, dtype=torch.float64, device=device)
            q = q / q.norm(dim=-1, keepdim=True)
            f = lambda: roma.unitquat_to_rotmat(q)
            dt = timeit(f, repeat=3)
            rows.append({"device": device, "batch": n, "ms": round(dt * 1000, 3),
                         "ms_per_1k": round(dt * 1000 / max(n, 1) * 1000, 4)})
    return rows


def _sdf_query_torch(sdf, points, device):
    t = torch.as_tensor(points, dtype=torch.float64, device=device)
    h = torch.cat([t, torch.ones(len(t), 1, dtype=torch.float64, device=device)], dim=1)
    Tr = torch.as_tensor(sdf.T_inv, dtype=torch.float64, device=device)
    f = (Tr @ h.T).T[:, :3]
    shape = torch.tensor(sdf.sdf.shape, device=device, dtype=torch.float64)
    f = torch.maximum(torch.minimum(f, shape - 1.0001), torch.zeros(3, device=device, dtype=torch.float64))
    i0 = torch.floor(f).long()
    frac = f - i0
    vol = torch.as_tensor(sdf.sdf, dtype=torch.float64, device=device)
    out = torch.zeros(len(t), dtype=torch.float64, device=device)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((frac[:, 0] if dx else 1 - frac[:, 0])
                     * (frac[:, 1] if dy else 1 - frac[:, 1])
                     * (frac[:, 2] if dz else 1 - frac[:, 2]))
                out = out + w * vol[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return out


def bench_sdf(n_points_list=(1000, 100000, 1000000)):
    from object_sdf import VoxelSDF

    sdf = VoxelSDF(gio.object_mesh_path("apple"), pitch=0.002)
    rng = np.random.default_rng(0)
    rows = []
    for n in n_points_list:
        pts = rng.uniform(-0.05, 0.05, (n, 3))
        for device in ["cpu", "numpy", "cuda"]:
            if device == "cuda" and not torch.cuda.is_available():
                continue
            if device == "numpy":
                f = lambda: sdf.query(pts)
            else:
                f = lambda: _sdf_query_torch(sdf, pts, device)
                _sdf_query_torch(sdf, pts[:10], device)  # warm up CUDA
            dt = timeit(f, repeat=3)
            rows.append({"device": device, "points": n, "ms": round(dt * 1000, 3),
                         "ms_per_100k": round(dt * 1000 / n * 100000, 3)})
    return rows


def bench_sapien(use_gpu_env: bool = True):
    """Try the PhysX GPU backend and report exactly where it stops."""
    import sapien

    out = {"torch_cuda": torch.cuda.is_available(), "sapien": sapien.__version__,
           "physx_gpu_enabled": False, "gpu_scene_ok": False, "steps_per_s_cpu": None,
           "steps_per_s_gpu": None, "error": None}
    sapien.set_log_level("error")

    def _scene(system=None):
        return sapien.Scene([system]) if system is not None else sapien.Scene()

    def _steps_per_s(scene, name):
        b = scene.create_actor_builder()
        b.add_box_collision(half_size=[0.05] * 3)
        b.set_initial_pose(sapien.Pose([0, 0, 1]))
        e = b.build()
        scene.set_timestep(1 / 240)
        t0 = time.perf_counter()
        for _ in range(200):
            scene.step()
        return 200 / (time.perf_counter() - t0)

    # PhysX GPU must be enabled before *any* scene/PhysX object is created, so the
    # GPU attempt comes first; a failure here leaves the CPU path untouched.
    if use_gpu_env:
        try:
            import ctypes

            alias = HERE.parent / "_sapien_home" / "cuda.dll"
            if alias.exists():
                ctypes.CDLL(str(alias), ctypes.RTLD_GLOBAL)
            if not sapien.physx.is_gpu_enabled():
                sapien.physx.enable_gpu()
            out["physx_gpu_enabled"] = bool(sapien.physx.is_gpu_enabled())
            gpu = sapien.physx.PhysxGpuSystem()
            out["gpu_system_constructed"] = type(gpu).__name__
            scene = _scene(gpu)
            out["gpu_scene_system"] = type(scene.physx_system).__name__
            out["steps_per_s_gpu"] = round(_steps_per_s(scene, "gpu"), 1)
            out["gpu_scene_ok"] = True
        except Exception as exc:  # noqa: BLE001
            prefix = "gpu: "
            out["error"] = (out["error"] + "; " if out["error"] else "") + f"{prefix}{type(exc).__name__}: {exc}"
            out["gpu_scene_ok"] = False

    try:
        scene = _scene()
        out["steps_per_s_cpu"] = round(_steps_per_s(scene, "cpu"), 1)
    except Exception as exc:  # noqa: BLE001
        out["error"] = (out["error"] + "; " if out["error"] else "") + f"cpu: {type(exc).__name__}: {exc}"
    return out


def main():
    result = {"sapien": bench_sapien()}
    if torch.cuda.is_available():
        result["fk"] = bench_fk()
        result["quat"] = bench_quat()
        result["sdf"] = bench_sdf()
    else:
        result["note"] = "CUDA not available; only CPU numbers produced"
    out = HERE / "logs" / "benchmark_cpu_gpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
