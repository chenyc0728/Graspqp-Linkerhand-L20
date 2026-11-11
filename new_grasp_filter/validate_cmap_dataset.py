"""Validate ``cmap_dataset_from_graspqp_v2.pt`` *through* ``CMapDataset.py``.

Two things make this non-trivial and are handled here without touching CMap's
files:

1. ``CMapDataset.py`` imports ``utils.hand_model`` which **does not exist** in
   this workspace (only ``Cmap_dataset/dataset`` + ``data_urdf`` were shipped).
   We inject a minimal ``utils.hand_model`` shim (backed by ``pytorch_kinematics``)
   that also implements the v2 identity mapping
   ``linkerhand -> left extended URDF``, ``linkerhand_right -> right extended``.
2. The module resolves its data through ``ROOT_DIR/data/...`` while the files
   live directly under ``Cmap_dataset/``.  A scoped ``os.path`` shim redirects
   those joins (and can point ``cmap_dataset.pt`` at our v2 file).

Checks:

* dataset construction ``len()`` / metadata filtering / ``__getitem__`` (validate
  mode) for our left-hand dataset and for the reference CMap dataset;
* ``robot_name`` alias behaviour (``linkerhand`` -> ``linkerhand_right``);
* URDF resolution for both identities;
* FK round-trip: ``q`` -> left extended URDF -> hand-base/fingertip poses vs the
  original GraspQP pose.

Run::

    python new_grasp_filter/validate_cmap_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_io as gio  # noqa: E402

CMAP_ROOT = gio.CMAP_ROOT
CMAP_DATASET_DIR = CMAP_ROOT / "dataset"

import itertools  # noqa: E402

_MODULE_COUNTER = itertools.count()

# --------------------------------------------------------------------------
# 1. minimal utils.hand_model shim (CMapDataset imports it)
# --------------------------------------------------------------------------
ROBOT_URDF = {
    "linkerhand": gio.cmap_left_urdf(repaired=True),      # LEFT (v2)
    "linkerhand_right": gio.CMAP_RIGHT_URDF,              # historical CMap right
    "linkerhand_right_raw": gio.CMAP_LEFT_URDF,           # broken original, for the report
}


def _urdf_from_meta(robot_name: str) -> Path:
    if robot_name in ROBOT_URDF:
        return ROBOT_URDF[robot_name]
    meta = json.loads(gio.CMAP_ASSETS_META.read_text(encoding="utf-8"))
    rel = meta["urdf_path"][robot_name]
    if rel.startswith("data/"):
        rel = rel[len("data/"):]
    return gio.CMAP_ROOT / rel


class _HandShim:
    """Only the surface used by ``CMapDataset`` in validate/``is_train`` paths."""

    def __init__(self, robot_name: str, device="cpu", num_points: int = 128):
        self.robot_name = robot_name
        self.urdf_path = _urdf_from_meta(robot_name)
        self.chain = gio.UrdfModel(self.urdf_path).pk_chain_on("cpu")
        self.pk_chain = self.chain
        self.dof = len(self.chain.get_joint_parameter_names())
        self.device = device
        # one small point cloud per link, in the link frame (identity local pose)
        link_meshes = gio.hand_link_meshes(self.urdf_path)
        self.links_pc = []
        self.link_names = []
        rng = np.random.default_rng(0)
        for name, path in link_meshes.items():
            import trimesh

            m = trimesh.load(path, force="mesh")
            idx = rng.choice(len(m.vertices), size=min(num_points, len(m.vertices)), replace=False)
            self.links_pc.append(torch.tensor(m.vertices[idx], dtype=torch.float32))
            self.link_names.append(name)

    def get_initial_q(self):
        return torch.zeros(self.dof, dtype=torch.float32)

    def get_transformed_links_pc(self, q=None):
        if q is None:
            q = self.get_initial_q()
        q = torch.as_tensor(q, dtype=torch.float64).reshape(1, -1)
        fk = self.chain.forward_kinematics(q)
        mats, pts = [], []
        for name, pc in zip(self.link_names, self.links_pc):
            T = fk[name].get_matrix()[0]
            mats.append(T.float())
            pts.append(pc.double() @ T[:3, :3].T + T[:3, 3])
        # CMapDataset stacks this directly -> must be (n_links, 4, 4), not a list
        return torch.cat(pts, dim=0), torch.stack(mats, dim=0)


def inject_hand_model_shim():
    utils = types.ModuleType("utils")
    hm = types.ModuleType("utils.hand_model")
    hm.create_hand_model = lambda robot_name, device="cpu", **kw: _HandShim(robot_name, device)
    utils.hand_model = hm
    sys.modules["utils"] = utils
    sys.modules["utils.hand_model"] = hm


# --------------------------------------------------------------------------
# 2. scoped path shim for ROOT_DIR/data/... and dataset redirection
# --------------------------------------------------------------------------
class _PathShim:
    """Redirect `ROOT_DIR/data/...` joins to the real layout (and swap dataset files).

    Paths are normalised with ``os.path.normpath`` before matching: CMapDataset
    builds ``os.path.join(ROOT_DIR, 'data/CMapDataset_filtered/...')`` which yields
    mixed separators on Windows.
    """

    def __init__(self, prefix_map, file_map=None):
        self.prefix_map = {os.path.normpath(k): v for k, v in prefix_map.items()}
        self.file_map = {os.path.normpath(k): v for k, v in (file_map or {}).items()}

    def join(self, *parts):
        p = os.path.normpath(os.path.join(*parts))
        if p in self.file_map:
            return self.file_map[p]
        for a, b in self.prefix_map.items():
            if p == a or p.startswith(a + os.sep):
                return os.path.normpath(b + p[len(a):])
        return p

    def __getattr__(self, item):
        return getattr(os.path, item)


def load_cmap_module(dataset_override: Path | None = None):
    """Load a **fresh** ``CMapDataset`` module object with scoped paths.

    A fresh module object per call matters: ``mod.os`` is set per instance, and
    ``importlib.reload`` would mutate the module already handed out (so a later
    ``load_cmap_module(dataset_override=...)`` used to silently re-point the
    *reference* dataset checks at our file).
    """
    inject_hand_model_shim()
    import importlib.util
    import itertools

    idx = next(_MODULE_COUNTER)
    path = CMAP_DATASET_DIR / "CMapDataset.py"
    spec = importlib.util.spec_from_file_location(f"_cmap_dataset_{idx}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prefix = {str(CMAP_ROOT / "data"): str(CMAP_ROOT)}
    files = {}
    if dataset_override is not None:
        files[os.path.normpath(str(CMAP_ROOT / "data/CMapDataset_filtered/cmap_dataset.pt"))] = str(dataset_override)
    mod.os = types.SimpleNamespace(path=_PathShim(prefix, files))
    return mod


# --------------------------------------------------------------------------
# 3. checks
# --------------------------------------------------------------------------
def check_dataset(mod, path: Path, robot_names, objects, is_train, label, n_getitem=0):
    out = {"label": label, "path": str(path), "robot_names": robot_names}
    ds = mod.CMapDataset(
        batch_size=1, robot_names=robot_names, is_train=is_train,
        debug_object_names=objects,
        # validate mode reads the fixed 512-point CMap PointCloud files; the class
        # requires num_points == 512 there (its own assertion / tensor shape).
        num_points=512 if not is_train else 64,
        object_pc_type="fixed" if not is_train else "random",
    )
    out["n_metadata"] = len(ds.metadata)
    out["len"] = len(ds)
    out["robots_in_metadata"] = sorted({m[2] for m in ds.metadata})
    out["objects_in_metadata"] = len({m[1] for m in ds.metadata})
    if ds.metadata:
        q = ds.metadata[0][0]
        out["q_shape"] = list(q.shape)
        out["q_dtype"] = str(q.dtype)
    if len(ds) and n_getitem:
        try:
            item = ds[0]
            out["getitem_ok"] = True
            out["getitem_robot"] = item["robot_name"]
            out["getitem_object"] = item["object_name"]
            out["getitem_initial_q"] = list(item["initial_q"].shape)
            out["getitem_object_pc"] = list(item["object_pc"].shape)
        except Exception as exc:  # noqa: BLE001
            out["getitem_ok"] = False
            out["getitem_error"] = f"{type(exc).__name__}: {exc}"
    return out


def check_alias(mod):
    aliases = getattr(mod, "ROBOT_ALIASES", {})
    return {"ROBOT_ALIASES": aliases, "applied_in_memory": aliases.get("linkerhand") == "linkerhand_right",
            "note": "alias is skipped when the file declares that robot in `info` (left-hand v2 data)"}


def check_urdf_resolution():
    out = {}
    for name in ["linkerhand", "linkerhand_right"]:
        try:
            h = _HandShim(name)
            out[name] = {"urdf": str(h.urdf_path.relative_to(gio.REPO_ROOT)), "dof": h.dof,
                         "pk_ok": True}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"urdf": str(_urdf_from_meta(name)), "pk_ok": False, "error": str(exc)}
    return out


def check_alias_behaviour(tmp_dir: Path):
    """Synthetic test of the alias rule (no such samples exist in the real files).

    * legacy payload (``info`` has no ``linkerhand``)  -> sample is renamed to
      ``linkerhand_right`` and selected with that robot name;
    * left-hand payload (``info`` declares ``linkerhand``) -> stays ``linkerhand``.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    q = torch.zeros(22, dtype=torch.float32)
    out = {}
    for tag, info, robots, expect in [
        ("legacy", {"allegro": {"robot_name": "allegro", "num_total": 0}},
         ["linkerhand_right"], "linkerhand_right"),
        ("left", {"linkerhand": {"robot_name": "linkerhand", "num_total": 0}},
         ["linkerhand"], "linkerhand"),
    ]:
        f = tmp_dir / f"alias_{tag}.pt"
        torch.save({"info": info, "metadata": [(q, "ycb+apple", "linkerhand")]}, f)
        mod = load_cmap_module(dataset_override=f)
        ds = mod.CMapDataset(batch_size=1, robot_names=robots, is_train=True,
                             debug_object_names=["ycb+apple"], num_points=32,
                             object_pc_type="random")
        out[tag] = {
            "n_metadata": len(ds.metadata),
            "robot_seen": ds.metadata[0][2] if ds.metadata else None,
            "expected": expect,
            "ok": bool(ds.metadata) and ds.metadata[0][2] == expect,
        }
    return out


def check_q_roundtrip(dataset_path: Path, n: int = 20, index_csvs=None, prefixes=None):
    """q (CMap) -> left extended URDF -> hand base pose vs the original GraspQP pose.

    The metadata order is reproduced exactly as ``convert_to_cmap.convert`` builds
    it: the inputs (index CSVs) in the given order, rows sorted by
    ``source/object/file``, feasible indices sorted, deduped by
    ``(sha256, index)`` -- so every q is paired with its source grasp.  Pass
    ``--index`` / ``--prefixes`` when validating a merged (multi-input) dataset;
    the default is the single contactdb index of the old v2 era.
    """
    import convert_to_cmap as c2c

    left = gio.UrdfModel(gio.cmap_left_urdf(repaired=True))
    chain = left.pk_chain_on("cpu")
    names = chain.get_joint_parameter_names()
    if index_csvs:
        paths = [Path(p).resolve() for p in index_csvs]
    else:
        paths = [HERE / "filtered_grasps" / "index.csv"]
    prefs = list(prefixes or [])
    if prefs and len(prefs) != len(paths):
        raise SystemExit(f"--prefixes has {len(prefs)} entries but --index has {len(paths)}")
    specs = [(p, prefs[i] if prefs else "ycb") for i, p in enumerate(paths)]
    pairs = list(c2c.iter_pairs(specs, False, True))

    data = torch.load(dataset_path, weights_only=False)
    md = data["metadata"]
    assert len(md) == len(pairs), f"metadata {len(md)} != expected {len(pairs)}"
    rng = np.random.default_rng(0)
    picks = rng.choice(len(md), size=min(n, len(md)), replace=False) if md else []
    worst_rot, worst_trans = 0.0, 0.0
    for k in picks:
        q, obj_name, _ = md[int(k)]
        row, i, path, run_name, _ = pairs[int(k)]
        g = gio.load_grasp_file(path)
        R0 = gio.quat_xyzw_to_matrix(g["root_pose"][i, 3:7])
        t0 = g["root_pose"][i, :3] + R0 @ gio.ROOT_TO_HAND_BASE
        qv = q.numpy().astype(np.float64)
        full21 = gio.build_full_angles(qv[6:22][None, :])[0]
        vec = np.zeros(len(names))
        for j, name in enumerate(names):
            if name in ("virtual_joint_x", "virtual_joint_y", "virtual_joint_z"):
                vec[j] = qv[("virtual_joint_x", "virtual_joint_y", "virtual_joint_z").index(name)]
            elif name in ("virtual_joint_roll", "virtual_joint_pitch", "virtual_joint_yaw"):
                vec[j] = qv[3 + ("virtual_joint_roll", "virtual_joint_pitch", "virtual_joint_yaw").index(name)]
            elif name in gio.CHAIN_JOINT_NAMES:
                vec[j] = full21[gio.CHAIN_JOINT_NAMES.index(name)]
        T = chain.forward_kinematics(torch.tensor(vec[None, :], dtype=torch.float64))["hand_base_link"].get_matrix()[0].numpy()
        worst_rot = max(worst_rot, float(np.abs(T[:3, :3] - R0).max()))
        worst_trans = max(worst_trans, float(np.abs(T[:3, 3] - t0).max()))
    return {"n_checked": len(picks), "max_abs_rotation_error": worst_rot,
            "max_abs_translation_error_m": worst_trans,
            "index_inputs": [{"index_csv": str(p), "prefix": pre} for p, pre in specs],
            "fk_backend": "pytorch_kinematics",
            "note": ("this env's pytorch_kinematics fork (renezurbruegg @92c3b97) adds "
                     "+/-1e-2 rad per revolute joint in Frame.get_transform, so the "
                     "rotation error is ~1e-2 here; the numpy FK gives ~2e-7 "
                     "(see verify_cmap_v3.py).")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "cmap_dataset_from_graspqp_v2.pt"))
    ap.add_argument("--index", nargs="+", default=None,
                    help="index CSV(s) used to build --dataset, in the same order as the "
                         "conversion (e.g. index_ycb.csv index.csv); default: filtered_grasps/index.csv")
    ap.add_argument("--prefixes", nargs="+", default=None,
                    help="object_name prefix per --index (e.g. ycb contactdb)")
    ap.add_argument("--out", default=str(HERE / "logs" / "validate_cmap_dataset.json"))
    args = ap.parse_args()
    dataset = Path(args.dataset).resolve()

    report = {"dataset": str(dataset)}
    # reference CMap dataset (default loader, allegro only to keep it fast)
    ref_mod = load_cmap_module()
    report["alias"] = check_alias(ref_mod)
    report["urdf_resolution"] = check_urdf_resolution()
    report["reference_dataset"] = check_dataset(
        ref_mod, CMAP_ROOT / "CMapDataset_filtered/cmap_dataset.pt",
        ["allegro"], ["ycb+lemon", "ycb+baseball", "contactdb+apple"], is_train=True,
        label="reference cmap_dataset.pt (allegro)",
    )
    # our dataset (v2 ycb-only or v3 ycb+contactdb) through the same class
    our_mod = load_cmap_module(dataset_override=dataset)
    data = torch.load(dataset, weights_only=False)
    objects = sorted(data["info"]["linkerhand"]["num_per_object"].keys())
    report["v2_dataset"] = check_dataset(
        our_mod, dataset, ["linkerhand"], objects, is_train=False,
        label=f"{dataset.name} (left linkerhand)", n_getitem=1,
    )
    report["v2_dataset"]["n_objects_arg"] = len(objects)
    report["v2_dataset"]["object_name_prefixes"] = sorted({o.split("+")[0] for o in objects})
    report["v2_dataset"]["object_name_example"] = objects[:3]
    # alias behaviour on the reference file: right-hand identity must be selectable
    report["reference_linkerhand_right"] = check_dataset(
        ref_mod, CMAP_ROOT / "CMapDataset_filtered/cmap_dataset.pt",
        ["linkerhand_right"], ["ycb+lemon"], is_train=True, label="reference with linkerhand_right",
    )
    report["alias_behaviour"] = check_alias_behaviour(HERE / "logs" / "alias_test")
    report["q_roundtrip"] = check_q_roundtrip(dataset, index_csvs=args.index,
                                              prefixes=args.prefixes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
