"""SAPIEN-based feasibility checker for GraspQP LinkerHand-L20 grasps.

Design (full rationale in REPORT.md):

* **Hand**: one convex hull per URDF link, as static SAPIEN actors.  The L20
  visual meshes are non-watertight high-poly shells (SAPIEN's dynamic
  triangle-mesh/SDF path warns about them) and PhysX does not generate
  triangle-mesh vs triangle-mesh contacts at all, so at least one side of every
  contact pair must be convex.  Convex hulls are watertight, cheap to cook and
  tight around the phalanges (measured hull->mesh error: mean 0.4-1.9 mm, max
  3-18 mm, the palm being the loosest).  The hulls are also what SAPIEN uses for
  the *lift* test.
* **Object**: SAPIEN dynamic triangle mesh built from the watertight
  ``<object>.stl`` (exact geometry, SDF collision).
* **Contacts / penetration**: SAPIEN's reported ``separation`` for mesh-convex
  pairs has a measured multi-millimetre bias (see ``REPORT.md`` "Calibration"),
  so millimetre-level penetration is computed geometrically against the exact
  object mesh with a 2 mm voxel SDF (``object_sdf.VoxelSDF``); SAPIEN contacts
  are recorded as an advisory signal.
* **Stability**: lift test.  The hand actors are teleported upward by 6 cm over
  40 physics steps while gravity is enabled for the object; a grasp that holds
  raises the object with the hand, a failed grasp leaves it falling.

Every threshold lives in :class:`CheckerConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import sapien
import trimesh

import grasp_io as gio
from object_sdf import VoxelSDF

ASSET_CACHE = Path(__file__).resolve().parent / "assets" / "hand_hulls"


@dataclass
class CheckerConfig:
    # --- geometric contact / penetration (exact mesh, voxel SDF) -------------
    sdf_pitch: float = 0.002            # m
    contact_dist: float = 0.002         # m, link considered "in contact" below this
    max_penetration: float = 0.015      # m, sanity bound (see REPORT: systematic asset bias)
    min_contact_links: int = 2          # distinct hand links in contact
    min_contact_points: int = 3         # hand vertices within contact_dist
    # --- SAPIEN collision ----------------------------------------------------
    contact_offset: float = 0.003       # m
    tight_contact: float = 0.001        # m, advisory "tight" SAPIEN contact
    # --- lift test -----------------------------------------------------------
    lift_height: float = 0.06           # m
    lift_steps: int = 40
    settle_steps: int = 5
    timestep: float = 1.0 / 240.0
    lift_min_rise: float = 0.025        # m
    explosion_dist: float = 1.0         # m
    # --- speed ---------------------------------------------------------------
    max_verts_per_link: int = 300
    skip_lift_above: float = 0.015      # m, do not simulate the lift beyond this overlap


@dataclass
class GraspResult:
    feasible: bool
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)
    contacts: List[dict] = field(default_factory=list)


def _hull_cache(link_path: Path) -> Path:
    ASSET_CACHE.mkdir(parents=True, exist_ok=True)
    return ASSET_CACHE / (link_path.stem + "_hull.obj")


def build_hand_hulls(links: Dict[str, Path], force: bool = False) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for name, path in links.items():
        cache = _hull_cache(path)
        if force or not cache.exists():
            mesh = trimesh.load(path, force="mesh")
            mesh.convex_hull.export(cache)
        out[name] = cache
    return out


def measure_hull_error(links: Dict[str, Path], n_samples: int = 400) -> Dict[str, dict]:
    """Hull-surface -> visual-mesh distance: how far the proxy sticks out."""
    stats = {}
    for name, path in links.items():
        vis = trimesh.load(path, force="mesh")
        hull = trimesh.load(_hull_cache(path), force="mesh")
        pts, _ = trimesh.sample.sample_surface(hull, n_samples)
        d = trimesh.proximity.closest_point(vis, pts)[1]
        stats[name] = {
            "hull_to_mesh_max_mm": float(d.max() * 1000),
            "hull_to_mesh_mean_mm": float(d.mean() * 1000),
        }
    return stats


class SapienGraspChecker:
    """Reusable SAPIEN scene for one object; the hand is re-posed per grasp."""

    def __init__(self, object_name: str, cfg: CheckerConfig, log=print):
        self.cfg = cfg
        self.object_name = object_name
        self.log = log
        self.urdf = gio.UrdfModel(gio.HAND_URDF)
        self.links = gio.hand_link_meshes()
        self.hulls = build_hand_hulls(self.links)
        self.object_mesh = gio.object_mesh_path(object_name)

        self.sdf = VoxelSDF(self.object_mesh, pitch=cfg.sdf_pitch)
        self.obj_mesh = self.sdf.mesh
        self._link_vertices = {}
        for name, path in self.links.items():
            m = trimesh.load(path, force="mesh")
            v = m.vertices
            step = max(1, len(v) // cfg.max_verts_per_link)
            self._link_vertices[name] = v[::step]

        sapien.set_log_level("error")
        self.scene = sapien.Scene()
        self.scene.set_timestep(cfg.timestep)
        self.scene.physx_system.set_scene_collision_id(0)
        self._build_hand()
        self._build_object()

    # ---------------------------------------------------------------- scene
    def _build_hand(self):
        self.hand_actors: Dict[str, object] = {}
        self.hand_components: Dict[str, object] = {}
        for name, hull in self.hulls.items():
            b = self.scene.create_actor_builder()
            b.set_name(name)
            b.add_convex_collision_from_file(str(hull), density=1000)
            b.set_initial_pose(sapien.Pose())
            # Kinematic (not static): a static actor teleported with set_pose does
            # not push dynamic bodies (verified), while a kinematic body driven by
            # set_kinematic_target does -- that is what makes the lift test work.
            actor = b.build_kinematic(name)
            comp = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            for sh in comp.get_collision_shapes():
                sh.set_contact_offset(self.cfg.contact_offset)
            self.hand_actors[name] = actor
            self.hand_components[name] = comp

    def _build_object(self):
        b = self.scene.create_actor_builder()
        b.set_name("object")
        b.add_nonconvex_collision_from_file(str(self.object_mesh), density=1000)
        b.set_initial_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self.object_actor = b.build(name="object")
        self.object_comp = self.object_actor.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        for sh in self.object_comp.get_collision_shapes():
            sh.set_contact_offset(self.cfg.contact_offset)
        self.object_mass = float(self.object_comp.mass)
        self.object_comp.disable_gravity = True
        self.object_comp.max_depenetration_velocity = 0.02
        self.object_comp.max_linear_velocity = 2.0
        self.object_comp.max_angular_velocity = 10.0

    # ------------------------------------------------------------ per grasp
    def _pose_matrices(self, root_pose: np.ndarray, joints: np.ndarray, dz: float = 0.0):
        Troot = np.eye(4)
        Troot[:3, :3] = gio.quat_xyzw_to_matrix(root_pose[3:7])
        Troot[:3, 3] = root_pose[:3] + np.array([0.0, 0.0, dz])
        T = gio.fk_batch(self.urdf, gio.build_full_angles(joints[None, :]))
        return {name: Troot @ T[name][0] for name in self.hand_actors}

    def _place_hand(self, root_pose: np.ndarray, joints: np.ndarray, dz: float = 0.0):
        for name, W in self._pose_matrices(root_pose, joints, dz).items():
            q_wxyz = gio.matrix_to_quat_xyzw(W[:3, :3])[[3, 0, 1, 2]]
            self.hand_components[name].set_kinematic_target(
                sapien.Pose(W[:3, 3].tolist(), q_wxyz.tolist())
            )

    def _reset_object(self):
        self.object_comp.set_entity_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self.object_comp.set_linear_velocity(np.zeros(3))
        self.object_comp.set_angular_velocity(np.zeros(3))
        self.object_comp.disable_gravity = True
        self.object_comp.wake_up()

    def _read_contacts(self) -> dict:
        pairs = self.scene.get_contacts()
        pts_all: List[dict] = []
        links = set()
        for c in pairs:
            names = [c.bodies[0].get_name(), c.bodies[1].get_name()]
            if "object" not in names:
                continue
            other = names[0] if names[1] == "object" else names[1]
            for p in c.points:
                pts_all.append(
                    {
                        "link": other,
                        "separation_mm": float(p.separation) * 1000.0,
                        "position": [float(x) for x in p.position],
                        "normal": [float(x) for x in p.normal],
                    }
                )
                links.add(other)
        tight = [p for p in pts_all if p["separation_mm"] < self.cfg.tight_contact * 1000]
        max_pen = max(0.0, -min((p["separation_mm"] for p in pts_all), default=0.0)) / 1000.0
        return {
            "n_points": len(pts_all),
            "n_tight_points": len(tight),
            "n_links": len(links),
            "contact_links": sorted(links),
            "max_penetration_m": max_pen,
            "points": pts_all,
        }

    def _geometric_metrics(self, matrices: Dict[str, np.ndarray]) -> dict:
        """Exact-mesh geometry: per-link min distance and penetration depth."""
        link_min = {}
        all_d = []
        for name, W in matrices.items():
            V = self._link_vertices[name] @ W[:3, :3].T + W[:3, 3]
            d = self.sdf.query(V)  # negative inside
            all_d.append(d)
            link_min[name] = float(-d.min())
        d = np.concatenate(all_d)
        in_contact = {k: v for k, v in link_min.items() if v < self.cfg.contact_dist}
        pen = float(max(0.0, -d.min()))
        return {
            "geo_min_distance_mm": float(-d.min()) * 1000.0,
            "geo_penetration_mm": pen * 1000.0,
            "geo_contact_links": sorted(in_contact),
            "geo_n_contact_links": len(in_contact),
            "geo_contact_points": int((d < self.cfg.contact_dist).sum()),
            "geo_vertices_inside": int((d < 0).sum()),
        }

    def check(self, root_pose: np.ndarray, joints: np.ndarray) -> GraspResult:
        cfg = self.cfg
        try:
            self._reset_object()
            matrices = self._pose_matrices(root_pose, joints, dz=0.0)
            self._place_hand(root_pose, joints, dz=0.0)

            geo = self._geometric_metrics(matrices)
            metrics: Dict[str, float] = dict(geo)

            # ---- decision gate 1: contact & penetration (exact geometry) ----
            # NOTE: the physics below is always executed so that the logs carry the
            # lift metrics for every grasp; the verdict itself still honours the
            # gate.  ``run_filter.py --finalize`` can therefore re-derive verdicts
            # from the logs without re-simulating.
            gate_reason = None
            if geo["geo_contact_points"] < cfg.min_contact_points:
                gate_reason = "insufficient_contacts"
            elif geo["geo_n_contact_links"] < cfg.min_contact_links:
                gate_reason = "too_few_contact_links"
            elif geo["geo_penetration_mm"] > cfg.max_penetration * 1000:
                gate_reason = "penetration"

            # ---- SAPIEN settle + advisory contacts --------------------------
            for _ in range(cfg.settle_steps):
                self.scene.step()
            c0 = self._read_contacts()
            metrics.update(
                {
                    "sapien_contact_points": c0["n_points"],
                    "sapien_tight_contact_points": c0["n_tight_points"],
                    "sapien_contact_links": c0["n_links"],
                    "sapien_max_penetration_mm": c0["max_penetration_m"] * 1000.0,
                }
            )

            # ---- decision gate 2: lift test ---------------------------------
            # A grasp whose mesh already overlaps the object by more than
            # ``skip_lift_above`` cannot become feasible under any threshold we
            # would accept, so the (relatively expensive) lift is skipped and the
            # log records ``lift_skipped``.
            if geo["geo_penetration_mm"] > cfg.skip_lift_above * 1000:
                metrics["lift_skipped"] = 1.0
                metrics.update({"lift_rise_mm": float("nan"), "lift_drift_mm": float("nan")})
                return GraspResult(False, gate_reason or "penetration", metrics, c0["points"])
            p0 = self.object_comp.get_entity_pose().p.copy()
            self.object_comp.disable_gravity = False
            dz = cfg.lift_height / cfg.lift_steps
            for i in range(cfg.lift_steps):
                self._place_hand(root_pose, joints, dz=dz * (i + 1))
                self.object_comp.wake_up()
                self.scene.step()
            c1 = self._read_contacts()
            p1 = self.object_comp.get_entity_pose().p.copy()
            rise = float(p1[2] - p0[2])
            metrics.update(
                {
                    "lift_rise_mm": rise * 1000.0,
                    "lift_drift_mm": float(np.linalg.norm(p1[:2] - p0[:2])) * 1000.0,
                    "sapien_contacts_after_lift": c1["n_points"],
                    "sapien_links_after_lift": c1["n_links"],
                }
            )
            if not np.isfinite(p1).all() or float(np.linalg.norm(p1)) > cfg.explosion_dist:
                return GraspResult(False, "unstable_simulation", metrics, c0["points"])
            if gate_reason is not None:
                return GraspResult(False, gate_reason, metrics, c0["points"])
            if rise < cfg.lift_min_rise:
                return GraspResult(False, "slips_on_lift", metrics, c0["points"])
            return GraspResult(True, "ok", metrics, c0["points"])
        except Exception as exc:
            return GraspResult(False, f"exception:{type(exc).__name__}", {"error": str(exc)}, [])


def load_grasps(path: Path):
    d = gio.load_grasp_file(path)
    return d["root_pose"], d["joints"], d["values"], d["contact_idx"]


__all__ = [
    "CheckerConfig", "GraspResult", "SapienGraspChecker", "build_hand_hulls",
    "measure_hull_error", "load_grasps",
]
