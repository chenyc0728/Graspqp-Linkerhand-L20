"""手动（人工）审核 CMapDataset 里的每条抓取：open3d 逐条可视化 + 按键保留/丢弃。

数据源是 ``cmap_dataset_from_graspqp_v3.pt``（不再从 ``filtered_grasps`` 重新组装），
人工决策逐条追加到 ``manual_review_decisions.jsonl``，退出时只把 **被按 y 保留** 的样本按
v3.pt 的 metadata 顺序、以完全相同的数据结构写成新 ``.pt``。

数据结构（从 v3.pt 实读，不是凭文档臆断）::

    {"info": {"linkerhand": {"robot_name": "linkerhand",
                             "num_total": int, "num_upper_object": int,
                             "num_per_object": {object_name: int}}},
     "metadata": [ (Tensor(22,) float32, object_name: str, robot_name: str), ... ]}

``q = [x, y, z, roll, pitch, yaw, j0..j15]``，平移/旋转是 **手基座 hand_base_link** 在物体坐标系
里的位姿，欧拉乘序 ``R = Rx(roll) @ Ry(pitch) @ Rz(yaw)``（与 URDF 的 rpy 相反，见 HANDOFF §5.2）。
手部几何一律用 ``grasp_io.cmap_left_urdf()`` 的修复版 URDF 做 FK（原文件坏了，见 HANDOFF §7-4），
贴的是 **视觉 mesh**（不是碰撞凸包）。

按键
----
======  ==========================================================
``y`` / 空格   保留（keep）
``n`` / ``d``  丢弃（reject）
``s``          暂缓 / 跳过（skip，**不写入** decisions，下次还会出现）
``u``          撤销上一条决策（从 decisions.jsonl 末尾弹出并恢复现场）
``r``          重置视角到默认
``v``          在 auto / front / top / side 之间切换视角
``q`` / ``Esc``退出（已做出的决策全部保留；输出的 .pt 只含 keep）
======  ==========================================================

断点续审
--------
启动时若 ``--decisions`` 已存在：默认 **只审未决策的**（keep / reject 都算已决策），并在终端打印
``已 keep / 已 reject / 未审`` 统计。``--review-all`` 强制重审（重审时以最新决策为准）。
``u`` 之后会回到被撤销的那条样本。

uid 与 v3 的对应关系
--------------------
``uid = "<sha256>:<grasp_index>"``，与 ``convert_to_cmap.py`` 的去重键完全一致。sha256/grasp_index
不在 v3.pt 里，因此脚本用与转换时**相同**的输入按 `(source, object, grasp_file)` 排序 + feasible 序号
升序 + ``(sha256, index)`` 去重重放一遍，并**逐条断言** object_name 与 v3.pt 的 metadata 顺序完全一致；
对不上就报错退出（提示用 ``--index-csvs`` 显式指定，或用 ``--uid-mode positional`` 兜底）。

典型用法
--------
::

    # 交互审核（需要 X11；ssh -X 或 VNC）
    python new_grasp_filter/manual_review.py \\
        --dataset new_grasp_filter/cmap_dataset_from_graspqp_v3.pt \\
        --output  new_grasp_filter/cmap_dataset_from_graspqp_v3_manual.pt

    # 只审某个物体、只看前 50 条未决策的
    python new_grasp_filter/manual_review.py --objects contactdb+alarm_clock --limit 50

    # 不弹窗，只按已有 decisions 重放并生成 .pt（CI / 断点校验）
    python new_grasp_filter/manual_review.py --dry-run

    # 无窗口环境批量出图（每个待审样本一张 PNG 到 --record-dir）
    python new_grasp_filter/manual_review.py --offscreen --record-dir new_grasp_filter/vis_output/manual_review

注意：本机 ``pytorch_kinematics`` 是 fork（每个 revolute 关节 +±0.01 rad，见 REPORT_v4 §2.2），
所以手部 FK 默认用 ``--fk-backend numpy``（几何正确、与 q 自洽）；``--fk-backend pk`` 可按字面复现
带偏置的手型（画面上指尖最多差约 4 cm）。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grasp_io as gio  # noqa: E402

APP_VERSION = "manual_review/v1"
ROBOT_NAME = "linkerhand"
FG = HERE / "filtered_grasps"
DEFAULT_DECISIONS = HERE / "manual_review_decisions.jsonl"
DEFAULT_RECORD_DIR = HERE / "vis_output" / "manual_review"
SUMMARY_JSON = HERE / "logs" / "manual_review_v5.json"

VIRTUAL_XYZ = ("virtual_joint_x", "virtual_joint_y", "virtual_joint_z")
VIRTUAL_RPY = ("virtual_joint_roll", "virtual_joint_pitch", "virtual_joint_yaw")
VIEW_ORDER = ("auto", "front", "top", "side")
KEEP, REJECT = "keep", "reject"
KEY_HELP = "y/space=keep  n/d=reject  s=skip  u=undo  r=reset  v=view  q/Esc=quit"

#: per-link colours for the fingers (hand_base is neutral grey)
LINK_COLORS = [
    [0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
    [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.30, 0.70, 0.70],
    [0.80, 0.80, 0.20], [0.40, 0.40, 0.90], [0.95, 0.35, 0.55], [0.35, 0.75, 0.45],
]
OBJECT_COLOR = [0.93, 0.62, 0.18]
HAND_BASE_COLOR = [0.72, 0.72, 0.72]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
@dataclass
class Sample:
    index: int              # position in v3.pt metadata
    uid: str                # "<sha256>:<grasp_index>" (or positional fallback)
    object_name: str        # "<prefix>+<obj>"
    robot_name: str
    q_tensor: torch.Tensor  # the exact tensor object from v3.pt (float32 (22,))
    source: str = ""        # index.csv `source` column, for the meta file only

    @property
    def prefix(self) -> str:
        return self.object_name.split("+", 1)[0]

    @property
    def q(self) -> np.ndarray:
        return self.q_tensor.numpy()

    @property
    def pose(self) -> Tuple[np.ndarray, np.ndarray]:
        q = self.q
        return q[:3], q[3:6]


def load_dataset(path: Path):
    """Load v3.pt and validate its structure (never assume, always read)."""
    path = Path(path).resolve()
    if not path.exists():
        raise SystemExit(f"[error] 数据集不存在: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or sorted(payload.keys()) != ["info", "metadata"]:
        raise SystemExit(f"[error] {path.name}: 顶层结构不是 {{'info','metadata'}}，实际 {type(payload)}"
                         f"/{getattr(payload, 'keys', lambda: None)()}")
    metadata = payload["metadata"]
    if not isinstance(metadata, list):
        raise SystemExit(f"[error] metadata 不是 list，而是 {type(metadata)}")
    for i, m in enumerate(metadata):
        if not (isinstance(m, tuple) and len(m) == 3):
            raise SystemExit(f"[error] metadata[{i}] 不是 3 元组: {type(m)}/{m!r}"[:300])
        if not (torch.is_tensor(m[0]) and tuple(m[0].shape) == (22,)
                and m[0].dtype == torch.float32):
            raise SystemExit(f"[error] metadata[{i}][0] 不是 float32 (22,) 张量: "
                             f"{type(m[0])}/{getattr(m[0], 'shape', None)}/{getattr(m[0], 'dtype', None)}")
        if not isinstance(m[1], str) or not isinstance(m[2], str):
            raise SystemExit(f"[error] metadata[{i}][1]/[2] 不是 str")
    robots = {m[2] for m in metadata}
    if len(robots) > 1:
        raise SystemExit(f"[error] robot_name 不唯一: {sorted(robots)}")
    return payload, metadata


def resolve_index_specs(prefixes: Sequence[str], explicit: Optional[Sequence[str]]):
    """Locate the index CSVs whose (sha256, grasp_index) order reproduces v3.pt."""
    if explicit:
        paths = [Path(p).resolve() for p in explicit]
        if len(paths) != len(prefixes):
            raise SystemExit(f"[error] --index-csvs 有 {len(paths)} 项，--prefixes 有 {len(prefixes)} 项")
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"[error] --index-csvs 不存在: {missing}")
        return list(zip(paths, prefixes))
    specs = []
    for pref in prefixes:
        for cand in (FG / f"index_{pref}.csv", FG / "index.csv"):
            if cand.exists():
                specs.append((cand.resolve(), pref))
                break
        else:
            raise SystemExit(f"[error] 找不到 {pref} 的索引 CSV（试过 {FG}/index_{pref}.csv 与 index.csv）；"
                             f"请用 --index-csvs 指定")
    return specs


def build_samples(metadata, index_specs, uid_mode: str, dataset_path: Path) -> List[Sample]:
    """Pair every q with its ``(sha256, grasp_index)`` uid, asserting the order matches."""
    samples = [
        Sample(index=i, uid="", object_name=m[1], robot_name=m[2], q_tensor=m[0])
        for i, m in enumerate(metadata)
    ]
    if uid_mode == "positional":
        for s in samples:
            s.uid = f"{dataset_path.stem}:{s.index}"
        log(f"[uid] positional mode -> {samples[0].uid} …（与 convert_to_cmap 的去重键无关，仅作兜底）")
        return samples

    import convert_to_cmap as c2c

    pairs = list(c2c.iter_pairs(index_specs, False, True))
    if len(pairs) != len(samples):
        raise SystemExit(
            f"[error] 索引重放出 {len(pairs)} 条，v3.pt 有 {len(samples)} 条 —— 输入不匹配。\n"
            f"        索引: {[str(p) for p, _ in index_specs]}\n"
            f"        用 --index-csvs 指定正确的索引，或 --uid-mode positional 兜底。")
    bad = [i for i, (p, s) in enumerate(zip(pairs, samples)) if p[3] != s.object_name]
    if bad:
        i = bad[0]
        raise SystemExit(
            f"[error] 索引与 v3.pt 的 object_name 顺序对不上（首个不一致: metadata[{i}]="
            f"{samples[i].object_name} vs 索引 {pairs[i][3]}）。\n"
            f"        用 --index-csvs 指定正确的索引，或 --uid-mode positional 兜底。")
    for s, p in zip(samples, pairs):
        s.uid = f"{p[0]['sha256']}:{p[1]}"
        s.source = p[4]
    dupes = len(samples) - len({s.uid for s in samples})
    if dupes:
        raise SystemExit(f"[error] 重放出 {dupes} 个重复 uid，索引与转换时的去重口径不一致")
    log(f"[uid] 与 convert_to_cmap 的去重键一致（{len(samples)} 条，例 {samples[0].uid}）")
    return samples


# --------------------------------------------------------------------------
# decisions (append-only JSONL, crash safe)
# --------------------------------------------------------------------------
class DecisionStore:
    """``decisions.jsonl``：一行一条，append + fsync；``undo`` 从末尾弹出。"""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.records: List[dict] = []
        self.by_uid: Dict[str, dict] = {}
        self.bad_lines = 0

    def load(self) -> "DecisionStore":
        self.records, self.by_uid, self.bad_lines = [], {}, 0
        if not self.path.exists():
            return self
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                self.bad_lines += 1
                continue
            if rec.get("decision") not in (KEEP, REJECT) or "uid" not in rec:
                self.bad_lines += 1
                continue
            self.records.append(rec)
            self.by_uid[rec["uid"]] = rec          # last write wins
        return self

    def append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())                   # 立即落盘：崩溃不丢决策
        self.records.append(rec)
        self.by_uid[rec["uid"]] = rec

    def undo(self) -> Optional[dict]:
        """Pop the last line and rewrite the file (state = reload from the file)."""
        if not self.records:
            return None
        popped = self.records.pop()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.records),
                       encoding="utf-8")
        os.replace(tmp, self.path)
        self.load()
        return popped

    def decision(self, uid: str) -> Optional[str]:
        rec = self.by_uid.get(uid)
        return rec["decision"] if rec else None

    def counts(self) -> Tuple[int, int]:
        kept = sum(1 for v in self.by_uid.values() if v["decision"] == KEEP)
        rej = sum(1 for v in self.by_uid.values() if v["decision"] == REJECT)
        return kept, rej


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
class HandGeometry:
    """Visual meshes of the left L20 + FK on CMap's repaired left-hand URDF."""

    def __init__(self, backend: str = "numpy"):
        self.backend = backend
        gio.set_backend("numpy" if backend == "numpy" else "torch")
        self.urdf = gio.UrdfModel(gio.cmap_left_urdf(repaired=True))
        # spec: 视觉网格来自 graspqp/assets/linkerhand_l20/meshes/*.STL
        paths = dict(gio.hand_link_meshes(gio.HAND_URDF))
        fallback = gio.hand_link_meshes(self.urdf.path)          # 同几何（已对拍）
        for name, p in fallback.items():
            if name not in paths or not Path(paths[name]).exists():
                paths[name] = p
        import trimesh

        self.vertex: Dict[str, np.ndarray] = {}
        self.face: Dict[str, np.ndarray] = {}
        for name, p in paths.items():
            if not Path(p).exists():
                continue
            m = trimesh.load(str(p), force="mesh", process=True)
            self.vertex[name] = np.asarray(m.vertices, dtype=np.float64)
            self.face[name] = np.asarray(m.faces, dtype=np.int32)
        self.link_names = sorted(self.vertex)
        self.n_tris = sum(len(f) for f in self.face.values())

    def transforms(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """link -> 4x4 world matrix, from the 22-dim CMap q."""
        q = np.asarray(q, dtype=np.float64)
        angles: Dict[str, float] = {}
        for i, n in enumerate(VIRTUAL_XYZ):
            angles[n] = float(q[i])
        for i, n in enumerate(VIRTUAL_RPY):
            angles[n] = float(q[3 + i])
        full = gio.build_full_angles(q[6:22][None, :])[0]
        for i, n in enumerate(gio.CHAIN_JOINT_NAMES):
            angles[n] = float(full[i])
        if self.backend == "numpy":
            return self.urdf.fk(angles)
        chain = self.urdf.pk_chain_on("cpu")
        names = chain.get_joint_parameter_names()
        vec = torch.tensor([[angles.get(n, 0.0) for n in names]], dtype=torch.float64)
        fk = chain.forward_kinematics(vec)
        return {k: v.get_matrix()[0].numpy() for k, v in fk.items()}

    def geoms(self, q: np.ndarray):
        import open3d as o3d

        T = self.transforms(q)
        out = []
        for j, name in enumerate(self.link_names):
            W = T.get(name)
            if W is None:
                continue
            V = self.vertex[name] @ W[:3, :3].T + W[:3, 3]
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                          o3d.utility.Vector3iVector(self.face[name]))
            m.compute_vertex_normals()
            color = HAND_BASE_COLOR if name == "hand_base_link" else LINK_COLORS[j % len(LINK_COLORS)]
            m.paint_uniform_color(color)
            out.append(m)
        return out, T


class ObjectGeometry:
    """CMap PointCloud 优先，缺失时回退到 object/<prefix>/<obj>/<obj>.stl；缓存当前物体。"""

    def __init__(self, geom_mode: str = "auto", point_size: float = 4.0):
        self.geom_mode = geom_mode
        self.point_size = point_size
        self._cache: Dict[str, object] = {}

    @staticmethod
    def pointcloud_path(object_name: str) -> Path:
        prefix, obj = object_name.split("+", 1)
        return gio.CMAP_ROOT / "PointCloud" / "object" / prefix / f"{obj}.pt"

    @staticmethod
    def mesh_path(object_name: str) -> Path:
        prefix, obj = object_name.split("+", 1)
        return gio.REPO_ROOT / "object" / prefix / obj / f"{obj}.stl"

    def get(self, object_name: str):
        import open3d as o3d

        if object_name in self._cache:
            return self._cache[object_name]
        if len(self._cache) > 2:                     # 只留最近两个，避免内存膨胀
            self._cache.pop(next(iter(self._cache)))
        pc_path, mesh_path = self.pointcloud_path(object_name), self.mesh_path(object_name)
        geom = None
        if self.geom_mode in ("auto", "points") and pc_path.exists():
            arr = torch.load(pc_path, map_location="cpu")
            pts = arr[:, :3].numpy().astype(np.float64) if torch.is_tensor(arr) else np.asarray(arr)[:, :3]
            geom = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            geom.paint_uniform_color(OBJECT_COLOR)
            self._cache[object_name] = geom
            return geom
        if self.geom_mode in ("auto", "mesh") and mesh_path.exists():
            import trimesh

            m = trimesh.load(str(mesh_path), force="mesh", process=True)   # HANDOFF §7-3
            geom = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(m.vertices, dtype=np.float64)),
                                             o3d.utility.Vector3iVector(np.asarray(m.faces, dtype=np.int32)))
            geom.compute_vertex_normals()
            geom.paint_uniform_color(OBJECT_COLOR)
            self._cache[object_name] = geom
            return geom
        raise SystemExit(f"[error] {object_name}: 既没有 PointCloud（{pc_path}）也没有网格（{mesh_path}）")


def burn_hud(png_path: Path, lines: List[str]) -> None:
    """把 HUD 文本烘进 PNG（--record-dir 的截图自解释；PIL 缺失就跳过）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:                            # pragma: no cover
        return
    try:
        img = Image.open(png_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        y = 8
        for line in lines:
            draw.text((10, y), line, fill=(255, 255, 255), font=font)
            y += 15
        img.save(png_path)
    except Exception:                            # pragma: no cover
        return


def overlay_geoms(hand_T: Dict[str, np.ndarray], q: np.ndarray, decided: Optional[str],
                  axes_size: float = 0.03):
    """三条坐标轴（物体原点 + 手基座）、q 平移点小球。"""
    import open3d as o3d

    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size)]
    W = hand_T.get("hand_base_link")
    if W is not None:
        hand_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size)
        hand_axes.transform(W)
        geoms.append(hand_axes)
    ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.004, resolution=12)
    ball.translate(np.asarray(q[:3], dtype=np.float64))
    ball.compute_vertex_normals()
    color = {"keep": [0.1, 0.8, 0.1], "reject": [0.9, 0.1, 0.1]}.get(decided or "", [0.95, 0.95, 0.95])
    ball.paint_uniform_color(color)
    geoms.append(ball)
    return geoms


# --------------------------------------------------------------------------
# on-screen HUD (this open3d build has no add_3d_label -> retitle the window)
# --------------------------------------------------------------------------
class X11Title:
    """Best-effort window-title HUD via X11 (WM_NAME + _NET_WM_NAME).

    本机 open3d 0.20 的 ``VisualizerWithKeyCallback`` 没有 ``add_3d_label``，
    所以把 i/N、object_name、q 前 6 维、是否已决策写进窗口标题；同样的信息也逐条
    打在终端。没有 X11（Wayland / 非 Linux / 找不到窗口）时静默降级，不影响审核。
    """

    def __init__(self, tag: str):
        self.dpy = None
        self.win = None
        self.enabled = False
        if not sys.platform.startswith("linux") or not os.environ.get("DISPLAY"):
            return
        try:
            import ctypes
            import ctypes.util

            x11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
            x11.XOpenDisplay.restype = ctypes.c_void_p
            x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
            x11.XDefaultRootWindow.restype = ctypes.c_ulong
            x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
            x11.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                                       ctypes.POINTER(ctypes.c_ulong),
                                       ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
                                       ctypes.POINTER(ctypes.c_uint)]
            x11.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
            x11.XStoreName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_char_p]
            x11.XInternAtom.restype = ctypes.c_ulong
            x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
            x11.XChangeProperty.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
                                            ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_char_p, ctypes.c_int]
            x11.XFlush.argtypes = [ctypes.c_void_p]
            dpy = x11.XOpenDisplay(os.environ["DISPLAY"].encode())
            if not dpy:
                return
            root = x11.XDefaultRootWindow(ctypes.c_void_p(dpy))

            def walk(w, out):
                r = ctypes.c_ulong()
                par = ctypes.c_ulong()
                ch = ctypes.POINTER(ctypes.c_ulong)()
                n = ctypes.c_uint()
                x11.XQueryTree(ctypes.c_void_p(dpy), ctypes.c_ulong(w), ctypes.byref(r),
                               ctypes.byref(par), ctypes.byref(ch), ctypes.byref(n))
                for i in range(n.value):
                    out.append(ch[i])
                    walk(ch[i], out)
                return out

            hit = None
            for w in walk(root, []):
                nm = ctypes.c_char_p()
                if x11.XFetchName(ctypes.c_void_p(dpy), ctypes.c_ulong(w), ctypes.byref(nm)) and nm.value:
                    if tag.encode() in nm.value:
                        hit = w
            if hit is None:
                return
            self.dpy, self.win = dpy, hit
            self.x11 = x11
            self.atom_net = x11.XInternAtom(ctypes.c_void_p(dpy), b"_NET_WM_NAME", 0)
            self.atom_utf8 = x11.XInternAtom(ctypes.c_void_p(dpy), b"UTF8_STRING", 0)
            self.enabled = True
        except Exception:                        # pragma: no cover - cosmetic only
            self.enabled = False

    def set_title(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            raw = text.encode("utf-8")
            self.x11.XStoreName(ctypes.c_void_p(self.dpy), ctypes.c_ulong(self.win), text.encode("ascii", "replace"))
            self.x11.XChangeProperty(ctypes.c_void_p(self.dpy), ctypes.c_ulong(self.win), ctypes.c_ulong(self.atom_net),
                                     ctypes.c_ulong(self.atom_utf8), 8, 0, raw, len(raw))
            self.x11.XFlush(ctypes.c_void_p(self.dpy))
        except Exception:                        # pragma: no cover - cosmetic only
            self.enabled = False


# --------------------------------------------------------------------------
# viewer
# --------------------------------------------------------------------------
class ReviewViewer:
    def __init__(self, queue: List[Sample], samples: List[Sample], store: DecisionStore, args,
                 hand: HandGeometry, objs: ObjectGeometry):
        import open3d as o3d

        self.o3d = o3d
        self.queue = queue
        self.samples = samples
        self.store = store
        self.args = args
        self.hand = hand
        self.objs = objs
        self.pos = 0
        self.view = args.view if args.view in VIEW_ORDER else "auto"
        self.n_decided_now = 0
        self.quitting = False

        self.window_tag = f"manual_review[{os.getpid()}]"
        self._title_text = ""
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        ok = self.vis.create_window(window_name=self.window_tag, width=args.width, height=args.height,
                                   visible=True)
        if not ok:
            raise SystemExit(
                "[error] 无法创建 open3d 窗口（没有可用的 X11/GLFW 显示）。\n"
                "        请在带显示器的桌面、或 `ssh -X`/`ssh -Y`/VNC 下运行；\n"
                "        无窗口环境请用 --dry-run（只按已有 decisions 生成 .pt）\n"
                "        或 --offscreen --record-dir <dir>（只出图）。")
        ro = self.vis.get_render_option()
        ro.background_color = np.array([0.08, 0.08, 0.09])
        ro.point_size = float(args.point_size)
        ro.mesh_show_back_face = True
        ro.light_on = True
        self._busy = False
        self._x11 = X11Title(self.window_tag)
        log(f"[view] 键位: {KEY_HELP}")
        self._register_keys()

    # -- helpers ---------------------------------------------------------
    def _register_keys(self) -> None:
        def on(decision):
            def cb(_vis):
                if self._busy:
                    return False
                self._busy = True
                try:
                    self._decide(decision)
                finally:
                    self._busy = False
                return False
            return cb

        self.vis.register_key_callback(ord("Y"), on(KEEP))
        self.vis.register_key_callback(32, on(KEEP))            # space
        self.vis.register_key_callback(ord("N"), on(REJECT))
        self.vis.register_key_callback(ord("D"), on(REJECT))
        self.vis.register_key_callback(ord("S"), self._wrap(self._skip))
        self.vis.register_key_callback(ord("U"), self._wrap(self._undo))
        self.vis.register_key_callback(ord("R"), self._wrap(self._reset_view))
        self.vis.register_key_callback(ord("V"), self._wrap(self._cycle_view))
        self.vis.register_key_callback(ord("Q"), self._wrap(self._quit))
        self.vis.register_key_callback(256, self._wrap(self._quit))   # Esc

    def _wrap(self, fn):
        def cb(_vis):
            if self._busy:                 # 防止回调里再进回调（open3d 会重入派发）
                return False
            self._busy = True
            try:
                fn()
            finally:
                self._busy = False
            return False
        return cb

    def _safe_reset_view(self) -> None:
        """reset_view_point 在空场景（已审完）上会抛 CreateCoordinateFrame(size<=0)。"""
        try:
            self.vis.reset_view_point(True)
        except Exception:
            return
        self._apply_view()

    @property
    def current(self) -> Optional[Sample]:
        return self.queue[self.pos] if 0 <= self.pos < len(self.queue) else None

    # -- drawing ---------------------------------------------------------
    def _draw(self, refit: bool = False) -> None:
        s = self.current
        self.vis.clear_geometries()
        if s is None:
            self._update_title(None)
            self.vis.update_renderer()
            return
        obj = self.objs.get(s.object_name)
        self.vis.add_geometry(obj)
        hand_geoms, T = self.hand.geoms(s.q)
        for g in hand_geoms:
            self.vis.add_geometry(g)
        decided = self.store.decision(s.uid)
        for g in overlay_geoms(T, s.q, decided):
            self.vis.add_geometry(g)
        self._update_title(s, decided)
        for line in self._hud(s, obj, decided)[:2]:
            log(line)
        if refit:
            self._safe_reset_view()
        self.vis.update_renderer()

    def _hud(self, s: Sample, obj, decided: Optional[str]) -> List[str]:
        t, rpy = s.pose
        kept, rej = self.store.counts()
        n_pending = len(self.samples) - kept - rej
        state = f"decided={decided.upper()}" if decided else "undecided"
        return [
            f"[{self.pos + 1}/{len(self.queue)}] meta#{s.index} {s.object_name} robot={s.robot_name} {state}",
            f"t=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}) rpy=({rpy[0]:+.3f},{rpy[1]:+.3f},{rpy[2]:+.3f})",
            f"uid={s.uid.split(':')[0][:10]}..:{s.uid.split(':')[-1]} kept={kept} rejected={rej} pending={n_pending} view={self.view}",
            "y/space=keep  n/d=reject  s=skip  u=undo  r=reset  v=view  q/Esc=quit",
        ]

    def _update_title(self, s: Optional[Sample], decided: Optional[str] = None) -> None:
        """On-screen HUD: this open3d build has no add_3d_label, so we update the X11 title."""
        if s is None:
            title = f"{self.window_tag} | queue finished -- press q/Esc to write the .pt"
        else:
            hud = self._hud(s, None, decided)
            title = f"{self.window_tag} | {hud[0]} | {hud[1]} | {hud[3]}"
        if title == self._title_text:
            return
        self._title_text = title
        if self._x11 is not None:
            self._x11.set_title(title)

    def _apply_view(self) -> None:
        ctrl = self.vis.get_view_control()
        presets = {
            "front": ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
            "top": ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
            "side": ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            "auto": ([0.55, -0.72, 0.42], [0.0, 0.0, 1.0]),
        }
        front, up = presets.get(self.view, presets["auto"])
        ctrl.set_front(np.asarray(front, dtype=float))
        ctrl.set_up(np.asarray(up, dtype=float))
        ctrl.set_zoom(0.55)

    # -- actions ---------------------------------------------------------
    def _record(self, s: Sample, decision: str) -> None:
        self.store.append({
            "dataset": str(self.args.dataset),
            "uid": s.uid,
            "object_name": s.object_name,
            "decision": decision,
            "ts": now_iso(),
            "app_version": APP_VERSION,
            "metadata_index": s.index,
            "prefix": s.prefix,
            "source": s.source,
        })

    def _maybe_capture(self, s: Sample, decision: str) -> None:
        if not self.args.record_dir:
            return
        buf = self.vis.capture_screen_float_buffer(do_render=True)
        arr = (np.asarray(buf) * 255).astype(np.uint8)
        out = Path(self.args.record_dir) / f"{s.index:05d}_{decision}_{s.object_name.replace('+', '_')}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        self.o3d.io.write_image(str(out), self.o3d.geometry.Image(arr))
        burn_hud(out, self._hud(s, None, decision))

    def _decide(self, decision: str) -> None:
        s = self.current
        if s is None:
            return
        self._record(s, decision)                 # 先落盘，再切下一条
        self._maybe_capture(s, decision)
        self.n_decided_now += 1
        log(f"[{self.pos + 1}/{len(self.queue)}] {decision:6s} {s.object_name:24s} {s.uid}")
        if self.n_decided_now % max(1, self.args.save_every) == 0:
            self._flush_outputs(quiet=False)
        self.pos += 1
        self._draw()

    def _skip(self) -> None:
        s = self.current
        if s is None:
            return
        log(f"[{self.pos + 1}/{len(self.queue)}] skip   {s.object_name} （不写入 decisions）")
        self.pos += 1
        self._draw()

    def _undo(self) -> None:
        popped = self.store.undo()
        if popped is None:
            log("[undo] decisions.jsonl 已经空了")
            return
        log(f"[undo] 撤销 uid={popped['uid']} （原决策 {popped['decision']}）")
        # 回到被撤销的那条样本：如果在本次队列里就跳过去，否则重放全部
        target = next((i for i, s in enumerate(self.queue) if s.uid == popped["uid"]), None)
        if target is None:
            log("[undo] 该样本不在本次队列（可能已过滤掉），按 --review-all 口径重新排队…")
            self.queue = [s for s in self.samples]
            target = next(i for i, s in enumerate(self.queue) if s.uid == popped["uid"])
        self.pos = target
        self._draw(refit=True)

    def _reset_view(self) -> None:
        if self.current is None:
            return
        self._safe_reset_view()
        self._draw()

    def _cycle_view(self) -> None:
        self.view = VIEW_ORDER[(VIEW_ORDER.index(self.view) + 1) % len(VIEW_ORDER)]
        if self.current is None:
            return
        self._apply_view()
        self._draw()

    def _quit(self) -> None:
        self.quitting = True
        self.vis.close()

    def _flush_outputs(self, quiet: bool = True) -> None:
        meta = write_outputs(self.samples, self.store, Path(self.args.output),
                            Path(self.args.dataset), Path(self.args.decisions), self.args,
                            quiet=quiet)
        if not quiet:
            log(f"[save] {self.args.output}  n={meta['n_samples']} (keep={meta['num_kept']})")
        self._last_meta = meta

    def run(self) -> dict:
        if not self.queue:
            log("[run] 没有需要审核的样本（全部已决策；用 --review-all 可重审）")
            self.vis.destroy_window()
            return write_outputs(self.samples, self.store, Path(self.args.output),
                                Path(self.args.dataset), Path(self.args.decisions), self.args)
        self._draw(refit=True)
        self.vis.run()
        self.vis.destroy_window()
        return write_outputs(self.samples, self.store, Path(self.args.output),
                             Path(self.args.dataset), Path(self.args.decisions), self.args,
                             quiet=False)


class OffscreenRenderer:
    """``--record-dir`` + ``--offscreen``：无交互，逐条出 PNG（不是交互模式）。"""

    def __init__(self, queue: List[Sample], store: DecisionStore, args,
                 hand: HandGeometry, objs: ObjectGeometry):
        import open3d as o3d

        self.o3d = o3d
        self.queue = queue
        self.store = store
        self.args = args
        self.hand = hand
        self.objs = objs
        self.view = args.view if args.view in VIEW_ORDER else "auto"
        self.vis = o3d.visualization.Visualizer()
        if not self.vis.create_window(window_name="manual_review_offscreen",
                                     width=args.width, height=args.height, visible=False):
            raise SystemExit("[error] 离屏窗口创建失败（需要可用的 GLX/EGL）")
        ro = self.vis.get_render_option()
        ro.background_color = np.array([0.08, 0.08, 0.09])
        ro.point_size = float(args.point_size)
        ro.mesh_show_back_face = True
        self.outdir = Path(args.record_dir)

    def _apply_view(self) -> None:
        ctrl = self.vis.get_view_control()
        presets = {
            "front": ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
            "top": ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
            "side": ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            "auto": ([0.55, -0.72, 0.42], [0.0, 0.0, 1.0]),
        }
        front, up = presets.get(self.view, presets["auto"])
        ctrl.set_front(np.asarray(front, dtype=float))
        ctrl.set_up(np.asarray(up, dtype=float))
        ctrl.set_zoom(0.55)

    def run(self) -> List[str]:
        self.outdir.mkdir(parents=True, exist_ok=True)
        written = []
        for i, s in enumerate(self.queue):
            self.vis.clear_geometries()
            self.vis.add_geometry(self.objs.get(s.object_name))
            geoms, T = self.hand.geoms(s.q)
            for g in geoms:
                self.vis.add_geometry(g)
            for g in overlay_geoms(T, s.q, self.store.decision(s.uid)):
                self.vis.add_geometry(g)
            self.vis.reset_view_point(True)
            self._apply_view()
            self.vis.poll_events()
            self.vis.update_renderer()
            arr = (np.asarray(self.vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
            out = self.outdir / f"{s.index:05d}_unreviewed_{s.object_name.replace('+', '_')}.png"
            self.o3d.io.write_image(str(out), self.o3d.geometry.Image(arr))
            t, rpy = s.pose
            burn_hud(out, [
                f"[{i + 1}/{len(self.queue)}] meta#{s.index} {s.object_name} robot={s.robot_name} "
                f"decided={self.store.decision(s.uid) or 'undecided'}",
                f"t=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}) rpy=({rpy[0]:+.3f},{rpy[1]:+.3f},{rpy[2]:+.3f}) "
                f"view={self.view} fk={self.args.fk_backend}",
            ])
            written.append(str(out))
            log(f"[{i + 1}/{len(self.queue)}] {out}")
        self.vis.destroy_window()
        return written


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def write_outputs(samples: List[Sample], store: DecisionStore, output: Path, dataset: Path,
                  decisions_file: Path, args, quiet: bool = True) -> dict:
    """把 decision==keep 的样本按 v3.pt 的 metadata 顺序写成 .pt + .meta.json。"""
    kept = [s for s in samples if store.decision(s.uid) == KEEP]
    rejected = [s for s in samples if store.decision(s.uid) == REJECT]
    robot = samples[0].robot_name if samples else ROBOT_NAME

    metadata = [(s.q_tensor, s.object_name, s.robot_name) for s in kept]   # 复用 v3 的张量对象
    per_object: Dict[str, int] = {}
    for s in kept:
        per_object[s.object_name] = per_object.get(s.object_name, 0) + 1

    def _by_prefix(seq: List[Sample]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in seq:
            out[s.prefix] = out.get(s.prefix, 0) + 1
        return out

    reviewed_at = now_iso()
    manual_review = {
        "source_dataset": str(dataset),
        "num_before": len(samples),
        "num_after": len(kept),
        "num_rejected": len(rejected),
        "num_unreviewed": len(samples) - len(kept) - len(rejected),
        "kept_by_prefix": _by_prefix(kept),
        "rejected_by_prefix": _by_prefix(rejected),
        "reviewed_at": reviewed_at,
        "decisions_file": str(decisions_file),
        "app_version": APP_VERSION,
        "keep_rule": "只有 decision==keep 的样本进入本文件；未审样本一律不保留",
    }
    info = {
        robot: {
            "robot_name": robot,
            "num_total": len(kept),
            "num_upper_object": max(per_object.values()) if per_object else 0,
            "num_per_object": per_object,
            "manual_review": manual_review,
        }
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"info": info, "metadata": metadata}, output)

    def _diff() -> Dict[str, Dict[str, int]]:
        before: Dict[str, int] = {}
        rej: Dict[str, int] = {}
        for s in samples:
            before[s.object_name] = before.get(s.object_name, 0) + 1
            if store.decision(s.uid) == REJECT:
                rej[s.object_name] = rej.get(s.object_name, 0) + 1
        rows = {}
        for name in sorted(before):
            k = per_object.get(name, 0)
            r = rej.get(name, 0)
            rows[name] = {"before": before[name], "kept": k, "rejected": r,
                          "unreviewed": before[name] - k - r}
        return rows

    obj_rows = _diff()
    meta = {
        "output": str(output),
        "source_dataset": str(dataset),
        "robot_name": robot,
        "n_samples": len(kept),
        "n_objects": len(per_object),
        "per_object": per_object,
        "per_prefix": _by_prefix(kept),
        "num_kept": len(kept),
        "num_rejected": len(rejected),
        "num_unreviewed": len(samples) - len(kept) - len(rejected),
        "num_before": len(samples),
        "decisions_file": str(decisions_file),
        "reviewed_at": reviewed_at,
        "app_version": APP_VERSION,
        "fk_backend": getattr(args, "fk_backend", "numpy"),
        "object_geom": getattr(args, "object_geom", "auto"),
        "diff_vs_source": obj_rows,
        "objects_fully_rejected": sorted(n for n, r in obj_rows.items() if r["kept"] == 0),
        "note": (
            "q = [x, y, z, roll, pitch, yaw, 16 joints]，手基座 hand_base_link 位姿，R = Rx@Ry@Rz；"
            "结构 / dtype / robot_name / object_name 前缀与 source_dataset 逐项对齐，"
            "只保留人工按键 keep 的样本。manual_review 统计同时写在 info[robot]['manual_review']。"
        ),
    }
    # per_source 来自索引 CSV 的 source 列（v3.pt 里没有该字段）
    src_counts: Dict[str, int] = {}
    for s in kept:
        if s.source:
            src_counts[s.source] = src_counts.get(s.source, 0) + 1
    meta["per_source"] = src_counts
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                               encoding="utf-8")

    # 可选：被丢弃的样本单独留一份，便于回溯
    if getattr(args, "rejected_output", None) and rejected:
        rej_meta = [(s.q_tensor, s.object_name, s.robot_name) for s in rejected]
        rej_per: Dict[str, int] = {}
        for s in rejected:
            rej_per[s.object_name] = rej_per.get(s.object_name, 0) + 1
        torch.save({"info": {robot: {"robot_name": robot, "num_total": len(rejected),
                                     "num_upper_object": max(rej_per.values()),
                                     "num_per_object": rej_per}},
                    "metadata": rej_meta}, Path(args.rejected_output))
        Path(args.rejected_output).with_suffix(".meta.json").write_text(
            json.dumps({"output": str(args.rejected_output), "source_dataset": str(dataset),
                        "num_rejected": len(rejected), "per_object": rej_per,
                        "decisions_file": str(decisions_file), "reviewed_at": reviewed_at},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    write_summary(samples, store, kept, rejected, output, dataset, decisions_file, obj_rows,
                  Path(getattr(args, "summary_json", SUMMARY_JSON)))
    if not quiet:
        log(f"[write] {output}  n={len(kept)} / {len(samples)}  "
            f"keep_prefix={_by_prefix(kept)}  rejected={len(rejected)}")
    return meta


def write_summary(samples: List[Sample], store: DecisionStore, kept: List[Sample],
                  rejected: List[Sample], output: Path, dataset: Path, decisions_file: Path,
                  obj_rows: Dict[str, Dict[str, int]], summary_path: Path = SUMMARY_JSON) -> None:
    kept_n, rej_n = store.counts()
    summary = {
        "app_version": APP_VERSION,
        "source_dataset": str(dataset),
        "output": str(output),
        "decisions_file": str(decisions_file),
        "generated_at": now_iso(),
        "totals": {
            "num_before": len(samples),
            "num_kept": len(kept),
            "num_rejected": len(rejected),
            "num_unreviewed": len(samples) - len(kept) - len(rejected),
            "decisions_recorded": len(store.records),
            "decisions_keep": kept_n,
            "decisions_reject": rej_n,
            "malformed_decision_lines": store.bad_lines,
        },
        "per_prefix": {
            "kept": _count_prefix(kept),
            "rejected": _count_prefix(rejected),
            "unreviewed": _count_prefix([s for s in samples
                                         if store.decision(s.uid) is None]),
        },
        "objects_fully_rejected": sorted(n for n, r in obj_rows.items() if r["kept"] == 0),
        "per_object": obj_rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _count_prefix(seq: List[Sample]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in seq:
        out[s.prefix] = out.get(s.prefix, 0) + 1
    return out


# --------------------------------------------------------------------------
# queue selection
# --------------------------------------------------------------------------
def select_queue(samples: List[Sample], store: DecisionStore, args) -> Tuple[List[Sample], dict]:
    pool = list(samples)
    prefixes = list(args.prefixes or [])
    if prefixes:
        pool = [s for s in pool if s.prefix in prefixes]
    if args.objects:
        wanted = set(args.objects)
        pool = [s for s in pool if s.object_name in wanted or s.object_name.split("+", 1)[1] in wanted]
    info = {"after_filters": len(pool)}
    if args.shuffle:
        import random

        rng = random.Random(args.seed)
        rng.shuffle(pool)
        info["shuffled"] = True
        info["seed"] = args.seed
    if not args.review_all:
        pool = [s for s in pool if store.decision(s.uid) is None]
    info["pending_after_resume_filter"] = len(pool)
    start = max(0, int(args.start or 0))
    pool = pool[start:]
    if args.limit:
        pool = pool[: int(args.limit)]
    info["queued"] = len(pool)
    return pool, info


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="按键：y/空格=保留  n/d=丢弃  s=暂缓(不写入)  u=撤销上一条  r=重置视角  "
               "v=切换视角  q/Esc=退出（只把 keep 写进输出 .pt）")
    ap.add_argument("--dataset", default=str(HERE / "cmap_dataset_from_graspqp_v3.pt"),
                    help="唯一数据源（v3.pt）；不从 filtered_grasps 重新组装")
    ap.add_argument("--output", default=str(HERE / "cmap_dataset_from_graspqp_v3_manual.pt"),
                    help="输出 .pt（只含 decision==keep 的样本），同名 .meta.json 一并写出")
    ap.add_argument("--prefixes", nargs="+", default=["ycb", "contactdb"],
                    help="只审这些 object_name 前缀（默认 ycb contactdb）")
    ap.add_argument("--objects", nargs="+", default=None,
                    help="只审这些物体（可写 contactdb+alarm_clock 或裸名 alarm_clock）")
    ap.add_argument("--start", type=int, default=0, help="跳过队列前 N 条（默认 0）")
    ap.add_argument("--limit", type=int, default=None, help="本次最多审 N 条（默认不限）")
    ap.add_argument("--shuffle", action="store_true", default=False,
                    help="打乱顺序（默认按 metadata 原顺序，便于按物体连续看）")
    ap.add_argument("--seed", type=int, default=0, help="--shuffle 的随机种子（默认 0）")
    ap.add_argument("--decisions", default=str(DEFAULT_DECISIONS),
                    help="决策 JSONL（逐条追加，可断点恢复）")
    ap.add_argument("--view", default="auto", choices=list(VIEW_ORDER),
                    help="初始视角（auto/front/top/side），运行中 v 键切换")
    ap.add_argument("--record-dir", default=None,
                    help="可选：把每次决策时的窗口截图（或 --offscreen 的逐条渲染）存到这里")
    ap.add_argument("--dry-run", action="store_true", default=False,
                    help="不弹窗：只按已有 decisions 重放并生成 .pt（CI / 断点校验）")
    ap.add_argument("--review-all", action="store_true", default=False,
                    help="强制重审已决策样本（以最新决策为准，会覆盖对应 uid 的旧决策）")
    ap.add_argument("--offscreen", action="store_true", default=False,
                    help="配合 --record-dir：无交互，逐条离屏渲染出 PNG（不是交互模式）")
    ap.add_argument("--index-csvs", nargs="+", default=None,
                    help="构造 uid 用的索引 CSV（默认自动找 filtered_grasps/index_<prefix>.csv）")
    ap.add_argument("--uid-mode", default="sha", choices=["sha", "positional"],
                    help="uid 口径：sha=(sha256:grasp_index)，与 convert_to_cmap 一致；positional=仅兜底")
    ap.add_argument("--fk-backend", default="numpy", choices=["numpy", "pk"],
                    help="手部 FK 后端：numpy（默认，几何正确）或 pk（本机 fork 有 ±0.01rad 偏置）")
    ap.add_argument("--object-geom", default="auto", choices=["auto", "points", "mesh"],
                    help="物体几何：auto=PointCloud 优先，缺失回退 STL；points/mesh 强制")
    ap.add_argument("--point-size", type=float, default=4.0, help="点云点大小（默认 4）")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--save-every", type=int, default=20,
                    help="每 N 条决策顺手落一次盘（崩溃保护，默认 20）")
    ap.add_argument("--summary-json", default=str(SUMMARY_JSON),
                    help="审核统计摘要写到哪（默认 new_grasp_filter/logs/manual_review_v5.json）")
    ap.add_argument("--rejected-output", default=None,
                    help="可选：把 decision==reject 的样本另存一份 .pt（默认不写；给了路径才写，"
                         "免得覆盖已有文件）")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.rejected_output == "":
        args.rejected_output = None

    dataset = Path(args.dataset).resolve()
    args.dataset = dataset
    payload, metadata = load_dataset(dataset)
    log(f"[load] {dataset.name}: metadata={len(metadata)}  "
        f"info.num_total={payload['info'][ROBOT_NAME]['num_total']}  "
        f"robot={payload['info'][ROBOT_NAME]['robot_name']}")

    specs = resolve_index_specs(args.prefixes, args.index_csvs)
    log(f"[index] {[ (str(p), pre) for p, pre in specs ]}")
    samples = build_samples(metadata, specs, args.uid_mode, dataset)

    store = DecisionStore(Path(args.decisions)).load()
    kept, rej = store.counts()
    log(f"[resume] decisions={store.path}  行数={len(store.records)}  已 keep={kept}  已 reject={rej}  "
        f"未审={len(samples) - kept - rej}" + (f"  跳过坏行={store.bad_lines}" if store.bad_lines else ""))
    unknown = set(store.by_uid) - {s.uid for s in samples}
    if unknown:
        log(f"[warn] decisions 里有 {len(unknown)} 个 uid 不在当前数据集（换数据集了？）例: "
            f"{sorted(unknown)[:3]}")

    queue, qinfo = select_queue(samples, store, args)
    log(f"[queue] {qinfo}  本次待审 {len(queue)} 条")

    if args.dry_run:
        log("[dry-run] 不弹窗，只按已有 decisions 重放")
        meta = write_outputs(samples, store, Path(args.output), dataset, Path(args.decisions),
                             args, quiet=False)
        _print_table(meta)
        return 0

    if args.offscreen:
        if not args.record_dir:
            raise SystemExit("[error] --offscreen 需要配合 --record-dir <dir>（只出图，不交互）")
        hand = HandGeometry(args.fk_backend)
        log(f"[hand] fk_backend={args.fk_backend}  links={len(hand.link_names)}  tris={hand.n_tris}  "
            f"urdf={hand.urdf.path}")
        objs = ObjectGeometry(args.object_geom, args.point_size)
        files = OffscreenRenderer(queue, store, args, hand, objs).run()
        log(f"[offscreen] 写了 {len(files)} 张 PNG -> {args.record_dir}")
        return 0

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise SystemExit(
            "[error] 没有 DISPLAY（无法交互）。请用 `ssh -X`/`ssh -Y`/VNC，"
            "或用 --dry-run（只重放 decisions）／--offscreen --record-dir <dir>（只出图）。")

    hand = HandGeometry(args.fk_backend)
    log(f"[hand] fk_backend={args.fk_backend}  links={len(hand.link_names)}  tris={hand.n_tris}  "
        f"urdf={hand.urdf.path}")
    if args.fk_backend == "pk":
        log("[warn] 本机 pytorch_kinematics 是 fork（每个 revolute 关节 +±0.01 rad），"
            "画面上的手会比 q 真实姿态偏最多约 4 cm（REPORT_v4 §2.2）")
    objs = ObjectGeometry(args.object_geom, args.point_size)
    viewer = ReviewViewer(queue, samples, store, args, hand, objs)
    meta = viewer.run()
    _print_table(meta)
    return 0


def _print_table(meta: dict) -> None:
    rows = meta["diff_vs_source"]
    touched = [n for n in sorted(rows) if rows[n]["kept"] or rows[n]["rejected"]]
    log("")
    log(f"{'object_name':32s} {'before':>7s} {'kept':>6s} {'rejected':>9s} {'unreviewed':>11s}")
    for name in touched:
        r = rows[name]
        log(f"{name:32s} {r['before']:7d} {r['kept']:6d} {r['rejected']:9d} {r['unreviewed']:11d}")
    log(f"{'TOTAL (reviewed objects)':32s} {sum(rows[n]['before'] for n in touched):7d} "
        f"{sum(rows[n]['kept'] for n in touched):6d} {sum(rows[n]['rejected'] for n in touched):9d} "
        f"{sum(rows[n]['unreviewed'] for n in touched):11d}")
    log(f"{'TOTAL (whole dataset)':32s} {meta['num_before']:7d} {meta['num_kept']:6d} "
        f"{meta['num_rejected']:9d} {meta['num_unreviewed']:11d}")
    fully = meta["objects_fully_rejected"]
    if meta["num_kept"] and any(rows[n]["rejected"] or rows[n]["kept"] for n in rows):
        rest = [n for n in fully if n not in touched]
        if rest:
            log(f"[note] 尚未审核（因此整体为 0）的物体 {len(rest)} 个: {rest[:6]}"
                + (" …" if len(rest) > 6 else ""))
    log(f"[out] {meta['output']}   .meta.json / summary json 已更新")


if __name__ == "__main__":
    sys.exit(main())
