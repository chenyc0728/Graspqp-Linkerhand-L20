"""Fast approximate signed-distance field for an object mesh (voxel based).

Why this exists: SAPIEN's reported ``separation`` for *triangle-mesh vs convex*
contacts carries a systematic bias of several millimetres (measured: a nominal
0 mm gap is reported as +6 mm, a 10 mm overlap as -2.4 mm -- see REPORT.md
"Calibration").  That is fine for "is there a contact" but not for millimetre
level penetration thresholds, so penetration depth is computed geometrically
against the exact object mesh with this SDF, while SAPIEN remains responsible
for the rigid-body/lift physics.

The grid comes straight from ``trimesh``'s voxelizer (2 mm default) so that the
voxel ``transform`` keeps care of the world <- index mapping, and the signed
distance is a Euclidean distance transform on the filled occupancy grid.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage


class VoxelSDF:
    def __init__(self, mesh_path, pitch: float = 0.002):
        self.mesh = trimesh.load(str(mesh_path), force="mesh")
        self.pitch = float(pitch)
        vox = self.mesh.voxelized(pitch=pitch).fill()
        self.vox = vox
        self.matrix = vox.matrix.astype(bool)  # True = inside
        self.T = np.asarray(vox.transform, dtype=np.float64)
        self.T_inv = np.linalg.inv(self.T)
        d_out = ndimage.distance_transform_edt(~self.matrix) * self.pitch
        d_in = ndimage.distance_transform_edt(self.matrix) * self.pitch
        self.sdf = d_out - d_in  # < 0 inside, > 0 outside
        self._shape = np.asarray(self.matrix.shape)

    def query(self, points: np.ndarray) -> np.ndarray:
        """Signed distance (m, negative = inside) for (N, 3) world points."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        f = (self.T_inv @ h.T).T[:, :3]
        f = np.clip(f, 0, self._shape - 1.0001)
        i0 = np.floor(f).astype(int)
        frac = f - i0
        out = np.zeros(len(pts))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (
                        (frac[:, 0] if dx else 1 - frac[:, 0])
                        * (frac[:, 1] if dy else 1 - frac[:, 1])
                        * (frac[:, 2] if dz else 1 - frac[:, 2])
                    )
                    out += w * self.sdf[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        return out
