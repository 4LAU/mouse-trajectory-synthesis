"""Corpus replay with similarity transform (translate + rotate + scale).

Instead of translating the donor and then correcting the endpoint with
a cubic ease, apply a full similarity transform so the donor's start
and end map exactly to the query's start and end. This preserves all
kinematic features (speed/acceleration ratios, curvature, direction
changes) while eliminating endpoint correction artifacts.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from experiments._common import Trajectory

_POOL_DIR = Path(os.environ.get("POOL_DIR", ""))
_full_pool_path = _POOL_DIR / "full_pool_offsets.npy" if _POOL_DIR.name else Path("training/full_pool_offsets.npy")

if _full_pool_path.exists():
    _pool_dir = _full_pool_path.parent
    _offsets = np.load(_pool_dir / "full_pool_offsets.npy")
    _meta = np.load(_pool_dir / "full_pool_meta.npy")
    _flat = np.load(_pool_dir / "pool_flat_i16.npy", mmap_mode="r")
    _t = np.load(_pool_dir / "pool_t_rel_f32.npy", mmap_mode="r")
else:
    raise FileNotFoundError("Full pool required for corpus_sim")

_N = len(_offsets) - 1
_pool_log_dist = _meta[:, 0]
_pool_angles = np.arctan2(_meta[:, 2], _meta[:, 1])
_pool_dist = np.exp(_pool_log_dist)
_pool_dx = _pool_dist * _meta[:, 1]
_pool_dy = _pool_dist * _meta[:, 2]

_rng = np.random.default_rng()
print(f"[corpus_sim] Pool: {_N:,} trajectories")


def _angle_diff(a, angles):
    d = np.abs(angles - a)
    return np.minimum(d, 2.0 * math.pi - d)


def generate_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> Trajectory:
    dx = end_x - start_x
    dy = end_y - start_y
    query_dist = math.hypot(dx, dy)
    query_angle = math.atan2(dy, dx)
    query_log_dist = math.log(max(query_dist, 1.0))

    ang_diff = _angle_diff(query_angle, _pool_angles)
    dist_diff = np.abs(_pool_log_dist - query_log_dist)

    candidates = np.where((ang_diff < math.pi / 3) & (dist_diff < 0.5))[0]
    if len(candidates) < 10:
        candidates = np.where((ang_diff < math.pi / 3) & (dist_diff < 1.0))[0]
    if len(candidates) < 5:
        candidates = np.where(dist_diff < 1.0)[0]
    if len(candidates) == 0:
        candidates = np.arange(_N)

    c_dx = _pool_dx[candidates]
    c_dy = _pool_dy[candidates]
    endpoint_err = (c_dx - dx) ** 2 + (c_dy - dy) ** 2

    K = min(3, len(candidates))
    if K < len(candidates):
        best_idx = np.argpartition(endpoint_err, K)[:K]
    else:
        best_idx = np.arange(len(candidates))

    chosen = int(candidates[best_idx[_rng.integers(0, len(best_idx))]])

    lo, hi = int(_offsets[chosen]), int(_offsets[chosen + 1])
    xy = np.array(_flat[lo:hi], dtype=np.float64)
    t_abs = np.array(_t[lo:hi], dtype=np.float64)

    n_pts = len(xy)
    if n_pts < 3:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    # Similarity transform: map donor start→end to query start→end
    donor_sx, donor_sy = xy[0, 0], xy[0, 1]
    donor_ex, donor_ey = xy[-1, 0], xy[-1, 1]
    donor_dx = donor_ex - donor_sx
    donor_dy = donor_ey - donor_sy
    donor_dist = math.hypot(donor_dx, donor_dy)

    if donor_dist < 1e-6:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    # Scale factor
    scale = query_dist / donor_dist

    # Rotation angle
    donor_angle = math.atan2(donor_dy, donor_dx)
    rot = query_angle - donor_angle
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)

    # Apply: translate to origin, rotate, scale, translate to query start
    centered_x = xy[:, 0] - donor_sx
    centered_y = xy[:, 1] - donor_sy

    new_x = (centered_x * cos_r - centered_y * sin_r) * scale + start_x
    new_y = (centered_x * sin_r + centered_y * cos_r) * scale + start_y

    # Scale timestamps proportionally to the scale factor
    # (longer distance → proportionally longer time, preserving speed ratios)
    result = [
        (float(new_x[i]), float(new_y[i]), float(t_abs[i]))
        for i in range(n_pts)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, t_abs[-1])

    return result
