"""Corpus replay with rotation+scale — match any donor to any query direction.

Unlike standard corpus replay (translate-only, angle-filtered), this approach:
1. Matches donors by log-distance only (no angle filter)
2. Rotates + scales the donor trajectory to exactly match the query endpoints
3. Preserves scale-invariant kinematic features: velocity_skewness,
   time_to_peak_velocity, angular_velocity, path_efficiency, num_direction_changes

This is generative: the rotated+scaled trajectory is novel (not in the corpus).
The full pool is accessible for any query direction, giving better donor matches.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from experiments._common import Trajectory

_POOL_DIR = Path("training")
_offsets = np.load(_POOL_DIR / "full_pool_offsets.npy")
_meta = np.load(_POOL_DIR / "full_pool_meta.npy")
_flat = np.load(_POOL_DIR / "pool_flat_i16.npy", mmap_mode="r")
_t_rel = np.load(_POOL_DIR / "pool_t_rel_f32.npy", mmap_mode="r")
_N = len(_offsets) - 1
_pool_log_dist = _meta[:, 0]
_pool_dist = np.exp(_pool_log_dist)
_pool_cos = _meta[:, 1]
_pool_sin = _meta[:, 2]
_pool_dx = _pool_dist * _pool_cos
_pool_dy = _pool_dist * _pool_sin

_DIST_THRESH = 0.15
_ENDPOINT_K = 5
_rng = np.random.default_rng()

print(f"[corpus_rotate] Pool: {_N:,} trajectories, dist_thresh={_DIST_THRESH}")


def generate_path(
    start_x: float, start_y: float,
    end_x: float, end_y: float,
) -> Trajectory:
    dx = end_x - start_x
    dy = end_y - start_y
    total_dist = math.hypot(dx, dy)

    if total_dist < 1.0:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    query_log_dist = math.log(total_dist)
    query_angle = math.atan2(dy, dx)

    # Match by distance only (no angle filter)
    dist_diff = np.abs(_pool_log_dist - query_log_dist)
    candidates = np.where(dist_diff < _DIST_THRESH)[0]

    if len(candidates) < 10:
        candidates = np.where(dist_diff < 0.3)[0]
    if len(candidates) < 5:
        candidates = np.where(dist_diff < 0.5)[0]
    if len(candidates) < 3:
        candidates = np.arange(_N)

    # Rank by distance match, pick from top-K
    c_dist_err = dist_diff[candidates]
    K = min(_ENDPOINT_K, len(candidates))
    if K < len(candidates):
        best_idx = np.argpartition(c_dist_err, K)[:K]
    else:
        best_idx = np.arange(len(candidates))

    chosen_local = int(best_idx[_rng.integers(0, len(best_idx))])
    chosen = int(candidates[chosen_local])

    # Extract donor trajectory
    lo, hi = int(_offsets[chosen]), int(_offsets[chosen + 1])
    n_pts = hi - lo
    if n_pts < 3:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    xy = np.array(_flat[lo:hi], dtype=np.float64)
    t_abs = np.array(_t_rel[lo:hi], dtype=np.float64)

    # Translate to origin
    ox, oy = xy[0, 0], xy[0, 1]
    xy[:, 0] -= ox
    xy[:, 1] -= oy

    # Compute donor's endpoint and angle
    donor_dx = xy[-1, 0]
    donor_dy = xy[-1, 1]
    donor_dist = math.hypot(donor_dx, donor_dy)
    if donor_dist < 1e-6:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    donor_angle = math.atan2(donor_dy, donor_dx)

    # Rotation angle and scale factor
    rotate = query_angle - donor_angle
    scale = total_dist / donor_dist

    cos_r = math.cos(rotate)
    sin_r = math.sin(rotate)

    # Apply rotation + scale + translate to start
    out_x = np.empty(n_pts)
    out_y = np.empty(n_pts)
    for i in range(n_pts):
        rx = xy[i, 0] * cos_r - xy[i, 1] * sin_r
        ry = xy[i, 0] * sin_r + xy[i, 1] * cos_r
        out_x[i] = rx * scale + start_x
        out_y[i] = ry * scale + start_y

    # Small endpoint correction for floating point
    err_x = end_x - out_x[-1]
    err_y = end_y - out_y[-1]
    if err_x * err_x + err_y * err_y > 0.01:
        for i in range(n_pts):
            frac = i / max(n_pts - 1, 1)
            out_x[i] += err_x * frac
            out_y[i] += err_y * frac

    result = [
        (float(out_x[i]), float(out_y[i]), float(t_abs[i]))
        for i in range(n_pts)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, result[-1][2])

    return result
