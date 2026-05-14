"""Corpus replay v2: magnitude-weighted endpoint correction.

Same retrieval as corpus_replay but replaces cubic-ease endpoint
correction (concentrated in last 25%) with magnitude-weighted
correction spread across all moving steps.
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
    raise FileNotFoundError("Full pool required")

_N = len(_offsets) - 1
_pool_log_dist = _meta[:, 0]
_pool_angles = np.arctan2(_meta[:, 2], _meta[:, 1])
_pool_dist = np.exp(_pool_log_dist)
_pool_dx = _pool_dist * _meta[:, 1]
_pool_dy = _pool_dist * _meta[:, 2]

_rng = np.random.default_rng()
print(f"[corpus_replay_v2] Pool: {_N:,} trajectories")


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
    query_angle = math.atan2(dy, dx)
    query_log_dist = math.log(max(math.hypot(dx, dy), 1.0))

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

    shift_x = start_x - xy[0, 0]
    shift_y = start_y - xy[0, 1]
    out_x = xy[:, 0] + shift_x
    out_y = xy[:, 1] + shift_y

    err_x = end_x - out_x[-1]
    err_y = end_y - out_y[-1]

    if err_x * err_x + err_y * err_y > 1.0:
        diffs_x = np.diff(out_x)
        diffs_y = np.diff(out_y)
        step_mags = np.sqrt(diffs_x ** 2 + diffs_y ** 2)
        moving = step_mags > 0.3
        total_moving = step_mags[moving].sum()

        if total_moving > 0.1:
            cum_cx, cum_cy = 0.0, 0.0
            for i in range(n_pts - 1):
                if moving[i]:
                    w = step_mags[i] / total_moving
                    cum_cx += err_x * w
                    cum_cy += err_y * w
                out_x[i + 1] += cum_cx
                out_y[i + 1] += cum_cy

    result = [
        (float(out_x[i]), float(out_y[i]), float(t_abs[i]))
        for i in range(n_pts)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, t_abs[-1])

    return result
