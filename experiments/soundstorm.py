"""Enhanced corpus rotate with duration-matched donor selection.

Improves on basic corpus rotate by matching donors on both distance
AND trajectory length (as a proxy for duration/speed). This gives
donors with similar kinematic profiles, fixing the direction-change
distribution mismatch that drives most of the classifier signal.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from experiments._common import DurationModel, Trajectory

_DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
_POOL_DIR = Path("training")
_offsets = np.load(_POOL_DIR / "full_pool_offsets.npy")
_meta = np.load(_POOL_DIR / "full_pool_meta.npy")
_flat = np.load(_POOL_DIR / "pool_flat_i16.npy", mmap_mode="r")
_t_rel = np.load(_POOL_DIR / "pool_t_rel_f32.npy", mmap_mode="r")
_N = len(_offsets) - 1
_pool_log_dist = _meta[:, 0]
_pool_lens = np.diff(_offsets).astype(np.int32)

_duration = DurationModel(_DATA_DIR)
_HZ = 125.0

_DIST_THRESH = 0.15
_LEN_RATIO_THRESH = float(os.environ.get("SS_LEN_RATIO", "0.3"))
_ENDPOINT_K = int(os.environ.get("SS_K", "5"))
_rng = np.random.default_rng()

print(f"[soundstorm] enhanced corpus rotate, "
      f"len_ratio_thresh={_LEN_RATIO_THRESH}, K={_ENDPOINT_K}, "
      f"pool={_N:,}")


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

    total_duration = _duration.sample(query_log_dist)
    expected_len = max(5, int(round(total_duration * _HZ)))

    dist_ok = np.abs(_pool_log_dist - query_log_dist) < _DIST_THRESH

    if _LEN_RATIO_THRESH > 0:
        log_len_ratio = np.abs(np.log(np.maximum(_pool_lens, 1) / expected_len))
        len_ok = log_len_ratio < _LEN_RATIO_THRESH
        candidates = np.where(dist_ok & len_ok)[0]

        if len(candidates) < 5:
            candidates = np.where(dist_ok & (log_len_ratio < 0.7))[0]
    else:
        candidates = np.where(dist_ok)[0]

    if len(candidates) < 10:
        candidates = np.where(np.abs(_pool_log_dist - query_log_dist) < 0.3)[0]
    if len(candidates) < 5:
        candidates = np.where(np.abs(_pool_log_dist - query_log_dist) < 0.5)[0]
    if len(candidates) < 3:
        candidates = np.arange(_N)

    dist_diff = np.abs(_pool_log_dist[candidates] - query_log_dist)
    K = min(_ENDPOINT_K, len(candidates))
    if K < len(candidates):
        best_idx = np.argpartition(dist_diff, K)[:K]
    else:
        best_idx = np.arange(len(candidates))

    chosen_local = int(best_idx[_rng.integers(0, len(best_idx))])
    chosen = int(candidates[chosen_local])

    lo, hi = int(_offsets[chosen]), int(_offsets[chosen + 1])
    n_pts = hi - lo
    if n_pts < 3:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    xy = np.array(_flat[lo:hi], dtype=np.float64)
    t_abs = np.array(_t_rel[lo:hi], dtype=np.float64)

    ox, oy = xy[0, 0], xy[0, 1]
    xy[:, 0] -= ox
    xy[:, 1] -= oy

    donor_dx = xy[-1, 0]
    donor_dy = xy[-1, 1]
    donor_dist = math.hypot(donor_dx, donor_dy)
    if donor_dist < 1e-6:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    donor_angle = math.atan2(donor_dy, donor_dx)
    rotate = query_angle - donor_angle
    scale = total_dist / donor_dist

    cos_r = math.cos(rotate)
    sin_r = math.sin(rotate)

    rx = xy[:, 0] * cos_r - xy[:, 1] * sin_r
    ry = xy[:, 0] * sin_r + xy[:, 1] * cos_r
    out_x = rx * scale + start_x
    out_y = ry * scale + start_y

    err_x = end_x - out_x[-1]
    err_y = end_y - out_y[-1]
    if err_x * err_x + err_y * err_y > 0.01:
        frac = np.linspace(0.0, 1.0, n_pts)
        out_x += err_x * frac
        out_y += err_y * frac

    result = [
        (float(out_x[i]), float(out_y[i]), float(t_abs[i]))
        for i in range(n_pts)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, result[-1][2])

    return result
