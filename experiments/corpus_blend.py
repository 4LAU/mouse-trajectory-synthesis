"""Corpus blend: retrieve K similar trajectories and blend them.

Instead of replaying a single trajectory (corpus replay), retrieve K=3
similar trajectories and create a weighted blend. The blended trajectory
is novel (doesn't exist in the pool), making this genuinely generative.
The model's "weights" are the blending coefficients.

The key insight: each real trajectory has correct stalls, timing, and
velocity profile. Blending K trajectories preserves these properties
while creating unique paths.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from experiments._common import Trajectory

# ---------------------------------------------------------------------------
# Load pool
# ---------------------------------------------------------------------------
_POOL_DIR = Path("training")
_offsets = np.load(_POOL_DIR / "full_pool_offsets.npy")
_meta = np.load(_POOL_DIR / "full_pool_meta.npy")
_flat = np.load(_POOL_DIR / "pool_flat_i16.npy", mmap_mode="r")
_t_rel = np.load(_POOL_DIR / "pool_t_rel_f32.npy", mmap_mode="r")
_N = len(_offsets) - 1
_pool_log_dist = _meta[:, 0]
_pool_angles = np.arctan2(_meta[:, 2], _meta[:, 1])
_pool_dist = np.exp(_pool_log_dist)
_pool_dx = _pool_dist * _meta[:, 1]
_pool_dy = _pool_dist * _meta[:, 2]

_rng = np.random.default_rng()
print(f"[corpus_blend] Pool: {_N:,} trajectories")

_K = 3
_ANGLE_THRESH = math.pi / 4


def _angle_diff(a: float, angles: np.ndarray) -> np.ndarray:
    d = np.abs(angles - a)
    return np.minimum(d, 2.0 * math.pi - d)


def _find_k_donors(dx: float, dy: float, k: int) -> list[int]:
    """Find K similar pool trajectories."""
    query_angle = math.atan2(dy, dx)
    query_log_dist = math.log(max(math.hypot(dx, dy), 1.0))

    ang_diff = _angle_diff(query_angle, _pool_angles)
    dist_diff = np.abs(_pool_log_dist - query_log_dist)

    candidates = np.where((ang_diff < _ANGLE_THRESH) & (dist_diff < 0.3))[0]
    if len(candidates) < k * 3:
        candidates = np.where((ang_diff < math.pi / 3) & (dist_diff < 0.5))[0]
    if len(candidates) < k * 2:
        candidates = np.where(dist_diff < 1.0)[0]
    if len(candidates) < k:
        candidates = np.arange(_N)

    # Rank by endpoint proximity
    c_dx = _pool_dx[candidates]
    c_dy = _pool_dy[candidates]
    endpoint_err = (c_dx - dx) ** 2 + (c_dy - dy) ** 2

    n_top = min(k * 3, len(candidates))
    if n_top < len(candidates):
        best_idx = np.argpartition(endpoint_err, n_top)[:n_top]
    else:
        best_idx = np.arange(len(candidates))

    # Pick K from top candidates
    chosen_idx = _rng.choice(best_idx, size=min(k, len(best_idx)), replace=False)
    return [int(candidates[i]) for i in chosen_idx]


def _extract_trajectory(idx: int):
    """Extract positions and timestamps from pool trajectory."""
    lo = int(_offsets[idx])
    hi = int(_offsets[idx + 1])
    n = hi - lo
    if n < 3:
        return None, None
    xy = np.array(_flat[lo:hi], dtype=np.float64)
    t = np.array(_t_rel[lo:hi], dtype=np.float64)
    return xy, t


def _resample(xy: np.ndarray, t: np.ndarray, n_out: int):
    """Resample trajectory to n_out evenly-spaced progress points."""
    n = len(xy)
    # Arc-length parameterization
    diffs = np.diff(xy, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cumlen[-1]
    if total_len < 1e-6:
        return np.tile(xy[0], (n_out, 1)), np.linspace(t[0], t[-1], n_out)

    # Normalize to [0, 1]
    s = cumlen / total_len
    s_out = np.linspace(0, 1, n_out)

    x_out = np.interp(s_out, s, xy[:, 0])
    y_out = np.interp(s_out, s, xy[:, 1])
    t_out = np.interp(s_out, s, t)

    return np.stack([x_out, y_out], axis=-1), t_out


def generate_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> Trajectory:
    dx = end_x - start_x
    dy = end_y - start_y
    total_dist = math.hypot(dx, dy)

    if total_dist < 1.0:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    # Find K donor trajectories
    donors = _find_k_donors(dx, dy, _K)

    trajs = []
    for d_idx in donors:
        xy, t = _extract_trajectory(d_idx)
        if xy is not None:
            trajs.append((xy, t))

    if len(trajs) == 0:
        dt = 0.008
        return [(start_x, start_y, 0.0), (end_x, end_y, dt)]

    if len(trajs) == 1:
        xy, t = trajs[0]
        shift_x = start_x - xy[0, 0]
        shift_y = start_y - xy[0, 1]
        n = len(xy)
        result = [
            (float(xy[i, 0] + shift_x), float(xy[i, 1] + shift_y), float(t[i]))
            for i in range(n)
        ]
        result[-1] = (end_x, end_y, result[-1][2])
        return result

    # Determine output length (median of donor lengths)
    n_out = int(np.median([len(xy) for xy, _ in trajs]))
    n_out = max(5, min(n_out, 500))

    # Resample all donors to same length and translate to start at origin
    resampled = []
    timestamps_list = []
    for xy, t in trajs:
        xy_r, t_r = _resample(xy, t, n_out)
        # Translate to origin
        xy_r -= xy_r[0]
        resampled.append(xy_r)
        timestamps_list.append(t_r)

    # Generate random blending weights (Dirichlet)
    weights = _rng.dirichlet(np.ones(len(resampled)) * 2.0)

    # Blend positions
    blended = np.zeros((n_out, 2))
    for w, xy_r in zip(weights, resampled):
        blended += w * xy_r

    # Blend timestamps
    blended_t = np.zeros(n_out)
    for w, t_r in zip(weights, timestamps_list):
        blended_t += w * t_r

    # Scale blended trajectory to match query displacement
    actual_end = blended[-1]
    actual_dist = math.hypot(actual_end[0], actual_end[1])
    if actual_dist > 1e-6:
        # Rotate and scale to match target
        target_angle = math.atan2(dy, dx)
        actual_angle = math.atan2(actual_end[1], actual_end[0])
        rotate = target_angle - actual_angle
        scale = total_dist / actual_dist

        cos_r = math.cos(rotate)
        sin_r = math.sin(rotate)
        for i in range(n_out):
            x, y = blended[i]
            blended[i, 0] = (x * cos_r - y * sin_r) * scale
            blended[i, 1] = (x * sin_r + y * cos_r) * scale

    # Translate to start position
    blended[:, 0] += start_x
    blended[:, 1] += start_y

    # Endpoint correction: cubic ease on last 25%
    err_x = end_x - blended[-1, 0]
    err_y = end_y - blended[-1, 1]
    if err_x * err_x + err_y * err_y > 4.0:
        n_correct = max(3, n_out // 4)
        for i in range(n_correct):
            idx = n_out - n_correct + i
            frac = ((i + 1) / n_correct) ** 3
            blended[idx, 0] += err_x * frac
            blended[idx, 1] += err_y * frac

    result = [
        (float(blended[i, 0]), float(blended[i, 1]), float(blended_t[i]))
        for i in range(n_out)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, result[-1][2])

    if len(result) < 3:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    return result
