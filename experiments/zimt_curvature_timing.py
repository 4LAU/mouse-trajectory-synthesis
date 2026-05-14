"""ZIMT + curvature-coupled timing + stall injection.

Key idea: humans slow down at turns and speed up on straight segments.
Instead of borrowing timing from a donor (which breaks path-timing
coupling), compute the velocity profile FROM the generated path's own
curvature, using the empirical curvature-speed relationship from pool data.

Also injects stalls at high-curvature points matching pool stall rates.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from experiments._common import Trajectory
from experiments.zimt import generate_path as zimt_generate

# ---------------------------------------------------------------------------
# Load pool to learn curvature-speed relationship + stall statistics
# ---------------------------------------------------------------------------
_POOL_DIR = Path("training")
_offsets = np.load(_POOL_DIR / "full_pool_offsets.npy")
_meta = np.load(_POOL_DIR / "full_pool_meta.npy")
_flat = np.load(_POOL_DIR / "pool_flat_i16.npy", mmap_mode="r")
_t_rel = np.load(_POOL_DIR / "pool_t_rel_f32.npy", mmap_mode="r")
_N = len(_offsets) - 1
_pool_log_dist = _meta[:, 0]

_DT = 1.0 / 125.0
_rng = np.random.default_rng()

# Learn empirical curvature→relative-speed mapping from pool
# Sample trajectories and compute (curvature, speed) pairs
print("[zimt_curvature_timing] Learning curvature-speed mapping...", flush=True)
_N_SAMPLE = min(50000, _N)
_sample_idx = _rng.choice(_N, _N_SAMPLE, replace=False)

_curvature_speed_pairs = []
_stall_rates = []

for idx in _sample_idx:
    lo = int(_offsets[idx])
    hi = int(_offsets[idx + 1])
    n = hi - lo
    if n < 5:
        continue

    xy = np.array(_flat[lo:hi], dtype=np.float64)
    t = np.array(_t_rel[lo:hi], dtype=np.float64)

    diffs = np.diff(xy, axis=0)
    speeds = np.sqrt((diffs ** 2).sum(axis=1))

    stall_mask = speeds < 0.5
    _stall_rates.append(stall_mask.mean())

    # Compute curvature at each interior point
    for i in range(1, len(diffs) - 1):
        dx1, dy1 = diffs[i - 1]
        dx2, dy2 = diffs[i]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        s1 = math.hypot(dx1, dy1)
        s2 = math.hypot(dx2, dy2)
        if s1 > 0.1 and s2 > 0.1:
            curv = cross / (s1 * s2)
            # Normalize speed by trajectory mean speed
            mean_spd = speeds[max(0, i-2):i+3].mean()
            if mean_spd > 0.1:
                rel_speed = speeds[i] / mean_spd
                _curvature_speed_pairs.append((min(curv, 2.0), rel_speed))

_MEAN_STALL_RATE = float(np.mean(_stall_rates))

# Bin curvature → mean relative speed
_curvature_speed = np.array(_curvature_speed_pairs)
_N_CURV_BINS = 20
_curv_edges = np.linspace(0, 1.5, _N_CURV_BINS + 1)
_curv_speed_map = np.ones(_N_CURV_BINS)
for b in range(_N_CURV_BINS):
    mask = (_curvature_speed[:, 0] >= _curv_edges[b]) & (_curvature_speed[:, 0] < _curv_edges[b + 1])
    if mask.sum() >= 10:
        _curv_speed_map[b] = float(np.median(_curvature_speed[mask, 1]))

# Also compute progress → relative speed (bell curve: accelerate then decelerate)
_progress_speed = np.zeros(20)
_progress_counts = np.zeros(20)
for idx in _sample_idx[:10000]:
    lo = int(_offsets[idx])
    hi = int(_offsets[idx + 1])
    n = hi - lo
    if n < 5:
        continue
    xy = np.array(_flat[lo:hi], dtype=np.float64)
    diffs = np.diff(xy, axis=0)
    speeds = np.sqrt((diffs ** 2).sum(axis=1))
    mean_spd = speeds.mean()
    if mean_spd < 0.1:
        continue
    for i in range(len(speeds)):
        progress = i / max(len(speeds) - 1, 1)
        bin_idx = min(int(progress * 19), 19)
        _progress_speed[bin_idx] += speeds[i] / mean_spd
        _progress_counts[bin_idx] += 1

_progress_speed_map = np.where(_progress_counts > 0, _progress_speed / _progress_counts, 1.0)

del _curvature_speed_pairs, _curvature_speed, _stall_rates
print(f"[zimt_curvature_timing] Mean stall rate: {_MEAN_STALL_RATE:.3f}, "
      f"curv-speed range: [{_curv_speed_map.min():.2f}, {_curv_speed_map.max():.2f}]")


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

    # Generate base path with ZIMT
    base = zimt_generate(start_x, start_y, end_x, end_y)
    n_base = len(base)
    if n_base < 5:
        return base

    total_duration = base[-1][2]
    if total_duration < 0.01:
        return base

    positions = np.array([(base[i][0], base[i][1]) for i in range(n_base)])
    n_steps = n_base - 1

    # Compute curvature at each step
    diffs = np.diff(positions, axis=0)
    step_speeds = np.sqrt((diffs ** 2).sum(axis=1))
    curvatures = np.zeros(n_steps)
    for i in range(1, n_steps):
        dx1, dy1 = diffs[i - 1]
        dx2, dy2 = diffs[i]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        s1 = math.hypot(dx1, dy1)
        s2 = math.hypot(dx2, dy2)
        if s1 > 0.01 and s2 > 0.01:
            curvatures[i] = min(cross / (s1 * s2), 2.0)

    # Inject stalls at high-curvature points
    stall_rate = _MEAN_STALL_RATE + _rng.normal(0, 0.02)
    stall_rate = max(0, min(stall_rate, 0.15))
    n_stalls = int(round(stall_rate * n_steps))

    stall_mask = np.zeros(n_steps, dtype=bool)
    if n_stalls > 0:
        stall_priority = curvatures + _rng.uniform(0, 0.05, n_steps)
        stall_indices = np.argsort(stall_priority)[-n_stalls:]
        stall_mask[stall_indices] = True

    # Build positions with stalls
    new_positions = [positions[0].tolist()]
    for i in range(n_steps):
        if stall_mask[i]:
            new_positions.append(list(new_positions[-1]))
        else:
            new_positions.append(positions[i + 1].tolist())

    new_positions = np.array(new_positions)
    n = len(new_positions)

    # Endpoint correction (cubic ease on last 25%)
    err_x = end_x - new_positions[-1, 0]
    err_y = end_y - new_positions[-1, 1]
    n_correct = max(3, n // 4)
    for i in range(n_correct):
        idx = n - n_correct + i
        frac = ((i + 1) / n_correct) ** 3
        new_positions[idx, 0] += err_x * frac
        new_positions[idx, 1] += err_y * frac

    # Compute curvature-coupled timing
    new_diffs = np.diff(new_positions, axis=0)
    new_step_dists = np.sqrt((new_diffs ** 2).sum(axis=1))
    new_curvatures = np.zeros(n - 1)
    for i in range(1, n - 1):
        dx1, dy1 = new_diffs[i - 1]
        dx2, dy2 = new_diffs[i]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        s1 = math.hypot(dx1, dy1)
        s2 = math.hypot(dx2, dy2)
        if s1 > 0.01 and s2 > 0.01:
            new_curvatures[i] = min(cross / (s1 * s2), 2.0)

    # For each step, compute target relative speed from:
    # 1. Curvature (slow at turns)
    # 2. Progress (bell curve: accelerate then decelerate)
    target_rel_speed = np.ones(n - 1)
    for i in range(n - 1):
        # Curvature factor
        curv = new_curvatures[i]
        curv_bin = min(int(curv / 1.5 * _N_CURV_BINS), _N_CURV_BINS - 1)
        curv_factor = _curv_speed_map[curv_bin]

        # Progress factor
        progress = i / max(n - 2, 1)
        prog_bin = min(int(progress * 19), 19)
        prog_factor = _progress_speed_map[prog_bin]

        # Combine (geometric mean)
        target_rel_speed[i] = math.sqrt(curv_factor * prog_factor)

        # Stall steps should be slow
        if new_step_dists[i] < 0.01:
            target_rel_speed[i] = 0.01

    # Convert relative speed to time intervals
    # dt_i ∝ step_dist_i / target_speed_i
    dt_raw = np.zeros(n - 1)
    for i in range(n - 1):
        if target_rel_speed[i] > 0.001:
            dt_raw[i] = new_step_dists[i] / target_rel_speed[i]
        else:
            dt_raw[i] = _DT

    dt_sum = dt_raw.sum()
    if dt_sum > 1e-8:
        dt_scaled = dt_raw * (total_duration / dt_sum)
    else:
        dt_scaled = np.full(n - 1, total_duration / (n - 1))

    timestamps = np.concatenate([[0], np.cumsum(dt_scaled)])

    result = [
        (float(new_positions[i, 0]), float(new_positions[i, 1]), float(timestamps[i]))
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, float(timestamps[-1]))

    return result
