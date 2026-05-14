"""ZIMT + stall injection v2: path-aware stall placement + cubic timing.

Improvements over v1:
  - Stalls placed based on curvature (where path turns sharply), not just
    borrowed from donor. This couples stalls to the actual generated path.
  - Cubic-ease endpoint correction instead of linear.
  - Timing from donor but with jitter to avoid exact matching.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from experiments._common import Trajectory
from experiments.zimt import generate_path as zimt_generate

# ---------------------------------------------------------------------------
# Load pool for donor timing
# ---------------------------------------------------------------------------
_POOL_DIR = Path("training")
_offsets = np.load(_POOL_DIR / "full_pool_offsets.npy")
_meta = np.load(_POOL_DIR / "full_pool_meta.npy")
_flat = np.load(_POOL_DIR / "pool_flat_i16.npy", mmap_mode="r")
_t_rel = np.load(_POOL_DIR / "pool_t_rel_f32.npy", mmap_mode="r")
_N = len(_offsets) - 1
_pool_log_dist = _meta[:, 0]
_pool_angles = np.arctan2(_meta[:, 2], _meta[:, 1])

_rng = np.random.default_rng()
print(f"[zimt_stall_inject_v2] Pool: {_N:,} trajectories")


def _angle_diff(a: float, angles: np.ndarray) -> np.ndarray:
    d = np.abs(angles - a)
    return np.minimum(d, 2.0 * math.pi - d)


def _find_donor(log_dist: float, angle: float) -> int:
    ang_diff = _angle_diff(angle, _pool_angles)
    dist_diff = np.abs(_pool_log_dist - log_dist)
    candidates = np.where((ang_diff < 0.5) & (dist_diff < 0.3))[0]
    if len(candidates) < 10:
        candidates = np.where((ang_diff < 1.0) & (dist_diff < 0.5))[0]
    if len(candidates) < 5:
        candidates = np.where(dist_diff < 1.0)[0]
    if len(candidates) == 0:
        candidates = np.arange(_N)
    return int(candidates[_rng.integers(0, len(candidates))])


def _get_donor_stall_rate_and_timing(donor_idx: int):
    """Get stall rate and normalized timestamps from donor."""
    lo = int(_offsets[donor_idx])
    hi = int(_offsets[donor_idx + 1])
    n = hi - lo
    if n < 3:
        return 0.0, None

    xy = np.array(_flat[lo:hi], dtype=np.float64)
    t = np.array(_t_rel[lo:hi], dtype=np.float64)

    diffs = np.diff(xy, axis=0)
    speeds = np.sqrt((diffs ** 2).sum(axis=1))
    stall_rate = (speeds < 0.5).mean()

    t_normalized = t / max(t[-1], 1e-6)
    return float(stall_rate), t_normalized


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

    base = zimt_generate(start_x, start_y, end_x, end_y)
    n_base = len(base)
    if n_base < 5:
        return base

    log_dist = math.log(max(total_dist, 1.0))
    angle = math.atan2(dy, dx)
    donor_idx = _find_donor(log_dist, angle)
    stall_rate, donor_t_norm = _get_donor_stall_rate_and_timing(donor_idx)

    total_duration = base[-1][2]
    if total_duration < 0.01:
        return base

    # Compute curvature at each step to decide where to place stalls
    positions = [(base[i][0], base[i][1]) for i in range(n_base)]
    n_steps = n_base - 1

    curvatures = np.zeros(n_steps)
    for i in range(1, n_steps):
        dx1 = positions[i][0] - positions[i - 1][0]
        dy1 = positions[i][1] - positions[i - 1][1]
        dx2 = positions[i + 1][0] - positions[i][0]
        dy2 = positions[i + 1][1] - positions[i][1]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        s1 = math.hypot(dx1, dy1)
        s2 = math.hypot(dx2, dy2)
        denom = max(s1 * s2, 1e-8)
        curvatures[i] = cross / denom

    # Place stalls at high-curvature points, matching donor's stall rate
    n_stalls = max(0, int(round(stall_rate * n_steps)))
    if n_stalls > 0 and n_stalls < n_steps:
        # Prefer high-curvature locations for stalls
        stall_priority = curvatures + _rng.uniform(0, 0.1, n_steps)
        stall_indices = set(np.argsort(stall_priority)[-n_stalls:])
    else:
        stall_indices = set()

    # Build new trajectory with stalls injected
    new_positions = [positions[0]]
    for i in range(n_steps):
        if i in stall_indices:
            new_positions.append(new_positions[-1])
        else:
            new_positions.append(positions[i + 1])

    # Endpoint correction with cubic ease
    n = len(new_positions)
    actual_end_x, actual_end_y = new_positions[-1]
    err_x = end_x - actual_end_x
    err_y = end_y - actual_end_y

    n_correct = max(3, n // 4)
    for i in range(n_correct):
        idx = n - n_correct + i
        frac = (i + 1) / n_correct
        weight = frac ** 3
        px, py = new_positions[idx]
        new_positions[idx] = (px + err_x * weight, py + err_y * weight)

    # Timing: borrow donor timestamps with small jitter
    if donor_t_norm is not None and len(donor_t_norm) >= 3:
        donor_ts = donor_t_norm * total_duration
        timestamps = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(donor_ts)),
            donor_ts,
        )
        # Add small jitter (1% of dt)
        dt = total_duration / max(n - 1, 1)
        jitter = _rng.normal(0, dt * 0.01, n)
        jitter[0] = 0
        jitter[-1] = 0
        timestamps = np.clip(timestamps + jitter, 0, total_duration)
        timestamps.sort()
    else:
        timestamps = np.linspace(0, total_duration, n)

    result = [
        (float(new_positions[i][0]), float(new_positions[i][1]), float(timestamps[i]))
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, float(timestamps[-1]))

    return result
