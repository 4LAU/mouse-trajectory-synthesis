"""ZIMT + pool stall injection.

Takes ZIMT's generated path and injects stall patterns (exact zero
displacements) borrowed from a similar real trajectory in the pool.
Also borrows the donor's timestamp distribution for coupled timing.

Rationale:
  ZIMT's #1 discriminative feature is curvature (which comes from stalls)
  and time_to_peak_velocity. Real trajectories have ~6% exact (0,0) stalls.
  ZIMT produces almost none. Injecting real stall patterns + timing from
  a similar pool trajectory should fix both issues simultaneously.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from experiments._common import Trajectory
from experiments.zimt import generate_path as zimt_generate

# ---------------------------------------------------------------------------
# Load pool for donor stall patterns
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
print(f"[zimt_stall_inject] Pool: {_N:,} trajectories for stall donation")


def _angle_diff(a: float, angles: np.ndarray) -> np.ndarray:
    d = np.abs(angles - a)
    return np.minimum(d, 2.0 * math.pi - d)


def _find_donor(log_dist: float, angle: float) -> int:
    """Find a similar pool trajectory by distance and angle."""
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


def _get_donor_stalls_and_timing(donor_idx: int):
    """Extract stall mask and relative timestamps from a donor trajectory."""
    lo = int(_offsets[donor_idx])
    hi = int(_offsets[donor_idx + 1])
    n = hi - lo
    if n < 3:
        return None, None

    xy = np.array(_flat[lo:hi], dtype=np.float64)
    t = np.array(_t_rel[lo:hi], dtype=np.float64)

    diffs = np.diff(xy, axis=0)
    speeds = np.sqrt((diffs ** 2).sum(axis=1))

    stall_mask = speeds < 0.5
    t_normalized = t / max(t[-1], 1e-6)

    return stall_mask, t_normalized


def generate_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> Trajectory:
    """Generate ZIMT trajectory, then inject donor stalls and timing."""
    dx = end_x - start_x
    dy = end_y - start_y
    total_dist = math.hypot(dx, dy)

    if total_dist < 1.0:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    # Generate base trajectory with ZIMT
    base = zimt_generate(start_x, start_y, end_x, end_y)
    n_base = len(base)
    if n_base < 5:
        return base

    # Find a similar donor trajectory
    log_dist = math.log(max(total_dist, 1.0))
    angle = math.atan2(dy, dx)
    donor_idx = _find_donor(log_dist, angle)
    stall_mask, donor_t_norm = _get_donor_stalls_and_timing(donor_idx)

    if stall_mask is None or len(stall_mask) < 3:
        return base

    total_duration = base[-1][2]
    if total_duration < 0.01:
        return base

    # Resample donor stall pattern to match base trajectory length
    n_donor_steps = len(stall_mask)
    n_base_steps = n_base - 1

    # Map each base step to a donor step by progress fraction
    base_stalls = np.zeros(n_base_steps, dtype=bool)
    for i in range(n_base_steps):
        progress = i / max(n_base_steps - 1, 1)
        donor_step = min(int(progress * (n_donor_steps - 1)), n_donor_steps - 1)
        base_stalls[i] = stall_mask[donor_step]

    # Inject stalls: where donor has stalls, set displacement to zero
    positions = [(base[0][0], base[0][1])]
    for i in range(n_base_steps):
        if base_stalls[i]:
            # Stall: repeat previous position
            positions.append(positions[-1])
        else:
            positions.append((base[i + 1][0], base[i + 1][1]))

    # Endpoint correction: redistribute the displacement we lost to stalls
    n = len(positions)
    actual_end_x, actual_end_y = positions[-1]
    err_x = end_x - actual_end_x
    err_y = end_y - actual_end_y

    # Only correct non-stall positions in the last 30%
    n_correct_zone = max(3, n // 3)
    correction_indices = []
    for i in range(n - n_correct_zone, n):
        step = i - 1
        if step >= 0 and step < n_base_steps and not base_stalls[step]:
            correction_indices.append(i)

    if len(correction_indices) > 0:
        for rank, idx in enumerate(correction_indices):
            frac = (rank + 1) / len(correction_indices)
            px, py = positions[idx]
            positions[idx] = (px + err_x * frac, py + err_y * frac)
    else:
        # Fallback: correct last 20% of all positions
        n_correct = max(3, n // 5)
        for i in range(n_correct):
            idx = n - n_correct + i
            frac = (i + 1) / n_correct
            px, py = positions[idx]
            positions[idx] = (px + err_x * frac, py + err_y * frac)

    # Borrow donor timing: interpolate donor timestamps to match our length
    donor_timestamps = donor_t_norm * total_duration
    timestamps = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(donor_timestamps)),
        donor_timestamps,
    )

    result = [
        (float(positions[i][0]), float(positions[i][1]), float(timestamps[i]))
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, float(timestamps[-1]))

    return result
