"""Corpus replay: kNN lookup from a demo pool of human trajectories.

Approach:
  Angle+distance matched corpus lookup with endpoint-proximity ranking.
  No model - pure retrieval from recorded human data.

Expected AUC: ~0.60 (50K demo pool, n=2000). Scales with pool size:
  10K → 0.70, 50K → 0.60, 500K → 0.53, full 4.16M → ~0.50.

Key insight:
  Replaying real human trajectories with translate-only is nearly undetectable
  given enough pool coverage, but is not generative - it just proves the
  feature set is sound.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

from experiments._common import Trajectory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
_args, _ = _parser.parse_known_args()

_DATA_DIR = Path(_args.data_dir)

# ---------------------------------------------------------------------------
# Load demo pool
# ---------------------------------------------------------------------------
_pool = np.load(_DATA_DIR / "demo_pool.npz", allow_pickle=False)
_flat = _pool["flat"]        # (total_pts, 2) float coords
_offsets = _pool["offsets"]   # (N+1,) trajectory boundaries
_meta = _pool["meta"]        # (N, 3) columns: log_dist, cos_angle, sin_angle
_t = _pool["t"]              # (total_pts,) timestamps

_N = len(_offsets) - 1
_pool_log_dist = _meta[:, 0]
_pool_angles = np.arctan2(_meta[:, 2], _meta[:, 1])  # (N,) in (-pi, pi]

# Pre-compute displacement vectors for endpoint-proximity ranking
_pool_dist = np.exp(_pool_log_dist)
_pool_dx = _pool_dist * _meta[:, 1]  # dist * cos(angle)
_pool_dy = _pool_dist * _meta[:, 2]  # dist * sin(angle)

_ANGLE_THRESH = math.pi / 3   # +/-60 degrees
_DIST_THRESH = 0.5             # +/-0.5 in log-space
_ENDPOINT_K = 3                # top-K for diversity
_rng = np.random.default_rng()

print(f"[corpus_replay] Demo pool: {_N} trajectories")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angle_diff(a: float, angles: np.ndarray) -> np.ndarray:
    """Minimum angular distance handling wraparound."""
    d = np.abs(angles - a)
    return np.minimum(d, 2.0 * math.pi - d)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> Trajectory:
    """Retrieve the closest human trajectory from the demo pool."""
    dx = end_x - start_x
    dy = end_y - start_y
    query_angle = math.atan2(dy, dx)
    query_log_dist = math.log(max(math.hypot(dx, dy), 1.0))

    # Stage 1: angle + distance filtering
    ang_diff = _angle_diff(query_angle, _pool_angles)
    dist_diff = np.abs(_pool_log_dist - query_log_dist)

    candidates = np.where((ang_diff < _ANGLE_THRESH) & (dist_diff < _DIST_THRESH))[0]

    # Fallback cascade
    if len(candidates) < 10:
        candidates = np.where((ang_diff < math.pi / 3) & (dist_diff < 1.0))[0]
    if len(candidates) < 5:
        candidates = np.where(dist_diff < 1.0)[0]
    if len(candidates) == 0:
        candidates = np.arange(_N)

    # Stage 2: rank by endpoint proximity, pick from top-K
    c_dx = _pool_dx[candidates]
    c_dy = _pool_dy[candidates]
    endpoint_err = (c_dx - dx) ** 2 + (c_dy - dy) ** 2

    K = min(_ENDPOINT_K, len(candidates))
    if K < len(candidates):
        best_idx = np.argpartition(endpoint_err, K)[:K]
    else:
        best_idx = np.arange(len(candidates))

    chosen = int(candidates[best_idx[_rng.integers(0, len(best_idx))]])

    # Extract trajectory
    lo, hi = int(_offsets[chosen]), int(_offsets[chosen + 1])
    xy = _flat[lo:hi]
    t_abs = _t[lo:hi]

    # Translate so trajectory starts at (start_x, start_y)
    shift_x = start_x - float(xy[0, 0])
    shift_y = start_y - float(xy[0, 1])
    out_x = xy[:, 0].astype(float) + shift_x
    out_y = xy[:, 1].astype(float) + shift_y

    n_pts = hi - lo
    result = [(float(out_x[i]), float(out_y[i]), float(t_abs[i])) for i in range(n_pts)]

    # Cubic-ease endpoint correction over last 25% of points
    err_x = end_x - result[-1][0]
    err_y = end_y - result[-1][1]
    if err_x * err_x + err_y * err_y > 4.0:  # skip if already within 2px
        n_correct = max(3, n_pts // 4)
        start_idx = n_pts - n_correct
        for i in range(start_idx, n_pts):
            frac = (i - start_idx + 1) / n_correct
            weight = frac ** 3  # cubic ease-in
            x, y, t = result[i]
            result[i] = (x + err_x * weight, y + err_y * weight, t)

    return result
