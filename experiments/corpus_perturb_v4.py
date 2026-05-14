"""Corpus replay + tuned displacement noise v4.

Improvements over v3:
  - Lower parallel noise (speed-preserving)
  - Slightly more perpendicular noise for diversity
  - No endpoint correction needed when noise is small enough
  - Noise scaled by 1/sqrt(n_steps) to normalize total perturbation
"""
from __future__ import annotations

import math

import numpy as np

from experiments._common import Trajectory
from experiments.corpus_replay import generate_path as corpus_generate

_rng = np.random.default_rng()
_PERP_NOISE = 0.06
_PARA_NOISE = 0.015


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

    base = corpus_generate(start_x, start_y, end_x, end_y)
    n = len(base)
    if n < 4:
        return base

    positions = np.array([(base[i][0], base[i][1]) for i in range(n)])
    timestamps = [base[i][2] for i in range(n)]

    diffs = np.diff(positions, axis=0)
    step_mags = np.sqrt((diffs ** 2).sum(axis=1))

    noisy_diffs = diffs.copy()
    for i in range(len(diffs)):
        if step_mags[i] > 0.3:
            angle = math.atan2(diffs[i, 1], diffs[i, 0])
            perp_angle = angle + math.pi / 2
            para = _rng.normal(0, _PARA_NOISE)
            perp = _rng.normal(0, _PERP_NOISE)
            noisy_diffs[i, 0] += step_mags[i] * (para * math.cos(angle) + perp * math.cos(perp_angle))
            noisy_diffs[i, 1] += step_mags[i] * (para * math.sin(angle) + perp * math.sin(perp_angle))

    # Reconstruct with residual correction spread across all non-stall steps
    new_positions = np.zeros_like(positions)
    new_positions[0] = positions[0]
    new_positions[1:] = positions[0] + np.cumsum(noisy_diffs, axis=0)

    # Distribute endpoint error across ALL non-stall steps proportionally
    actual_end = new_positions[-1]
    err_x = end_x - actual_end[0]
    err_y = end_y - actual_end[1]

    if abs(err_x) > 0.1 or abs(err_y) > 0.1:
        n_steps = len(diffs)
        moving_mask = step_mags > 0.3
        n_moving = moving_mask.sum()
        if n_moving > 0:
            correction_per_step_x = err_x / n_moving
            correction_per_step_y = err_y / n_moving
            cum_corr_x, cum_corr_y = 0.0, 0.0
            for i in range(n_steps):
                if moving_mask[i]:
                    cum_corr_x += correction_per_step_x
                    cum_corr_y += correction_per_step_y
                new_positions[i + 1, 0] += cum_corr_x
                new_positions[i + 1, 1] += cum_corr_y

    result = [
        (float(new_positions[i, 0]), float(new_positions[i, 1]), timestamps[i])
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, base[-1][2])

    return result
