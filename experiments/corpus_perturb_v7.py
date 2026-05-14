"""Corpus replay + magnitude scaling perturbation.

Instead of adding displacement noise (which changes direction counts),
scale each step's displacement by a random factor near 1.0. This
preserves direction changes exactly but subtly alters the speed profile.

Also adds very small perpendicular jitter for spatial diversity.
"""
from __future__ import annotations

import math

import numpy as np

from experiments._common import Trajectory
from experiments.corpus_replay import generate_path as corpus_generate

_rng = np.random.default_rng()
_SCALE_STD = 0.03       # std of multiplicative scale factor
_PERP_NOISE = 0.015     # tiny perpendicular jitter


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
    if n < 5:
        return base

    positions = np.array([(base[i][0], base[i][1]) for i in range(n)])
    timestamps = [base[i][2] for i in range(n)]

    diffs = np.diff(positions, axis=0)
    step_mags = np.sqrt((diffs ** 2).sum(axis=1))
    n_steps = len(diffs)

    noisy_diffs = diffs.copy()
    for i in range(n_steps):
        if step_mags[i] > 0.3:
            # Multiplicative scale (preserves direction)
            scale = 1.0 + _rng.normal(0, _SCALE_STD)
            scale = max(0.9, min(1.1, scale))
            noisy_diffs[i] *= scale

            # Tiny perpendicular jitter
            angle = math.atan2(diffs[i, 1], diffs[i, 0])
            perp_angle = angle + math.pi / 2
            perp = _rng.normal(0, _PERP_NOISE) * step_mags[i]
            noisy_diffs[i, 0] += perp * math.cos(perp_angle)
            noisy_diffs[i, 1] += perp * math.sin(perp_angle)

    new_positions = np.zeros_like(positions)
    new_positions[0] = positions[0]
    new_positions[1:] = positions[0] + np.cumsum(noisy_diffs, axis=0)

    # Magnitude-weighted endpoint correction
    actual_end = new_positions[-1]
    err_x = end_x - actual_end[0]
    err_y = end_y - actual_end[1]

    if abs(err_x) > 0.01 or abs(err_y) > 0.01:
        moving_mask = step_mags > 0.3
        total_moving_mag = step_mags[moving_mask].sum()
        if total_moving_mag > 0.1:
            cum_corr_x, cum_corr_y = 0.0, 0.0
            for i in range(n_steps):
                if moving_mask[i]:
                    weight = step_mags[i] / total_moving_mag
                    cum_corr_x += err_x * weight
                    cum_corr_y += err_y * weight
                new_positions[i + 1, 0] += cum_corr_x
                new_positions[i + 1, 1] += cum_corr_y

    result = [
        (float(new_positions[i, 0]), float(new_positions[i, 1]), timestamps[i])
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, base[-1][2])

    return result
