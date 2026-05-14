"""Corpus replay + displacement noise (preserves curvature structure).

Instead of adding noise to positions (which distorts curvature), add
noise to step displacements (dx, dy). This preserves the curvature
structure better because the noise is local, not cumulative.

The displacement noise is scaled relative to each step's magnitude
to avoid creating obviously artificial speed patterns.
"""
from __future__ import annotations

import math

import numpy as np

from experiments._common import Trajectory
from experiments.corpus_replay import generate_path as corpus_generate

_rng = np.random.default_rng()
_NOISE_SCALE = 0.08  # fraction of step displacement


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

    # Compute step displacements
    positions = np.array([(base[i][0], base[i][1]) for i in range(n)])
    timestamps = [base[i][2] for i in range(n)]

    diffs = np.diff(positions, axis=0)
    step_mags = np.sqrt((diffs ** 2).sum(axis=1))

    # Add proportional noise to displacements
    # For stall steps (magnitude < 0.5), don't add noise (preserve stalls)
    noisy_diffs = diffs.copy()
    for i in range(len(diffs)):
        if step_mags[i] > 0.5:
            # Noise proportional to step magnitude, perpendicular bias
            angle = math.atan2(diffs[i, 1], diffs[i, 0])
            perp_angle = angle + math.pi / 2
            # Mostly perpendicular noise (preserves speed, changes direction slightly)
            para = _rng.normal(0, _NOISE_SCALE * 0.3)
            perp = _rng.normal(0, _NOISE_SCALE)
            noisy_diffs[i, 0] += step_mags[i] * (para * math.cos(angle) + perp * math.cos(perp_angle))
            noisy_diffs[i, 1] += step_mags[i] * (para * math.sin(angle) + perp * math.sin(perp_angle))

    # Reconstruct positions
    new_positions = np.zeros_like(positions)
    new_positions[0] = positions[0]
    new_positions[1:] = positions[0] + np.cumsum(noisy_diffs, axis=0)

    # Endpoint correction: cubic ease on last 25%
    actual_end = new_positions[-1]
    err_x = end_x - actual_end[0]
    err_y = end_y - actual_end[1]
    n_correct = max(3, n // 4)
    for i in range(n_correct):
        idx = n - n_correct + i
        frac = ((i + 1) / n_correct) ** 3
        new_positions[idx, 0] += err_x * frac
        new_positions[idx, 1] += err_y * frac

    result = [
        (float(new_positions[i, 0]), float(new_positions[i, 1]), timestamps[i])
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, base[-1][2])

    return result
