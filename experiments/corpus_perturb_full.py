"""Corpus replay + position noise + timestamp jitter.

Applies both spatial and temporal perturbation to pool trajectories.
Position noise: 0.3% of total distance, spatially correlated.
Time jitter: 0.5% of step duration, preserving monotonicity.
"""
from __future__ import annotations

import math

import numpy as np

from experiments._common import Trajectory
from experiments.corpus_replay import generate_path as corpus_generate

_rng = np.random.default_rng()
_POS_NOISE = 0.003
_TIME_NOISE = 0.005


def _smooth_noise(n: int, scale: float) -> np.ndarray:
    raw = _rng.normal(0, scale / max(n, 1) ** 0.5, (n, 2))
    smoothed = np.cumsum(raw, axis=0)
    taper = np.sin(np.linspace(0, math.pi, n)) ** 2
    smoothed[:, 0] *= taper
    smoothed[:, 1] *= taper
    return smoothed


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
    if n < 3:
        return base

    # Position noise
    noise = _smooth_noise(n, _POS_NOISE * total_dist)

    # Time jitter
    total_duration = base[-1][2]
    dt_mean = total_duration / max(n - 1, 1)
    time_jitter = _rng.normal(0, _TIME_NOISE * dt_mean, n)
    time_jitter[0] = 0
    time_jitter[-1] = 0

    result = []
    for i in range(n):
        x, y, t = base[i]
        result.append((x + noise[i, 0], y + noise[i, 1], t + time_jitter[i]))

    # Fix monotonicity of timestamps
    for i in range(1, n):
        if result[i][2] <= result[i-1][2]:
            result[i] = (result[i][0], result[i][1], result[i-1][2] + 1e-6)

    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, base[-1][2])

    return result
