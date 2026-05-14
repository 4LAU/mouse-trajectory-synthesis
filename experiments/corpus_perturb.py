"""Corpus replay + minimal perturbation.

Retrieve a real trajectory, then add small correlated noise to positions.
The noise is spatially correlated (smooth) and scaled to ~1-2% of total
displacement to minimally disturb kinematic features while making each
trajectory technically novel.

This tests the hypothesis: how much perturbation can a real trajectory
tolerate before it becomes detectable?
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from experiments._common import Trajectory
from experiments.corpus_replay import generate_path as corpus_generate

_rng = np.random.default_rng()

# Noise scale as fraction of total distance
_NOISE_SCALE = 0.01


def _smooth_noise(n: int, scale: float) -> np.ndarray:
    """Generate spatially correlated noise using cumulative Gaussian."""
    raw = _rng.normal(0, scale / max(n, 1) ** 0.5, (n, 2))
    smoothed = np.cumsum(raw, axis=0)
    # Taper at endpoints to preserve start/end
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

    # Get a real trajectory from corpus replay
    base = corpus_generate(start_x, start_y, end_x, end_y)
    n = len(base)
    if n < 3:
        return base

    # Add smooth position noise
    noise = _smooth_noise(n, _NOISE_SCALE * total_dist)

    result = []
    for i in range(n):
        x, y, t = base[i]
        result.append((x + noise[i, 0], y + noise[i, 1], t))

    # Preserve exact start and end
    result[0] = (start_x, start_y, base[0][2])
    result[-1] = (end_x, end_y, base[-1][2])

    return result
