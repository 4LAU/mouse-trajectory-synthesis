"""Corpus replay + very low perturbation (0.3% noise scale)."""
from __future__ import annotations

import math

import numpy as np

from experiments._common import Trajectory
from experiments.corpus_replay import generate_path as corpus_generate

_rng = np.random.default_rng()
_NOISE_SCALE = 0.003


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

    noise = _smooth_noise(n, _NOISE_SCALE * total_dist)

    result = []
    for i in range(n):
        x, y, t = base[i]
        result.append((x + noise[i, 0], y + noise[i, 1], t))

    result[0] = (start_x, start_y, base[0][2])
    result[-1] = (end_x, end_y, base[-1][2])

    return result
