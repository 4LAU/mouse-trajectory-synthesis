"""Corpus replay + smooth correlated displacement noise.

Uses a low-frequency perturbation (GP-like smooth curve) instead of
independent per-step noise. This avoids adding spurious direction
changes while still making each trajectory unique.
"""
from __future__ import annotations

import math

import numpy as np

from experiments._common import Trajectory
from experiments.corpus_replay import generate_path as corpus_generate

_rng = np.random.default_rng()
_AMPLITUDE = 0.03  # max perpendicular deviation as fraction of total_dist
_N_HARMONICS = 3    # number of sine waves to superpose


def _smooth_perturbation(n: int, total_dist: float) -> np.ndarray:
    """Generate smooth perpendicular perturbation using sum of low-freq sines."""
    t = np.linspace(0, 1, n)
    perturb = np.zeros(n)
    for _ in range(_N_HARMONICS):
        freq = _rng.uniform(0.5, 3.0)
        phase = _rng.uniform(0, 2 * math.pi)
        amp = _rng.normal(0, _AMPLITUDE * total_dist / _N_HARMONICS)
        perturb += amp * np.sin(2 * math.pi * freq * t + phase)

    # Taper at endpoints
    taper = np.sin(np.linspace(0, math.pi, n))
    perturb *= taper
    return perturb


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

    # Direction of travel
    travel_angle = math.atan2(dy, dx)
    perp_x = -math.sin(travel_angle)
    perp_y = math.cos(travel_angle)

    # Smooth perpendicular perturbation
    perturb = _smooth_perturbation(n, total_dist)

    new_positions = positions.copy()
    new_positions[:, 0] += perturb * perp_x
    new_positions[:, 1] += perturb * perp_y

    # Fix endpoints exactly
    new_positions[0] = [start_x, start_y]

    # Distribute endpoint error smoothly
    actual_end = new_positions[-1]
    err_x = end_x - actual_end[0]
    err_y = end_y - actual_end[1]
    if abs(err_x) > 0.01 or abs(err_y) > 0.01:
        progress = np.linspace(0, 1, n)
        new_positions[:, 0] += err_x * progress
        new_positions[:, 1] += err_y * progress

    result = [
        (float(new_positions[i, 0]), float(new_positions[i, 1]), timestamps[i])
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, base[-1][2])

    return result
