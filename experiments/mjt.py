"""Minimum-Jerk Trajectory (MJT) generator with realistic noise.

Physics-based mouse trajectory synthesis using the minimum-jerk model
with Ornstein-Uhlenbeck heading noise for naturalistic curvature.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from experiments._common import DurationModel, Trajectory

_TRAIN_DIR = Path("./training")
_HZ = 125.0
_duration = DurationModel(_TRAIN_DIR, std_mult=0.7)
_rng = np.random.default_rng()

# Human feature statistics for calibration
# angular_velocity_std ~ 3.5, curvature_mean ~ 0.11 (median)
# path_efficiency ~ 0.949 (median)


def _mjt_speed_profile(n_points: int, peak_speed: float,
                        skew_alpha: float = 0.0) -> np.ndarray:
    """Generate minimum-jerk speed profile.

    skew_alpha shifts the peak earlier (positive) or later (negative).
    Human TTPV ≈ 0.345 means the peak is at 34.5% of duration.
    Standard MJT peaks at 50%. We need skew_alpha ~ 0.3 to shift to 35%.
    """
    tau = np.linspace(0, 1, n_points)
    if skew_alpha > 0:
        tau = np.power(tau, 1.0 / (1.0 + skew_alpha))
    speed = 30 * tau**2 * (1 - tau)**2
    speed *= peak_speed / (speed.max() + 1e-12)
    return speed


def generate_path(
    start_x: float, start_y: float,
    end_x: float, end_y: float,
) -> Trajectory:
    dx = end_x - start_x
    dy = end_y - start_y
    total_dist = math.hypot(dx, dy)

    if total_dist < 1.0:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    log_dist = math.log(total_dist)
    target_angle = math.atan2(dy, dx)
    duration = _duration.sample(log_dist)
    n_points = max(5, int(round(duration * _HZ)))

    peak_speed = total_dist / duration * 1.875
    skew_alpha = _rng.normal(0.3, 0.1)
    speed = _mjt_speed_profile(n_points, peak_speed, max(skew_alpha, 0.05))

    speed_noise = 1.0 + _rng.normal(0, 0.03, n_points)
    speed *= np.clip(speed_noise, 0.85, 1.15)

    dt = duration / n_points
    heading = np.full(n_points, target_angle)

    ou_sigma = _rng.uniform(0.3, 0.8)
    ou_theta = _rng.uniform(3.0, 8.0)
    dh = np.zeros(n_points)
    for i in range(1, n_points):
        dh[i] = dh[i-1] - ou_theta * dh[i-1] * dt + ou_sigma * _rng.normal() * math.sqrt(dt)

    heading = target_angle + dh

    vx = speed * np.cos(heading) * dt
    vy = speed * np.sin(heading) * dt
    cx = np.cumsum(vx)
    cy = np.cumsum(vy)

    actual_end_x = cx[-1] if len(cx) > 0 else 0
    actual_end_y = cy[-1] if len(cy) > 0 else 0
    err_x = (end_x - start_x) - actual_end_x
    err_y = (end_y - start_y) - actual_end_y

    correction_x = err_x * np.linspace(0, 1, n_points)
    correction_y = err_y * np.linspace(0, 1, n_points)
    cx += correction_x
    cy += correction_y

    px = start_x + cx
    py = start_y + cy

    times = np.arange(n_points) * dt
    return list(zip(px.tolist(), py.tolist(), times.tolist()))
