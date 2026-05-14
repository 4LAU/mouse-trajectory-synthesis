"""ZIMT with time-warped speed profile.

Shifts peak speed earlier (~0.45 -> ~0.35) by warping the time-to-position
mapping via beta CDF. Preserves ZIMT's natural speed variations — only shifts
WHEN they occur, doesn't replace the profile.

Base: ZIMT magcorr. Config via env vars:
  RETIMED_PEAK     - mean peak time fraction (default 0.35)
  RETIMED_PEAK_STD - per-trajectory variation (default 0.12)
  RETIMED_WARP     - warp concentration k (default 3.0, 0=off)
  RETIMED_SMOOTH   - position smoothing window (default 1=off)
"""
from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import beta as beta_dist

from experiments._common import Trajectory
from experiments.zimt_magcorr import generate_path as _magcorr_generate

_PEAK = float(os.environ.get("RETIMED_PEAK", "0.35"))
_PEAK_STD = float(os.environ.get("RETIMED_PEAK_STD", "0.12"))
_WARP_K = float(os.environ.get("RETIMED_WARP", "3.0"))
_SMOOTH = int(os.environ.get("RETIMED_SMOOTH", "1"))
_HZ = 125.0
_DT = 1.0 / _HZ

_rng = np.random.default_rng()

print(f"[zimt_retimed] peak={_PEAK}+/-{_PEAK_STD}, warp_k={_WARP_K}, smooth={_SMOOTH}")


def generate_path(
    start_x: float, start_y: float,
    end_x: float, end_y: float,
) -> Trajectory:
    base = _magcorr_generate(start_x, start_y, end_x, end_y)
    if len(base) < 6:
        return base

    pts = np.array(base, dtype=np.float64)
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    N = len(x)
    total_dur = t[-1]
    if total_dur <= 0:
        return base

    # Optional path smoothing (centered moving average)
    if _SMOOTH > 1:
        h = _SMOOTH // 2
        xs, ys = x.copy(), y.copy()
        for i in range(h, N - h):
            xs[i] = np.mean(x[i - h : i + h + 1])
            ys[i] = np.mean(y[i - h : i + h + 1])
        xs[0], ys[0] = start_x, start_y
        xs[-1], ys[-1] = end_x, end_y
        x, y = xs, ys

    if _WARP_K > 0:
        peak = float(np.clip(_rng.normal(_PEAK, _PEAK_STD), 0.10, 0.70))
        a = 1.0 + peak * _WARP_K
        b = 1.0 + (1.0 - peak) * _WARP_K

        t_norm = np.linspace(0.0, 1.0, N)
        g = beta_dist.cdf(t_norm, a, b) * (N - 1)

        old_idx = np.arange(N, dtype=np.float64)
        cs_x = CubicSpline(old_idx, x, bc_type='clamped')
        cs_y = CubicSpline(old_idx, y, bc_type='clamped')
        x = cs_x(g)
        y = cs_y(g)

    # Endpoint correction
    err_x = end_x - x[-1]
    err_y = end_y - y[-1]
    if err_x * err_x + err_y * err_y > 0.01:
        sm = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        moving = sm > 0.3
        total_mov = sm[moving].sum() if moving.any() else 0.0
        if total_mov > 0.1:
            cx, cy = 0.0, 0.0
            for i in range(len(sm)):
                if moving[i]:
                    w = sm[i] / total_mov
                    cx += err_x * w
                    cy += err_y * w
                x[i + 1] += cx
                y[i + 1] += cy

    t_out = np.linspace(0.0, total_dur, N)
    result = [(float(x[i]), float(y[i]), float(t_out[i])) for i in range(N)]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, result[-1][2])
    return result
