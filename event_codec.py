"""Canonical event-stream encode for WS7.

Turns a raw pool trajectory (integer pixel positions, ms-resolution timestamps)
into the (dt, dx, dy) event sequence the model trains on. This is the same
transform validated in experiments/event_replay.py MODE=split, which scored
0.50-0.52 through the full eval pipeline. Training prep and any future re-eval
must use this one implementation so representations never drift.

Steps:
1. Clamp negative dt to 0 (out-of-order timestamps count as duplicates).
2. Merge events with dt == 0 into the previous event (sum displacements).
3. Split any jump with |dx| or |dy| > vocab_max into collinear integer
   sub-events sharing the dt equally. The piecewise-linear path is unchanged
   up to sub-pixel rounding, so nothing is clipped.
"""
from __future__ import annotations

import numpy as np

VOCAB_MAX = 63


def encode_events(xy: np.ndarray, t: np.ndarray, vocab_max: int = VOCAB_MAX):
    """Encode one trajectory.

    Parameters
    ----------
    xy : (n, 2) integer pixel positions
    t : (n,) timestamps in seconds

    Returns
    -------
    dt : (m,) float64 seconds, all > 0
    dx, dy : (m,) int64 in [-vocab_max, vocab_max]
    or None if fewer than 2 events survive.
    """
    dt = np.maximum(np.diff(t.astype(np.float64)), 0.0)
    d = np.diff(xy.astype(np.int64), axis=0)
    if len(dt) == 0:
        return None

    keep = dt > 0
    groups = np.cumsum(keep)
    n_groups = int(groups[-1]) if len(groups) else 0
    if n_groups < 2:
        return None
    gidx = np.maximum(groups - 1, 0)
    dx_m = np.bincount(gidx, weights=d[:, 0], minlength=n_groups)
    dy_m = np.bincount(gidx, weights=d[:, 1], minlength=n_groups)
    dt_m = np.bincount(gidx, weights=dt, minlength=n_groups)

    over = np.maximum(np.abs(dx_m), np.abs(dy_m)) > vocab_max
    if over.any():
        dx_s, dy_s, dt_s = [], [], []
        for j in range(n_groups):
            if not over[j]:
                dx_s.append(dx_m[j]); dy_s.append(dy_m[j]); dt_s.append(dt_m[j])
                continue
            k = int(np.ceil(max(abs(dx_m[j]), abs(dy_m[j])) / (vocab_max - 1)))
            fx = np.round(np.linspace(0, dx_m[j], k + 1))
            fy = np.round(np.linspace(0, dy_m[j], k + 1))
            dx_s.extend(np.diff(fx)); dy_s.extend(np.diff(fy))
            dt_s.extend([dt_m[j] / k] * k)
        dx_m = np.asarray(dx_s); dy_m = np.asarray(dy_s); dt_m = np.asarray(dt_s)

    return dt_m, dx_m.astype(np.int64), dy_m.astype(np.int64)


# --- WS7b polar view: speed + heading increment -----------------------------
#
# Motion is re-expressed as speed s = hypot(dx, dy) and a heading INCREMENT
# dtheta relative to the previous motion event, so directional smoothness is
# a property of the representation. Ticks (dx == dy == 0, ~15% of events)
# carry s = 0 and no heading; heading persists through them. dtheta of the
# first motion event is 0 and its absolute heading theta0 is side information.


def to_polar(dx, dy):
    """Polar view of an encoded event stream.

    Returns (s, dtheta, theta0). s and dtheta are float64 arrays the same
    length as dx/dy; dtheta is 0 at ticks and at the first motion event.
    theta0 is the absolute heading of the first motion event (0.0 if the
    stream has no motion events).
    """
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    s = np.hypot(dx, dy)
    dtheta = np.zeros(len(s))
    theta0 = 0.0
    idx = np.flatnonzero(s > 0)
    if len(idx):
        th = np.arctan2(dy[idx], dx[idx])
        theta0 = float(th[0])
        inc = np.diff(th)
        dtheta[idx[1:]] = (inc + np.pi) % (2.0 * np.pi) - np.pi
    return s, dtheta, theta0


def from_polar(s, dtheta, theta0):
    """Decode the polar view back to float displacements. No rounding:
    the model's output space is continuous, and whether that survives the
    detector is exactly what the WS7b replay gate measures."""
    s = np.asarray(s, dtype=np.float64)
    dtheta = np.asarray(dtheta, dtype=np.float64)
    motion = s > 0
    heading = np.cumsum(np.where(motion, dtheta, 0.0))
    idx = np.flatnonzero(motion)
    if len(idx):
        heading = heading - heading[idx[0]] + theta0
    dx = np.where(motion, s * np.cos(heading), 0.0)
    dy = np.where(motion, s * np.sin(heading), 0.0)
    return dx, dy


def quantize_dtheta(dtheta, n_bins):
    """Snap heading increments to n_bins uniform bins centred on 0, so the
    45-degree lattice of 1px moves stays exact when 8 divides n_bins."""
    w = 2.0 * np.pi / n_bins
    return np.round(np.asarray(dtheta, dtype=np.float64) / w) * w


def quantize_speed(s, n_bins, s_max=90.0):
    """Snap speeds to n_bins log-uniform bin centres over [1, s_max].
    Ticks (s == 0) pass through untouched."""
    s = np.asarray(s, dtype=np.float64)
    out = s.copy()
    m = s > 0
    if m.any():
        w = np.log(s_max) / n_bins
        out[m] = np.exp(np.round(np.log(np.maximum(s[m], 1.0)) / w) * w)
    return out
