"""WS7b go/no-go gate: real events through the speed+heading representation.

Replays real pool trajectories through the canonical event encode
(event_codec.encode_events, the exact transform WS7 validated at 0.50-0.52),
then through the WS7b polar view (speed + heading increment) and back, and
decodes to (x, y, t) WITHOUT rounding to integer pixels. The WS7b model's
output space is continuous speed and binned dtheta, so this measures whether
that output space itself is detectable before any training is spent.

A plain float round-trip is lossless and proves nothing, so the variants
that matter apply the quantization the model heads will impose:

  POLAR_BINS   dtheta bins (0 = exact float, no binning)
  POLAR_SBINS  log-speed bins (0 = exact float)
  POLAR_ROUND  1 = round decoded positions to integer pixels (the sensor
               emits integers; binning alone leaves every position off the
               pixel grid, which the finer-binning sweep showed is itself
               detectable at ~0.55 regardless of bin resolution)

Gate: every variant intended for the model must hold RF OOB <= 0.52 at
n=2000. If the continuous decode alone blows past that, integer
displacements matter and WS7b switches to a categorical head over
realizable integer displacements instead.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from event_codec import (
    encode_events,
    from_polar,
    quantize_dtheta,
    quantize_speed,
    to_polar,
)

_POOL_DIR = Path(__file__).resolve().parent.parent / "training"

N_BINS = int(os.environ.get("POLAR_BINS", "0"))
S_BINS = int(os.environ.get("POLAR_SBINS", "0"))
ROUND = os.environ.get("POLAR_ROUND", "0") == "1"

_flat = np.load(_POOL_DIR / "pool_flat_i16.npy", mmap_mode="r")
_t_rel = np.load(_POOL_DIR / "pool_t_rel_f32.npy", mmap_mode="r")
_offsets = np.load(_POOL_DIR / "full_pool_offsets.npy")

_distances = np.load(_POOL_DIR.parent / "data" / "human_distances.npy")
_dist_order = np.argsort(_distances)
_dist_sorted = _distances[_dist_order]

_rng = np.random.default_rng(12345)

print(f"[event_replay_polar] dtheta bins={N_BINS or 'exact'} "
      f"speed bins={S_BINS or 'exact'}")


def _encode_decode(xy: np.ndarray, t: np.ndarray):
    enc = encode_events(xy, t)
    if enc is None:
        return None
    dt, dx, dy = enc

    s, dtheta, theta0 = to_polar(dx, dy)
    if N_BINS:
        dtheta = quantize_dtheta(dtheta, N_BINS)
    if S_BINS:
        s = quantize_speed(s, S_BINS)
    dxf, dyf = from_polar(s, dtheta, theta0)

    x = np.concatenate([[xy[0, 0]], xy[0, 0] + np.cumsum(dxf)])
    y = np.concatenate([[xy[0, 1]], xy[0, 1] + np.cumsum(dyf)])
    if ROUND:
        x = np.round(x)
        y = np.round(y)
    tt = np.concatenate([[0.0], np.cumsum(dt)])
    return list(zip(x.tolist(), y.tolist(), tt.tolist()))


def generate_path(sx, sy, ex, ey):
    return generate_paths([(sx, sy, ex, ey)])[0]


def generate_paths(specs) -> list:
    used: set[int] = set()
    out = []
    for sx, sy, ex, ey in specs:
        target = float(np.hypot(ex - sx, ey - sy))
        pos = int(np.searchsorted(_dist_sorted, target))
        for _ in range(16):
            j = int(np.clip(pos + int(_rng.integers(-32, 33)), 0, len(_dist_sorted) - 1))
            traj_idx = int(_dist_order[j])
            if traj_idx not in used:
                break
        used.add(traj_idx)

        s, e = int(_offsets[traj_idx]), int(_offsets[traj_idx + 1])
        if e - s < 5:
            out.append(None)
            continue
        xy = np.asarray(_flat[s:e], dtype=np.float64)
        t = np.asarray(_t_rel[s:e], dtype=np.float64)
        xy = xy - xy[0] + np.array([sx, sy])
        out.append(_encode_decode(xy, t))
    return out
