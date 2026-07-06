"""WS7 feasibility check 3: real event streams through the decode + resample path.

Replays real pool trajectories, re-encoded through the exact (dt, dx, dy) event
representation the event-stream model would emit, then decoded back to
(x, y, t) tuples. evaluate.py applies the standard 125Hz resample during
feature extraction, so this measures the upper bound of the representation:
if real events do not score near 0.50, the decode pipeline or the
representation is wrong and the WS7 premise fails cheaply.

Modes via EVENT_REPLAY_MODE env var:
  pure      - replay raw events untouched (validates the pipeline itself)
  roundtrip - merge non-positive-dt events and clip dx/dy to the model
              vocabulary (default, validates the representation)

Trajectories are matched to the requested distance (nearest in the pool,
sampled without replacement among near neighbours).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_POOL_DIR = Path(__file__).resolve().parent.parent / "training"

MODE = os.environ.get("EVENT_REPLAY_MODE", "roundtrip")
VOCAB_MAX = int(os.environ.get("EVENT_REPLAY_VOCAB", "63"))

_flat = np.load(_POOL_DIR / "pool_flat_i16.npy", mmap_mode="r")
_t_rel = np.load(_POOL_DIR / "pool_t_rel_f32.npy", mmap_mode="r")
_offsets = np.load(_POOL_DIR / "full_pool_offsets.npy")

# Per-trajectory straight-line distance, aligned with the pool
_distances = np.load(_POOL_DIR.parent / "data" / "human_distances.npy")
_dist_order = np.argsort(_distances)
_dist_sorted = _distances[_dist_order]

_rng = np.random.default_rng(12345)
_clip_steps = 0
_merge_steps = 0
_total_steps = 0


def _encode_decode(xy: np.ndarray, t: np.ndarray) -> list[tuple[float, float, float]]:
    """Round-trip a real trajectory through the (dt, dx, dy) event representation."""
    global _clip_steps, _merge_steps, _total_steps

    dt = np.diff(t)
    d = np.diff(xy, axis=0)
    _total_steps += len(dt)

    if MODE in ("roundtrip", "split"):
        # Out-of-order timestamps count as duplicates: no time advances
        dt = np.maximum(dt, 0.0)
        # Merge events with non-positive dt into the previous event
        keep = dt > 0
        _merge_steps += int((~keep).sum())
        # Sum displacements of merged runs onto the last kept event before them
        groups = np.cumsum(keep)  # group id per step, merged steps join the previous
        n_groups = int(groups[-1]) if len(groups) else 0
        if n_groups == 0:
            return [(float(xy[0, 0]), float(xy[0, 1]), 0.0)]
        # Steps before the first kept step fold forward into group 0
        gidx = np.maximum(groups - 1, 0)
        dx_m = np.bincount(gidx, weights=d[:, 0], minlength=n_groups)
        dy_m = np.bincount(gidx, weights=d[:, 1], minlength=n_groups)
        dt_m = np.bincount(gidx, weights=dt, minlength=n_groups)
        if MODE == "split":
            # Split jumps larger than the vocabulary into collinear integer
            # sub-events summing to the same displacement and dt. The
            # piecewise-linear path is unchanged up to sub-pixel rounding,
            # so nothing needs clipping.
            over = np.maximum(np.abs(dx_m), np.abs(dy_m)) > VOCAB_MAX
            if over.any():
                dx_s, dy_s, dt_s = [], [], []
                for j in range(n_groups):
                    if not over[j]:
                        dx_s.append(dx_m[j]); dy_s.append(dy_m[j]); dt_s.append(dt_m[j])
                        continue
                    k = int(np.ceil(max(abs(dx_m[j]), abs(dy_m[j])) / (VOCAB_MAX - 1)))
                    fx = np.round(np.linspace(0, dx_m[j], k + 1))
                    fy = np.round(np.linspace(0, dy_m[j], k + 1))
                    dx_s.extend(np.diff(fx)); dy_s.extend(np.diff(fy))
                    dt_s.extend([dt_m[j] / k] * k)
                    _clip_steps += 1
                dx_m = np.asarray(dx_s); dy_m = np.asarray(dy_s); dt_m = np.asarray(dt_s)
        else:
            # Clip displacements to the model vocabulary
            _clip_steps += int((np.abs(dx_m) > VOCAB_MAX).sum() + (np.abs(dy_m) > VOCAB_MAX).sum())
            dx_m = np.clip(dx_m, -VOCAB_MAX, VOCAB_MAX)
            dy_m = np.clip(dy_m, -VOCAB_MAX, VOCAB_MAX)
    else:
        dt_m, dx_m, dy_m = dt, d[:, 0], d[:, 1]

    # Decode: integrate events into positions at cumulative timestamps
    x = np.concatenate([[xy[0, 0]], xy[0, 0] + np.cumsum(dx_m)])
    y = np.concatenate([[xy[0, 1]], xy[0, 1] + np.cumsum(dy_m)])
    tt = np.concatenate([[0.0], np.cumsum(dt_m)])
    return list(zip(x.tolist(), y.tolist(), tt.tolist()))


def generate_path(sx, sy, ex, ey):
    return generate_paths([(sx, sy, ex, ey)])[0]


def generate_paths(specs) -> list:
    used: set[int] = set()
    out = []
    for sx, sy, ex, ey in specs:
        target = float(np.hypot(ex - sx, ey - sy))
        pos = int(np.searchsorted(_dist_sorted, target))
        # Sample among the 64 nearest-distance neighbours, skip already-used
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
        # Translate to the requested start point (pure translation, no rounding)
        xy = xy - xy[0] + np.array([sx, sy])
        out.append(_encode_decode(xy, t))

    if _total_steps:
        print(f"  [event_replay mode={MODE}] merged {_merge_steps:,} zero-dt steps, "
              f"clipped {_clip_steps:,} of {_total_steps:,} displacement components")
    return out
