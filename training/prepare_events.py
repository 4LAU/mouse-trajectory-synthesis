"""Prepare event-stream training data for the WS7 model.

Encodes every pool trajectory into a (dt, dx, dy) event sequence via
event_codec.encode_events (the split-mode transform validated at 0.50-0.52),
pads to MAX_EVENTS, and saves:

  events_dx.npy    (N, MAX_EVENTS) int8   displacement in [-63, 63]
  events_dy.npy    (N, MAX_EVENTS) int8
  events_dt.npy    (N, MAX_EVENTS) float16  gap in milliseconds
  events_len.npy   (N,) int32   real event count
  events_cond.npy  (N, 4) float32  [log_dist, log_dur, cos, sin]

Trajectories with fewer than 5 or more than MAX_EVENTS events are dropped.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from event_codec import encode_events

OUTPUT_DIR = Path(__file__).resolve().parent
POOL_DIR = OUTPUT_DIR
MAX_EVENTS = 256
MIN_EVENTS = 5


def main():
    flat = np.load(POOL_DIR / "pool_flat_i16.npy", mmap_mode="r")
    t_rel = np.load(POOL_DIR / "pool_t_rel_f32.npy", mmap_mode="r")
    offsets = np.load(POOL_DIR / "full_pool_offsets.npy")
    meta = np.load(POOL_DIR / "full_pool_meta.npy")
    n_traj = len(offsets) - 1
    print(f"Pool: {n_traj:,} trajectories", flush=True)

    all_dx = np.zeros((n_traj, MAX_EVENTS), dtype=np.int8)
    all_dy = np.zeros((n_traj, MAX_EVENTS), dtype=np.int8)
    all_dt = np.zeros((n_traj, MAX_EVENTS), dtype=np.float16)
    all_len = np.zeros(n_traj, dtype=np.int32)
    all_cond = np.zeros((n_traj, 4), dtype=np.float32)

    count = 0
    dropped_short = dropped_long = dropped_bad = 0
    t0 = time.time()

    for i in range(n_traj):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s < MIN_EVENTS:
            dropped_short += 1
            continue
        xy = np.asarray(flat[s:e])
        t = np.asarray(t_rel[s:e], dtype=np.float64)

        enc = encode_events(xy, t)
        if enc is None:
            dropped_bad += 1
            continue
        dt_s, dx, dy = enc
        m = len(dt_s)
        if m < MIN_EVENTS:
            dropped_short += 1
            continue
        if m > MAX_EVENTS:
            dropped_long += 1
            continue
        duration = float(dt_s.sum())
        if duration < 1e-4:
            dropped_bad += 1
            continue

        all_dx[count, :m] = dx
        all_dy[count, :m] = dy
        all_dt[count, :m] = (dt_s * 1000.0).astype(np.float16)
        all_len[count] = m
        all_cond[count] = [
            float(meta[i, 0]),
            float(np.log(duration)),
            float(meta[i, 1]),
            float(meta[i, 2]),
        ]
        count += 1

        if count % 500_000 == 0:
            el = time.time() - t0
            print(f"  {count:,} encoded (dropped {dropped_short + dropped_long + dropped_bad:,}) "
                  f"[{el:.0f}s, {count / el:.0f}/s]", flush=True)

    print(f"\nEncoded {count:,} / {n_traj:,} "
          f"(short {dropped_short:,}, long {dropped_long:,}, bad {dropped_bad:,})", flush=True)
    print(f"Drop fraction for length > {MAX_EVENTS}: {dropped_long / n_traj:.4f}")

    np.save(OUTPUT_DIR / "events_dx.npy", all_dx[:count])
    np.save(OUTPUT_DIR / "events_dy.npy", all_dy[:count])
    np.save(OUTPUT_DIR / "events_dt.npy", all_dt[:count])
    np.save(OUTPUT_DIR / "events_len.npy", all_len[:count])
    np.save(OUTPUT_DIR / "events_cond.npy", all_cond[:count])
    print("Saved events_{dx,dy,dt,len,cond}.npy")


if __name__ == "__main__":
    main()
