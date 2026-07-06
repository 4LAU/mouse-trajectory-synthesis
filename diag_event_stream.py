"""WS7 feasibility checks 1 and 2: raw event stream statistics.

Check 1: are pool timestamps pre-resample (non-uniform dt)? If dt is exactly
8ms everywhere, the pool is already 125Hz and the event stream must be inferred.

Check 2: events per trajectory, dt distribution, dx/dy vocabulary range.
Sets sequence length and categorical head sizes for the event-stream model.

Reads a random sample of trajectories from the memmapped pool.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

POOL_DIR = Path(__file__).resolve().parent / "training"
N_SAMPLE = 20000
SEED = 42


def main():
    flat = np.load(POOL_DIR / "pool_flat_i16.npy", mmap_mode="r")
    t_rel = np.load(POOL_DIR / "pool_t_rel_f32.npy", mmap_mode="r")
    offsets = np.load(POOL_DIR / "full_pool_offsets.npy")
    n_traj = len(offsets) - 1
    print(f"Pool: {n_traj:,} trajectories, flat shape {flat.shape} dtype {flat.dtype}, "
          f"t shape {t_rel.shape} dtype {t_rel.dtype}")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_traj, size=min(N_SAMPLE, n_traj), replace=False)

    all_dt = []
    all_dx = []
    all_dy = []
    n_events = []
    n_dupe_ts = 0
    n_total_steps = 0
    for i in idx:
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s < 2:
            continue
        xy = np.asarray(flat[s:e], dtype=np.int64)
        t = np.asarray(t_rel[s:e], dtype=np.float64)
        dt = np.diff(t)
        d = np.diff(xy, axis=0)
        all_dt.append(dt)
        all_dx.append(d[:, 0])
        all_dy.append(d[:, 1])
        n_events.append(e - s)
        n_dupe_ts += int((dt <= 0).sum())
        n_total_steps += len(dt)

    dt = np.concatenate(all_dt)
    dx = np.concatenate(all_dx)
    dy = np.concatenate(all_dy)
    n_events = np.asarray(n_events)

    print(f"\n=== Check 1: is dt non-uniform (raw events)? ===")
    pos_dt = dt[dt > 0]
    print(f"dt percentiles (ms): p1={np.percentile(pos_dt,1)*1e3:.2f} "
          f"p25={np.percentile(pos_dt,25)*1e3:.2f} p50={np.percentile(pos_dt,50)*1e3:.2f} "
          f"p75={np.percentile(pos_dt,75)*1e3:.2f} p99={np.percentile(pos_dt,99)*1e3:.2f} "
          f"max={pos_dt.max()*1e3:.1f}")
    print(f"dt unique values in sample of 100k: {len(np.unique(np.round(pos_dt[:100000]*1e3, 3)))}")
    frac_8ms = float(np.mean(np.abs(pos_dt - 0.008) < 1e-4))
    print(f"fraction of dt within 0.1ms of 8ms (125Hz): {frac_8ms:.3f}")
    print(f"non-positive dt steps: {n_dupe_ts:,} / {n_total_steps:,} "
          f"({100*n_dupe_ts/max(n_total_steps,1):.2f}%)")
    verdict = "RAW EVENTS (non-uniform dt)" if frac_8ms < 0.5 else "ALREADY RESAMPLED (uniform 8ms)"
    print(f"VERDICT: {verdict}")

    print(f"\n=== Check 2: event stream statistics ===")
    print(f"events per trajectory: p1={np.percentile(n_events,1):.0f} "
          f"p25={np.percentile(n_events,25):.0f} p50={np.percentile(n_events,50):.0f} "
          f"p75={np.percentile(n_events,75):.0f} p99={np.percentile(n_events,99):.0f} "
          f"max={n_events.max()}")
    for name, v in [("dx", dx), ("dy", dy)]:
        print(f"{name}: min={v.min()} max={v.max()} "
              f"p0.5={np.percentile(v,0.5):.0f} p99.5={np.percentile(v,99.5):.0f} "
              f"|{name}|<=15 covers {100*np.mean(np.abs(v)<=15):.2f}% "
              f"|{name}|<=31 covers {100*np.mean(np.abs(v)<=31):.2f}% "
              f"|{name}|<=63 covers {100*np.mean(np.abs(v)<=63):.2f}%")
    zero_step = np.mean((dx == 0) & (dy == 0))
    print(f"fraction of events with dx=dy=0 (pure time tick): {100*zero_step:.2f}%")

    counts, edges = np.histogram(np.clip(pos_dt * 1e3, 0, 100), bins=[0, 2, 4, 6, 8, 10, 12, 16, 24, 40, 100])
    print("dt histogram (ms bins):")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        print(f"  [{lo:>3.0f},{hi:>3.0f}): {100*c/len(pos_dt):6.2f}%")


if __name__ == "__main__":
    main()
