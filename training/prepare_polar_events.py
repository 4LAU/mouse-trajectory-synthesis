"""Prepare WS7b polar event training data from the WS7 event arrays.

Derives the speed + heading-increment view from events_{dx,dy,cond}.npy
(produced by prepare_events.py) and saves:

  events_s2.npy   (N, 256) int16  squared speed dx^2 + dy^2, EXACT (0 = tick)
  events_dth.npy  (N, 256) int16  heading increment on a 2^15-per-turn lattice
                                  (0.011 deg resolution), 0 at ticks and pads

Conventions (must match models/event_stream_polar.py and the eval decode):
- dtheta is defined between consecutive MOTION events; heading persists
  through ticks (dx = dy = 0).
- The FIRST motion event's dtheta is relative to the conditioning angle
  atan2(sin, cos) from events_cond, so the decoder needs no side information:
  heading starts at the requested target angle plus the first increment.
- Any train-time binning coarser than the 2^15 lattice is derived from these
  arrays; the quantization error here (0.006 deg) is far below the 0.52-gate
  sensitivity measured in eval_polar_b*.log.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent
LATTICE = 32768  # dtheta lattice steps per full turn


def main():
    dx = np.load(DIR / "events_dx.npy", mmap_mode="r")
    dy = np.load(DIR / "events_dy.npy", mmap_mode="r")
    cond = np.load(DIR / "events_cond.npy")
    lens = np.load(DIR / "events_len.npy")
    n, m = dx.shape
    print(f"{n:,} trajectories x {m} events", flush=True)

    theta_cond = np.arctan2(cond[:, 3].astype(np.float64), cond[:, 2].astype(np.float64))

    all_s2 = np.zeros((n, m), dtype=np.int16)
    all_dth = np.zeros((n, m), dtype=np.int16)

    scale = LATTICE / (2.0 * np.pi)
    t0 = time.time()
    chunk = 100_000
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        dxc = np.asarray(dx[c0:c1], dtype=np.int32)
        dyc = np.asarray(dy[c0:c1], dtype=np.int32)
        s2 = dxc * dxc + dyc * dyc
        all_s2[c0:c1] = s2.astype(np.int16)

        th = np.arctan2(dyc, dxc)  # only meaningful where s2 > 0
        for i in range(c1 - c0):
            k = lens[c0 + i]
            idx = np.flatnonzero(s2[i, :k] > 0)
            if len(idx) == 0:
                continue
            thi = th[i, idx]
            inc = np.empty(len(idx))
            inc[0] = thi[0] - theta_cond[c0 + i]
            inc[1:] = np.diff(thi)
            inc = (inc + np.pi) % (2.0 * np.pi) - np.pi
            q = np.round(inc * scale).astype(np.int32)
            q[q == LATTICE // 2] = -(LATTICE // 2)  # +pi and -pi share a bin
            all_dth[c0 + i, idx] = q.astype(np.int16)

        if c1 % 500_000 < chunk:
            el = time.time() - t0
            print(f"  {c1:,} done [{el:.0f}s, {c1 / el:.0f}/s]", flush=True)

    np.save(DIR / "events_s2.npy", all_s2)
    np.save(DIR / "events_dth.npy", all_dth)
    print(f"Saved events_s2.npy, events_dth.npy in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
