"""Precompute polar (speed, delta_heading) arrays for CANDI training.

train_candi.py converts (dx, dy) to polar inside the DataLoader on every
epoch, which is the training speed bottleneck. This script does the
conversion once and stores the result, using vectorized operations that
reproduce _to_polar exactly.

Input:  zimt_dxdy.npy, zimt_stall.npy, zimt_lengths.npy
Output: zimt_polar_spd.npy, zimt_polar_dh.npy  (N, S) float32
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def polar_chunk(dxdy: np.ndarray, stall: np.ndarray, lengths: np.ndarray):
    """Vectorized _to_polar over a (B, S, 2) chunk. Padding stays zero."""
    B, S, _ = dxdy.shape
    dx = dxdy[:, :, 0].astype(np.float64)
    dy = dxdy[:, :, 1].astype(np.float64)
    speed = np.sqrt(dx ** 2 + dy ** 2)
    heading = np.arctan2(dy, dx)

    pad = np.arange(S)[None, :] >= lengths[:, None]
    invalid = (stall.astype(bool)) | (speed < 1e-10)
    invalid |= pad

    # forward-fill heading over invalid steps, 0.0 if invalid from the start
    idx = np.where(~invalid, np.arange(S)[None, :], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    rows = np.arange(B)[:, None]
    filled = np.where(idx >= 0, heading[rows, np.maximum(idx, 0)], 0.0)

    dh = np.empty_like(filled)
    dh[:, 0] = filled[:, 0]
    dh[:, 1:] = np.diff(filled, axis=1)
    dh[:, 1:] = (dh[:, 1:] + np.pi) % (2 * np.pi) - np.pi

    speed[pad] = 0.0
    dh[pad] = 0.0
    return speed.astype(np.float32), dh.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--chunk", type=int, default=20000)
    parser.add_argument("--verify", type=int, default=200,
                        help="Number of trajectories to check against _to_polar")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dxdy = np.load(data_dir / "zimt_dxdy.npy", mmap_mode="r")
    stall = np.load(data_dir / "zimt_stall.npy", mmap_mode="r")
    lengths = np.load(data_dir / "zimt_lengths.npy")
    N, S, _ = dxdy.shape
    print(f"{N:,} trajectories, seq len {S}")

    spd_out = np.lib.format.open_memmap(
        data_dir / "zimt_polar_spd.npy", mode="w+", dtype=np.float32, shape=(N, S))
    dh_out = np.lib.format.open_memmap(
        data_dir / "zimt_polar_dh.npy", mode="w+", dtype=np.float32, shape=(N, S))

    t0 = time.time()
    for lo in range(0, N, args.chunk):
        hi = min(lo + args.chunk, N)
        spd, dh = polar_chunk(
            np.asarray(dxdy[lo:hi]), np.asarray(stall[lo:hi]),
            lengths[lo:hi].astype(np.int64))
        spd_out[lo:hi] = spd
        dh_out[lo:hi] = dh
        print(f"  {hi:,}/{N:,} ({time.time() - t0:.0f}s)", flush=True)

    spd_out.flush()
    dh_out.flush()

    if args.verify > 0:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_candi import _to_polar
        rng = np.random.default_rng(0)
        worst = 0.0
        for i in rng.choice(N, size=min(args.verify, N), replace=False):
            L = min(int(lengths[i]), S)
            ref_spd, ref_dh = _to_polar(np.asarray(dxdy[i]), np.asarray(stall[i]), L)
            worst = max(worst,
                        float(np.abs(spd_out[i, :L] - ref_spd).max()),
                        float(np.abs(dh_out[i, :L] - ref_dh).max()))
        print(f"verify: max abs diff vs _to_polar = {worst:.2e}")
        if worst > 1e-5:
            raise SystemExit("MISMATCH: precomputed polar differs from _to_polar")

    print("done")


if __name__ == "__main__":
    main()
