"""
Prepare training data for ZIMT: per-trajectory (dx, dy) sequences with stall labels.

Input: train_positions.npy, train_timestamps.npy, train_n_real.npy, train_conditions.npy
Output:
  - zimt_dxdy.npy:       (N_traj, max_seq_len, 2) float32, padded displacements
  - zimt_stall.npy:       (N_traj, max_seq_len) uint8, stall flags
  - zimt_lengths.npy:     (N_traj,) int32, actual sequence lengths
  - zimt_conditions.npy:  (N_traj, 4) float32, trajectory conditions
  - zimt_endpoints.npy:   (N_traj, 4) float32, (start_x, start_y, end_x, end_y) normalized
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent
HZ = 125.0
DT = 1.0 / HZ
MAX_SEQ_LEN = 256
STALL_THRESHOLD = 1e-6


def resample_to_125hz(positions, timestamps, n_real):
    n = int(n_real)
    if n < 3:
        return None, None, None

    x = positions[:n, 0].astype(np.float64)
    y = positions[:n, 1].astype(np.float64)
    t = timestamps[:n].astype(np.float64)

    t = np.maximum.accumulate(t)
    duration = t[-1] - t[0]
    if duration <= 0:
        return None, None, None

    n_out = max(3, int(round(duration * HZ)))
    t_out = np.arange(n_out) * DT
    if t_out[-1] > t[-1]:
        t_out = t_out[t_out <= t[-1]]
        n_out = len(t_out)
    if n_out < 3:
        return None, None, None

    x_out = np.interp(t_out, t, x)
    y_out = np.interp(t_out, t, y)

    dx = np.diff(x_out).astype(np.float32)
    dy = np.diff(y_out).astype(np.float32)
    dxdy = np.stack([dx, dy], axis=-1)
    n_steps = n_out - 1

    endpoints = np.array([x_out[0], y_out[0], x_out[-1], y_out[-1]], dtype=np.float32)

    return dxdy, n_steps, endpoints


def main():
    parser = argparse.ArgumentParser(description="Prepare ZIMT training data")
    parser.add_argument("--max-traj", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    args = parser.parse_args()

    print("Loading training data...")
    t0 = time.time()

    positions = np.load(TRAINING_DIR / "train_positions.npy", mmap_mode="r")
    timestamps = np.load(TRAINING_DIR / "train_timestamps.npy", mmap_mode="r")
    n_real = np.load(TRAINING_DIR / "train_n_real.npy")
    conditions = np.load(TRAINING_DIR / "train_conditions.npy", mmap_mode="r")

    n_traj = len(n_real)
    max_traj = min(args.max_traj or n_traj, n_traj)
    print(f"  {n_traj} total, processing {max_traj}, loaded in {time.time()-t0:.1f}s")

    all_dxdy = []
    all_stall = []
    all_lengths = []
    all_conditions = []
    all_endpoints = []
    n_stall_steps = 0
    n_total_steps = 0

    for i in range(max_traj):
        if i % 100_000 == 0 and i > 0:
            stall_pct = 100 * n_stall_steps / max(n_total_steps, 1)
            print(f"  {i}/{max_traj} ({len(all_dxdy)} valid, "
                  f"{n_total_steps} steps, {stall_pct:.1f}% stalls)")

        result = resample_to_125hz(positions[i], timestamps[i], n_real[i])
        if result[0] is None:
            continue

        dxdy, n_steps, endpoints = result

        if n_steps < 3:
            continue

        seq_len = min(n_steps, args.max_seq_len)
        dxdy_padded = np.zeros((args.max_seq_len, 2), dtype=np.float32)
        dxdy_padded[:seq_len] = dxdy[:seq_len]

        disp_mag = np.sqrt(dxdy[:seq_len, 0] ** 2 + dxdy[:seq_len, 1] ** 2)
        stall_flags = np.zeros(args.max_seq_len, dtype=np.uint8)
        stall_flags[:seq_len] = (disp_mag < STALL_THRESHOLD).astype(np.uint8)

        all_dxdy.append(dxdy_padded)
        all_stall.append(stall_flags)
        all_lengths.append(seq_len)
        all_conditions.append(conditions[i])
        all_endpoints.append(endpoints)
        n_total_steps += seq_len
        n_stall_steps += int(stall_flags[:seq_len].sum())

    n_valid = len(all_dxdy)
    stall_pct = 100 * n_stall_steps / max(n_total_steps, 1)
    print(f"\nResults:")
    print(f"  {n_valid} valid trajectories out of {max_traj}")
    print(f"  {n_total_steps} total steps")
    print(f"  {n_stall_steps} stall steps ({stall_pct:.2f}%)")
    print(f"  Sequence lengths: mean={np.mean(all_lengths):.1f}, "
          f"median={np.median(all_lengths):.1f}, "
          f"p95={np.percentile(all_lengths, 95):.1f}, "
          f"max={np.max(all_lengths)}")

    dxdy_arr = np.array(all_dxdy, dtype=np.float32)
    stall_arr = np.array(all_stall, dtype=np.uint8)
    lengths_arr = np.array(all_lengths, dtype=np.int32)
    conditions_arr = np.array(all_conditions, dtype=np.float32)
    endpoints_arr = np.array(all_endpoints, dtype=np.float32)

    # Analyze displacement distribution (non-stall only)
    flat_dxdy = np.concatenate([d[:l] for d, l in zip(all_dxdy, all_lengths)])
    flat_stall = np.concatenate([s[:l] for s, l in zip(all_stall, all_lengths)])
    motion_mask = flat_stall == 0
    motion_dxdy = flat_dxdy[motion_mask]
    print(f"\nMotion displacement stats (non-stall):")
    print(f"  dx: mean={motion_dxdy[:,0].mean():.6f}, std={motion_dxdy[:,0].std():.6f}")
    print(f"  dy: mean={motion_dxdy[:,1].mean():.6f}, std={motion_dxdy[:,1].std():.6f}")

    np.save(TRAINING_DIR / "zimt_dxdy.npy", dxdy_arr)
    np.save(TRAINING_DIR / "zimt_stall.npy", stall_arr)
    np.save(TRAINING_DIR / "zimt_lengths.npy", lengths_arr)
    np.save(TRAINING_DIR / "zimt_conditions.npy", conditions_arr)
    np.save(TRAINING_DIR / "zimt_endpoints.npy", endpoints_arr)

    print(f"\nSaved to {TRAINING_DIR}:")
    print(f"  zimt_dxdy.npy:       {dxdy_arr.shape}")
    print(f"  zimt_stall.npy:      {stall_arr.shape}")
    print(f"  zimt_lengths.npy:    {lengths_arr.shape}")
    print(f"  zimt_conditions.npy: {conditions_arr.shape}")
    print(f"  zimt_endpoints.npy:  {endpoints_arr.shape}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
