"""
Prepare training data for VQ-VAE: extract (dx, dy) displacement pairs at 125Hz.

Input: arc-length resampled trajectories (positions, timestamps, n_real)
Output:
  - vqvae_dxdy.npy: (N_total_steps, 2) all displacement pairs
  - vqvae_traj_offsets.npy: (N_traj+1,) index boundaries per trajectory
  - vqvae_conditions.npy: (N_traj, 4) conditions per trajectory
  - vqvae_speeds.npy: (N_total_steps,) speed at each step
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

GEN_DIR = Path(__file__).resolve().parent

HZ = 125.0
DT = 1.0 / HZ


def resample_to_125hz(positions, timestamps, n_real):
    """Resample a single trajectory from arc-length to 125Hz."""
    n = int(n_real)
    if n < 3:
        return None, None

    x = positions[:n, 0].astype(np.float64)
    y = positions[:n, 1].astype(np.float64)
    t = timestamps[:n].astype(np.float64)

    # Timestamps should be monotonic
    t = np.maximum.accumulate(t)
    duration = t[-1] - t[0]
    if duration <= 0:
        return None, None

    # Resample at 125Hz
    n_out = max(3, int(round(duration * HZ)))
    t_out = np.arange(n_out) * DT
    if t_out[-1] > t[-1]:
        t_out = t_out[t_out <= t[-1]]
        n_out = len(t_out)
    if n_out < 3:
        return None, None

    x_out = np.interp(t_out, t, x)
    y_out = np.interp(t_out, t, y)

    # Compute (dx, dy) per step
    dx = np.diff(x_out)
    dy = np.diff(y_out)

    return np.stack([dx, dy], axis=-1).astype(np.float32), n_out - 1


def main():
    print("Loading training data...")
    t0 = time.time()

    positions = np.load(GEN_DIR / "train_positions.npy", mmap_mode="r")
    timestamps = np.load(GEN_DIR / "train_timestamps.npy", mmap_mode="r")
    n_real = np.load(GEN_DIR / "train_n_real.npy")
    conditions = np.load(GEN_DIR / "train_conditions.npy", mmap_mode="r")

    n_traj = len(n_real)
    print(f"  {n_traj} trajectories, loaded in {time.time()-t0:.1f}s")

    # Process in chunks for memory efficiency
    # Subsample for speed: use first 500K trajectories (plenty for VQ-VAE)
    max_traj = min(500_000, n_traj)
    print(f"  Processing {max_traj} trajectories...")

    all_dxdy = []
    offsets = [0]
    valid_conditions = []
    all_speeds = []
    n_valid = 0
    n_stall_steps = 0
    n_total_steps = 0

    for i in range(max_traj):
        if i % 50000 == 0 and i > 0:
            print(f"  {i}/{max_traj} ({n_valid} valid, {n_total_steps} steps, "
                  f"{n_stall_steps} stalls = {100*n_stall_steps/max(n_total_steps,1):.1f}%)...")

        dxdy, n_steps = resample_to_125hz(positions[i], timestamps[i], n_real[i])
        if dxdy is None:
            continue

        speed = np.sqrt(dxdy[:, 0]**2 + dxdy[:, 1]**2) / DT
        stalls = speed < 1.0  # stall = speed < 1 px/s

        all_dxdy.append(dxdy)
        all_speeds.append(speed)
        offsets.append(offsets[-1] + n_steps)
        valid_conditions.append(conditions[i])
        n_valid += 1
        n_total_steps += n_steps
        n_stall_steps += int(stalls.sum())

    print(f"\nResults:")
    print(f"  {n_valid} valid trajectories out of {max_traj}")
    print(f"  {n_total_steps} total steps")
    print(f"  {n_stall_steps} stall steps ({100*n_stall_steps/n_total_steps:.2f}%)")

    # Concatenate and save
    print("Concatenating and saving...")
    dxdy_all = np.concatenate(all_dxdy, axis=0)
    speeds_all = np.concatenate(all_speeds, axis=0)
    offsets_arr = np.array(offsets, dtype=np.int64)
    conditions_arr = np.array(valid_conditions, dtype=np.float32)

    # Analyze distribution
    print(f"\n(dx, dy) distribution:")
    print(f"  dx: mean={dxdy_all[:,0].mean():.4f}, std={dxdy_all[:,0].std():.4f}, "
          f"min={dxdy_all[:,0].min():.4f}, max={dxdy_all[:,0].max():.4f}")
    print(f"  dy: mean={dxdy_all[:,1].mean():.4f}, std={dxdy_all[:,1].std():.4f}, "
          f"min={dxdy_all[:,1].min():.4f}, max={dxdy_all[:,1].max():.4f}")
    print(f"  speed: mean={speeds_all.mean():.1f}, std={speeds_all.std():.1f}, "
          f"max={speeds_all.max():.1f}")

    # Percentiles
    for p in [1, 5, 25, 50, 75, 95, 99]:
        dx_p = np.percentile(dxdy_all[:, 0], p)
        dy_p = np.percentile(dxdy_all[:, 1], p)
        spd_p = np.percentile(speeds_all, p)
        print(f"  P{p:02d}: dx={dx_p:.3f}, dy={dy_p:.3f}, speed={spd_p:.1f}")

    # Stall analysis
    stall_mask = speeds_all < 1.0
    print(f"\nStall analysis:")
    print(f"  Stall steps (speed < 1 px/s): {stall_mask.sum()} ({100*stall_mask.mean():.2f}%)")
    near_zero = speeds_all < 0.01
    print(f"  Near-zero (speed < 0.01): {near_zero.sum()} ({100*near_zero.mean():.2f}%)")

    # Save
    np.save(GEN_DIR / "vqvae_dxdy.npy", dxdy_all)
    np.save(GEN_DIR / "vqvae_speeds.npy", speeds_all)
    np.save(GEN_DIR / "vqvae_traj_offsets.npy", offsets_arr)
    np.save(GEN_DIR / "vqvae_conditions.npy", conditions_arr)

    print(f"\nSaved: dxdy={dxdy_all.shape}, offsets={offsets_arr.shape}, "
          f"conditions={conditions_arr.shape}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
