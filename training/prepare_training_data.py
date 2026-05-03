"""Prepare training data for the generative trajectory model.

Strategy:
  1. Preprocess each raw trajectory at 125Hz (same as feature extractor)
  2. Origin-translate positions
  3. If n_points <= N (192): store exact positions, pad with last point
     If n_points > N: subsample to N indices (keeping timestamps)
  4. Store: positions (N, 2), n_real (int), condition (4,), timestamps (N,)

Condition vector: [log_dist, log_duration, cos_angle, sin_angle]
  - log_dist and cos/sin_angle come from pool_meta
  - log_duration computed from raw timestamps

Output files:
  {split}_positions.npy   (N_split, 192, 2) float32
  {split}_conditions.npy  (N_split, 4) float32
  {split}_n_real.npy      (N_split,) uint8 -- number of real points (cap 192)
  {split}_timestamps.npy  (N_split, 192) float32 -- timestamps (padded)

Split: 90% train, 5% val, 5% test
Stratified by 4 distance bands from log_dist quartiles.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from features import resample_trajectory

N_RESAMPLE = 192
MIN_POINTS = 5
FEATURE_HZ = 125.0
OUTPUT_DIR = Path(__file__).resolve().parent
# Pool data directory -- adjust this path to where your pool .npy files live
POOL_DIR = OUTPUT_DIR


def process_trajectory(xy, t, n=N_RESAMPLE):
    """Process a single trajectory into the training representation.

    Returns
    -------
    positions : (n, 2) float32
    n_real : int
    timestamps : (n,) float32
    duration : float
    """
    duration = float(t[-1] - t[0])
    if duration < 1e-8:
        return None

    raw_traj = list(zip(xy[:, 0].tolist(), xy[:, 1].tolist(), t.tolist()))
    pp = resample_trajectory(raw_traj)
    pp_arr = np.array(pp, dtype=np.float64)
    n_pp = len(pp_arr)

    if n_pp < MIN_POINTS:
        return None

    xy_pp = pp_arr[:, :2].copy()
    t_pp = pp_arr[:, 2] - pp_arr[0, 2]

    # Origin-translate
    xy_pp[:, 0] -= xy_pp[0, 0]
    xy_pp[:, 1] -= xy_pp[0, 1]

    # Distance-normalize so inference coordinate scale matches checkpoints
    total_dist = np.hypot(xy_pp[-1, 0], xy_pp[-1, 1])
    if total_dist > 1e-6:
        xy_pp[:, :2] /= total_dist

    positions = np.empty((n, 2), dtype=np.float32)
    timestamps = np.empty(n, dtype=np.float32)

    if n_pp <= n:
        # Fits: store exact, pad with last point
        n_real = n_pp
        positions[:n_pp] = xy_pp.astype(np.float32)
        positions[n_pp:] = xy_pp[-1].astype(np.float32)
        timestamps[:n_pp] = t_pp.astype(np.float32)
        # Pad timestamps with last timestamp
        timestamps[n_pp:] = t_pp[-1].astype(np.float32)
    else:
        # Too many: subsample to n indices
        n_real = n
        sel = np.round(np.linspace(0, n_pp - 1, n)).astype(int)
        positions[:] = xy_pp[sel].astype(np.float32)
        timestamps[:] = t_pp[sel].astype(np.float32)

    return positions, n_real, timestamps, duration


def main():
    print("=== Preparing Training Data ===\n")

    # Load pool data
    print("Loading pool data...")
    flat_i16 = np.load(POOL_DIR / "pool_flat_i16.npy", mmap_mode="r")
    t_rel = np.load(POOL_DIR / "pool_t_rel_f32.npy", mmap_mode="r")
    offsets = np.load(POOL_DIR / "full_pool_offsets.npy")
    meta = np.load(POOL_DIR / "full_pool_meta.npy")  # (N, 3) = [log_dist, cos_angle, sin_angle]
    n_traj = len(offsets) - 1

    print(f"Pool: {n_traj:,} trajectories")
    print(f"Flat coords: {flat_i16.shape}, dtype={flat_i16.dtype}")
    print(f"Timestamps: {t_rel.shape}, dtype={t_rel.dtype}")
    print(f"Meta: {meta.shape}, dtype={meta.dtype}")

    # Pre-allocate output arrays
    # Estimate: ~95% will be usable
    max_usable = n_traj
    all_positions = np.empty((max_usable, N_RESAMPLE, 2), dtype=np.float32)
    all_conditions = np.empty((max_usable, 4), dtype=np.float32)
    all_n_real = np.empty(max_usable, dtype=np.uint8)
    all_timestamps = np.empty((max_usable, N_RESAMPLE), dtype=np.float32)

    count = 0
    skipped = 0
    t0 = time.time()

    for i in range(n_traj):
        s, e = int(offsets[i]), int(offsets[i + 1])
        n_pts = e - s
        if n_pts < MIN_POINTS:
            skipped += 1
            continue

        # Convert int16 coords to float64
        xy = flat_i16[s:e].astype(np.float64)

        # t_rel is already zero-based relative timestamps per trajectory
        t_raw = t_rel[s:e].astype(np.float64)

        duration = float(t_raw[-1] - t_raw[0])
        if duration < 1e-8:
            skipped += 1
            continue

        result = process_trajectory(xy, t_raw)
        if result is None:
            skipped += 1
            continue

        positions, n_real, timestamps, dur = result

        # Build condition vector: [log_dist, log_duration, cos_angle, sin_angle]
        log_dist = float(meta[i, 0])
        cos_angle = float(meta[i, 1])
        sin_angle = float(meta[i, 2])
        log_duration = float(np.log(max(dur, 1e-8)))

        all_positions[count] = positions
        all_conditions[count] = [log_dist, log_duration, cos_angle, sin_angle]
        all_n_real[count] = min(n_real, 255)  # uint8 cap
        all_timestamps[count] = timestamps
        count += 1

        if count % 500_000 == 0:
            elapsed = time.time() - t0
            rate = count / elapsed
            eta = (n_traj - i) / rate if rate > 0 else 0
            print(f"  Processed {count:,} / ~{n_traj:,} "
                  f"(skipped {skipped:,}) "
                  f"[{elapsed:.0f}s, {rate:.0f}/s, ETA {eta:.0f}s]")

    elapsed = time.time() - t0
    print(f"\nDone: {count:,} usable trajectories, {skipped:,} skipped ({elapsed:.0f}s)")

    # Trim to actual count
    all_positions = all_positions[:count]
    all_conditions = all_conditions[:count]
    all_n_real = all_n_real[:count]
    all_timestamps = all_timestamps[:count]

    # Stratified split by log_dist quartiles
    print("\nSplitting data (stratified by distance bands)...")
    log_dists = all_conditions[:, 0]
    quartiles = np.percentile(log_dists, [25, 50, 75])
    bands = np.digitize(log_dists, quartiles)  # 0, 1, 2, 3

    rng = np.random.default_rng(42)
    indices = np.arange(count)
    rng.shuffle(indices)

    # Sort by band for stratification
    train_idx = []
    val_idx = []
    test_idx = []

    for band in range(4):
        band_mask = bands[indices] == band
        band_indices = indices[band_mask]
        n_band = len(band_indices)

        n_val = max(1, int(n_band * 0.05))
        n_test = max(1, int(n_band * 0.05))
        n_train = n_band - n_val - n_test

        train_idx.append(band_indices[:n_train])
        val_idx.append(band_indices[n_train:n_train + n_val])
        test_idx.append(band_indices[n_train + n_val:])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    # Shuffle within splits
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    print(f"  Train: {len(train_idx):,}")
    print(f"  Val:   {len(val_idx):,}")
    print(f"  Test:  {len(test_idx):,}")

    # Save
    print("\nSaving files...")
    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        pos_path = OUTPUT_DIR / f"{split_name}_positions.npy"
        cond_path = OUTPUT_DIR / f"{split_name}_conditions.npy"
        nreal_path = OUTPUT_DIR / f"{split_name}_n_real.npy"
        ts_path = OUTPUT_DIR / f"{split_name}_timestamps.npy"

        np.save(pos_path, all_positions[split_idx])
        np.save(cond_path, all_conditions[split_idx])
        np.save(nreal_path, all_n_real[split_idx])
        np.save(ts_path, all_timestamps[split_idx])

        # Report sizes
        pos_size = pos_path.stat().st_size / (1024 ** 3)
        cond_size = cond_path.stat().st_size / (1024 ** 2)
        nreal_size = nreal_path.stat().st_size / (1024 ** 2)
        ts_size = ts_path.stat().st_size / (1024 ** 3)
        print(f"  {split_name}: positions={pos_size:.2f}GB, conditions={cond_size:.1f}MB, "
              f"n_real={nreal_size:.1f}MB, timestamps={ts_size:.2f}GB")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
