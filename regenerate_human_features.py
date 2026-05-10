"""Regenerate human_eval_features.npy from the full trajectory pool.

Ensures human features are computed with the same features.py used for
synthetic features at evaluation time. Run this if you modify features.py
or suspect the precomputed features are stale.

Requires: training/pool_flat_i16.npy, training/pool_t_rel_f32.npy,
          training/full_pool_offsets.npy (from prepare_training_data.py)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from features import extract_feature_matrix


def main():
    parser = argparse.ArgumentParser(description="Regenerate human_eval_features.npy")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--training-dir", default="./training")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    data_dir = Path(args.data_dir)

    print("Loading full pool...")
    offsets = np.load(training_dir / "full_pool_offsets.npy")
    flat = np.load(training_dir / "pool_flat_i16.npy", mmap_mode="r")
    t = np.load(training_dir / "pool_t_rel_f32.npy", mmap_mode="r")
    n_pool = len(offsets) - 1
    print(f"Pool: {n_pool} trajectories")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_pool, size=args.n_samples, replace=False)

    print(f"Extracting {args.n_samples} trajectories...")
    trajs = []
    for idx in indices:
        s, e = int(offsets[idx]), int(offsets[idx + 1])
        xy = flat[s:e].astype(np.float64)
        ts = t[s:e].astype(np.float64)
        traj = [(float(xy[j, 0]), float(xy[j, 1]), float(ts[j])) for j in range(len(xy))]
        trajs.append(traj)

    print("Computing features...")
    feats = extract_feature_matrix(trajs)
    print(f"Valid features: {len(feats)}/{args.n_samples}")

    out_path = data_dir / "human_eval_features.npy"
    np.save(out_path, feats)
    print(f"Saved {out_path} ({feats.shape})")


if __name__ == "__main__":
    main()
