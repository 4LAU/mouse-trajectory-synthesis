"""Analyze WHY derivative feature correlations break in synthetic data."""
import os
import math
import numpy as np

os.environ["CANDI_CKPT"] = "candi_polar_best.pt"
os.environ["CANDI_CFG"] = "0.0"
os.environ["CANDI_GUIDE"] = "0.5"
os.environ["CANDI_CORRECT"] = "rotate"
os.environ["CANDI_STEPS"] = "50"
os.environ["CANDI_ETA"] = "0.0"
os.environ["CANDI_CANDIDATES"] = "1"
os.environ["CANDI_SMOOTH_DH"] = "0.0"
os.environ["CANDI_SMOOTH_POS"] = "0"

from experiments.candi import generate_path
from features import extract_features, resample_trajectory, FEATURE_NAMES

# Generate some trajectories and analyze speed profiles
rng = np.random.default_rng(42)
distances = np.load("data/human_distances.npy")

n = 200
feat_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}

synth_features = []
for i in range(n):
    dist = float(rng.choice(distances))
    angle = float(rng.uniform(0, 2 * np.pi))
    ex = 960 + dist * np.cos(angle)
    ey = 540 + dist * np.sin(angle)
    traj = generate_path(960, 540, ex, ey)
    resampled = resample_trajectory(traj)
    feats = extract_features(resampled)
    if feats is not None:
        synth_features.append(feats)

synth = np.array(synth_features)
human = np.load("data/human_eval_features.npy")[:n]

# Check correlations for derivative features
deriv_names = ["mean_velocity", "mean_acceleration", "std_acceleration",
               "max_acceleration", "mean_jerk", "std_jerk"]
deriv_idx = [feat_idx[n] for n in deriv_names]

print("=== Human correlations (derivative features) ===")
h_sub = human[:, deriv_idx]
h_corr = np.corrcoef(h_sub.T)
for i, ni in enumerate(deriv_names):
    for j, nj in enumerate(deriv_names):
        if j > i:
            print(f"  {ni:20s} x {nj:20s}  r={h_corr[i,j]:+.3f}")

print("\n=== Synthetic correlations (derivative features) ===")
s_sub = synth[:, deriv_idx]
s_corr = np.corrcoef(s_sub.T)
for i, ni in enumerate(deriv_names):
    for j, nj in enumerate(deriv_names):
        if j > i:
            print(f"  {ni:20s} x {nj:20s}  r={s_corr[i,j]:+.3f}")

# Distribution comparison
print("\n=== Feature distributions (mean +/- std) ===")
print(f"  {'Feature':25s} {'Human':>20s} {'Synthetic':>20s}")
for i, name in enumerate(FEATURE_NAMES):
    hm, hs = human[:, i].mean(), human[:, i].std()
    sm, ss = synth[:, i].mean(), synth[:, i].std()
    print(f"  {name:25s} {hm:8.3f} +/- {hs:8.3f}   {sm:8.3f} +/- {ss:8.3f}")

# Check if the issue is per-trajectory or distributional
print("\n=== Per-trajectory derivative consistency ===")
# For each synthetic trajectory, check if its internal derivatives are consistent
speed_acc_corrs = []
for feats in synth_features:
    # These are summary stats, not per-timestep
    pass

# Actually check raw speed profiles
print("\n=== Speed profile analysis (5 samples) ===")
for i in range(5):
    dist = float(rng.choice(distances))
    angle = float(rng.uniform(0, 2 * np.pi))
    ex = 960 + dist * np.cos(angle)
    ey = 540 + dist * np.sin(angle)
    traj = generate_path(960, 540, ex, ey)
    pts = np.array(resample_trajectory(traj))
    dt = np.maximum(np.diff(pts[:, 2]), 1e-6)
    ds = np.sqrt(np.diff(pts[:, 0])**2 + np.diff(pts[:, 1])**2)
    speed = ds / dt
    acc = np.diff(speed) / dt[:-1]
    print(f"  Traj {i}: len={len(pts)}, speed: mean={speed.mean():.1f} std={speed.std():.1f}, "
          f"acc: mean={acc.mean():.1f} std={acc.std():.1f}, "
          f"speed_range=[{speed.min():.1f}, {speed.max():.1f}]")
