"""Diagnose mean_acceleration and mean_jerk distributions in human vs synthetic."""
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CANDI_CKPT"] = "candi_polar_flow_best.pt"
os.environ["CANDI_CFG"] = "0.0"
os.environ["CANDI_STEPS"] = "200"
os.environ["CANDI_ETA"] = "0.0"
os.environ["CANDI_CANDIDATES"] = "1"
os.environ["CANDI_GUIDE"] = "0.3"
os.environ["CANDI_CORRECT"] = "rotate"
os.environ["CANDI_SPEED_SKEW"] = "0.3"
os.environ["CANDI_PERP_SCALE"] = "0.7"

from features import extract_features, resample_trajectory, FEATURE_NAMES

human_feats = np.load("data/human_eval_features.npy")
print(f"Human features: {human_feats.shape}")

# Key feature indices
idx_mean_vel = FEATURE_NAMES.index("mean_velocity")
idx_mean_acc = FEATURE_NAMES.index("mean_acceleration")
idx_mean_jerk = FEATURE_NAMES.index("mean_jerk")
idx_max_vel = FEATURE_NAMES.index("max_velocity")
idx_std_acc = FEATURE_NAMES.index("std_acceleration")
idx_vel_skew = FEATURE_NAMES.index("velocity_skewness")

print("\n=== HUMAN DATA ===")
h_acc = human_feats[:, idx_mean_acc]
h_jerk = human_feats[:, idx_mean_jerk]
h_vel = human_feats[:, idx_mean_vel]
h_max_vel = human_feats[:, idx_max_vel]

print(f"mean_acceleration: mean={np.mean(h_acc):.6f}, std={np.std(h_acc):.6f}, "
      f"min={np.min(h_acc):.6f}, max={np.max(h_acc):.6f}")
print(f"mean_jerk:         mean={np.mean(h_jerk):.6f}, std={np.std(h_jerk):.6f}, "
      f"min={np.min(h_jerk):.6f}, max={np.max(h_jerk):.6f}")
print(f"mean_velocity:     mean={np.mean(h_vel):.6f}, std={np.std(h_vel):.6f}")
print(f"max_velocity:      mean={np.mean(h_max_vel):.6f}, std={np.std(h_max_vel):.6f}")

print(f"\nCorrelations in human data:")
print(f"  mean_acc × mean_vel:  {np.corrcoef(h_acc, h_vel)[0,1]:.4f}")
print(f"  mean_acc × max_vel:   {np.corrcoef(h_acc, h_max_vel)[0,1]:.4f}")
print(f"  mean_acc × mean_jerk: {np.corrcoef(h_acc, h_jerk)[0,1]:.4f}")
print(f"  mean_jerk × mean_vel: {np.corrcoef(h_jerk, h_vel)[0,1]:.4f}")

# Check percentiles
print(f"\nmean_acceleration percentiles:")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  p{p}: {np.percentile(h_acc, p):.6f}")

print(f"\nmean_jerk percentiles:")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  p{p}: {np.percentile(h_jerk, p):.6f}")

# Check if mean_acc is near zero or has real signal
print(f"\nFraction with mean_acc > 0: {np.mean(h_acc > 0):.3f}")
print(f"Fraction with mean_acc > 1: {np.mean(h_acc > 1):.3f}")
print(f"Fraction with mean_jerk > 0: {np.mean(h_jerk > 0):.3f}")

# Check relationship with trajectory speed
# Bin by mean_velocity and look at mean_acceleration
vel_bins = np.percentile(h_vel, [0, 25, 50, 75, 100])
for i in range(4):
    mask = (h_vel >= vel_bins[i]) & (h_vel < vel_bins[i+1] + 0.001)
    if mask.sum() > 0:
        print(f"\nVelocity bin [{vel_bins[i]:.2f}, {vel_bins[i+1]:.2f}]:")
        print(f"  n={mask.sum()}, mean_acc={np.mean(h_acc[mask]):.6f}, "
              f"mean_jerk={np.mean(h_jerk[mask]):.6f}")

# Now generate synthetic and compare
print("\n\n=== GENERATING SYNTHETIC DATA ===")
from experiments.candi import generate_path

rng = np.random.default_rng(42)
targets = np.load("data/eval_targets.npz")
sx, sy, ex, ey = targets["start_x"], targets["start_y"], targets["end_x"], targets["end_y"]

n_gen = 50
synth_feats = []
for i in range(n_gen):
    idx = i % len(sx)
    traj = generate_path(float(sx[idx]), float(sy[idx]), float(ex[idx]), float(ey[idx]))
    resampled = resample_trajectory(traj)
    f = extract_features(resampled)
    if f is not None and not np.any(np.isnan(f)):
        synth_feats.append(f)
    if (i+1) % 10 == 0:
        print(f"  Generated {i+1}/{n_gen}", flush=True)

synth = np.array(synth_feats)
print(f"\nSynthetic features: {synth.shape}")

s_acc = synth[:, idx_mean_acc]
s_jerk = synth[:, idx_mean_jerk]
s_vel = synth[:, idx_mean_vel]
s_max_vel = synth[:, idx_max_vel]

print("\n=== SYNTHETIC DATA ===")
print(f"mean_acceleration: mean={np.mean(s_acc):.6f}, std={np.std(s_acc):.6f}, "
      f"min={np.min(s_acc):.6f}, max={np.max(s_acc):.6f}")
print(f"mean_jerk:         mean={np.mean(s_jerk):.6f}, std={np.std(s_jerk):.6f}, "
      f"min={np.min(s_jerk):.6f}, max={np.max(s_jerk):.6f}")
print(f"mean_velocity:     mean={np.mean(s_vel):.6f}, std={np.std(s_vel):.6f}")

print(f"\nCorrelations in synthetic data:")
print(f"  mean_acc × mean_vel:  {np.corrcoef(s_acc, s_vel)[0,1]:.4f}")
print(f"  mean_acc × max_vel:   {np.corrcoef(s_acc, s_max_vel)[0,1]:.4f}")
print(f"  mean_acc × mean_jerk: {np.corrcoef(s_acc, s_jerk)[0,1]:.4f}")
print(f"  mean_jerk × mean_vel: {np.corrcoef(s_jerk, s_vel)[0,1]:.4f}")

print(f"\nFraction with mean_acc > 0: {np.mean(s_acc > 0):.3f}")
print(f"Fraction with mean_jerk > 0: {np.mean(s_jerk > 0):.3f}")

# Directly analyze a few speed profiles
print("\n\n=== ANALYZING SPEED PROFILES ===")
for i in range(min(5, n_gen)):
    idx = i % len(sx)
    traj = generate_path(float(sx[idx]), float(sy[idx]), float(ex[idx]), float(ey[idx]))
    resampled = resample_trajectory(traj)
    pts = np.array(resampled)
    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])
    dt = np.maximum(np.diff(pts[:, 2]), 1e-6)
    speed = np.sqrt(dx**2 + dy**2) / dt
    acc = np.diff(speed) / dt[:-1]

    peak_idx = np.argmax(speed)
    print(f"\nTrajectory {i}: len={len(speed)}, peak_idx={peak_idx}/{len(speed)}, "
          f"peak_frac={peak_idx/len(speed):.2f}")
    print(f"  speed[0:3]={speed[:3]}, speed[-3:]={speed[-3:]}")
    print(f"  mean_acc={np.mean(acc):.6f}, sum_acc={np.sum(acc):.6f}")
    print(f"  speed[-1] - speed[0] = {speed[-1] - speed[0]:.6f}")
    # Check acceleration in first half vs second half
    mid = len(acc) // 2
    print(f"  mean_acc first half={np.mean(acc[:mid]):.4f}, second half={np.mean(acc[mid:]):.4f}")
