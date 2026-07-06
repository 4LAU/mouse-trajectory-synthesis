"""Diagnose: does PERP_HP actually decouple curvature from max_deviation?
Generate same trajectories with different PERP_HP and compare features.
"""
import os, sys, math
import numpy as np

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
import experiments.candi as candi_mod

human = np.load("data/human_eval_features.npy")
dists = np.load("data/human_distances.npy")

idx_curv = FEATURE_NAMES.index("curvature_mean")
idx_curv_std = FEATURE_NAMES.index("curvature_std")
idx_max_dev = FEATURE_NAMES.index("max_deviation")
idx_ang_std = FEATURE_NAMES.index("angular_velocity_std")
idx_ang_mean = FEATURE_NAMES.index("angular_velocity_mean")
idx_vel_skew = FEATURE_NAMES.index("velocity_skewness")

print("Human medians:")
print(f"  curvature_mean:    {np.median(human[:, idx_curv]):.4f}")
print(f"  curvature_std:     {np.median(human[:, idx_curv_std]):.4f}")
print(f"  max_deviation:     {np.median(human[:, idx_max_dev]):.4f}")
print(f"  ang_vel_std:       {np.median(human[:, idx_ang_std]):.4f}")
print(f"  ang_vel_mean:      {np.median(human[:, idx_ang_mean]):.4f}")
print(f"  vel_skewness:      {np.median(human[:, idx_vel_skew]):.4f}")

for hp_val in [1.0, 1.3, 1.5, 2.0, 2.5, 3.0]:
    candi_mod._PERP_HP = hp_val
    candi_mod._PERP_HP_WIN = 21
    rng = np.random.default_rng(42)
    feats = []
    for i in range(20):
        d = float(rng.choice(dists))
        ang = float(rng.uniform(0, 2 * math.pi))
        ex = 960.0 + d * math.cos(ang)
        ey = 540.0 + d * math.sin(ang)
        traj = candi_mod.generate_path(960.0, 540.0, ex, ey)
        f = extract_features(resample_trajectory(traj))
        if f is not None and not np.any(np.isnan(f)):
            feats.append(f)
    feats = np.array(feats)
    print(f"\nPERP_HP={hp_val}: (n={len(feats)})")
    print(f"  curvature_mean:    {np.median(feats[:, idx_curv]):.4f}")
    print(f"  curvature_std:     {np.median(feats[:, idx_curv_std]):.4f}")
    print(f"  max_deviation:     {np.median(feats[:, idx_max_dev]):.4f}")
    print(f"  ang_vel_std:       {np.median(feats[:, idx_ang_std]):.4f}")
    print(f"  ang_vel_mean:      {np.median(feats[:, idx_ang_mean]):.4f}")
    print(f"  vel_skewness:      {np.median(feats[:, idx_vel_skew]):.4f}")

# Also test with different window sizes at HP=1.5
print("\n\n=== WINDOW SIZE SWEEP (PERP_HP=1.5) ===")
for win in [7, 11, 15, 21, 31, 41]:
    candi_mod._PERP_HP = 1.5
    candi_mod._PERP_HP_WIN = win
    rng = np.random.default_rng(42)
    feats = []
    for i in range(20):
        d = float(rng.choice(dists))
        ang = float(rng.uniform(0, 2 * math.pi))
        ex = 960.0 + d * math.cos(ang)
        ey = 540.0 + d * math.sin(ang)
        traj = candi_mod.generate_path(960.0, 540.0, ex, ey)
        f = extract_features(resample_trajectory(traj))
        if f is not None and not np.any(np.isnan(f)):
            feats.append(f)
    feats = np.array(feats)
    print(f"  win={win:3d}: curv={np.median(feats[:, idx_curv]):.4f}, "
          f"max_dev={np.median(feats[:, idx_max_dev]):.4f}, "
          f"ang_std={np.median(feats[:, idx_ang_std]):.2f}")
