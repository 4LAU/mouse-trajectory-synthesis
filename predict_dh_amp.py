"""Quick test: what does DH_AMP do to curvature and angular velocity features?
Generates 10 trajectories with different DH_AMP values to predict optimal range.
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
rng = np.random.default_rng(42)

idx_curv_mean = FEATURE_NAMES.index("curvature_mean")
idx_curv_std = FEATURE_NAMES.index("curvature_std")
idx_ang_std = FEATURE_NAMES.index("angular_velocity_std")
idx_ang_mean = FEATURE_NAMES.index("angular_velocity_mean")
idx_max_dev = FEATURE_NAMES.index("max_deviation")
idx_vel_skew = FEATURE_NAMES.index("velocity_skewness")

def gen_samples(dh_amp, n=10):
    candi_mod._DH_AMP = dh_amp
    feats = []
    for i in range(n):
        d = float(rng.choice(dists))
        ang = float(rng.uniform(0, 2 * math.pi))
        ex = 960.0 + d * math.cos(ang)
        ey = 540.0 + d * math.sin(ang)
        traj = candi_mod.generate_path(960.0, 540.0, ex, ey)
        f = extract_features(resample_trajectory(traj))
        if f is not None and not np.any(np.isnan(f)):
            feats.append(f)
    return np.array(feats)

print("Human medians:")
print(f"  curvature_mean:  {np.median(human[:, idx_curv_mean]):.4f}")
print(f"  curvature_std:   {np.median(human[:, idx_curv_std]):.4f}")
print(f"  ang_vel_std:     {np.median(human[:, idx_ang_std]):.4f}")
print(f"  ang_vel_mean:    {np.median(human[:, idx_ang_mean]):.4f}")
print(f"  max_deviation:   {np.median(human[:, idx_max_dev]):.4f}")
print(f"  vel_skewness:    {np.median(human[:, idx_vel_skew]):.4f}")

for amp in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]:
    s = gen_samples(amp, n=10)
    if len(s) == 0:
        print(f"\ndh_amp={amp}: no valid samples")
        continue
    print(f"\ndh_amp={amp}: (n={len(s)})")
    print(f"  curvature_mean:  {np.median(s[:, idx_curv_mean]):.4f}")
    print(f"  curvature_std:   {np.median(s[:, idx_curv_std]):.4f}")
    print(f"  ang_vel_std:     {np.median(s[:, idx_ang_std]):.4f}")
    print(f"  ang_vel_mean:    {np.median(s[:, idx_ang_mean]):.4f}")
    print(f"  max_deviation:   {np.median(s[:, idx_max_dev]):.4f}")
    print(f"  vel_skewness:    {np.median(s[:, idx_vel_skew]):.4f}")
