"""Quick diagnostic: compare human vs synthetic feature distributions for top gaps."""
import os
import sys
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

human = np.load("data/human_eval_features.npy")

# Key feature indices with highest WD
features_of_interest = [
    "time_to_peak_velocity",
    "curvature_mean",
    "curvature_std",
    "num_direction_changes",
    "velocity_skewness",
    "angular_velocity_std",
    "movement_duration",
    "angular_velocity_mean",
]

print("=== HUMAN FEATURE DISTRIBUTIONS ===")
for name in features_of_interest:
    idx = FEATURE_NAMES.index(name)
    vals = human[:, idx]
    std = np.std(vals)
    print(f"\n{name}:")
    print(f"  mean={np.mean(vals):.4f}, std={std:.4f}, median={np.median(vals):.4f}")
    print(f"  p5={np.percentile(vals, 5):.4f}, p25={np.percentile(vals, 25):.4f}, "
          f"p75={np.percentile(vals, 75):.4f}, p95={np.percentile(vals, 95):.4f}")

# Generate synthetic
print("\n\n=== GENERATING 30 SYNTHETIC (best config) ===")
from experiments.candi import generate_path

rng = np.random.default_rng(42)
dists = np.load("data/human_distances.npy")
import math
synth_feats = []
for i in range(30):
    d = float(rng.choice(dists))
    ang = float(rng.uniform(0, 2 * math.pi))
    ex = 960.0 + d * math.cos(ang)
    ey = 540.0 + d * math.sin(ang)
    traj = generate_path(960.0, 540.0, ex, ey)
    f = extract_features(resample_trajectory(traj))
    if f is not None and not np.any(np.isnan(f)):
        synth_feats.append(f)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/30", flush=True)

synth = np.array(synth_feats)
print(f"\nSynthetic: {synth.shape[0]} valid")

print("\n=== COMPARISON (feature: human_mean -> synth_mean, direction of gap) ===")
for name in features_of_interest:
    idx = FEATURE_NAMES.index(name)
    h = human[:, idx]
    s = synth[:, idx]
    h_std = np.std(h)
    if h_std < 1e-10:
        continue
    direction = "synth HIGH" if np.mean(s) > np.mean(h) else "synth LOW"
    wd = float(np.mean(np.abs(np.sort(s/h_std) - np.sort(np.random.choice(h/h_std, len(s), replace=False)))))
    print(f"\n{name}:")
    print(f"  human  mean={np.mean(h):.4f}  std={np.std(h):.4f}  median={np.median(h):.4f}")
    print(f"  synth  mean={np.mean(s):.4f}  std={np.std(s):.4f}  median={np.median(s):.4f}")
    print(f"  >> {direction}")
