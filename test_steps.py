"""Quick test: how does sampling step count affect quality?
Test 50, 100, 200, 400 steps with the best config.
"""
import os, sys, math
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CANDI_CKPT"] = "candi_polar_flow_best.pt"
os.environ["CANDI_CFG"] = "0.0"
os.environ["CANDI_ETA"] = "0.0"
os.environ["CANDI_CANDIDATES"] = "1"
os.environ["CANDI_GUIDE"] = "0.3"
os.environ["CANDI_CORRECT"] = "rotate"
os.environ["CANDI_SPEED_SKEW"] = "0.3"
os.environ["CANDI_PERP_SCALE"] = "0.7"

# Start with 200 steps to import the module
os.environ["CANDI_STEPS"] = "200"

from features import extract_features, resample_trajectory, FEATURE_NAMES
import experiments.candi as candi_mod
import time

dists = np.load("data/human_distances.npy")
rng = np.random.default_rng(42)

def gen_and_measure(n_steps, n_traj=15):
    candi_mod._N_SAMPLE_STEPS = n_steps
    feats = []
    t0 = time.time()
    for i in range(n_traj):
        d = float(rng.choice(dists))
        ang = float(rng.uniform(0, 2 * math.pi))
        ex = 960.0 + d * math.cos(ang)
        ey = 540.0 + d * math.sin(ang)
        traj = candi_mod.generate_path(960.0, 540.0, ex, ey)
        f = extract_features(resample_trajectory(traj))
        if f is not None and not np.any(np.isnan(f)):
            feats.append(f)
    elapsed = time.time() - t0
    return np.array(feats), elapsed

human = np.load("data/human_eval_features.npy")
from features import normalized_wasserstein_by_feature

for steps in [50, 100, 200]:
    rng = np.random.default_rng(42)  # reset seed for fair comparison
    feats, elapsed = gen_and_measure(steps, n_traj=15)
    if len(feats) < 5:
        print(f"\nsteps={steps}: too few valid ({len(feats)})")
        continue

    wds = normalized_wasserstein_by_feature(human[:len(feats)], feats)
    mean_wd = np.mean(wds)
    max_wd = np.max(wds)

    print(f"\nsteps={steps}: {elapsed:.1f}s for {len(feats)} traj ({elapsed/len(feats):.1f}s/traj)")
    print(f"  mean WD={mean_wd:.4f}, max WD={max_wd:.4f}")
    # Top 5 WDs
    ranked = sorted(zip(FEATURE_NAMES, wds), key=lambda x: x[1], reverse=True)
    for name, d in ranked[:5]:
        print(f"  {name:30s} {d:.4f}")
