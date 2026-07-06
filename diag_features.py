"""Diagnose feature distribution gaps between human and DDPM synthetic data.

Generates trajectories on CPU and compares distributions of top discriminating
features. Outputs: direction of gap (too high/low), percentiles, shapes.
"""
import math
import os
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from models.candi import CANDIModel
from experiments._common import DurationModel
from features import (FEATURE_NAMES, extract_features, resample_trajectory,
                      extract_feature_matrix, normalized_wasserstein_by_feature)

TRAIN_DIR = "training"
DATA_DIR = "data"
HZ = 125.0
N = 200

ckpt = torch.load(f"{TRAIN_DIR}/candi_polar_best.pt", map_location="cpu", weights_only=False)
cfg = ckpt["config"]
data_scale = ckpt["data_scale"]
model = CANDIModel(**cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

duration_model = DurationModel(TRAIN_DIR, std_mult=0.7)
human_features = np.load(f"{DATA_DIR}/human_eval_features.npy")
distances = np.load(f"{DATA_DIR}/human_distances.npy")

rng = np.random.default_rng(42)
spd_s = float(data_scale[0])
dh_s = float(data_scale[1])

print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
print(f"Human features: {human_features.shape}")
print(f"Generating {N} synthetic trajectories on CPU...")

synth_feats = []
for i in range(N):
    d_idx = rng.integers(len(distances))
    total_dist = float(distances[d_idx])
    angle = rng.uniform(0, 2 * math.pi)
    log_dist = math.log(total_dist)
    duration = duration_model.sample(log_dist)
    log_dur = math.log(duration)
    seq_len = max(5, min(int(round(duration * HZ)), cfg["max_seq_len"]))

    cond = torch.tensor([[log_dist, log_dur, math.cos(angle), math.sin(angle)]])
    with torch.no_grad():
        raw, stall = model.sample(cond, seq_len, n_steps=50, eta=0.0,
                                   cfg_scale=0.0, pred_type="x0")
    raw_np = raw[0].numpy()
    stall_np = stall[0].numpy()

    speed = np.clip(raw_np[:, 0] / spd_s, 0, None)
    dh = raw_np[:, 1] / dh_s
    speed[stall_np > 0.5] = 0.0
    dh[stall_np > 0.5] = 0.0

    heading = np.cumsum(dh)
    cum_x = np.cumsum(speed * np.cos(heading))
    cum_y = np.cumsum(speed * np.sin(heading))

    raw_mag = math.hypot(cum_x[-1], cum_y[-1])
    if raw_mag > 1e-8:
        tgt_mag = 1.0
        scale = tgt_mag / raw_mag
        raw_ang = math.atan2(cum_y[-1], cum_x[-1])
        rot = angle - raw_ang
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rx = (cum_x * cos_r - cum_y * sin_r) * scale
        ry = (cum_x * sin_r + cum_y * cos_r) * scale
        cum_x, cum_y = rx, ry

    out_x = cum_x * total_dist
    out_y = cum_y * total_dist

    dt = 1.0 / HZ
    traj = [(0.0, 0.0, 0.0)]
    for j in range(seq_len):
        traj.append((float(out_x[j]), float(out_y[j]), (j + 1) * dt))

    resampled = resample_trajectory(traj)
    feats = extract_features(resampled)
    if feats is not None:
        synth_feats.append(feats)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N}", flush=True)

synth_arr = np.array(synth_feats)
print(f"Valid: {len(synth_feats)}/{N}\n")

# Per-feature comparison
wd = normalized_wasserstein_by_feature(human_features, synth_arr)
ranked = sorted(zip(range(len(FEATURE_NAMES)), FEATURE_NAMES, wd),
                key=lambda x: -x[2])

print(f"{'Feature':>25s}  {'WD':>6s}  {'H_mean':>10s}  {'S_mean':>10s}  {'Direction':>10s}  {'H_std':>10s}  {'S_std':>10s}  {'StdRatio':>8s}")
print("-" * 105)
for idx, name, d in ranked:
    h_mean = human_features[:, idx].mean()
    s_mean = synth_arr[:, idx].mean()
    h_std = human_features[:, idx].std()
    s_std = synth_arr[:, idx].std()
    direction = "TOO HIGH" if s_mean > h_mean else "TOO LOW"
    std_ratio = s_std / h_std if h_std > 1e-8 else float('inf')
    print(f"{name:>25s}  {d:6.4f}  {h_mean:10.4f}  {s_mean:10.4f}  {direction:>10s}  {h_std:10.4f}  {s_std:10.4f}  {std_ratio:8.3f}")

# Percentile comparison for top 5
print("\n--- Percentile comparison (top 5 discriminating features) ---")
pcts = [5, 25, 50, 75, 95]
for idx, name, d in ranked[:5]:
    h_p = np.percentile(human_features[:, idx], pcts)
    s_p = np.percentile(synth_arr[:, idx], pcts)
    print(f"\n  {name} (WD={d:.4f}):")
    print(f"    {'':>8s}  {'p5':>10s}  {'p25':>10s}  {'p50':>10s}  {'p75':>10s}  {'p95':>10s}")
    print(f"    {'Human':>8s}  {h_p[0]:10.4f}  {h_p[1]:10.4f}  {h_p[2]:10.4f}  {h_p[3]:10.4f}  {h_p[4]:10.4f}")
    print(f"    {'Synth':>8s}  {s_p[0]:10.4f}  {s_p[1]:10.4f}  {s_p[2]:10.4f}  {s_p[3]:10.4f}  {s_p[4]:10.4f}")
