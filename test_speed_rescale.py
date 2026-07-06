"""Test speed rescaling post-processing on CPU.

Hypothesis: The speed-block correlation gap comes from inconsistent speed
scaling across trajectories. If we rescale each trajectory's speed to match
the expected scale from conditioning, correlations should improve.
"""
import math
import os
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from models.candi import CANDIModel
from experiments._common import DurationModel
from features import FEATURE_NAMES, extract_features, resample_trajectory, extract_feature_matrix, normalized_wasserstein_by_feature

TRAIN_DIR = "training"
DATA_DIR = "data"
HZ = 125.0

ckpt = torch.load(f"{TRAIN_DIR}/candi_polar_best.pt", map_location="cpu", weights_only=False)
cfg = ckpt["config"]
data_scale = ckpt["data_scale"]

model = CANDIModel(**cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded on CPU, {sum(p.numel() for p in model.parameters()):,} params")

duration_model = DurationModel(TRAIN_DIR, std_mult=0.7)

human_features = np.load(f"{DATA_DIR}/human_eval_features.npy")
distances = np.load(f"{DATA_DIR}/human_distances.npy")
print(f"Human features: {human_features.shape}")

rng = np.random.default_rng(42)
N = 300
spd_s = float(data_scale[0])
dh_s = float(data_scale[1])


def generate_one(total_dist, angle, rescale=False):
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

    moving = stall_np < 0.5
    speed[~moving] = 0.0
    dh[~moving] = 0.0

    if rescale and moving.sum() > 0:
        actual_mean = speed[moving].mean()
        expected_mean = total_dist / duration / moving.sum()
        if actual_mean > 1e-8:
            scale_factor = expected_mean / actual_mean
            speed[moving] *= scale_factor

    heading = np.cumsum(dh)
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    cum_x = np.cumsum(vx)
    cum_y = np.cumsum(vy)

    raw_mag = math.hypot(cum_x[-1], cum_y[-1])
    if raw_mag > 1e-8:
        tgt_mag = 1.0
        scale = tgt_mag / raw_mag
        raw_ang = math.atan2(cum_y[-1], cum_x[-1])
        tgt_ang = angle
        rot = tgt_ang - raw_ang
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rx = (cum_x * cos_r - cum_y * sin_r) * scale
        ry = (cum_x * sin_r + cum_y * cos_r) * scale
        cum_x, cum_y = rx, ry

    out_x = cum_x * total_dist
    out_y = cum_y * total_dist

    dt = 1.0 / HZ
    traj = [(0.0, 0.0, 0.0)]
    for i in range(seq_len):
        traj.append((float(out_x[i]), float(out_y[i]), (i + 1) * dt))
    return traj


def run_experiment(rescale, label):
    print(f"\n--- {label} ---")
    synth_feats = []
    for i in range(N):
        d_idx = rng.integers(len(distances))
        total_dist = float(distances[d_idx])
        angle = rng.uniform(0, 2 * math.pi)
        traj = generate_one(total_dist, angle, rescale=rescale)
        resampled = resample_trajectory(traj)
        feats = extract_features(resampled)
        if feats is not None:
            synth_feats.append(feats)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N}", flush=True)

    synth_arr = np.array(synth_feats)
    print(f"  Valid: {len(synth_feats)}/{N}")

    # Check speed block correlations
    speed_idx = [0, 1, 2, 4, 5, 6, 7, 8]
    speed_names = ['mean_vel', 'std_vel', 'max_vel', 'mean_acc', 'std_acc',
                   'max_acc', 'mean_jerk', 'std_jerk']

    corr = np.corrcoef(synth_arr[:, speed_idx].T)
    print("  Speed block correlations (synth):")
    for i, n in enumerate(speed_names):
        row = ' '.join(f'{corr[i,j]:+.3f}' for j in range(len(speed_names)))
        print(f'    {n:>10s}  {row}')

    # Key correlations
    print("\n  Key correlation gaps:")
    human_corr = np.corrcoef(human_features[:, speed_idx].T)
    for i in range(len(speed_idx)):
        for j in range(i+1, len(speed_idx)):
            gap = abs(human_corr[i,j] - corr[i,j])
            if gap > 0.5:
                print(f"    {speed_names[i]:>10s} x {speed_names[j]:<10s}  "
                      f"human={human_corr[i,j]:+.3f}  synth={corr[i,j]:+.3f}  gap={gap:.3f}")

    # Wasserstein distances
    print("\n  Wasserstein distances (top 5):")
    wd = normalized_wasserstein_by_feature(human_features, synth_arr)
    pairs = sorted(zip(FEATURE_NAMES, wd), key=lambda x: -x[1])
    for name, d in pairs[:5]:
        print(f"    {name:>25s}  {d:.4f}")


run_experiment(rescale=False, label="Baseline (no rescale)")
run_experiment(rescale=True, label="With speed rescale")
