"""Check if path geometry gaps vary by distance quantile."""
import math
import os
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from models.candi import CANDIModel
from experiments._common import DurationModel
from features import extract_features, resample_trajectory

TRAIN_DIR = "training"
DATA_DIR = "data"
HZ = 125.0
N = 300

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

# Feature indices for path geometry
PE_IDX = 10   # path_efficiency
MD_IDX = 11   # max_deviation
CM_IDX = 12   # curvature_mean
CS_IDX = 13   # curvature_std
NDC_IDX = 14  # num_direction_changes

# Generate with distance tracking
results = []
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
                                   cfg_scale=2.0, pred_type="x0")
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
        scale = 1.0 / raw_mag
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
        results.append((total_dist, feats))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N}", flush=True)

print(f"Valid: {len(results)}/{N}\n")

# Split by distance quantile
dists = np.array([r[0] for r in results])
feats = np.array([r[1] for r in results])
q33, q67 = np.percentile(dists, [33, 67])

bins = [("Short (<{:.0f}px)".format(q33), dists < q33),
        ("Med ({:.0f}-{:.0f}px)".format(q33, q67), (dists >= q33) & (dists < q67)),
        ("Long (>{:.0f}px)".format(q67), dists >= q67)]

# Also need human distances for binning human features
h_distances = np.load(f"{DATA_DIR}/human_distances.npy")
# The human features are pre-computed without distances paired... we can use h_distances
# but we don't know which feature row corresponds to which distance.
# So just report synthetic by distance and overall human as reference.

print(f"Distance quantiles: q33={q33:.0f}px, q67={q67:.0f}px\n")

feat_names = ["path_efficiency", "max_deviation", "curvature_mean",
              "curvature_std", "num_dir_changes"]
feat_idxs = [PE_IDX, MD_IDX, CM_IDX, CS_IDX, NDC_IDX]

for fi, fn in zip(feat_idxs, feat_names):
    h_med = np.median(human_features[:, fi])
    h_mean = np.mean(human_features[:, fi])
    print(f"\n{fn}:")
    print(f"  {'Group':>20s}  {'N':>4s}  {'Mean':>10s}  {'Median':>10s}  {'Std':>10s}  {'p5':>10s}  {'p95':>10s}")
    print(f"  {'Human (all)':>20s}  {len(human_features):>4d}  {h_mean:10.4f}  {h_med:10.4f}  {human_features[:, fi].std():10.4f}  {np.percentile(human_features[:, fi], 5):10.4f}  {np.percentile(human_features[:, fi], 95):10.4f}")
    for label, mask in bins:
        subset = feats[mask, fi]
        if len(subset) == 0:
            continue
        print(f"  {label:>20s}  {len(subset):>4d}  {subset.mean():10.4f}  {np.median(subset):10.4f}  {subset.std():10.4f}  {np.percentile(subset, 5):10.4f}  {np.percentile(subset, 95):10.4f}")
    syn_all = feats[:, fi]
    print(f"  {'Synth (all)':>20s}  {len(syn_all):>4d}  {syn_all.mean():10.4f}  {np.median(syn_all):10.4f}  {syn_all.std():10.4f}  {np.percentile(syn_all, 5):10.4f}  {np.percentile(syn_all, 95):10.4f}")
