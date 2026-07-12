"""Quick AUC screening for model checkpoints.

Generates 200 trajectories and computes approximate AUC. Much faster than
the full evaluate.py (2000 trajectories). Use for checkpoint screening,
not final results.

Usage:
    python quick_eval.py [checkpoint_name] [--guide G] [--correct rotate]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from models.candi import CANDIModel
from experiments._common import DurationModel
from features import (FEATURE_NAMES, extract_features, resample_trajectory,
                      normalized_wasserstein_by_feature)

TRAIN_DIR = "training"
DATA_DIR = "data"
HZ = 125.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="candi_polar_best.pt")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--guide", type=float, default=0.3)
    parser.add_argument("--correct", default="rotate", choices=["rotate", "additive"])
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--jitter", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ckpt_path = f"{TRAIN_DIR}/{args.checkpoint}"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    data_scale = ckpt["data_scale"]
    pred_type = ckpt.get("pred_type", "x0")
    epoch = ckpt.get("epoch", "?")

    model = CANDIModel(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    duration_model = DurationModel(TRAIN_DIR, std_mult=0.7)
    human_features = np.load(f"{DATA_DIR}/human_eval_features.npy")
    distances = np.load(f"{DATA_DIR}/human_distances.npy")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Checkpoint: {args.checkpoint} (epoch {epoch}, pred_type={pred_type})")
    print(f"Model: {n_params:,} params, guide={args.guide}, correct={args.correct}, "
          f"cfg={args.cfg}, steps={args.steps}, jitter={args.jitter}")

    rng = np.random.default_rng(args.seed)
    spd_s = float(data_scale[0])
    dh_s = float(data_scale[1])
    N = args.n

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
            if pred_type == "flow":
                raw, stall = model.flow_sample(cond, seq_len,
                                                n_steps=args.steps, cfg_scale=args.cfg)
            else:
                raw, stall = model.sample(cond, seq_len,
                                           n_steps=args.steps, eta=0.0,
                                           cfg_scale=args.cfg, pred_type=pred_type)

        raw_np = raw[0].numpy()
        stall_np = stall[0].numpy()

        speed = np.clip(raw_np[:, 0] / spd_s, 0, None)
        dh = raw_np[:, 1] / dh_s
        speed[stall_np > 0.5] = 0.0
        dh[stall_np > 0.5] = 0.0

        if args.jitter > 0:
            noise = rng.normal(0, args.jitter, size=dh.shape) * speed
            noise[stall_np > 0.5] = 0.0
            dh = dh + noise

        heading = np.cumsum(dh)
        cum_x = np.cumsum(speed * np.cos(heading))
        cum_y = np.cumsum(speed * np.sin(heading))

        if args.correct == "rotate":
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
        traj[-1] = (total_dist * math.cos(angle), total_dist * math.sin(angle), traj[-1][2])

        resampled = resample_trajectory(traj)
        feats = extract_features(resampled)
        if feats is not None:
            synth_feats.append(feats)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N}", flush=True)

    synth_arr = np.array(synth_feats)
    n_valid = len(synth_feats)
    print(f"Valid: {n_valid}/{N}")

    if n_valid < 50:
        print("Too few valid trajectories. Aborting.")
        return

    n_use = min(len(human_features), n_valid)
    h_bal = human_features[:n_use]
    s_bal = synth_arr[:n_use]

    X = np.vstack([h_bal, s_bal])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])

    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                  random_state=args.seed)
    clf.fit(X, y)
    oob_proba = clf.oob_decision_function_[:, 1]
    auc = roc_auc_score(y, oob_proba)

    print(f"\nval_auc: {auc:.4f} (approx, N={n_use})")

    wd = normalized_wasserstein_by_feature(h_bal, s_bal)
    ranked = sorted(zip(FEATURE_NAMES, wd), key=lambda x: -x[1])
    print(f"\nTop Wasserstein distances:")
    for name, d in ranked[:8]:
        print(f"  {name:>25s}  {d:.4f}")

    importances = clf.feature_importances_
    ranked_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    print(f"\nRF feature importances:")
    for name, imp in ranked_imp[:8]:
        print(f"  {name:>25s}  {imp:.4f}")


if __name__ == "__main__":
    main()
