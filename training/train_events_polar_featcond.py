"""Stage 3: feature-conditioned fine-tune of the WS7b polar event model.

Both fine-tuning signals (fixed statistics, adversarial critic) made the 4M
base WORSE, and the correlation diagnosis says why: the remaining detector
gap is per-trajectory cross-feature coherence (in humans mean_acc, std_acc,
std_jerk and friends move together; in samples they do not), and a masked
token model with per-position heads has no global variable that could carry
that. So give it one: condition the trunk on the trajectory's own 18
detector features (z-scored), teacher-forced on real data with the standard
pretraining losses. The projection is zero-initialized, so training starts
exactly at the pretrained model and the conditioning pathway only grows as
it earns loss.

At sampling time the feature vector comes from a kernel density estimate
over real feature vectors (a bank stored in this checkpoint): draw a bank
row, add Gaussian noise of bandwidth EVENT_FEAT_BW. That is a generative
density over movement characters, so the pipeline stays level 3.

Run:
    .venv/Scripts/python.exe training/train_events_polar_featcond.py \
        --load-from event_polar_4m.pt --save-name event_polar_4m_fc_v1.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.event_stream_polar import EventStreamPolarModel, TICK_CLASS, S_PAD_CLASS, TH_BINS  # noqa: E402
from training.train_events_polar import PolarEventDataset  # noqa: E402
from training.train_events_polar_dm import (  # noqa: E402
    build_value_tables, detector_features, real_batch_values, stream_to_frames,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    ckpt = torch.load(data_dir / args.load_from, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    dt_mean, dt_std = float(ckpt["dt_mean"]), float(ckpt["dt_std"])
    cfg["feat_dim"] = args.n_feat
    model = EventStreamPolarModel(**cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("feat_embed") for k in missing), missing
    print(f"Loaded {args.load_from} (epoch {ckpt.get('epoch')}), "
          f"feat_embed fresh ({len(missing)} tensors, zero-init output)", flush=True)

    print("Loading polar event data...", flush=True)
    s2 = np.load(data_dir / "events_s2.npy", mmap_mode="r")
    dth = np.load(data_dir / "events_dth.npy", mmap_mode="r")
    dt = np.load(data_dir / "events_dt.npy", mmap_mode="r")
    lengths = np.load(data_dir / "events_len.npy")
    conditions = np.load(data_dir / "events_cond.npy")
    N = len(lengths)
    rng = np.random.default_rng(123)
    idx = np.sort(rng.choice(N, min(N, 400_000), replace=False))
    ds = PolarEventDataset(s2[idx], dth[idx], dt[idx], lengths[idx],
                           conditions[idx], cfg["max_seq_len"], dt_mean, dt_std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.num_workers > 0)
    print(f"  {len(ds):,} trajectories", flush=True)

    tables = build_value_tables(device)

    def batch_features(s_cls, th_cls, dt_z, real, cond):
        dt_s = torch.exp(dt_z * dt_std + dt_mean).clamp(0.1, 1000.0) / 1000.0
        speed, motion, tick, cos_th, sin_th = real_batch_values(s_cls, th_cls, tables)
        x, y, fmask = stream_to_frames(speed, motion, cos_th, sin_th,
                                       dt_s, real, cond, args.n_frames)
        return detector_features(x, y, fmask)

    # feature standardization + the sampling bank, both from real data. The
    # bank keeps each row's log-distance so eval can draw a movement
    # character consistent with the requested distance.
    stats_feats, stats_cond = [], []
    with torch.no_grad():
        for bi, (dt_z, s_cls, th_cls, real, cond) in enumerate(dl):
            dt_z, s_cls, th_cls, real, cond = (
                x.to(device) for x in (dt_z, s_cls, th_cls, real, cond))
            stats_feats.append(batch_features(s_cls, th_cls, dt_z, real, cond))
            stats_cond.append(cond[:, 0])
            if bi >= 63:
                break
    sf = torch.cat(stats_feats)
    f_mu, f_sd = sf.mean(0), sf.std(0).clamp(min=1e-4)
    bank = ((sf - f_mu) / f_sd).clamp(-10.0, 10.0)[:args.bank_size].cpu()
    bank_log_dist = torch.cat(stats_cond)[:args.bank_size].cpu()
    n_feat = sf.shape[1]
    assert n_feat == args.n_feat, (n_feat, args.n_feat)
    print(f"  feature stats over {len(sf)} real trajectories, "
          f"bank {len(bank)} x {n_feat}", flush=True)

    for p in model.dt_head.parameters():
        p.requires_grad_(False)
    g_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(g_params, lr=args.lr, weight_decay=0.0)

    save_path = data_dir / args.save_name
    latest_path = save_path.with_stem(save_path.stem + "_latest")
    start_step = 0
    if args.auto_resume and latest_path.exists():
        rck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(rck["model_state_dict"])
        opt.load_state_dict(rck["opt_state_dict"])
        start_step = rck["step"]
        print(f"  Resumed at step {start_step}", flush=True)

    model.train()
    step_i = start_step
    t0 = time.time()
    ema = None
    data_iter = iter(dl)
    while step_i < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        dt_z, s_cls, th_cls, real, cond = (x.to(device) for x in batch)
        B = dt_z.shape[0]

        with torch.no_grad():
            feat = ((batch_features(s_cls, th_cls, dt_z, real, cond) - f_mu)
                    / f_sd).clamp(-10.0, 10.0)

        t_cont = torch.rand(B, device=device)
        t_int = (t_cont * (model.n_steps - 1)).long()
        dt_noisy, _, velocity = model.q_flow(dt_z, t_cont)
        s_m, th_m, mask = model.q_mask_joint(s_cls, th_cls, t_int)
        v_pred, s_logits, th_logits = model(
            dt_noisy, s_m, th_m, t_cont * (model.n_steps - 1), cond, s_cls,
            feat=feat,
        )
        w_flow = real + (1.0 - real) * 0.1
        flow_loss = ((v_pred - velocity) ** 2 * w_flow).sum() / w_flow.sum().clamp(1)
        ce_s = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]),
                               s_cls.reshape(-1), reduction="none").view(B, -1)
        ws = mask.float() * (real + (1.0 - real) * 0.15)
        s_loss = (ce_s * ws).sum() / ws.sum().clamp(1)
        motion = (s_cls > TICK_CLASS) & (s_cls < S_PAD_CLASS)
        ce_th = F.cross_entropy(th_logits.reshape(-1, th_logits.shape[-1]),
                                th_cls.clamp(max=TH_BINS - 1).reshape(-1),
                                reduction="none").view(B, -1)
        wt = (mask & motion).float()
        th_loss = (ce_th * wt).sum() / wt.sum().clamp(1)
        loss = flow_loss + s_loss + th_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(g_params, 1.0)
        opt.step()

        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        step_i += 1
        if step_i % 50 == 0 or step_i == 1:
            print(f"  step {step_i:5d}/{args.steps} | loss {ema:.4f} "
                  f"(flow {flow_loss.item():.3f} s {s_loss.item():.3f} "
                  f"th {th_loss.item():.3f}) | {time.time() - t0:.0f}s", flush=True)
        if step_i % args.save_every == 0 or step_i == args.steps:
            out = {
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "config": cfg, "dt_mean": dt_mean, "dt_std": dt_std,
                "feat_mu": f_mu.cpu(), "feat_sd": f_sd.cpu(),
                "feat_bank": bank, "feat_bank_log_dist": bank_log_dist,
                "step": step_i, "epoch": ckpt.get("epoch"),
            }
            torch.save(out, latest_path)
            torch.save(out, save_path)

    print(f"Done. Final loss (ema): {ema:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--load-from", default="event_polar_4m.pt")
    parser.add_argument("--save-name", default="event_polar_4m_fc_v1.pt")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n-feat", type=int, default=18)
    parser.add_argument("--n-frames", type=int, default=256)
    parser.add_argument("--bank-size", type=int, default=8192)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--auto-resume", action="store_true")
    train(parser.parse_args())
