"""Stage 2c: adversarial fine-tune of the movement-character-conditioned model.

The plain adversarial pass on the 4M base failed the same way the fixed-stat
DM did: the critic found a real gap (D gap grew 0.03 -> 1.80 and never closed)
that per-position heads cannot act on, because the gap is a global
per-trajectory property. The featcond model (fc_v2) now carries exactly that
global variable, teacher-forced during training. This script reruns the critic
with the character vector active on both the generation pass and the anchor
pass, so the generator finally has the pathway the criticism refers to.

Differences from train_events_polar_adv.py:
  - loads a featcond checkpoint (feat_dim > 0 required) and keeps its
    feat_mu/feat_sd/feat_bank so the saved checkpoint stays eval-compatible
  - each trajectory's own 18 detector features (z-scored by the checkpoint's
    stats, teacher-forced, same recipe as featcond training) are passed as
    feat to partial_reveal, st_complete, and the anchor forward
  - the critic input stays the resampled detector features z-scored by
    freshly measured real-data stats, exactly as in the plain adv script

Run:
    .venv/Scripts/python.exe training/train_events_polar_advfc.py --steps 800 \
        --load-from event_polar_4m_fc_v2.pt --save-name event_polar_4m_advfc_v1.pt
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
from training.train_events_polar_adv import FeatureCritic  # noqa: E402
from training.train_events_polar_dm import (  # noqa: E402
    build_value_tables, detector_features, partial_reveal,
    real_batch_values, st_complete, stream_to_frames,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    ckpt = torch.load(data_dir / args.load_from, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    assert cfg.get("feat_dim", 0) > 0, "load-from must be a featcond checkpoint"
    dt_mean, dt_std = float(ckpt["dt_mean"]), float(ckpt["dt_std"])
    feat_mu = ckpt["feat_mu"].to(device)
    feat_sd = ckpt["feat_sd"].to(device)
    model = EventStreamPolarModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {args.load_from} (feat_dim {cfg['feat_dim']})", flush=True)

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
    dl = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.num_workers > 0)
    print(f"  {len(ds):,} trajectories", flush=True)

    tables = build_value_tables(device)

    def resampled_features(vals, dt_s, real, cond):
        speed, motion, tick, cos_th, sin_th = vals
        x, y, fmask = stream_to_frames(speed, motion, cos_th, sin_th,
                                       dt_s, real, cond, args.n_frames)
        return detector_features(x, y, fmask)

    # critic feature standardization from real data (same recipe as adv stage)
    stats_feats = []
    with torch.no_grad():
        for bi, (dt_z, s_cls, th_cls, real, cond) in enumerate(dl):
            dt_z, s_cls, th_cls, real, cond = (
                x.to(device) for x in (dt_z, s_cls, th_cls, real, cond))
            dt_s = torch.exp(dt_z * dt_std + dt_mean).clamp(0.1, 1000.0) / 1000.0
            vals = real_batch_values(s_cls, th_cls, tables)
            stats_feats.append(resampled_features(vals, dt_s, real, cond))
            if bi >= 30:
                break
    sf = torch.cat(stats_feats)
    f_mu, f_sd = sf.mean(0), sf.std(0).clamp(min=1e-4)
    n_feat = sf.shape[1]
    print(f"  critic stats over {len(sf)} real trajectories ({n_feat} features)",
          flush=True)

    critic = FeatureCritic(n_feat, args.critic_width).to(device)

    for p in model.dt_head.parameters():
        p.requires_grad_(False)
    g_params = [p for p in model.parameters() if p.requires_grad]
    opt_g = torch.optim.AdamW(g_params, lr=args.lr, weight_decay=0.0)
    opt_d = torch.optim.Adam(critic.parameters(), lr=args.critic_lr,
                             betas=(0.5, 0.999))

    save_path = data_dir / args.save_name
    latest_path = save_path.with_stem(save_path.stem + "_latest")
    start_step = 0
    if args.auto_resume and latest_path.exists():
        rck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(rck["model_state_dict"])
        critic.load_state_dict(rck["critic_state_dict"])
        opt_g.load_state_dict(rck["opt_g_state_dict"])
        opt_d.load_state_dict(rck["opt_d_state_dict"])
        start_step = rck["step"]
        print(f"  Resumed at step {start_step}", flush=True)

    def zc(f):
        return ((f - f_mu) / f_sd).clamp(-10.0, 10.0)

    def zfeat(f):
        return ((f - feat_mu) / feat_sd).clamp(-10.0, 10.0)

    model.train()
    step_i = start_step
    t0 = time.time()
    ema_gap, ema_anchor = None, None
    data_iter = iter(dl)
    while step_i < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        dt_z, s_cls, th_cls, real, cond = (x.to(device) for x in batch)
        B2 = dt_z.shape[0]
        h = B2 // 2  # first half generates, second half is the real reference

        dt_s = torch.exp(dt_z * dt_std + dt_mean).clamp(0.1, 1000.0) / 1000.0

        # each trajectory's own movement character, teacher-forced (same
        # recipe as featcond training, z-scored by the checkpoint's stats)
        with torch.no_grad():
            all_vals = real_batch_values(s_cls, th_cls, tables)
            feat = zfeat(resampled_features(all_vals, dt_s, real, cond))

        r = float(np.random.default_rng(step_i).uniform(args.reveal_min, args.reveal_max))
        s_tok, th_tok, masked = partial_reveal(
            model, dt_z[:h], cond[:h], real[:h], r,
            args.reveal_steps, args.choice_temp, device, feat=feat[:h],
        )
        gen_vals = st_complete(
            model, dt_z[:h], s_tok, th_tok, masked, cond[:h], real[:h],
            tables, args.tau, feat=feat[:h],
        )
        gen_f = zc(resampled_features(gen_vals, dt_s[:h], real[:h], cond[:h]))

        with torch.no_grad():
            ref_f = zc(resampled_features(
                real_batch_values(s_cls[h:], th_cls[h:], tables),
                dt_s[h:], real[h:], cond[h:]))

        # critic update(s) on detached generator features
        gen_f_d = gen_f.detach()
        for _ in range(args.critic_iters):
            d_real = critic(ref_f)
            d_fake = critic(gen_f_d)
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        gap = (d_real.mean() - d_fake.mean()).item()

        # generator update: fool the critic, stay anchored to pretraining
        warm = step_i < args.critic_warmup
        adv = -critic(gen_f).mean() if not warm else gen_f.new_zeros(())

        t_cont = torch.rand(B2 - h, device=device)
        t_int = (t_cont * (model.n_steps - 1)).long()
        dt_noisy, _, velocity = model.q_flow(dt_z[h:], t_cont)
        s_m, th_m, mask_a = model.q_mask_joint(s_cls[h:], th_cls[h:], t_int)
        v_pred, s_logits, th_logits = model(
            dt_noisy, s_m, th_m, t_cont * (model.n_steps - 1), cond[h:], s_cls[h:],
            feat=feat[h:],
        )
        w_flow = real[h:] + (1.0 - real[h:]) * 0.1
        flow_loss = ((v_pred - velocity) ** 2 * w_flow).sum() / w_flow.sum().clamp(1)
        ce_s = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]),
                               s_cls[h:].reshape(-1), reduction="none").view(B2 - h, -1)
        ws = mask_a.float() * (real[h:] + (1.0 - real[h:]) * 0.15)
        s_loss = (ce_s * ws).sum() / ws.sum().clamp(1)
        motion_a = (s_cls[h:] > TICK_CLASS) & (s_cls[h:] < S_PAD_CLASS)
        ce_th = F.cross_entropy(th_logits.reshape(-1, th_logits.shape[-1]),
                                th_cls[h:].clamp(max=TH_BINS - 1).reshape(-1),
                                reduction="none").view(B2 - h, -1)
        wt = (mask_a & motion_a).float()
        th_loss = (ce_th * wt).sum() / wt.sum().clamp(1)
        anchor = flow_loss + s_loss + th_loss

        loss = args.adv_weight * adv + args.anchor_weight * anchor
        opt_g.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(g_params, 1.0)
        opt_g.step()

        ema_gap = gap if ema_gap is None else 0.95 * ema_gap + 0.05 * gap
        ema_anchor = anchor.item() if ema_anchor is None else 0.95 * ema_anchor + 0.05 * anchor.item()
        step_i += 1

        if step_i % 20 == 0 or step_i == 1:
            print(f"  step {step_i:4d}/{args.steps} | D gap {ema_gap:.4f} "
                  f"(real {d_real.mean().item():+.3f} fake {d_fake.mean().item():+.3f}) | "
                  f"anchor {ema_anchor:.3f} (flow {flow_loss.item():.3f} "
                  f"s {s_loss.item():.3f} th {th_loss.item():.3f}) | "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step_i % args.save_every == 0 or step_i == args.steps:
            out = {
                "model_state_dict": model.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "opt_g_state_dict": opt_g.state_dict(),
                "opt_d_state_dict": opt_d.state_dict(),
                "config": cfg, "dt_mean": dt_mean, "dt_std": dt_std,
                "feat_mu": ckpt["feat_mu"], "feat_sd": ckpt["feat_sd"],
                "feat_bank": ckpt["feat_bank"],
                "feat_bank_log_dist": ckpt["feat_bank_log_dist"],
                "step": step_i, "epoch": ckpt.get("epoch"),
            }
            torch.save(out, latest_path)
            torch.save(out, save_path)
            torch.save(out, save_path.with_stem(save_path.stem + f"_s{step_i}"))

    print(f"Done. Final D gap (ema): {ema_gap:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--load-from", default="event_polar_4m_fc_v2.pt")
    parser.add_argument("--save-name", default="event_polar_4m_advfc_v1.pt")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--critic-width", type=int, default=256)
    parser.add_argument("--critic-iters", type=int, default=1)
    parser.add_argument("--critic-warmup", type=int, default=50,
                        help="steps of critic-only training before adversarial "
                             "gradients reach the generator")
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--reveal-min", type=float, default=0.2)
    parser.add_argument("--reveal-max", type=float, default=0.9)
    parser.add_argument("--reveal-steps", type=int, default=12)
    parser.add_argument("--choice-temp", type=float, default=8.0)
    parser.add_argument("--n-frames", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--auto-resume", action="store_true")
    train(parser.parse_args())
