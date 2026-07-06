"""Stage 2b: adversarial feature-space fine-tune of the WS7b polar event model.

The fixed-statistic DM stage (train_events_polar_dm.py) is saturated: quant and
cov sit at their estimator floors on the 4M base, and fine-tuning with them made
the score WORSE (0.695 -> 0.710 at seed 42). Whatever still separates generated
trajectories from human ones is not on the fixed list. So let a critic find it:
a small MLP on the same 18 differentiable detector features, trained with a
hinge loss to separate real from generated, while the generator trains to fool
it. This is the direct analog of the eval itself (RF on 18 features), made
differentiable.

Everything else is inherited unchanged from the DM harness: partial no-grad
MaskGIT reveal, straight-through Gumbel completion, lattice-snapped integer
decode, 125Hz resample, dt head frozen, pretraining losses as anchor.

Stability choices: spectral norm on every critic linear (bounded critic
gradients without a GP), hinge loss, critic lr 1e-4 vs generator 1e-5, features
z-scored by real-data stats and clamped to +/-10 exactly as in the DM stage.

Run:
    .venv/Scripts/python.exe training/train_events_polar_adv.py --steps 800 \
        --load-from event_polar_4m.pt --save-name event_polar_4m_adv_v1.pt
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
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.event_stream_polar import EventStreamPolarModel, TICK_CLASS, S_PAD_CLASS, TH_BINS  # noqa: E402
from training.train_events_polar import PolarEventDataset  # noqa: E402
from training.train_events_polar_dm import (  # noqa: E402
    build_value_tables, detector_features, event_features, partial_reveal,
    real_batch_values, st_complete, stream_to_frames,
)


class FeatureCritic(nn.Module):
    def __init__(self, n_feat: int, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(n_feat, width)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(width, width)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(width, 1)),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    ckpt = torch.load(data_dir / args.load_from, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    dt_mean, dt_std = float(ckpt["dt_mean"]), float(ckpt["dt_std"])
    model = EventStreamPolarModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {args.load_from} (epoch {ckpt.get('epoch')})", flush=True)

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

    def features_from_values(vals, dt_s, real, cond):
        speed, motion, tick, cos_th, sin_th = vals
        if args.feature_space == "event":
            return event_features(speed, motion, tick, cos_th, sin_th, dt_s, real)
        x, y, fmask = stream_to_frames(speed, motion, cos_th, sin_th,
                                       dt_s, real, cond, args.n_frames)
        return detector_features(x, y, fmask)

    # global feature standardization from real data (same recipe as DM stage)
    stats_feats = []
    with torch.no_grad():
        for bi, (dt_z, s_cls, th_cls, real, cond) in enumerate(dl):
            dt_z, s_cls, th_cls, real, cond = (
                x.to(device) for x in (dt_z, s_cls, th_cls, real, cond))
            dt_s = torch.exp(dt_z * dt_std + dt_mean).clamp(0.1, 1000.0) / 1000.0
            vals = real_batch_values(s_cls, th_cls, tables)
            stats_feats.append(features_from_values(vals, dt_s, real, cond))
            if bi >= 30:
                break
    sf = torch.cat(stats_feats)
    f_mu, f_sd = sf.mean(0), sf.std(0).clamp(min=1e-4)
    n_feat = sf.shape[1]
    print(f"  feature stats over {len(sf)} real trajectories "
          f"({args.feature_space} space, {n_feat} features)", flush=True)

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

        r = float(np.random.default_rng(step_i).uniform(args.reveal_min, args.reveal_max))
        s_tok, th_tok, masked = partial_reveal(
            model, dt_z[:h], cond[:h], real[:h], r,
            args.reveal_steps, args.choice_temp, device,
        )
        gen_vals = st_complete(
            model, dt_z[:h], s_tok, th_tok, masked, cond[:h], real[:h],
            tables, args.tau,
        )
        gen_f = zc(features_from_values(gen_vals, dt_s[:h], real[:h], cond[:h]))

        with torch.no_grad():
            ref_vals = real_batch_values(s_cls[h:], th_cls[h:], tables)
            ref_f = zc(features_from_values(ref_vals, dt_s[h:], real[h:], cond[h:]))

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
                "step": step_i, "epoch": ckpt.get("epoch"),
            }
            torch.save(out, latest_path)
            torch.save(out, save_path)
            torch.save(out, save_path.with_stem(save_path.stem + f"_s{step_i}"))

    print(f"Done. Final D gap (ema): {ema_gap:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--load-from", default="event_polar_4m.pt")
    parser.add_argument("--save-name", default="event_polar_4m_adv_v1.pt")
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
    parser.add_argument("--choice-temp", type=float, default=7.0)
    parser.add_argument("--feature-space", choices=["event", "resampled"],
                        default="resampled")
    parser.add_argument("--n-frames", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--auto-resume", action="store_true")
    train(parser.parse_args())
