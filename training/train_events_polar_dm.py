"""Stage 2: distribution-matching fine-tune of the WS7b polar event model.

WS3 proved the mechanics on CANDI (MMD fell 0.78 -> 0.16) but that backbone
was already at its representation ceiling and the score did not move. The
event backbone is not: real events replayed through this representation score
0.507. The remaining WS7b gaps after the sampler fix (tick over-generation,
angular velocity per second, the signed mean-acceleration correlation
structure) are all directly expressible as differentiable statistics of the
generated event stream, so we train on them.

Per step:
1. Take a real batch. Keep its dt sequence and conditioning (timing was right
   in WS7 and stays untouched; the dt head is excluded from MMD gradients and
   anchored by the standard flow loss).
2. No-grad: partially sample s/dtheta with the SAME sampler used at eval
   (MaskGIT, Gumbel choice order, annealed choice temperature), stopping at a
   random reveal fraction so the model sees every stage of its own sampling
   process as context.
3. With grad: one forward pass completes the remaining masked positions via
   straight-through Gumbel-softmax on both heads (hard tokens forward, soft
   gradients back).
4. Differentiable event-level features on the completed stream (real dt +
   generated speeds/headings), multi-bandwidth RBF MMD against the same
   features on an independent real batch.
5. Loss = mmd_weight * MMD + anchor CE (s, th) + flow MSE on a standard
   masked real batch, so the pretrained solution anchors the fine-tune.

Run (short, crash-resumable):
    .venv/Scripts/python.exe training/train_events_polar_dm.py --steps 600 \
        --save-name event_polar_dm_v1.pt
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
from models.event_stream_polar import (  # noqa: E402
    N_S_CLASSES, S_MASK_TOKEN, S_PAD_CLASS, TH_BINS, TH_MASK_TOKEN,
    TH_NULL_CLASS, TICK_CLASS, EventStreamPolarModel, class_to_dtheta,
    class_to_speed,
)
from training.train_events_polar import PolarEventDataset  # noqa: E402


def build_value_tables(device):
    """Class-index -> physical-value lookup tables (constant, no grad)."""
    s_idx = torch.arange(N_S_CLASSES, device=device)
    speed_vals = class_to_speed(s_idx).float()          # tick/PAD -> 0
    motion_vals = ((s_idx > TICK_CLASS) & (s_idx < S_PAD_CLASS)).float()
    tick_vals = (s_idx == TICK_CLASS).float()
    th_idx = torch.arange(TH_BINS, device=device)
    th_ang = class_to_dtheta(th_idx).float()
    return speed_vals, motion_vals, tick_vals, torch.cos(th_ang), torch.sin(th_ang), th_ang


def masked_mean(x, w, dim=1):
    return (x * w).sum(dim) / w.sum(dim).clamp(min=1e-6)


def masked_std(x, w, dim=1):
    m = masked_mean(x, w, dim).unsqueeze(dim)
    var = ((x - m) ** 2 * w).sum(dim) / w.sum(dim).clamp(min=1e-6)
    return (var + 1e-8).sqrt()


def event_features(speed, motion, tick, cos_th, sin_th, dt_s, real):
    """Differentiable per-trajectory features of an event stream.

    speed: (B,T) px per event. motion/tick: (B,T) soft indicators.
    cos_th/sin_th: (B,T) heading-increment components (motion positions).
    dt_s: (B,T) seconds (real, constant). real: (B,T) validity mask.
    Returns (B, F).
    """
    dt = dt_s.clamp(min=5e-4)
    v = speed / dt                                     # px/s per event
    wt = real * dt                                     # time weights

    mean_v = masked_mean(v, wt)
    std_v = masked_std(v, wt)
    max_v = (v * real).amax(dim=1)

    dt_mid = ((dt[:, 1:] + dt[:, :-1]) * 0.5).clamp(min=5e-4)
    m_acc = real[:, 1:] * real[:, :-1]
    acc = (v[:, 1:] - v[:, :-1]) / dt_mid
    mean_acc = masked_mean(acc, m_acc)                 # SIGNED: telescoping
    mean_abs_acc = masked_mean(acc.abs(), m_acc)
    std_acc = masked_std(acc, m_acc)

    m_jerk = m_acc[:, 1:] * m_acc[:, :-1]
    jerk = (acc[:, 1:] - acc[:, :-1]) / dt_mid[:, 1:]
    mean_abs_jerk = masked_mean(jerk.abs(), m_jerk)
    std_jerk = masked_std(jerk, m_jerk)

    # heading increment as an angle; motion-gated
    dth = torch.atan2(sin_th, cos_th) * motion
    w_mot = real * motion
    ang_v = dth / dt                                   # rad/s
    mean_abs_w = masked_mean(ang_v.abs(), w_mot * dt)
    std_w = masked_std(ang_v, w_mot * dt)
    mean_abs_dth = masked_mean(dth.abs(), w_mot)

    tick_frac = masked_mean(tick, real)
    stall_time = masked_mean(tick, real * dt)

    heading = torch.cumsum(dth, dim=1)
    ex = (speed * torch.cos(heading) * real).sum(dim=1)
    ey = (speed * torch.sin(heading) * real).sum(dim=1)
    d_straight = (ex ** 2 + ey ** 2 + 1e-6).sqrt()
    d_travel = (speed * real).sum(dim=1).clamp(min=1e-6)
    path_eff = d_straight / d_travel

    dur = wt.sum(dim=1)

    def lg(x):
        return torch.log1p(x.clamp(min=0))

    return torch.stack([
        lg(mean_v), lg(std_v), lg(max_v),
        mean_acc / 1e4, lg(mean_abs_acc), lg(std_acc),
        lg(mean_abs_jerk), lg(std_jerk),
        lg(mean_abs_w), lg(std_w), lg(mean_abs_dth * 57.3),
        tick_frac, stall_time, path_eff, lg(dur * 10.0),
    ], dim=1)


def resample_positions(px, py, t_ev, t_end, n_frames, hz=125.0):
    """Differentiable 125Hz linear-interpolation resample.

    px/py: (B, T+1) positions including the start point. t_ev: (B, T+1)
    cumulative times (constant, real). t_end: (B,). Returns (x, y, fmask):
    (B, M) frame positions and validity. Gradients flow through positions
    only, matching features.resample_trajectory (time axis is real data).
    """
    B = px.shape[0]
    dev = px.device
    tau = (torch.arange(n_frames, device=dev).float() / hz).unsqueeze(0).expand(B, -1)
    fmask = (tau <= t_end.unsqueeze(1)).float()
    idx = torch.searchsorted(t_ev.detach(), tau.contiguous(), right=True) - 1
    idx = idx.clamp(min=0, max=px.shape[1] - 2)
    t0 = t_ev.gather(1, idx)
    t1 = t_ev.gather(1, idx + 1)
    w = ((tau - t0) / (t1 - t0).clamp(min=1e-6)).clamp(0.0, 1.0)
    x = px.gather(1, idx) * (1 - w) + px.gather(1, idx + 1) * w
    y = py.gather(1, idx) * (1 - w) + py.gather(1, idx + 1) * w
    return x, y, fmask


def detector_features(x, y, fmask, hz=125.0):
    """Differentiable analogs of the 18 features in features.py, computed on
    the resampled frames. Gen and real batches go through this SAME function,
    so shared approximation bias cancels in the MMD."""
    dt = 1.0 / hz
    m1 = fmask[:, 1:] * fmask[:, :-1]                  # segment validity
    dx = (x[:, 1:] - x[:, :-1]) * m1
    dy = (y[:, 1:] - y[:, :-1]) * m1
    ds = (dx ** 2 + dy ** 2 + 1e-8).sqrt()
    speed = ds / dt
    vx, vy = dx / dt, dy / dt

    mean_v = masked_mean(speed, m1)
    std_v = masked_std(speed, m1)
    max_v = (speed * m1).amax(dim=1)
    skew_v = masked_mean(((speed - mean_v.unsqueeze(1)) / std_v.unsqueeze(1).clamp(min=1e-4)) ** 3, m1)

    m2 = m1[:, 1:] * m1[:, :-1]
    acc = (speed[:, 1:] - speed[:, :-1]) / dt
    mean_acc = masked_mean(acc, m2)
    std_acc = masked_std(acc, m2)
    max_acc = (acc.abs() * m2).amax(dim=1)

    m3 = m2[:, 1:] * m2[:, :-1]
    jerk = (acc[:, 1:] - acc[:, :-1]) / dt
    mean_jerk = masked_mean(jerk, m3)
    std_jerk = masked_std(jerk, m3)

    # endpoints: last valid frame ~ frame count per row
    n_last = fmask.sum(dim=1).long().clamp(min=2) - 1
    xe = x.gather(1, n_last.unsqueeze(1)).squeeze(1)
    ye = y.gather(1, n_last.unsqueeze(1)).squeeze(1)
    ldx, ldy = xe - x[:, 0], ye - y[:, 0]
    d_straight = (ldx ** 2 + ldy ** 2 + 1e-8).sqrt()
    d_travel = (ds * m1).sum(dim=1).clamp(min=1e-6)
    path_eff = d_straight / d_travel
    perp = (ldy.unsqueeze(1) * (x - x[:, :1]) - ldx.unsqueeze(1) * (y - y[:, :1])).abs() \
        / d_straight.unsqueeze(1).clamp(min=1e-6)
    max_dev = (perp * fmask).amax(dim=1)

    ax = (vx[:, 1:] - vx[:, :-1]) / dt
    ay = (vy[:, 1:] - vy[:, :-1]) / dt
    # clamp well above zero: 1/speed^3 gradients on sub-pixel frames are the
    # dominant source of destabilizing gradient noise (v2 diverged on this)
    speed_mid = speed[:, :-1].clamp(min=30.0)
    cross = (vx[:, :-1] * ay - vy[:, :-1] * ax).abs()
    curv = (cross / speed_mid ** 3).clamp(max=1e4)
    curv_mean = masked_mean(curv, m2)
    curv_std = masked_std(curv, m2)

    # heading change between consecutive segments via atan2(cross, dot) on
    # segments normalized by a CLAMPED length: the value is identical (atan2
    # is scale-invariant) but the 1/|segment|^2 gradient blowup on sub-pixel
    # frames is capped at 1/0.5. This keeps gradient flow at slow frames,
    # which is where the angular-velocity gap (the top remaining feature
    # gap) actually lives; a hard detach there froze it.
    r = (dx ** 2 + dy ** 2 + 1e-12).sqrt().clamp(min=0.5)
    ux, uy = dx / r, dy / r
    cross_seg = ux[:, :-1] * uy[:, 1:] - uy[:, :-1] * ux[:, 1:]
    dot_seg = ux[:, :-1] * ux[:, 1:] + uy[:, :-1] * uy[:, 1:]
    d_ang = torch.atan2(cross_seg, dot_seg + 1e-9)
    s_soft = torch.tanh(d_ang / 0.05)
    sign_flip = 0.5 * (1.0 - s_soft[:, 1:] * s_soft[:, :-1])
    n_dir_changes = (sign_flip * m3).sum(dim=1)

    duration = fmask.sum(dim=1) * dt
    t_norm = torch.arange(speed.shape[1], device=x.device).float() / speed.shape[1]
    peak_w = torch.softmax(speed / speed.amax(dim=1, keepdim=True).clamp(min=1e-3) * 25.0
                           + (m1 - 1.0) * 1e4, dim=1)
    time_to_peak = (peak_w * t_norm.unsqueeze(0)).sum(dim=1)

    omega = (d_ang / dt).clamp(-1e6, 1e6)
    omega_mean = masked_mean(omega.abs(), m2)
    omega_std = masked_std(omega, m2)

    def lg(v):
        return torch.log1p(v.clamp(min=0))

    return torch.stack([
        lg(mean_v), lg(std_v), lg(max_v), skew_v.clamp(-8, 8),
        mean_acc / 1e4, lg(std_acc), lg(max_acc),
        mean_jerk / 1e6, lg(std_jerk),
        path_eff, lg(max_dev),
        lg(curv_mean * 1e3), lg(curv_std * 1e3),
        lg(n_dir_changes), lg(duration * 10.0), time_to_peak,
        lg(omega_mean), lg(omega_std),
    ], dim=1)


def stream_to_frames(speed, motion, cos_th, sin_th, dt_s, real, cond, n_frames,
                     snap=2.5):
    """Event values -> lattice-snapped, integer-rounded positions ->
    resampled frames. Mirrors the eval decode (EVENT_SNAP=2.5): slow steps
    are emitted as whole lattice steps (straight-through round), which is
    what removes the manufactured slow-frame angular jitter."""
    dth = torch.atan2(sin_th, cos_th) * motion
    angle0 = torch.atan2(cond[:, 3], cond[:, 2]).unsqueeze(1)
    heading = angle0 + torch.cumsum(dth, dim=1)
    step_x = speed * torch.cos(heading) * real
    step_y = speed * torch.sin(heading) * real
    if snap > 0:
        slow = ((speed > 0) & (speed < snap)).float()
        sx_r = step_x + (step_x.round() - step_x).detach()
        sy_r = step_y + (step_y.round() - step_y).detach()
        step_x = slow * sx_r + (1 - slow) * step_x
        step_y = slow * sy_r + (1 - slow) * step_y
    B = speed.shape[0]
    z = speed.new_zeros(B, 1)
    px = torch.cat([z, torch.cumsum(step_x, dim=1)], dim=1)
    py = torch.cat([z, torch.cumsum(step_y, dim=1)], dim=1)
    px = px + (px.round() - px).detach()               # STE integer grid
    py = py + (py.round() - py).detach()
    t_ev = torch.cumsum(dt_s * real, dim=1)
    t_ev = torch.cat([z, t_ev], dim=1)
    # freeze time past the real region so searchsorted never lands there
    t_ev = t_ev + (1.0 - torch.cat([torch.ones_like(z), real], dim=1)) * 1e6
    t_end = (dt_s * real).sum(dim=1)
    return resample_positions(px, py, t_ev, t_end, n_frames)


def mmd_rbf(x, y, bandwidths=(0.25, 0.5, 1.0, 2.0, 4.0)):
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2
    loss = x.new_zeros(())
    for bw in bandwidths:
        loss = loss + (torch.exp(-xx / (2 * bw)).mean()
                       + torch.exp(-yy / (2 * bw)).mean()
                       - 2 * torch.exp(-xy / (2 * bw)).mean())
    return loss


def quantile_loss(x, y):
    """Per-feature 1D Wasserstein via sorted quantile differences. This is
    the differentiable analog of the eval's per-feature Wasserstein table."""
    xs, _ = x.sort(dim=0)
    ys, _ = y.sort(dim=0)
    return (xs - ys).abs().mean()


def cov_loss(x, y):
    """Covariance matching: the detector's remaining edge is correlation
    structure (mean_acceleration vs everything), which the kernel MMD is
    nearly blind to at feasible batch sizes."""
    xc = x - x.mean(0, keepdim=True)
    yc = y - y.mean(0, keepdim=True)
    cx = xc.t() @ xc / max(x.shape[0] - 1, 1)
    cy = yc.t() @ yc / max(y.shape[0] - 1, 1)
    return (cx - cy).pow(2).mean()


def match_loss(x, y, w_mmd, w_quant, w_cov):
    # z-scored inputs; clamp so a single degenerate trajectory cannot blow
    # up the quantile/covariance terms (curvature and jerk have wild tails)
    x = x.clamp(-10.0, 10.0)
    y = y.clamp(-10.0, 10.0)
    parts = {
        "mmd": mmd_rbf(x, y),
        "quant": quantile_loss(x, y),
        "cov": cov_loss(x, y),
    }
    total = w_mmd * parts["mmd"] + w_quant * parts["quant"] + w_cov * parts["cov"]
    return total, parts


def real_batch_values(s_cls, th_cls, tables):
    """Token tensors -> (speed, motion, tick, cos, sin) via hard lookup."""
    speed_vals, motion_vals, tick_vals, cos_t, sin_t, _ = tables
    s = s_cls.clamp(max=N_S_CLASSES - 1)
    th = th_cls.clamp(max=TH_BINS - 1)
    null = th_cls >= TH_NULL_CLASS
    return (
        speed_vals[s],
        motion_vals[s],
        tick_vals[s],
        torch.where(null, torch.ones_like(speed_vals[s]), cos_t[th]),
        torch.where(null, torch.zeros_like(speed_vals[s]), sin_t[th]),
    )


@torch.no_grad()
def partial_reveal(model, dt_z, cond, real, reveal_frac, n_steps, choice_temp, device,
                   feat=None):
    """Run the eval sampler until reveal_frac of the real region is unmasked.

    Mirrors EventStreamPolarModel.sample (gumbel order) but only over the
    real region: PAD tail positions hold their PAD/NULL tokens so trajectory
    length stays real. Returns (s_tok, th_tok, masked) at the stop point.
    """
    B, T = dt_z.shape
    s_tok = torch.full((B, T), S_MASK_TOKEN, dtype=torch.long, device=device)
    th_tok = torch.full((B, T), TH_MASK_TOKEN, dtype=torch.long, device=device)
    pad_region = real < 0.5
    s_tok[pad_region] = S_PAD_CLASS
    th_tok[pad_region] = TH_NULL_CLASS

    n_real = real.sum(dim=1, keepdim=True)
    step = 1.0 / n_steps
    for i in range(n_steps):
        masked = s_tok == S_MASK_TOKEN
        frac_revealed = 1.0 - masked.float().sum(dim=1, keepdim=True) / n_real.clamp(min=1)
        if (frac_revealed >= reveal_frac).all():
            break
        t_cont = 1.0 - i * step
        t_scaled = torch.full((B,), t_cont * (model.n_steps - 1), device=device)
        x_feat = model.trunk(dt_z, s_tok, th_tok, t_scaled, cond, feat)
        s_logits = model.s_head(x_feat)

        t_next = max(t_cont - step, 0.0)
        target_revealed = model.sqrt_ab[int(t_next * (model.n_steps - 1))] * n_real.squeeze(1)
        current_revealed = n_real.squeeze(1) - masked.float().sum(1)
        n_new = (target_revealed - current_revealed).clamp(min=0)

        s_probs = torch.softmax(s_logits, dim=-1)
        s_new = torch.multinomial(s_probs.view(-1, s_probs.shape[-1]), 1).view(B, T)
        s_for_th = torch.where(masked, s_new, s_tok.clamp(max=N_S_CLASSES - 1))
        th_l = model.th_logits(x_feat, s_for_th)
        th_probs = torch.softmax(th_l, dim=-1)
        th_new = torch.multinomial(th_probs.view(-1, th_probs.shape[-1]), 1).view(B, T)
        motion = (s_new > TICK_CLASS) & (s_new < S_PAD_CLASS)
        th_new = torch.where(motion, th_new, torch.full_like(th_new, TH_NULL_CLASS))

        conf = s_probs.gather(-1, s_new.unsqueeze(-1)).squeeze(-1)
        th_conf = th_probs.gather(-1, th_new.clamp(max=TH_BINS - 1).unsqueeze(-1)).squeeze(-1)
        conf = torch.where(motion, conf * th_conf, conf)
        g = -torch.log(-torch.log(torch.rand_like(conf).clamp(1e-9, 1.0)))
        score = torch.log(conf.clamp(min=1e-9)) + choice_temp * (1.0 - i / n_steps) * g
        score = torch.where(masked, score, torch.full_like(score, -1e9))

        rank = score.argsort(dim=-1, descending=True)
        k = n_new.long().clamp(max=T)
        arange = torch.arange(T, device=device).unsqueeze(0)
        take = arange < k.unsqueeze(1)
        reveal = torch.zeros_like(masked)
        reveal.scatter_(1, rank, take)
        reveal &= masked
        s_tok[reveal] = s_new[reveal]
        th_tok[reveal] = th_new[reveal]

    return s_tok, th_tok, s_tok == S_MASK_TOKEN


def st_complete(model, dt_z, s_tok, th_tok, masked, cond, real, tables, tau,
                feat=None):
    """Gradient pass: complete masked positions with ST Gumbel-softmax and
    return full (speed, motion, tick, cos, sin) event values."""
    speed_vals, motion_vals, tick_vals, cos_t, sin_t, _ = tables
    B, T = dt_z.shape
    # present the t whose training mask schedule matches the actual masked
    # fraction, so the trunk sees an in-distribution (t, mask) pair
    masked_frac = masked.float().sum() / real.sum().clamp(min=1)
    t_idx = (model.sqrt_ab - (1.0 - masked_frac)).abs().argmin().float()
    t_scaled = torch.full((B,), float(t_idx), device=dt_z.device)
    x_feat = model.trunk(dt_z, s_tok, th_tok, t_scaled, cond, feat)

    s_logits = model.s_head(x_feat)
    y_s = F.gumbel_softmax(s_logits, tau=tau, hard=True)          # (B,T,Cs)
    s_ctx = y_s @ model.s_ctx_embed.weight                        # soft embed
    th_l = model.th_head(model.th_norm(x_feat + s_ctx))
    y_th = F.gumbel_softmax(th_l, tau=tau, hard=True)             # (B,T,256)

    sp_soft = y_s @ speed_vals
    mo_soft = y_s @ motion_vals
    tk_soft = y_s @ tick_vals
    cos_soft = y_th @ cos_t
    sin_soft = y_th @ sin_t

    sp_r, mo_r, tk_r, cos_r, sin_r = real_batch_values(s_tok, th_tok, tables)
    m = masked.float()
    speed = m * sp_soft + (1 - m) * sp_r
    motion = m * mo_soft + (1 - m) * mo_r
    tick = m * tk_soft + (1 - m) * tk_r
    # gate generated heading by its own motion indicator
    cos_th = m * (mo_soft * cos_soft + (1 - mo_soft)) + (1 - m) * cos_r
    sin_th = m * mo_soft * sin_soft + (1 - m) * sin_r
    return speed, motion, tick, cos_th, sin_th


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

    # global feature standardization from real data
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
    sfz = (sf - f_mu) / f_sd
    floors = {"mmd": [], "quant": [], "cov": []}
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(20):
        p = torch.randperm(len(sfz), generator=g)
        a, b = sfz[p[:args.batch_size]], sfz[p[args.batch_size:2 * args.batch_size]]
        _, parts = match_loss(a, b, 1, 1, 1)
        for k in floors:
            floors[k].append(parts[k].item())
    floor_str = " ".join(f"{k}={np.mean(v):.4f}" for k, v in floors.items())
    print(f"  feature stats over {len(sf)} real trajectories "
          f"({args.feature_space} space); real-vs-real floors at "
          f"n={args.batch_size}: {floor_str}", flush=True)

    # dt head stays put; everything else fine-tunes
    for p in model.dt_head.parameters():
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    save_path = data_dir / args.save_name
    latest_path = save_path.with_stem(save_path.stem + "_latest")
    start_step = 0
    if args.auto_resume and latest_path.exists():
        rck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(rck["model_state_dict"])
        optimizer.load_state_dict(rck["optimizer_state_dict"])
        start_step = rck["step"]
        print(f"  Resumed at step {start_step}", flush=True)

    model.train()
    step_i = start_step
    t0 = time.time()
    ema_mmd, ema_anchor = None, None
    data_iter = iter(dl)
    while step_i < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        dt_z, s_cls, th_cls, real, cond = (x.to(device) for x in batch)
        B2 = dt_z.shape[0]
        h = B2 // 2  # first half generates, second half is the MMD reference

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
        gen_f = features_from_values(gen_vals, dt_s[:h], real[:h], cond[:h])

        with torch.no_grad():
            ref_vals = real_batch_values(s_cls[h:], th_cls[h:], tables)
            ref_f = features_from_values(ref_vals, dt_s[h:], real[h:], cond[h:])

        mmd, parts = match_loss((gen_f - f_mu) / f_sd, (ref_f - f_mu) / f_sd,
                                args.w_mmd, args.w_quant, args.w_cov)

        # anchor: standard pretraining losses on the reference half
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

        loss = args.mmd_weight * mmd + args.anchor_weight * anchor
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        ema_mmd = mmd.item() if ema_mmd is None else 0.95 * ema_mmd + 0.05 * mmd.item()
        ema_anchor = anchor.item() if ema_anchor is None else 0.95 * ema_anchor + 0.05 * anchor.item()
        step_i += 1

        if step_i % 20 == 0 or step_i == 1:
            print(f"  step {step_i:4d}/{args.steps} | match {ema_mmd:.4f} "
                  f"(mmd {parts['mmd'].item():.3f} quant {parts['quant'].item():.3f} "
                  f"cov {parts['cov'].item():.3f}) | "
                  f"anchor {ema_anchor:.3f} (flow {flow_loss.item():.3f} "
                  f"s {s_loss.item():.3f} th {th_loss.item():.3f}) | "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step_i % args.save_every == 0 or step_i == args.steps:
            out = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg, "dt_mean": dt_mean, "dt_std": dt_std,
                "step": step_i, "epoch": ckpt.get("epoch"),
            }
            torch.save(out, latest_path)
            torch.save(out, save_path)
            torch.save(out, save_path.with_stem(save_path.stem + f"_s{step_i}"))

    print(f"Done. Final MMD (ema): {ema_mmd:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--load-from", default="event_polar_best.pt")
    parser.add_argument("--save-name", default="event_polar_dm_v1.pt")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mmd-weight", type=float, default=4.0,
                        help="overall weight on the combined match loss")
    parser.add_argument("--w-mmd", type=float, default=1.0)
    parser.add_argument("--w-quant", type=float, default=2.0)
    parser.add_argument("--w-cov", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--reveal-min", type=float, default=0.2)
    parser.add_argument("--reveal-max", type=float, default=0.9)
    parser.add_argument("--reveal-steps", type=int, default=12)
    parser.add_argument("--choice-temp", type=float, default=4.0)
    parser.add_argument("--feature-space", choices=["event", "resampled"],
                        default="resampled")
    parser.add_argument("--n-frames", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()
    train(args)
