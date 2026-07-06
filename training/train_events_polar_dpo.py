"""Preference fine-tune: teach fc_v2 to prefer SIR winners over losers.

Imitating SIR winners failed (train_events_polar_distill.py, July 6): the
winners are the model's own samples, so max-likelihood on them mostly
re-teaches the current marginals and drifts the model off the real-token
manifold. This script uses the judge's signal differently: each corpus item
is a (winner, loser) PAIR from the same candidate pool, and the loss is the
Diffusion-DPO objective, push the model's denoising loss down on the winner
and up on the loser RELATIVE to a frozen reference copy. The gradient is
the contrast between two whole sequences, which is exactly the
trajectory-level judgment SIR applies, and the reference anchoring is an
implicit KL leash against the drift that killed plain distillation. No
sampling happens during training, so the ST-Gumbel gradient path that
killed the adversarial fine-tunes is not involved.

Pairs come from make_distill_corpus.py with DISTILL_SAVE_LOSER=1 (winner =
judge's best of the K pool, loser = judge's worst).

Run:
    .venv/Scripts/python.exe training/train_events_polar_dpo.py \
        --load-from event_polar_4m_fc_v2.pt --save-name event_polar_4m_dpo_v1.pt
"""
from __future__ import annotations

import argparse
import copy
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.event_stream_polar import EventStreamPolarModel, TICK_CLASS, S_PAD_CLASS, TH_BINS  # noqa: E402
from training.train_events_polar_dm import (  # noqa: E402
    build_value_tables, detector_features, real_batch_values, stream_to_frames,
)


class PairCorpusDataset(Dataset):
    """(winner, loser) token pairs from the same SIR candidate pool."""

    def __init__(self, w, l, cond):
        self.w = w
        self.l = l
        self.cond = cond

    def __len__(self):
        return len(self.cond)

    @staticmethod
    def _seq(dt_z, s_cls, th_cls, L, seq_len):
        real = np.zeros(seq_len, dtype=np.float32)
        real[:int(L)] = 1.0
        return (torch.from_numpy(dt_z.astype(np.float32)),
                torch.from_numpy(s_cls.astype(np.int64)),
                torch.from_numpy(th_cls.astype(np.int64)),
                torch.from_numpy(real))

    def __getitem__(self, idx):
        seq_len = self.w[0].shape[1]
        w = self._seq(self.w[0][idx], self.w[1][idx], self.w[2][idx],
                      self.w[3][idx], seq_len)
        l = self._seq(self.l[0][idx], self.l[1][idx], self.l[2][idx],
                      self.l[3][idx], seq_len)
        return (*w, *l, torch.from_numpy(self.cond[idx].astype(np.float32)))


def load_pairs(pattern):
    shards = sorted(glob.glob(pattern))
    assert shards, f"no pair shards match {pattern}"
    parts = [np.load(s) for s in shards]
    def cat(key):
        return np.concatenate([p[key] for p in parts])
    w = (cat("dt_z"), cat("s_cls"), cat("th_cls"), cat("length"))
    l = (cat("dt_z_l"), cat("s_cls_l"), cat("th_cls_l"), cat("length_l"))
    cond = cat("cond")
    gap = cat("logw_w") - cat("logw_l")
    print(f"  pairs: {len(shards)} shards, {len(cond):,} pairs, "
          f"judge gap median {np.median(gap):.2f} logits", flush=True)
    return w, l, cond


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    ckpt = torch.load(data_dir / args.load_from, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    dt_mean, dt_std = float(ckpt["dt_mean"]), float(ckpt["dt_std"])
    model = EventStreamPolarModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    f_mu = ckpt["feat_mu"].to(device)
    f_sd = ckpt["feat_sd"].to(device)
    print(f"Loaded {args.load_from} (epoch {ckpt.get('epoch')}, "
          f"step {ckpt.get('step')}), frozen reference copy made", flush=True)

    dl = DataLoader(
        PairCorpusDataset(*load_pairs(str(data_dir / args.corpus))),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True)

    tables = build_value_tables(device)

    def batch_features(s_cls, th_cls, dt_z, real, cond):
        dt_s = torch.exp(dt_z * dt_std + dt_mean).clamp(0.1, 1000.0) / 1000.0
        speed, motion, tick, cos_th, sin_th = real_batch_values(s_cls, th_cls, tables)
        x, y, fmask = stream_to_frames(speed, motion, cos_th, sin_th,
                                       dt_s, real, cond, args.n_frames)
        return detector_features(x, y, fmask)

    def seq_loss(net, dt_z, s_cls, th_cls, real, cond, feat, t_cont, t_int):
        """Per-sequence masked CE (s + th heads), the pretraining loss
        without the batch reduction. dt flow excluded: that head is frozen
        and timing already matches humans."""
        B = dt_z.shape[0]
        dt_noisy, _, _ = net.q_flow(dt_z, t_cont)
        s_m, th_m, mask = net.q_mask_joint(s_cls, th_cls, t_int)
        _, s_logits, th_logits = net(
            dt_noisy, s_m, th_m, t_cont * (net.n_steps - 1), cond, s_cls,
            feat=feat,
        )
        ce_s = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]),
                               s_cls.reshape(-1), reduction="none").view(B, -1)
        ws = mask.float() * real
        s_l = (ce_s * ws).sum(dim=1) / ws.sum(dim=1).clamp(1)
        motion = (s_cls > TICK_CLASS) & (s_cls < S_PAD_CLASS)
        ce_th = F.cross_entropy(th_logits.reshape(-1, th_logits.shape[-1]),
                                th_cls.clamp(max=TH_BINS - 1).reshape(-1),
                                reduction="none").view(B, -1)
        wt = (mask & motion).float()
        th_l = (ce_th * wt).sum(dim=1) / wt.sum(dim=1).clamp(1)
        return s_l + th_l

    for p in model.dt_head.parameters():
        p.requires_grad_(False)
    g_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(g_params, lr=args.lr, weight_decay=0.0)

    save_path = data_dir / args.save_name

    def save_ckpt(path, step):
        torch.save({
            "model_state_dict": model.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "config": cfg, "dt_mean": dt_mean, "dt_std": dt_std,
            "feat_mu": f_mu.cpu(), "feat_sd": f_sd.cpu(),
            "feat_bank": ckpt["feat_bank"],
            "feat_bank_log_dist": ckpt["feat_bank_log_dist"],
            "step": step, "epoch": ckpt.get("epoch"),
        }, path)

    model.train()
    step_i = 0
    t0 = time.time()
    ema = None
    acc_ema = None
    data_iter = iter(dl)
    while step_i < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        (dt_w, s_w, th_w, real_w,
         dt_l, s_l_, th_l_, real_l, cond) = (x.to(device) for x in batch)
        B = dt_w.shape[0]

        with torch.no_grad():
            feat_w = ((batch_features(s_w, th_w, dt_w, real_w, cond) - f_mu)
                      / f_sd).clamp(-10.0, 10.0)
            feat_l = ((batch_features(s_l_, th_l_, dt_l, real_l, cond) - f_mu)
                      / f_sd).clamp(-10.0, 10.0)

        # one t per pair: winner and loser are corrupted equally hard
        t_cont = torch.rand(B, device=device)
        t_int = (t_cont * (model.n_steps - 1)).long()

        lw_pol = seq_loss(model, dt_w, s_w, th_w, real_w, cond, feat_w, t_cont, t_int)
        ll_pol = seq_loss(model, dt_l, s_l_, th_l_, real_l, cond, feat_l, t_cont, t_int)
        with torch.no_grad():
            lw_ref = seq_loss(ref, dt_w, s_w, th_w, real_w, cond, feat_w, t_cont, t_int)
            ll_ref = seq_loss(ref, dt_l, s_l_, th_l_, real_l, cond, feat_l, t_cont, t_int)

        adv = (lw_ref - lw_pol) - (ll_ref - ll_pol)
        loss = -F.logsigmoid(args.beta * adv).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(g_params, 1.0)
        opt.step()

        with torch.no_grad():
            acc = (adv > 0).float().mean().item()
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        acc_ema = acc if acc_ema is None else 0.95 * acc_ema + 0.05 * acc
        step_i += 1
        if step_i % 50 == 0 or step_i == 1:
            print(f"  step {step_i:5d}/{args.steps} | dpo {ema:.4f} | "
                  f"pref-acc {acc_ema:.3f} | "
                  f"margin {adv.mean().item():+.4f} | {time.time() - t0:.0f}s",
                  flush=True)
        if step_i % args.snapshot_every == 0:
            save_ckpt(save_path.with_stem(save_path.stem + f"_s{step_i}"), step_i)
    save_ckpt(save_path, step_i)
    print(f"Done. dpo {ema:.4f} pref-acc {acc_ema:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--load-from", default="event_polar_4m_fc_v2.pt")
    parser.add_argument("--save-name", default="event_polar_4m_dpo_v1.pt")
    parser.add_argument("--corpus", default="distill_pairs_b*.npz")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--n-frames", type=int, default=256)
    parser.add_argument("--snapshot-every", type=int, default=250)
    train(parser.parse_args())
