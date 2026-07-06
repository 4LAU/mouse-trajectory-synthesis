"""Distill SIR selection into the feature-conditioned 4M model.

The SIR pipeline reaches its AUC by drawing K candidates and keeping the one
a human-vs-synthetic judge prefers. That is an inference-time system. This
script bakes the judge's preference into the weights: fine-tune fc_v2 on the
SIR-selected token corpus (training/make_distill_corpus.py) with the exact
pretraining objective, so a SINGLE draft from the distilled model lands
where the selected draft used to. The checkpoint's feature bank, mu and sd
pass through unchanged: sampling still draws movement characters from the
real-human bank; only how the model realizes them shifts.

The dt head stays frozen (timing texture already matches humans) and the
learning rate is low: the corpus is the model's own output, reweighted, so
this is a nudge toward its best modes, not new information. Snapshots every
500 steps let the eval pick the best point on the drift-vs-gain curve.

Optional --real-frac mixes real-trajectory batches back in as an anchor
against self-training drift.

Run:
    .venv/Scripts/python.exe training/train_events_polar_distill.py \
        --load-from event_polar_4m_fc_v2.pt --save-name event_polar_4m_distill_v1.pt
"""
from __future__ import annotations

import argparse
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
from training.train_events_polar import PolarEventDataset  # noqa: E402
from training.train_events_polar_dm import (  # noqa: E402
    build_value_tables, detector_features, real_batch_values, stream_to_frames,
)


class DistillCorpusDataset(Dataset):
    """SIR-selected token sequences, already at max_seq_len and z-scored dt."""

    def __init__(self, dt_z, s_cls, th_cls, cond, lengths):
        self.dt_z = dt_z
        self.s_cls = s_cls
        self.th_cls = th_cls
        self.cond = cond
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        L = int(self.lengths[idx])
        real = np.zeros(self.dt_z.shape[1], dtype=np.float32)
        real[:L] = 1.0
        return (
            torch.from_numpy(self.dt_z[idx].astype(np.float32)),
            torch.from_numpy(self.s_cls[idx].astype(np.int64)),
            torch.from_numpy(self.th_cls[idx].astype(np.int64)),
            torch.from_numpy(real),
            torch.from_numpy(self.cond[idx].astype(np.float32)),
        )


def load_corpus(pattern):
    shards = sorted(glob.glob(pattern))
    assert shards, f"no corpus shards match {pattern}"
    parts = [np.load(s) for s in shards]
    dt_z = np.concatenate([p["dt_z"] for p in parts])
    s_cls = np.concatenate([p["s_cls"] for p in parts])
    th_cls = np.concatenate([p["th_cls"] for p in parts])
    cond = np.concatenate([p["cond"] for p in parts])
    lengths = np.concatenate([p["length"] for p in parts])
    print(f"  corpus: {len(shards)} shards, {len(lengths):,} trajectories, "
          f"median length {int(np.median(lengths))}", flush=True)
    return dt_z, s_cls, th_cls, cond, lengths


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    ckpt = torch.load(data_dir / args.load_from, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    dt_mean, dt_std = float(ckpt["dt_mean"]), float(ckpt["dt_std"])
    model = EventStreamPolarModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    f_mu = ckpt["feat_mu"].to(device)
    f_sd = ckpt["feat_sd"].to(device)
    print(f"Loaded {args.load_from} (epoch {ckpt.get('epoch')}, "
          f"step {ckpt.get('step')}), feat pathway intact", flush=True)

    dl = DataLoader(
        DistillCorpusDataset(*load_corpus(str(data_dir / args.corpus))),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True)

    real_iter = None
    if args.real_frac > 0:
        s2 = np.load(data_dir / "events_s2.npy", mmap_mode="r")
        dth = np.load(data_dir / "events_dth.npy", mmap_mode="r")
        dt = np.load(data_dir / "events_dt.npy", mmap_mode="r")
        lengths = np.load(data_dir / "events_len.npy")
        conditions = np.load(data_dir / "events_cond.npy")
        rng = np.random.default_rng(123)
        idx = np.sort(rng.choice(len(lengths), min(len(lengths), 400_000),
                                 replace=False))
        real_dl = DataLoader(
            PolarEventDataset(s2[idx], dth[idx], dt[idx], lengths[idx],
                              conditions[idx], cfg["max_seq_len"], dt_mean, dt_std),
            batch_size=args.batch_size, shuffle=True, num_workers=0,
            pin_memory=True, drop_last=True)
        real_iter = iter(real_dl)
        print(f"  real anchor: {len(idx):,} trajectories at frac {args.real_frac}",
              flush=True)

    tables = build_value_tables(device)

    def batch_features(s_cls, th_cls, dt_z, real, cond):
        dt_s = torch.exp(dt_z * dt_std + dt_mean).clamp(0.1, 1000.0) / 1000.0
        speed, motion, tick, cos_th, sin_th = real_batch_values(s_cls, th_cls, tables)
        x, y, fmask = stream_to_frames(speed, motion, cos_th, sin_th,
                                       dt_s, real, cond, args.n_frames)
        return detector_features(x, y, fmask)

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
    step_i = start_step
    t0 = time.time()
    ema = None
    mix_rng = np.random.default_rng(7)
    data_iter = iter(dl)
    while step_i < args.steps:
        use_real = real_iter is not None and mix_rng.random() < args.real_frac
        src = real_iter if use_real else data_iter
        try:
            batch = next(src)
        except StopIteration:
            if use_real:
                real_iter = iter(real_dl)
                batch = next(real_iter)
            else:
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
        if step_i % args.snapshot_every == 0:
            save_ckpt(save_path.with_stem(save_path.stem + f"_s{step_i}"), step_i)
        if step_i % args.save_every == 0 or step_i == args.steps:
            save_ckpt(latest_path, step_i)
            save_ckpt(save_path, step_i)

    print(f"Done. Final loss (ema): {ema:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--load-from", default="event_polar_4m_fc_v2.pt")
    parser.add_argument("--save-name", default="event_polar_4m_distill_v1.pt")
    parser.add_argument("--corpus", default="distill_corpus_b*.npz")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--real-frac", type=float, default=0.0)
    parser.add_argument("--n-frames", type=int, default=256)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--auto-resume", action="store_true")
    train(parser.parse_args())
