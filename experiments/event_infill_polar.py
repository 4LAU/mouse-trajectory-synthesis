"""Masked-infill hybrid: regenerate a fraction of a real event stream.

Level 2 on the purity scale. For each requested endpoint pair, pick a real
trajectory of matching distance, mask INFILL_FRAC of its speed/dtheta tokens
at random positions, and let the MaskGIT sampler re-infill them. The real dt
sequence is kept (the dt head has been frozen since the DM stage; timing was
never the gap). Decode uses the full eval contract from
experiments/event_stream_polar (integer rounding, EVENT_SNAP, same env knobs),
and the start angle is the requested one, so the borrowed shape is rotated to
the requested direction exactly like the corpus replay baseline.

The two endpoints of the dial are known: INFILL_FRAC=0 is event replay
(RF OOB 0.507) and INFILL_FRAC=1 with real timing is close to pure generation
(~0.70). Sweeping the fraction traces the purity-vs-detectability frontier.

Env knobs (on top of the EVENT_* knobs inherited from event_stream_polar):
  INFILL_FRAC   fraction of s/dtheta tokens regenerated (default 0.5)
  INFILL_STEPS  reveal iterations for the completion (default 12)
  INFILL_MODE   "random" scatters masked positions, "span" masks one
                contiguous stretch at a random offset (default random)
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

import experiments.event_stream_polar as base
from models.event_stream_polar import (
    N_S_CLASSES, S_MASK_TOKEN, S_PAD_CLASS, TH_BINS, TH_MASK_TOKEN,
    TH_NULL_CLASS, TICK_CLASS, dth_lattice_to_class, s2_to_class,
)

_TRAIN_DIR = Path(os.environ.get("TRAIN_DIR", "./training"))
_DEVICE = base._DEVICE
_MODEL = base._model
_CFG = base._cfg

_FRAC = float(os.environ.get("INFILL_FRAC", "0.5"))
_STEPS = int(os.environ.get("INFILL_STEPS", "12"))
_MODE = os.environ.get("INFILL_MODE", "random")
_CHOICE_TEMP = base._CHOICE_TEMP
_TEMP = base._TEMP

_s2 = np.load(_TRAIN_DIR / "events_s2.npy", mmap_mode="r")
_dth = np.load(_TRAIN_DIR / "events_dth.npy", mmap_mode="r")
_dt = np.load(_TRAIN_DIR / "events_dt.npy", mmap_mode="r")
_lengths = np.load(_TRAIN_DIR / "events_len.npy")
_conds = np.load(_TRAIN_DIR / "events_cond.npy")

_log_dist = _conds[:, 0].astype(np.float64)
_order = np.argsort(_log_dist)
_sorted_log_dist = _log_dist[_order]
_rng = np.random.default_rng(12345)

print(f"[event_infill_polar] frac={_FRAC} steps={_STEPS} mode={_MODE} "
      f"pool={len(_lengths):,} trajectories")


def _pick_real(target_dist: float, used: set[int]) -> int:
    pos = int(np.searchsorted(_sorted_log_dist, math.log(max(target_dist, 1e-3))))
    for _ in range(16):
        j = int(np.clip(pos + int(_rng.integers(-32, 33)), 0, len(_order) - 1))
        idx = int(_order[j])
        if idx not in used and int(_lengths[idx]) >= 5:
            used.add(idx)
            return idx
    used.add(idx)
    return idx


@torch.no_grad()
def _complete(dt_z, s_tok, th_tok, cond, n_real):
    """MaskGIT completion of partially masked token grids, gumbel order,
    t conditioning matched to the actual masked fraction (same trick as the
    DM stage's st_complete)."""
    B, T = s_tok.shape
    for i in range(_STEPS):
        masked = s_tok == S_MASK_TOKEN
        n_masked = masked.float().sum(dim=1)
        if not masked.any():
            break
        frac_masked = (n_masked.sum() / n_real.sum().clamp(min=1)).item()
        t_idx = (_MODEL.sqrt_ab - (1.0 - frac_masked)).abs().argmin().float()
        t_scaled = torch.full((B,), float(t_idx), device=_DEVICE)

        x_feat = _MODEL.trunk(dt_z, s_tok, th_tok, t_scaled, cond)
        s_logits = _MODEL.s_head(x_feat) / _TEMP
        s_probs = torch.softmax(s_logits, dim=-1)
        s_new = torch.multinomial(s_probs.view(-1, s_probs.shape[-1]), 1).view(B, T)
        s_for_th = torch.where(masked, s_new, s_tok.clamp(max=N_S_CLASSES - 1))
        th_l = _MODEL.th_logits(x_feat, s_for_th) / _TEMP
        th_probs = torch.softmax(th_l, dim=-1)
        th_new = torch.multinomial(th_probs.view(-1, th_probs.shape[-1]), 1).view(B, T)
        motion = (s_new > TICK_CLASS) & (s_new < S_PAD_CLASS)
        th_new = torch.where(motion, th_new, torch.full_like(th_new, TH_NULL_CLASS))

        conf = s_probs.gather(-1, s_new.unsqueeze(-1)).squeeze(-1)
        th_conf = th_probs.gather(-1, th_new.clamp(max=TH_BINS - 1).unsqueeze(-1)).squeeze(-1)
        conf = torch.where(motion, conf * th_conf, conf)
        g = -torch.log(-torch.log(torch.rand_like(conf).clamp(1e-9, 1.0)))
        score = torch.log(conf.clamp(min=1e-9)) \
            + _CHOICE_TEMP * (1.0 - i / _STEPS) * g
        score = torch.where(masked, score, torch.full_like(score, -1e9))

        k = torch.ceil(n_masked / max(_STEPS - i, 1)).long().clamp(max=T)
        rank = score.argsort(dim=-1, descending=True)
        arange = torch.arange(T, device=_DEVICE).unsqueeze(0)
        take = arange < k.unsqueeze(1)
        reveal = torch.zeros_like(masked)
        reveal.scatter_(1, rank, take)
        reveal &= masked
        s_tok[reveal] = s_new[reveal]
        th_tok[reveal] = th_new[reveal]

    # safety: any straggler masked position falls back to its greedy token
    left = s_tok == S_MASK_TOKEN
    if left.any():
        s_tok[left] = s_new[left]
        th_tok[left] = torch.where(
            (s_new[left] > TICK_CLASS) & (s_new[left] < S_PAD_CLASS),
            th_new[left], torch.full_like(th_new[left], TH_NULL_CLASS))
    return s_tok, th_tok


def generate_paths(specs: list) -> list:
    results: list = [None] * len(specs)
    used: set[int] = set()
    pending = []
    for idx, (sx, sy, ex, ey) in enumerate(specs):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            results[idx] = [(sx, sy, 0.0), (ex, ey, 0.008)]
            continue
        pending.append((idx, sx, sy, math.atan2(ey - sy, ex - sx),
                        _pick_real(dist, used)))

    seq_len = _CFG["max_seq_len"]
    for c0 in range(0, len(pending), base._EVAL_BATCH):
        chunk = pending[c0:c0 + base._EVAL_BATCH]
        B = len(chunk)
        dt_z = torch.zeros(B, seq_len, dtype=torch.float32)
        s_tok = torch.full((B, seq_len), S_PAD_CLASS, dtype=torch.long)
        th_tok = torch.full((B, seq_len), TH_NULL_CLASS, dtype=torch.long)
        cond = torch.zeros(B, 4, dtype=torch.float32)
        n_real = torch.zeros(B)

        for k, (_, _, _, angle, ri) in enumerate(chunk):
            L = min(int(_lengths[ri]), seq_len)
            n_real[k] = L
            dt_ms = np.asarray(_dt[ri, :L], dtype=np.float32)
            dt_z[k, :L] = torch.from_numpy(
                (np.log(np.maximum(dt_ms, 0.05)) - base._DT_MEAN) / base._DT_STD)
            s2 = torch.from_numpy(np.asarray(_s2[ri, :L], dtype=np.int64))
            dth = torch.from_numpy(np.asarray(_dth[ri, :L], dtype=np.int64))
            s_tok[k, :L] = s2_to_class(s2)
            th_tok[k, :L] = torch.where(
                s2 > 0, dth_lattice_to_class(dth),
                torch.full_like(dth, TH_NULL_CLASS))
            # rotate the borrowed shape to the requested direction: tokens are
            # direction relative, the conditioning angle sets the frame
            cond[k] = torch.tensor([
                _conds[ri, 0], _conds[ri, 1], math.cos(angle), math.sin(angle)])
            if _FRAC > 0:
                n_mask = int(round(_FRAC * L))
                if n_mask > 0:
                    if _MODE == "span":
                        start = int(_rng.integers(0, L - n_mask + 1))
                        mp = np.arange(start, start + n_mask)
                    else:
                        mp = _rng.choice(L, size=n_mask, replace=False)
                    s_tok[k, mp] = S_MASK_TOKEN
                    th_tok[k, mp] = TH_MASK_TOKEN

        dt_z, s_tok, th_tok, cond, n_real = (
            t.to(_DEVICE) for t in (dt_z, s_tok, th_tok, cond, n_real))
        if _FRAC > 0:
            s_tok, th_tok = _complete(dt_z, s_tok, th_tok, cond, n_real)

        dt_np = dt_z.float().cpu().numpy()
        s_np = s_tok.cpu().numpy()
        th_np = th_tok.cpu().numpy()
        for k, (oi, sx, sy, angle, _) in enumerate(chunk):
            results[oi] = base._decode(dt_np[k], s_np[k], th_np[k], sx, sy, angle)
    return results


def generate_path(sx, sy, ex, ey):
    return generate_paths([(sx, sy, ex, ey)])[0]
