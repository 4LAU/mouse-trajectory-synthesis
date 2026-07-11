"""Stage 2d (pilot): trajectory-level GRPO fine-tune of the WS7b polar model.

Why this and not another critic/DPO pass: three independent gradient-based
fine-tunes already died on the same wall (EXPERIMENTS.md, July 4-6). Plain
adversarial (train_events_polar_adv.py) and conditioning-aware adversarial
(train_events_polar_advfc.py) both showed the critic finding a real,
growing gap that the generator's per-position heads could never close --
"the gap is a global per-trajectory property... per-position token heads
cannot coordinate global outcomes" (July 5). Preference learning (DPO)
avoided the straight-through Gumbel path but collapsed off-manifold instead:
pure-model RF OOB rose monotonically from the 0.6470 control to 0.9782 by
step 1500, textbook Goodhart -- the judge's ranking signal is real but only
usable as a selection-time filter, never as a training gradient into this
architecture (July 6, 18:25).

GRPO sidesteps both failure modes structurally, not just by tuning knobs:
  - No straight-through estimator, ever. A trajectory is sampled with the
    EXACT eval-time MaskGIT/Gumbel-reveal procedure in a fully no-grad
    rollout pass (rollout_no_grad below, copied from
    EventStreamPolarModel.sample() because that method is @torch.no_grad
    and cannot be edited in place). Rewards and per-group advantages are
    computed on the decoded trajectories. Only then does a REPLAY pass
    (replay_backward) rerun each recorded reveal step's forward with grad,
    compute that step's contribution to the REINFORCE + KL loss, and call
    backward() immediately so no more than ONE step's graph is ever alive.
    The gradient never crosses a sampling operation or a time step; it only
    flows through each step's own logits into the log-prob of the token
    that step already committed to (a pure score-function estimator). This
    is what "coordinates whole-trajectory outcomes": the trajectory-level
    reward is broadcast, via the advantage, onto the log-prob of every
    token that trajectory contains, so a single global judgment CAN move
    every position's weights in the same direction -- the exact capability
    the adversarial critics lacked.
  - The reward RF is refit only every --refresh-every iterations and frozen
    in between (a moving but intermittently-static target, not a live
    adversary), and every update carries a per-token KL penalty against a
    FROZEN copy of the pretrained model. Both are direct guards against the
    DPO collapse: nothing here differentiates through the judge, and the KL
    leash is the anchor DPO's reference-model term was supposed to provide
    but couldn't stop in weight space.

Honest-split protocol: ALL in-training evals, best-checkpoint selection,
and auto-stop use a VALIDATION human sample drawn from the training pool
with the 2000 seed-42 eval indices excluded (built/cached at startup, see
build_val_human_features). data/human_eval_features.npy -- the headline
eval humans -- is never loaded as an eval class by this trainer; it is
reserved for exactly one manual post-training evaluation. See RL_PILOT.md.

See RL_PILOT.md at the repo root for the full design writeup, the smoke
test transcript, and launch instructions. This file is an authoring-only
pilot: nothing here has been trained for real.

Run (real, GPU, in supervised bursts because this machine bluescreens
under sustained GPU load):
    .venv/Scripts/python.exe training/train_events_polar_grpo.py \
        --iters 500 --device cuda --max-hours 1.5
    ...then repeatedly, until done:
    .venv/Scripts/python.exe training/train_events_polar_grpo.py \
        --iters 500 --device cuda --max-hours 1.5 --resume

Smoke test (CPU, tiny, exercises the whole loop incl. checkpoint/resume):
    .venv/Scripts/python.exe training/train_events_polar_grpo.py \
        --iters 2 --group-size 2 --specs-per-iter 4 --device cpu --smoke
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments._common import DurationModel  # noqa: E402
from features import FEATURE_NAMES, extract_features, resample_trajectory  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    N_S_CLASSES, S_MASK_TOKEN, S_PAD_CLASS, TH_BINS, TH_MASK_TOKEN,
    TH_NULL_CLASS, TICK_CLASS, EventStreamPolarModel, class_to_dtheta,
    class_to_speed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pure fc_v2, single-sample (no SIR), RF OOB at N=2000 -- EXPERIMENTS.md
# "Preference learning verdict" (July 6, 18:25). The pilot's success bar is
# beating this without the tail canaries shrinking. NOTE: that number was
# measured against the EVAL humans; in-training numbers here are against the
# VALIDATION humans, so treat it as a reference line, not an exact target.
BASELINE_FC_V2_AUC_N2000 = 0.6470

STD_JERK_IDX = FEATURE_NAMES.index("std_jerk")
CURVATURE_STD_IDX = FEATURE_NAMES.index("curvature_std")
PATH_EFFICIENCY_IDX = FEATURE_NAMES.index("path_efficiency")

# Eval-humans draw, fixed by the headline protocol (regenerate_human_features
# .py and experiments/novelty_check.py both reproduce it):
# np.random.default_rng(42).choice(n_pool, 2000, replace=False).
EVAL_HUMANS_SEED = 42
EVAL_HUMANS_N = 2000


# ---------------------------------------------------------------------------
# Decode: copied from experiments/event_stream_polar.py's _decode (that file
# is a locked experiment module and is not edited here). TICKMERGE (a
# cosmetic post-process) is dropped for pilot simplicity; SNAP and ROUND --
# the two decode-contract choices that actually move the detector score --
# are kept, matching the "locked recipe" (EXPERIMENTS.md, July 6, 06:15).
# ---------------------------------------------------------------------------
def decode_trajectory(dt_z, s_cls, th_cls, sx, sy, angle, dt_mean, dt_std,
                       snap=2.5, round_=True):
    """(dt_z, s_cls, th_cls) numpy arrays for ONE trajectory -> (x, y, t)
    list, or None if too short. Also returns n, the decoded real-event
    count (first PAD position), used to length-normalize the REINFORCE
    log-prob / KL sums."""
    pad = s_cls >= S_PAD_CLASS
    n = int(np.argmax(pad)) if pad.any() else len(s_cls)
    if n < 2:
        return None

    s_cls_t = torch.from_numpy(s_cls[:n].astype(np.int64))
    th_cls_t = torch.from_numpy(th_cls[:n].astype(np.int64))
    s = class_to_speed(s_cls_t).numpy()
    dth = class_to_dtheta(th_cls_t).numpy()

    motion = s_cls[:n] > TICK_CLASS
    heading = angle + np.cumsum(np.where(motion, dth, 0.0))
    dx = np.where(motion, s * np.cos(heading), 0.0)
    dy = np.where(motion, s * np.sin(heading), 0.0)
    if snap > 0:
        slow = motion & (s > 0) & (s < snap)
        dx = np.where(slow, np.round(dx), dx)
        dy = np.where(slow, np.round(dy), dy)

    dt_ms = np.exp(dt_z[:n] * dt_std + dt_mean)
    dt_s = np.clip(dt_ms, 0.1, 1000.0) / 1000.0

    x = np.concatenate([[sx], sx + np.cumsum(dx)])
    y = np.concatenate([[sy], sy + np.cumsum(dy)])
    if round_:
        x = np.round(x)
        y = np.round(y)
    t = np.concatenate([[0.0], np.cumsum(dt_s)])
    return list(zip(x.tolist(), y.tolist(), t.tolist())), n


def traj_to_features(traj):
    f = extract_features(resample_trajectory(traj))
    if f is None or not np.all(np.isfinite(f)):
        return None
    return f


# ---------------------------------------------------------------------------
# Validation human sample (honest-split guard).
#
# The headline eval humans (data/human_eval_features.npy) must never drive
# best-checkpoint selection or auto-stop: that is model selection on the
# eval sample, the exact leakage the July 5 "SIR leakage audit" fixed on the
# selection side. This builds a fresh VALIDATION sample from the same 4.16M
# pool with (a) the 2000 seed-42 eval indices excluded by index, and (b) any
# row whose 18-feature vector matches a row of the reward reference
# (data/human_ref_features_sir.npy) excluded by feature match -- that file's
# generating script/seed is not in the repo (EXPERIMENTS.md provenance says
# only "4000 drawn from the pool, eval indices excluded"), so index-level
# disjointness from it cannot be proven and feature-level screening is the
# honest fallback. Result is cached (uncommitted) so real runs pay the cost
# once.
# ---------------------------------------------------------------------------
def build_val_human_features(train_dir: Path, cache_path: Path,
                              sir_features: np.ndarray, n_val: int,
                              val_seed: int) -> np.ndarray:
    if cache_path.exists():
        feats = np.load(cache_path)
        print(f"[grpo] loaded cached validation humans: {feats.shape} "
              f"({cache_path})", flush=True)
        return feats

    print(f"[grpo] building validation human sample (n={n_val}, "
          f"seed={val_seed}) from the full pool...", flush=True)
    offsets = np.load(train_dir / "full_pool_offsets.npy")
    flat = np.load(train_dir / "pool_flat_i16.npy", mmap_mode="r")
    t_arr = np.load(train_dir / "pool_t_rel_f32.npy", mmap_mode="r")
    n_pool = len(offsets) - 1

    def pool_feats(idx: int):
        s, e = int(offsets[idx]), int(offsets[idx + 1])
        xy = flat[s:e].astype(np.float64)
        ts = t_arr[s:e].astype(np.float64)
        traj = [(float(xy[j, 0]), float(xy[j, 1]), float(ts[j]))
                for j in range(len(xy))]
        return traj_to_features(traj)

    eval_idx = np.random.default_rng(EVAL_HUMANS_SEED).choice(
        n_pool, size=EVAL_HUMANS_N, replace=False)

    # sanity: prove the reconstructed eval indices ARE the headline eval
    # sample before trusting the exclusion (same gate novelty_check.py uses,
    # first rows only -- cheap). human_eval_features.npy has exactly 2000
    # rows, so row i corresponds to eval index i with nothing dropped.
    eval_feats_cached = np.load(REPO_ROOT / "data" / "human_eval_features.npy")
    assert len(eval_feats_cached) == EVAL_HUMANS_N
    for i in range(5):
        f = pool_feats(int(eval_idx[i]))
        assert f is not None and np.allclose(f, eval_feats_cached[i], atol=1e-6), (
            "reconstructed eval index does not reproduce "
            "data/human_eval_features.npy; refusing to build the validation "
            "sample from unverified indices")

    mask = np.ones(n_pool, dtype=bool)
    mask[eval_idx] = False
    remaining = np.flatnonzero(mask)
    draw = np.random.default_rng(val_seed).choice(
        remaining, size=min(n_val + 1000, len(remaining)), replace=False)

    sir_keys = {np.round(row, 6).tobytes() for row in sir_features}
    rows, n_sir_hits = [], 0
    for idx in draw:
        f = pool_feats(int(idx))
        if f is None:
            continue
        if np.round(f, 6).tobytes() in sir_keys:
            n_sir_hits += 1
            continue
        rows.append(f)
        if len(rows) >= n_val:
            break
    feats = np.asarray(rows)
    if len(feats) < n_val:
        raise RuntimeError(f"could only build {len(feats)}/{n_val} validation rows")
    np.save(cache_path, feats)
    print(f"[grpo] validation humans built: {feats.shape}, "
          f"{n_sir_hits} rows dropped as reward-reference feature matches, "
          f"cached to {cache_path}", flush=True)
    return feats


# ---------------------------------------------------------------------------
# Spec / movement-character sampling (mirrors generate_paths() in
# experiments/event_stream_polar.py: distance from the human empirical
# distribution, uniform angle, duration from the binned empirical prior,
# and -- for featcond checkpoints -- an independent KDE draw per sample from
# the checkpoint's real-feature bank, nearest by log-distance).
# ---------------------------------------------------------------------------
def make_condition_batch(n_distinct, group_size, human_distances, duration_model,
                          rng, device):
    dists = rng.choice(human_distances, size=n_distinct)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=n_distinct)
    log_dist = np.log(np.maximum(dists, 1e-6))
    log_dur = np.array([
        math.log(max(duration_model.sample(float(ld)), 1e-3)) for ld in log_dist
    ], dtype=np.float64)
    cond = np.stack(
        [log_dist, log_dur, np.cos(angles), np.sin(angles)], axis=1
    ).astype(np.float32)
    if group_size > 1:
        cond = np.repeat(cond, group_size, axis=0)
        log_dist_rep = np.repeat(log_dist, group_size)
    else:
        log_dist_rep = log_dist
    return torch.from_numpy(cond).to(device), log_dist_rep.astype(np.float32)


def draw_feat(feat_bank, fb_order, fb_sorted_ld, log_dist_rep, bw, win, device):
    B = len(log_dist_rep)
    ld_t = torch.from_numpy(log_dist_rep).to(device)
    pos = torch.searchsorted(fb_sorted_ld, ld_t.contiguous())
    jit = torch.randint(-win, win + 1, (B,), device=device)
    pos = (pos + jit).clamp(0, len(fb_order) - 1)
    return feat_bank[fb_order[pos]] + bw * torch.randn(
        B, feat_bank.shape[1], device=device)


def sample_chunked(model, cond, feat, seq_len, n_steps, temperature,
                    th_temperature, order, choice_temp, max_batch=500):
    """model.sample() in bounded-size chunks: peak batch memory never exceeds
    max_batch regardless of the total n asked for (reward refresh asks for
    4000, the big eval for 2000; unchunked, either could OOM the 4070)."""
    outs = []
    with torch.no_grad():
        for i in range(0, cond.shape[0], max_batch):
            c = cond[i:i + max_batch]
            f = feat[i:i + max_batch] if feat is not None else None
            outs.append(model.sample(
                c, seq_len, n_steps=n_steps, temperature=temperature,
                th_temperature=th_temperature, order=order,
                choice_temp=choice_temp, feat=f))
    return (torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs]),
            torch.cat([o[2] for o in outs]))


# ---------------------------------------------------------------------------
# GRPO pass 1: no-grad rollout, a copy of EventStreamPolarModel.sample()
# (that method is @torch.no_grad and lives in a tracked module file, so it
# is copied here rather than edited) that additionally RECORDS, per reveal
# step: the input dt_z, the reveal mask, the diffusion time presented to the
# trunk, and the temperature-scaled log-prob of the tokens revealed at that
# step (for the pass-2 consistency assert). Because tokens never change
# after they are revealed, the (s_tok, th_tok) input state of any step can
# be reconstructed in pass 2 from the final tokens plus the cumulative
# reveal masks, so the full token grids are not stored per step.
#
# Nothing here carries gradient (torch.no_grad on the whole pass), so the
# GPU never holds more than one forward's activations during rollout: the
# memory profile is the same as plain inference sampling, plus the small
# recorded tensors (dt_in float32 + reveal bool per step, ~35 MB total at
# B=256, T=256, ~100 steps).
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_no_grad(model, cond, feat, seq_len, n_steps, temperature,
                     th_temperature, order, choice_temp, device):
    B = cond.shape[0]
    temp = max(temperature, 1e-4)
    th_temp = max(th_temperature if th_temperature is not None else temperature, 1e-4)

    state = {
        "dt_z": torch.randn(B, seq_len, device=device),
        "s_tok": torch.full((B, seq_len), S_MASK_TOKEN, dtype=torch.long, device=device),
        "th_tok": torch.full((B, seq_len), TH_MASK_TOKEN, dtype=torch.long, device=device),
    }
    step = 1.0 / n_steps
    steps = []

    def do_step(i, force_all=False):
        t_cont = max(1.0 - i * step, 0.0)
        t_scaled = torch.full((B,), t_cont * (model.n_steps - 1), device=device)
        masked = state["s_tok"] == S_MASK_TOKEN
        dt_in = state["dt_z"].clone()

        x_feat = model.trunk(state["dt_z"], state["s_tok"], state["th_tok"],
                             t_scaled, cond, feat)
        v_pred = model.dt_head(x_feat).squeeze(-1)
        state["dt_z"] = state["dt_z"] - step * v_pred

        if not force_all:
            t_next = max(t_cont - step, 0.0)
            n_target = int(round(
                float(model.sqrt_ab[int(t_next * (model.n_steps - 1))]) * seq_len))
            n_new = n_target - int(seq_len - masked[0].sum().item())
            if n_new <= 0:
                return

        s_logits = model.s_head(x_feat)
        s_logp_temp = F.log_softmax(s_logits / temp, dim=-1)
        s_probs_temp = s_logp_temp.exp()
        s_new = torch.multinomial(
            s_probs_temp.reshape(-1, s_probs_temp.shape[-1]), 1).view(B, seq_len)
        s_for_th = torch.where(masked, s_new, state["s_tok"].clamp(max=N_S_CLASSES - 1))
        th_l = model.th_logits(x_feat, s_for_th)
        th_logp_temp = F.log_softmax(th_l / th_temp, dim=-1)
        th_probs_temp = th_logp_temp.exp()
        th_new_raw = torch.multinomial(
            th_probs_temp.reshape(-1, th_probs_temp.shape[-1]), 1).view(B, seq_len)
        motion = (s_new > TICK_CLASS) & (s_new < S_PAD_CLASS)

        # log-prob and confidence gathered on the RAW sampled th index
        # (0..TH_BINS-1) before it is overwritten with NULL below --
        # mirrors the exact order of EventStreamPolarModel.sample().
        logp_s = s_logp_temp.gather(-1, s_new.unsqueeze(-1)).squeeze(-1)
        logp_th = th_logp_temp.gather(-1, th_new_raw.unsqueeze(-1)).squeeze(-1)
        logp_tok = logp_s + motion.float() * logp_th

        conf = s_probs_temp.gather(-1, s_new.unsqueeze(-1)).squeeze(-1)
        th_conf = th_probs_temp.gather(-1, th_new_raw.unsqueeze(-1)).squeeze(-1)
        conf = torch.where(motion, conf * th_conf, conf)

        th_new = torch.where(motion, th_new_raw,
                             torch.full_like(th_new_raw, TH_NULL_CLASS))

        if force_all:
            reveal = masked.clone()
        else:
            if order == "random":
                score = torch.rand_like(conf)
            elif order == "gumbel":
                g = -torch.log(-torch.log(torch.rand_like(conf).clamp(1e-9, 1.0)))
                anneal = choice_temp * (1.0 - i / n_steps)
                score = torch.log(conf.clamp(min=1e-9)) + anneal * g
            else:
                score = conf
            score = torch.where(masked, score, torch.full_like(score, -1e9))
            rank = score.argsort(dim=-1, descending=True)
            reveal = torch.zeros_like(masked)
            reveal.scatter_(1, rank[:, :n_new], True)
            reveal &= masked

        state["s_tok"] = torch.where(reveal, s_new, state["s_tok"])
        state["th_tok"] = torch.where(reveal, th_new, state["th_tok"])
        steps.append({
            "t_scaled": float(t_cont * (model.n_steps - 1)),
            "dt_in": dt_in,
            "reveal": reveal,
            "logp": (logp_tok * reveal.float()).sum(dim=1),
        })

    for i in range(n_steps):
        do_step(i)
    if (state["s_tok"] == S_MASK_TOKEN).any():
        do_step(n_steps, force_all=True)

    return {"steps": steps, "final_s": state["s_tok"],
            "final_th": state["th_tok"], "final_dt": state["dt_z"]}


# ---------------------------------------------------------------------------
# Checkpointed trunk forward: line-for-line the same computation as
# EventStreamPolarModel.trunk() (models/event_stream_polar.py, not edited in
# place for the same reason rollout_no_grad is a copy -- see the module
# docstring), except each CANDIBlock layer runs under
# torch.utils.checkpoint instead of being called directly. Checkpointing
# discards each layer's intermediate activations right after computing its
# output and RECOMPUTES that one layer's forward during backward instead of
# keeping it resident; it changes nothing about the arithmetic (verified
# bit-identical against model.trunk() under no_grad, same inputs), only when
# each layer's activations exist in memory.
#
# This is the actual fix for the 1000+s/iter replay wall, not a batching
# trick: profiling (scratch_mem_layers.py during this optimization pass)
# showed the un-checkpointed grad-enabled trunk forward alone -- ONE replay
# step, before the reference-model pass or backward -- holds ~9.2 GB on this
# 8 GB 4070, i.e. every single replay step was already overflowing physical
# VRAM and silently spilling into WDDM's shared system-memory fallback (CUDA
# does not hard-OOM on Windows in that case, it just runs at PCIe speed
# instead of VRAM speed, which is consistent with the observed ~11s/step).
# Per-layer checkpointing cuts the peak to ~2.3 GB for the same step, which
# comfortably fits alongside the frozen reference model's no-grad forward
# (no graph, no retained activations, already cheap) with headroom to spare.
# ---------------------------------------------------------------------------
def checkpointed_trunk(model, dt_noisy, s_tok, th_tok, t, cond, feat):
    B, T = dt_noisy.shape
    x = (
        model.dt_proj(dt_noisy.unsqueeze(-1))
        + model.s_embed(s_tok)
        + model.th_embed(th_tok)
        + model.pos_embed(torch.arange(T, device=dt_noisy.device))
    )
    t_emb = model.time_embed(t)
    combined = t_emb + model.cond_embed(cond)
    if model.feat_embed is not None and feat is not None:
        combined = combined + model.feat_embed(feat)
    for layer in model.layers:
        x = torch_checkpoint.checkpoint(layer, x, combined, use_reentrant=False)
    return model.norm(x)


# ---------------------------------------------------------------------------
# GRPO pass 2: replay each recorded reveal step's forward WITH grad, compute
# that step's contribution to the total loss, and backward() it immediately
# so the graph is freed per step -- at no point does the backward have to
# hold more than one step's stored forward, which is what makes B=256 x
# ~100 reveal steps feasible on the 4070 (a single backward over the whole
# rollout would have to keep ~100 forwards' activations alive at once).
#
# The decomposition is exact: the single-backward loss would be
#   mean_valid(-adv_j * sum_i logp_ij / n_j) + beta * mean_valid(sum_i kl_ij / n_j)
# which equals the sum over steps i of
#   sum_j (-adv_j * logp_ij + beta * kl_ij) * w_j / n_valid,   w_j = 1 / n_j
# (w_j = 0 for invalid trajectories), so each step's partial loss carries
# the same global denominator and the accumulated gradient is identical to
# the single backward.
#
# Replay only needs the log-prob of tokens at REVEALED positions, and both
# the log-prob gather and the th-head conditioning are position-local (the
# th head sees the trunk feature plus the speed class at the SAME position),
# so the discarded s/th samples pass 1 drew at not-yet-revealed positions
# never enter any replayed quantity. The recomputed logp is asserted against
# pass 1's recorded value for the first few steps (same weights, eval mode,
# same inputs -> equal to float tolerance).
#
# The policy trunk forward runs through checkpointed_trunk (see above) --
# the only change from the original per-step loop is HOW that forward's
# memory is managed, not what it computes; the per-step accumulation
# (Python float += pg_part.item()) is left exactly as it was so the printed
# pg/kl aggregates stay bit-for-bit reproducible against the pre-change
# code, not just close. --amp (default off) additionally wraps both the
# policy and reference forwards in bf16 autocast for a further speed lever;
# it is the only thing in this function that can move replay-logp-err off
# exactly 0.0 (autocast changes matmul precision), so it defaults OFF and
# every other change here is exact.
# ---------------------------------------------------------------------------
def replay_backward(model, ref_model, rollout, cond, feat, adv_w, inv_n,
                     n_valid, beta, temperature, th_temperature, device,
                     assert_first_k=3, amp=False):
    temp = max(temperature, 1e-4)
    th_temp = max(th_temperature if th_temperature is not None else temperature, 1e-4)
    final_s, final_th = rollout["final_s"], rollout["final_th"]
    B, T = final_s.shape
    revealed_before = torch.zeros(B, T, dtype=torch.bool, device=device)
    total_pg, total_kl, max_logp_err = 0.0, 0.0, 0.0
    amp_enabled = amp and device.type == "cuda"

    motion_all = (final_s > TICK_CLASS) & (final_s < S_PAD_CLASS)
    th_gather = final_th.clamp(max=TH_BINS - 1)
    final_s_clamped = final_s.clamp(max=N_S_CLASSES - 1)

    for si, rec in enumerate(rollout["steps"]):
        reveal = rec["reveal"]
        s_in = torch.where(revealed_before, final_s,
                           torch.full_like(final_s, S_MASK_TOKEN))
        th_in = torch.where(revealed_before, final_th,
                            torch.full_like(final_th, TH_MASK_TOKEN))
        t_scaled = torch.full((B,), rec["t_scaled"], device=device)
        # position-local th conditioning: at revealed positions the sampled
        # speed class is final_s; other positions are never gathered.
        s_cond_tok = torch.where(reveal, final_s, s_in).clamp(max=N_S_CLASSES - 1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            x_feat = checkpointed_trunk(model, rec["dt_in"], s_in, th_in, t_scaled, cond, feat)
            s_logits = model.s_head(x_feat)
            th_l = model.th_logits(x_feat, s_cond_tok)
        s_logits = s_logits.float()
        th_l = th_l.float()

        logp_s = F.log_softmax(s_logits / temp, dim=-1).gather(
            -1, final_s_clamped.unsqueeze(-1)).squeeze(-1)
        logp_th = F.log_softmax(th_l / th_temp, dim=-1).gather(
            -1, th_gather.unsqueeze(-1)).squeeze(-1)
        logp_tok = logp_s + motion_all.float() * logp_th
        logp_step = (logp_tok * reveal.float()).sum(dim=1)

        if si < assert_first_k:
            err = (logp_step.detach() - rec["logp"]).abs().max().item()
            max_logp_err = max(max_logp_err, err)
            assert err < 1e-2, (
                f"pass-2 replay logprob mismatch at step {si}: max abs err "
                f"{err:.6f}. Replay is not reproducing the rollout forward; "
                f"the REINFORCE gradient would be wrong. Aborting.")

        # KL(policy || frozen reference), full categorical, at temperature=1
        # (the anchor measures actual weight drift, not the decoding knob).
        with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            ref_x = ref_model.trunk(rec["dt_in"], s_in, th_in, t_scaled, cond, feat)
            ref_logq_s = F.log_softmax(ref_model.s_head(ref_x).float(), dim=-1)
            ref_logq_th = F.log_softmax(ref_model.th_logits(ref_x, s_cond_tok).float(), dim=-1)
        logp_s_raw = F.log_softmax(s_logits, dim=-1)
        kl_s = (logp_s_raw.exp() * (logp_s_raw - ref_logq_s)).sum(-1)
        logp_th_raw = F.log_softmax(th_l, dim=-1)
        kl_th = (logp_th_raw.exp() * (logp_th_raw - ref_logq_th)).sum(-1)
        kl_tok = kl_s + motion_all.float() * kl_th
        kl_step = (kl_tok * reveal.float()).sum(dim=1)

        pg_part = (-(adv_w * logp_step) * inv_n).sum() / n_valid
        kl_part = beta * (kl_step * inv_n).sum() / n_valid
        (pg_part + kl_part).backward()
        total_pg += pg_part.item()
        total_kl += kl_part.item()

        revealed_before = revealed_before | reveal

    return total_pg, total_kl, max_logp_err


# ---------------------------------------------------------------------------
# Reward RF: fit ONCE per refresh window (frozen in between), 18 detector
# features (features.py), n_estimators=100. Human side is the 4000-row pool
# reference data/human_ref_features_sir.npy (disjoint from the eval human
# class per its EXPERIMENTS.md provenance; the same file SIR selection uses
# for the same leakage reason).
# ---------------------------------------------------------------------------
def generate_features(model, cfg, dt_mean, dt_std, duration_model, human_distances,
                       feat_bank, fb_order, fb_sorted_ld, has_feat, n, sample_steps,
                       order, choice_temp, temperature, th_temperature, snap,
                       feat_bw, feat_win, device, rng, sample_batch):
    """n fresh pure-inference samples -> 18-feature matrix (valid rows only)."""
    cond, log_dist_rep = make_condition_batch(n, 1, human_distances, duration_model,
                                              rng, device)
    feat = (draw_feat(feat_bank, fb_order, fb_sorted_ld, log_dist_rep, feat_bw,
                      feat_win, device) if has_feat else None)
    dt_z, s_tok, th_tok = sample_chunked(
        model, cond, feat, cfg["max_seq_len"], sample_steps, temperature,
        th_temperature, order, choice_temp, max_batch=sample_batch)
    dt_np, s_np, th_np = dt_z.float().cpu().numpy(), s_tok.cpu().numpy(), th_tok.cpu().numpy()
    angle_np = torch.atan2(cond[:, 3], cond[:, 2]).cpu().numpy()
    feats = []
    for i in range(n):
        dec = decode_trajectory(dt_np[i], s_np[i], th_np[i], 0.0, 0.0,
                                 float(angle_np[i]), dt_mean, dt_std, snap=snap)
        if dec is None:
            continue
        f = traj_to_features(dec[0])
        if f is not None:
            feats.append(f)
    return np.asarray(feats)


# ---------------------------------------------------------------------------
# Replay buffer: a capped ring buffer of past iterations' valid rollout
# feature rows, mixed into each RF refit alongside the fresh refresh_n draw
# so a refit is never made from a single newest batch alone (the batch right
# after a policy update is the one most likely to have drifted toward
# whatever the frozen RF currently rewards). cap<=0 disables it entirely --
# every method below is then a no-op and fit_reward_rf falls back to the
# pre-change fresh-samples-only behavior.
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, cap, dim):
        self.cap = cap
        self.dim = dim
        self.buf = np.zeros((cap, dim), dtype=np.float32) if cap > 0 else None
        self.count = 0
        self.ptr = 0

    def add(self, rows):
        if self.cap <= 0 or len(rows) == 0:
            return
        for row in np.asarray(rows, dtype=np.float32):
            self.buf[self.ptr] = row
            self.ptr = (self.ptr + 1) % self.cap
            self.count = min(self.count + 1, self.cap)

    def sample(self, k, rng):
        if self.cap <= 0 or self.count == 0 or k <= 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        idx = rng.choice(self.count, size=min(k, self.count), replace=False)
        return self.buf[idx]

    def valid_rows(self):
        if self.cap <= 0 or self.count == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.buf[:self.count].copy()

    def restore(self, rows):
        if self.cap <= 0 or rows is None or len(rows) == 0:
            return
        k = min(len(rows), self.cap)
        self.buf[:k] = rows[-k:]
        self.count = k
        self.ptr = k % self.cap


def fit_reward_rf(gen_kwargs, human_ref, n, n_trees, seed, replay_buf=None, min_valid=20):
    X_synth = generate_features(n=n, **gen_kwargs)
    if len(X_synth) < max(min_valid, n * 0.2):
        return None, len(X_synth), 0
    X_synth_all = X_synth
    n_buf = 0
    if replay_buf is not None:
        buf_rows = replay_buf.sample(n, np.random.default_rng(seed))
        n_buf = len(buf_rows)
        if n_buf:
            X_synth_all = np.vstack([X_synth, buf_rows])
    # human class stays the full reward_human_ref, untrimmed -- synth can now
    # be up to 2x that (fresh + replay), so class_weight rebalances the RF
    # rather than throwing away human or synth rows to match counts.
    X = np.vstack([human_ref, X_synth_all])
    y = np.concatenate([np.zeros(len(human_ref)), np.ones(len(X_synth_all))])
    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=seed,
                                class_weight="balanced")
    rf.fit(X, y)
    return rf, len(X_synth), n_buf


def rf_reward(rf, X, clip=4.0):
    p = np.clip(rf.predict_proba(X)[:, 1], 1e-4, 1.0 - 1e-4)
    logit = np.log(p / (1.0 - p))
    return np.clip(-logit, -clip, clip)


# ---------------------------------------------------------------------------
# Eval gate: pure inference-time generation (model.sample(), the real
# no_grad method -- byte-for-byte the same code path production eval uses,
# chunked for memory), fresh RF vs the VALIDATION humans (never the eval
# humans -- see build_val_human_features), plus the two tail canaries from
# the July 8 residual analysis (synthetic p99 / human p99 for std_jerk and
# curvature_std; 0.85 and 0.56 for the selected sets -- if RL pushes these
# DOWN, that is the Goodhart tail-shrinkage failure mode starting).
# UPDATE July 10: the curvature_std canary is informational only, not a
# strike condition -- its human p99 anchor is dominated by a handful of
# pathological traces (human p50 is 0.345 vs a p99 near 152455), so the
# ratio sits near zero regardless of model quality and fired false strikes
# at iter 100/200 while detector AUC was improving. Only std_jerk remains
# a strike condition; the synthetic curvature_std bulk is separately
# guarded by the --curv-floor-penalty hinge.
# ---------------------------------------------------------------------------
def run_eval(gen_kwargs, human_pool, n, seed, n_trees, label, min_valid=10):
    X_synth = generate_features(n=n, **gen_kwargs)
    n_use = min(len(X_synth), len(human_pool))
    if n_use < max(min_valid, n * 0.2):
        print(f"  [eval:{label}] too few valid trajectories "
              f"({len(X_synth)}/{n}), skipping this eval", flush=True)
        return None
    Xs, Xh = X_synth[:n_use], human_pool[:n_use]
    X = np.vstack([Xh, Xs])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    rf = RandomForestClassifier(n_estimators=n_trees, oob_score=True, n_jobs=-1,
                                random_state=seed)
    rf.fit(X, y)
    auc = roc_auc_score(y, rf.oob_decision_function_[:, 1])
    tail_std_jerk = float(np.percentile(Xs[:, STD_JERK_IDX], 99)
                          / max(np.percentile(Xh[:, STD_JERK_IDX], 99), 1e-9))
    tail_curv = float(np.percentile(Xs[:, CURVATURE_STD_IDX], 99)
                      / max(np.percentile(Xh[:, CURVATURE_STD_IDX], 99), 1e-9))
    return {"label": label, "n": n_use, "auc": float(auc),
            "tail_std_jerk": tail_std_jerk, "tail_curvature_std": tail_curv}


# ---------------------------------------------------------------------------
# Checkpointing: atomic write (tmp then rename), so a mid-write bluescreen
# never leaves a truncated checkpoint (this machine bluescreens under
# sustained GPU load; --max-hours exists for the same reason).
# ---------------------------------------------------------------------------
def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(args):
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t_start = time.time()

    smoke_n = args.smoke_n
    sample_steps = min(args.sample_steps, 8) if args.smoke else args.sample_steps
    refresh_n = smoke_n if args.smoke else args.refresh_n
    eval_n_small = smoke_n if args.smoke else args.eval_n_small
    eval_n_big = smoke_n if args.smoke else args.eval_n_big
    refresh_every = 1 if args.smoke else args.refresh_every
    eval_every = 1 if args.smoke else args.eval_every
    eval_big_every = 2 if args.smoke else args.eval_big_every
    min_valid_refresh = 3 if args.smoke else 20
    min_valid_eval = 3 if args.smoke else 10

    print(f"[grpo] device={device} smoke={args.smoke} sample_steps={sample_steps} "
          f"group_size={args.group_size} specs_per_iter={args.specs_per_iter} "
          f"beta={args.beta} lr={args.lr} max_hours={args.max_hours} "
          f"baseline_fc_v2_auc_n2000={BASELINE_FC_V2_AUC_N2000} (eval-humans "
          f"reference line; in-training numbers are vs VALIDATION humans)",
          flush=True)

    # --- load policy + frozen reference (always the ORIGINAL checkpoint,
    # never the resumed/partially-trained policy, so the KL anchor cannot
    # itself drift under --resume) ---
    ckpt_path = data_dir / args.load_from
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    dt_mean, dt_std = float(ckpt["dt_mean"]), float(ckpt["dt_std"])
    has_feat = cfg.get("feat_dim", 0) > 0

    model = EventStreamPolarModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[grpo] loaded policy from {args.load_from} (epoch {ckpt.get('epoch')}, "
          f"feat_dim {cfg.get('feat_dim', 0)})", flush=True)

    ref_model = EventStreamPolarModel(**cfg).to(device)
    ref_model.load_state_dict(ckpt["model_state_dict"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    feat_bank = fb_order = fb_sorted_ld = None
    if has_feat:
        feat_bank = ckpt["feat_bank"].to(device)
        fb_ld = ckpt["feat_bank_log_dist"]
        fb_order = torch.argsort(fb_ld).to(device)
        fb_sorted_ld = fb_ld.sort().values.to(device)

    for p in model.dt_head.parameters():
        p.requires_grad_(False)
    g_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(g_params, lr=args.lr, weight_decay=0.0)

    # --- human pools ---
    human_distances = np.load(data_dir / "human_distances.npy")
    duration_model = DurationModel(data_dir, std_mult=args.dur_std)
    reward_human_ref = np.load(args.reward_human_ref)

    # Tail-penalty anchors (the same std_jerk/curvature_std canaries run_eval
    # tracks): human p99, recomputed fresh every run -- NOT checkpointed,
    # since reward_human_ref never changes mid-run so there is nothing to
    # preserve across --resume.
    h99_std_jerk = float(np.percentile(reward_human_ref[:, STD_JERK_IDX], 99))
    h99_curvature_std = float(np.percentile(reward_human_ref[:, CURVATURE_STD_IDX], 99))
    h50_curvature_std = float(np.percentile(reward_human_ref[:, CURVATURE_STD_IDX], 50))

    # Tail-support BONUS anchors (opt-in, default-inert): reward samples that
    # already sit in the human tail/low-tail instead of only penalizing the
    # human-exceeding overshoot tail above. Same source array, same "not
    # checkpointed" reasoning as the p99/p50 anchors above.
    h75_std_jerk = float(np.percentile(reward_human_ref[:, STD_JERK_IDX], 75))
    h25_path_efficiency = float(np.percentile(reward_human_ref[:, PATH_EFFICIENCY_IDX], 25))

    replay_buf = ReplayBuffer(args.replay_cap, dim=reward_human_ref.shape[1])

    val_human_pool = build_val_human_features(
        Path(args.train_pool_dir), Path(args.val_human_cache),
        reward_human_ref, args.val_n, args.val_seed)
    if args.smoke:
        reward_human_ref = reward_human_ref[:smoke_n]
        val_human_pool = val_human_pool[:smoke_n]
    print(f"[grpo] reward-RF human ref: {len(reward_human_ref)} rows "
          f"({args.reward_human_ref}); VALIDATION humans for all in-training "
          f"evals/selection/auto-stop: {len(val_human_pool)} rows "
          f"({args.val_human_cache}). Eval humans are NOT used by this trainer. "
          f"tail-penalty anchors (p99): std_jerk {h99_std_jerk:.3f} "
          f"curvature_std {h99_curvature_std:.3f} | curv-floor anchor (p50): "
          f"curvature_std {h50_curvature_std:.3f} | tail-bonus anchors: "
          f"std_jerk p75 {h75_std_jerk:.3f} path_efficiency p25 "
          f"{h25_path_efficiency:.3f} | replay buffer cap "
          f"{args.replay_cap}", flush=True)

    gen_kwargs = dict(
        model=model, cfg=cfg, dt_mean=dt_mean, dt_std=dt_std,
        duration_model=duration_model, human_distances=human_distances,
        feat_bank=feat_bank, fb_order=fb_order, fb_sorted_ld=fb_sorted_ld,
        has_feat=has_feat, sample_steps=sample_steps, order=args.order,
        choice_temp=args.choice_temp, temperature=args.temperature,
        th_temperature=args.th_temperature, snap=args.snap, feat_bw=args.feat_bw,
        feat_win=args.feat_win, device=device, rng=rng,
        sample_batch=args.sample_batch)

    save_path = data_dir / args.save_name
    latest_path = save_path.with_stem(save_path.stem + "_latest")
    best_path = save_path.with_stem(save_path.stem + "_best")

    start_iter = 0
    reward_rf = None
    last_refresh_iter = -1
    best_auc = None            # best-so-far big-eval AUC vs validation humans
    iter0_auc = None           # pre-update baseline (recorded once)
    iter0_tail_sj = None
    iter0_tail_cv = None
    bad_big_evals = 0          # consecutive bad big evals (patience 3)

    if args.resume and latest_path.exists():
        rck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(rck["model_state_dict"])
        optimizer.load_state_dict(rck["opt_state_dict"])
        start_iter = rck["grpo_iter"]
        reward_rf = rck.get("reward_rf")
        last_refresh_iter = rck.get("reward_rf_refresh_iter", -1)
        best_auc = rck.get("best_auc")
        iter0_auc = rck.get("iter0_auc")
        iter0_tail_sj = rck.get("iter0_tail_std_jerk")
        iter0_tail_cv = rck.get("iter0_tail_curvature_std")
        bad_big_evals = rck.get("bad_big_evals", 0)
        replay_buf.restore(rck.get("replay_buffer", None))
        print(f"[grpo] resumed at iter {start_iter} (best_auc="
              f"{best_auc if best_auc is not None else 'n/a'}, iter0_auc="
              f"{iter0_auc if iter0_auc is not None else 'n/a'}, "
              f"bad_big_evals={bad_big_evals}, reward_rf "
              f"{'present' if reward_rf is not None else 'MISSING, will refit'})",
              flush=True)
        if replay_buf.count > 0:
            print(f"[grpo] restored replay buffer: {replay_buf.count} rows", flush=True)
        else:
            print("[grpo] restored replay buffer: empty", flush=True)

    def maybe_checkpoint(it):
        atomic_save({
            "model_state_dict": model.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
            "config": cfg, "dt_mean": dt_mean, "dt_std": dt_std,
            "feat_mu": ckpt.get("feat_mu"), "feat_sd": ckpt.get("feat_sd"),
            "feat_bank": ckpt.get("feat_bank"),
            "feat_bank_log_dist": ckpt.get("feat_bank_log_dist"),
            "grpo_iter": it, "reward_rf": reward_rf,
            "reward_rf_refresh_iter": last_refresh_iter,
            "replay_buffer": replay_buf.valid_rows(),
            "best_auc": best_auc, "iter0_auc": iter0_auc,
            "iter0_tail_std_jerk": iter0_tail_sj,
            "iter0_tail_curvature_std": iter0_tail_cv,
            "bad_big_evals": bad_big_evals, "args": vars(args),
        }, latest_path)

    model.eval()  # exact inference-time behavior throughout: dropout off,
    #               cond-dropout off; grad still flows in the replay pass.

    # --- iter-0 baseline: one big eval BEFORE any update, so the auto-stop
    # thresholds (best + 0.02, 0.9 * iter-0 tails) are anchored to what the
    # pretrained model actually reads against the validation humans ---
    if iter0_auc is None:
        res0 = run_eval(gen_kwargs, val_human_pool, eval_n_big, args.seed,
                        args.rf_trees, label=f"N={eval_n_big} BASELINE(iter0)",
                        min_valid=min_valid_eval)
        if res0 is None:
            raise RuntimeError("iter-0 baseline eval failed to produce enough "
                               "valid trajectories; cannot anchor auto-stop")
        iter0_auc = res0["auc"]
        iter0_tail_sj = res0["tail_std_jerk"]
        iter0_tail_cv = res0["tail_curvature_std"]
        best_auc = iter0_auc
        print(f"  >>> BASELINE iter 0: {res0['label']} AUC {iter0_auc:.4f} "
              f"(n={res0['n']}) | tail std_jerk {iter0_tail_sj:.3f} "
              f"curvature_std {iter0_tail_cv:.3f}", flush=True)
        maybe_checkpoint(start_iter)

    t0 = time.time()
    it = start_iter
    stop_reason = None
    while it < args.iters:
        # --- reward RF refresh (frozen between windows) ---
        if reward_rf is None or (it - max(last_refresh_iter, 0)) >= refresh_every:
            rf_t0 = time.time()
            new_rf, n_valid_rf, n_buf_rf = fit_reward_rf(
                gen_kwargs, reward_human_ref, refresh_n, args.rf_trees,
                args.seed + it, replay_buf=replay_buf, min_valid=min_valid_refresh)
            last_refresh_iter = it
            if new_rf is None:
                print(f"[grpo] iter {it}: reward-RF refresh got too few valid "
                      f"trajectories ({n_valid_rf}/{refresh_n}), keeping previous RF",
                      flush=True)
            else:
                reward_rf = new_rf
                print(f"[grpo] iter {it}: refreshed reward RF on {n_valid_rf} fresh "
                      f"samples vs {len(reward_human_ref)} training-pool humans "
                      f"+ {n_buf_rf} replay rows ({time.time() - rf_t0:.1f}s)",
                      flush=True)

        # --- pass 1: no-grad rollout of G trajectories per spec ---
        it_t0 = time.time()
        cond, log_dist_rep = make_condition_batch(
            args.specs_per_iter, args.group_size, human_distances, duration_model,
            rng, device)
        feat = (draw_feat(feat_bank, fb_order, fb_sorted_ld, log_dist_rep,
                          args.feat_bw, args.feat_win, device) if has_feat else None)
        rollout = rollout_no_grad(
            model, cond, feat, cfg["max_seq_len"], sample_steps,
            args.temperature, args.th_temperature, args.order, args.choice_temp,
            device)
        angle_np = torch.atan2(cond[:, 3], cond[:, 2]).cpu().numpy()
        gen_time = time.time() - it_t0

        dt_np = rollout["final_dt"].float().cpu().numpy()
        s_np = rollout["final_s"].cpu().numpy()
        th_np = rollout["final_th"].cpu().numpy()

        B = cond.shape[0]
        group_id = np.arange(B) // args.group_size
        feats, valid_idx = [], []
        for i in range(B):
            dec = decode_trajectory(dt_np[i], s_np[i], th_np[i], 0.0, 0.0,
                                     float(angle_np[i]), dt_mean, dt_std,
                                     snap=args.snap)
            if dec is None:
                continue
            traj, n_real = dec
            f = traj_to_features(traj)
            if f is None:
                continue
            feats.append(f)
            valid_idx.append((i, max(n_real, 1)))

        replay_buf.add(feats)  # every iteration, valid or not, feeds the
        #                        buffer -- it is a source of refit diversity,
        #                        independent of whether THIS update runs

        if len(valid_idx) < 0.5 * B or reward_rf is None:
            print(f"  iter {it + 1:4d}/{args.iters} | too few valid trajectories "
                  f"({len(valid_idx)}/{B}) or no reward RF yet, skipping update "
                  f"({gen_time:.1f}s)", flush=True)
            it += 1
            continue

        # --- reward + per-group z-scored advantage ---
        X_synth = np.asarray(feats)
        rewards = rf_reward(reward_rf, X_synth, clip=args.reward_clip)

        # one-sided per-sample tail penalty: pulls down samples whose
        # std_jerk / curvature_std exceed the human p99, subtracted INSIDE
        # the reward (before the group-relative advantage normalization
        # below) so it survives that normalization -- a batch-constant
        # penalty would be nulled by the group-mean subtraction, but this one
        # depends on each sample's own features and so differentiates within
        # a group. Not re-clipped: the RF term above is already clipped, and
        # this term is allowed to push a sample's reward below -reward_clip.
        if args.tail_penalty > 0:
            sj, cv = X_synth[:, STD_JERK_IDX], X_synth[:, CURVATURE_STD_IDX]
            pen_sj = np.where(sj > 0, np.maximum(0.0, np.log(
                np.maximum(sj, 1e-12) / max(h99_std_jerk, 1e-9))), 0.0)
            pen_cv = np.where(cv > 0, np.maximum(0.0, np.log(
                np.maximum(cv, 1e-12) / max(h99_curvature_std, 1e-9))), 0.0)
            tail_pen = args.tail_penalty * (pen_sj + pen_cv)
            rewards = rewards - tail_pen
        else:
            tail_pen = np.zeros(len(X_synth))

        # per-sample undershoot brake on curvature_std: pushes samples whose
        # curvature_std falls BELOW the human median back up, subtracted
        # INSIDE the reward for the same group-normalization reason as
        # tail_pen above. Deliberately NOT gated on cv > 0 like pen_sj/pen_cv
        # above -- cv == 0 (a fully degenerate, zero-curvature trajectory) is
        # the worst undershoot case and must get the full clipped penalty,
        # not a free pass. std_jerk is left alone here (0.822 of human p99,
        # already healthy, and already covered by the overshoot hinge above).
        if args.curv_floor_penalty > 0:
            cv = X_synth[:, CURVATURE_STD_IDX]
            curv_floor_pen = args.curv_floor_penalty * np.clip(np.maximum(
                0.0, np.log(h50_curvature_std / np.maximum(cv, 1e-12))), 0.0, 3.0)
            rewards = rewards - curv_floor_pen
        else:
            curv_floor_pen = np.zeros(len(X_synth))

        # one-sided per-sample tail-SUPPORT bonuses: reward (rather than
        # penalize) samples that already sit in the human tail/low-tail,
        # added INSIDE the reward for the same group-normalization reason as
        # tail_pen/curv_floor_pen above -- a batch-constant bonus would be
        # nulled by the group-mean subtraction. Both default to 0.0 (fully
        # inert) so existing resume commands are unchanged.
        if args.jerk_tail_bonus > 0:
            sj = X_synth[:, STD_JERK_IDX]
            jerk_bonus = args.jerk_tail_bonus * np.where(sj > 0, np.clip(
                np.log(np.maximum(sj, 1e-12) / max(h75_std_jerk, 1e-9)),
                0.0, 1.0), 0.0)
            rewards = rewards + jerk_bonus
        else:
            jerk_bonus = np.zeros(len(X_synth))

        if args.pe_tail_bonus > 0:
            pe = X_synth[:, PATH_EFFICIENCY_IDX]
            pe_bonus = args.pe_tail_bonus * np.where(pe > 0, np.clip(
                np.log(max(h25_path_efficiency, 1e-9) / np.maximum(pe, 1e-12)),
                0.0, 1.0), 0.0)
            rewards = rewards + pe_bonus
        else:
            pe_bonus = np.zeros(len(X_synth))

        advantages = np.zeros(len(valid_idx))
        by_group = {}
        for k, (i, _) in enumerate(valid_idx):
            by_group.setdefault(group_id[i], []).append(k)
        for members in by_group.values():
            r = rewards[members]
            if len(r) > 1 and r.std() > 1e-6:
                advantages[members] = (r - r.mean()) / r.std()
            # else leave at 0: single-member or degenerate group

        # full-batch advantage / length weights (0 for invalid trajectories,
        # so they contribute nothing to either loss term)
        adv_w = torch.zeros(B, device=device)
        inv_n = torch.zeros(B, device=device)
        for k, (i, n_real) in enumerate(valid_idx):
            adv_w[i] = float(advantages[k])
            inv_n[i] = 1.0 / float(n_real)
        n_valid = len(valid_idx)

        # --- pass 2: per-step replay + immediate backward ---
        optimizer.zero_grad()
        loss_pg, loss_kl, logp_err = replay_backward(
            model, ref_model, rollout, cond, feat, adv_w, inv_n, n_valid,
            args.beta, args.temperature, args.th_temperature, device,
            amp=args.amp)
        grad_norm = torch.nn.utils.clip_grad_norm_(g_params, args.clip_grad)
        optimizer.step()
        del rollout

        iter_time = time.time() - it_t0
        tailpen_frac = float((tail_pen > 0).mean()) if len(tail_pen) else 0.0
        jbonus_frac = float((jerk_bonus > 0).mean()) if len(jerk_bonus) else 0.0
        pebonus_frac = float((pe_bonus > 0).mean()) if len(pe_bonus) else 0.0
        print(f"  iter {it + 1:4d}/{args.iters} | loss {loss_pg + loss_kl:+.4f} "
              f"(pg {loss_pg:+.4f} kl {loss_kl:.4f}) | reward {rewards.mean():+.3f} | "
              f"tailpen {tail_pen.mean():.3f} ({tailpen_frac * 100:.1f}%) | "
              f"curvpen {curv_floor_pen.mean():.3f} | "
              f"jbonus {jerk_bonus.mean():.3f} ({jbonus_frac * 100:.1f}%) | "
              f"pebonus {pe_bonus.mean():.3f} ({pebonus_frac * 100:.1f}%) | "
              f"valid {n_valid}/{B} | grad {grad_norm:.3f} | replay-logp-err "
              f"{logp_err:.2e} | gen {gen_time:.1f}s total {iter_time:.1f}s",
              flush=True)

        it += 1

        # --- eval gate (all vs VALIDATION humans) ---
        ran_eval = False
        if it % eval_every == 0:
            res = run_eval(gen_kwargs, val_human_pool, eval_n_small, args.seed,
                           args.rf_trees, label=f"N={eval_n_small} TREND-ONLY",
                           min_valid=min_valid_eval)
            if res is not None:
                print(f"  >>> EVAL iter {it}: {res['label']} AUC {res['auc']:.4f} "
                      f"(n={res['n']}, vs val humans) | tail std_jerk "
                      f"{res['tail_std_jerk']:.3f} curvature_std "
                      f"{res['tail_curvature_std']:.3f} "
                      f"(small-N reads ~0.3 optimistic, trend only)", flush=True)
            ran_eval = True

        if it % eval_big_every == 0:
            res = run_eval(gen_kwargs, val_human_pool, eval_n_big, args.seed,
                           args.rf_trees, label=f"N={eval_n_big} TRUSTWORTHY",
                           min_valid=min_valid_eval)
            if res is not None:
                auc_big = res["auc"]
                tail_sj, tail_cv = res["tail_std_jerk"], res["tail_curvature_std"]
                print(f"  >>> EVAL iter {it}: {res['label']} AUC {auc_big:.4f} "
                      f"(n={res['n']}, vs val humans; iter0 {iter0_auc:.4f}, "
                      f"eval-humans reference line {BASELINE_FC_V2_AUC_N2000}) | "
                      f"tail std_jerk {tail_sj:.3f} (iter0 {iter0_tail_sj:.3f}) "
                      f"curvature_std {tail_cv:.3f} (iter0 {iter0_tail_cv:.3f})",
                      flush=True)

                # bad = clear regression beyond noise: AUC above best-so-far
                # by more than 0.02 (~2 SE at N=2000), or a tail canary down
                # more than 10% from its iter-0 value. Patience: 3 consecutive
                # bad big evals. Single bad readings only warn (loudly).
                bad = False
                if auc_big > best_auc + 0.02:
                    print(f"  !!! N={eval_n_big} AUC {auc_big:.4f} exceeds best "
                          f"{best_auc:.4f} + 0.02 (regression beyond noise)",
                          flush=True)
                    bad = True
                if tail_sj < 0.9 * iter0_tail_sj:
                    print(f"  !!! tail canary std_jerk {tail_sj:.3f} below 90% of "
                          f"iter-0 value {iter0_tail_sj:.3f} (Goodhart collapse "
                          f"starting)", flush=True)
                    bad = True
                if tail_cv < 0.9 * iter0_tail_cv:
                    print(f"  ~~~ tail canary curvature_std {tail_cv:.3f} below "
                          f"90% of iter-0 value {iter0_tail_cv:.3f} (informational "
                          f"only, not a strike: human p99 anchor is dominated by "
                          f"rare pathological traces; bulk is guarded by the "
                          f"curv-floor penalty)", flush=True)
                bad_big_evals = bad_big_evals + 1 if bad else 0

                if auc_big < best_auc:
                    best_auc = auc_big
                    atomic_save({
                        "model_state_dict": model.state_dict(),
                        "config": cfg, "dt_mean": dt_mean, "dt_std": dt_std,
                        "feat_mu": ckpt.get("feat_mu"), "feat_sd": ckpt.get("feat_sd"),
                        "feat_bank": ckpt.get("feat_bank"),
                        "feat_bank_log_dist": ckpt.get("feat_bank_log_dist"),
                        "grpo_iter": it, "grpo_auc_val_nbig": auc_big,
                        "tail_std_jerk": tail_sj, "tail_curvature_std": tail_cv,
                    }, best_path)
                    print(f"  *** new best N={eval_n_big} AUC {auc_big:.4f} "
                          f"(vs val humans), saved {best_path.name}", flush=True)

                if bad_big_evals >= args.auto_stop_patience:
                    stop_reason = (f"auto-stop: {bad_big_evals} consecutive bad "
                                   f"big evals (patience {args.auto_stop_patience})")
            ran_eval = True

        if ran_eval:
            maybe_checkpoint(it)

        if stop_reason:
            print(f"[grpo] {stop_reason}", flush=True)
            break

        if args.max_hours is not None and (time.time() - t_start) > args.max_hours * 3600:
            maybe_checkpoint(it)
            print(f"[grpo] time budget reached ({args.max_hours}h), checkpointed "
                  f"at iter {it}; resume with --resume", flush=True)
            return

    maybe_checkpoint(it)
    print(f"[grpo] done at iter {it}/{args.iters} in {time.time() - t0:.1f}s. "
          f"best big-eval AUC vs val humans: {best_auc:.4f} "
          f"(iter0 {iter0_auc:.4f}; eval-humans reference line "
          f"{BASELINE_FC_V2_AUC_N2000}). Final headline number: run the standard "
          f"eval against the untouched eval humans ONCE, manually, on the best "
          f"checkpoint.", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description="Trajectory-level GRPO pilot for the "
                                            "polar event-stream model")
    p.add_argument("--data-dir", default="training")
    p.add_argument("--load-from", default="event_polar_4m_fc_v2.pt")
    p.add_argument("--save-name", default="event_polar_4m_grpo_v1.pt")
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--beta", type=float, default=0.05, help="KL-anchor coefficient")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--group-size", type=int, default=8, help="G, trajectories per spec")
    p.add_argument("--specs-per-iter", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--refresh-every", type=int, default=50)
    p.add_argument("--refresh-n", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-hours", type=float, default=None,
                   help="wall-clock budget; checkpoint and exit 0 when exceeded "
                        "(this machine bluescreens under sustained GPU load, so "
                        "real runs are launched in supervised bursts)")
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--sample-batch", type=int, default=500,
                   help="max batch for every no-grad model.sample() call "
                        "(reward refresh, evals); bounds peak sampling memory")
    p.add_argument("--order", default="gumbel", choices=["conf", "random", "gumbel"])
    p.add_argument("--choice-temp", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--th-temperature", type=float, default=None)
    p.add_argument("--snap", type=float, default=2.5)
    p.add_argument("--feat-bw", type=float, default=0.25)
    p.add_argument("--feat-win", type=int, default=256)
    p.add_argument("--dur-std", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-big-every", type=int, default=100)
    p.add_argument("--eval-n-small", type=int, default=500)
    p.add_argument("--eval-n-big", type=int, default=2000)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast on the pass-2 replay forward (policy + "
                        "frozen reference), opt-in and OFF by default: it is "
                        "the only lever in this file that can move "
                        "replay-logp-err off exactly 0.0. Checkpointed replay "
                        "(always on, exact) already gets the real 32x8x100 "
                        "config to the speed target without it.")
    p.add_argument("--reward-clip", type=float, default=4.0)
    p.add_argument("--rf-trees", type=int, default=100)
    p.add_argument("--replay-cap", type=int, default=20000,
                   help="ring-buffer capacity of past iterations' valid "
                        "rollout feature rows, mixed into each RF refit "
                        "alongside the fresh refresh_n draw; 0 disables the "
                        "buffer (fresh-samples-only refit, prior behavior)")
    p.add_argument("--tail-penalty", type=float, default=2.0,
                   help="one-sided per-sample penalty coefficient on "
                        "std_jerk/curvature_std exceeding their human p99 "
                        "(log-ratio, subtracted from reward before the "
                        "group-relative advantage normalization); 0 disables")
    p.add_argument("--curv-floor-penalty", type=float, default=3.0,
                   help="one-sided per-sample penalty coefficient pushing "
                        "curvature_std UP toward the human median (log-ratio, "
                        "clipped at 3.0, subtracted from reward before the "
                        "group-relative advantage normalization); 0 disables")
    p.add_argument("--jerk-tail-bonus", type=float, default=0.0,
                   help="one-sided per-sample BONUS coefficient rewarding "
                        "std_jerk above the human p75 (log-ratio, clipped at "
                        "1.0, added to reward before the group-relative "
                        "advantage normalization); 0.0 (default) disables, "
                        "fully inert")
    p.add_argument("--pe-tail-bonus", type=float, default=0.0,
                   help="one-sided per-sample BONUS coefficient rewarding "
                        "path_efficiency below the human p25 (log-ratio, "
                        "clipped at 1.0, added to reward before the "
                        "group-relative advantage normalization); 0.0 "
                        "(default) disables, fully inert")
    p.add_argument("--auto-stop-patience", type=int, default=3,
                   help="consecutive bad BIG evals before auto-stop")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reward-human-ref",
                   default=str(REPO_ROOT / "data" / "human_ref_features_sir.npy"),
                   help="4000-row training-pool human reference (disjoint from "
                        "the eval human class) for fitting the reward RF")
    p.add_argument("--val-human-cache",
                   default=str(REPO_ROOT / "data" / "human_val_features_grpo.npy"),
                   help="cache path for the validation human sample used by all "
                        "in-training evals/selection/auto-stop (built at startup "
                        "if missing; leave uncommitted)")
    p.add_argument("--val-n", type=int, default=2000)
    p.add_argument("--val-seed", type=int, default=20260709,
                   help="draw seed for the validation human sample (eval indices "
                        "excluded by index, reward-reference rows by feature match)")
    p.add_argument("--train-pool-dir", default=str(REPO_ROOT / "training"),
                   help="directory with full_pool_offsets/pool_flat_i16/"
                        "pool_t_rel_f32 for building the validation sample")
    p.add_argument("--smoke", action="store_true",
                   help="internal/CI-only: shrink sample-steps, refresh/eval "
                        "sample sizes, and eval cadence to tiny constants so the "
                        "loop can be exercised end to end on CPU in seconds. Not "
                        "for real runs.")
    p.add_argument("--smoke-n", type=int, default=8,
                   help="sample count used everywhere --smoke shrinks a size to")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
