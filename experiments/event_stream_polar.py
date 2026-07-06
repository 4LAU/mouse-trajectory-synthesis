"""WS7b polar event-stream model experiment.

Generates (dt, speed, heading-increment) event sequences, integrates heading
from the conditioning angle, decodes to positions ROUNDED TO INTEGER PIXELS
(the replay gate showed off-grid positions alone cost ~0.05 AUC), and returns
the raw event trajectory. evaluate.py applies the standard 125Hz resample in
feature extraction. Pure T3: no post-processing.

Env knobs:
  EVENT_CKPT   checkpoint name in training/ (default event_polar_best.pt)
  EVENT_STEPS  sampler steps (default 100)
  EVENT_TEMP   softmax temperature (default 1.0)
  EVENT_ROUND  1 = integer-pixel decode (default 1; 0 only for diagnostics)
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments._common import DurationModel, Trajectory, get_device
from models.event_stream_polar import (
    S_PAD_CLASS, TICK_CLASS, EventStreamPolarModel, class_to_dtheta,
    class_to_speed,
)

torch.manual_seed(42)

_TRAIN_DIR = Path(os.environ.get("TRAIN_DIR", "./training"))
_DEVICE = get_device()

_ckpt_name = os.environ.get("EVENT_CKPT", "event_polar_best.pt")
_ckpt = torch.load(_TRAIN_DIR / _ckpt_name, map_location=_DEVICE, weights_only=False)
_cfg = _ckpt["config"]
_DT_MEAN = float(_ckpt["dt_mean"])
_DT_STD = float(_ckpt["dt_std"])

_model = EventStreamPolarModel(**_cfg).to(_DEVICE)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()

_duration = DurationModel(_TRAIN_DIR, std_mult=float(os.environ.get("EVENT_DUR_STD", "0.7")))
_N_STEPS = int(os.environ.get("EVENT_STEPS", "100"))
_TEMP = float(os.environ.get("EVENT_TEMP", "1.0"))
_TH_TEMP = float(os.environ.get("EVENT_TH_TEMP", "0") or 0) or None
_ORDER = os.environ.get("EVENT_ORDER", "conf")
_CHOICE_TEMP = float(os.environ.get("EVENT_CHOICE_TEMP", "0"))
_ROUND = os.environ.get("EVENT_ROUND", "1") == "1"
# Merge a generated tick into the following event when both neighbours are
# fast motion (>= 10 px). Humans emit ~4.5 mid-flight ticks per 1000 events;
# the sampler's worst paths alternate tick/motion at 1 ms and that timing
# jitter dominates the angular-velocity features.
_TICKMERGE = os.environ.get("EVENT_TICKMERGE", "0") == "1"
_TICKMERGE_MIN = float(os.environ.get("EVENT_TICKMERGE_MIN", "10"))
# Snap slow steps (s < threshold px) to the integer lattice as whole steps.
# Human slow motion is natively lattice-aligned (repeated identical 1px
# steps, occasional direction changes); rounding a smooth off-lattice path
# instead alternates lattice directions nearly every step, which manufactures
# a 3x angular-velocity excess at slow frames. 0 disables.
_SNAP = float(os.environ.get("EVENT_SNAP", "0"))
_EVAL_BATCH = int(os.environ.get("EVENT_EVAL_BATCH", "256"))
# Feature-conditioned checkpoints (train_events_polar_featcond.py) carry a
# bank of z-scored real feature vectors; sample the "movement character"
# conditioning from a KDE over it: bank row + N(0, bw). EVENT_FEAT_BW sets
# the bandwidth, EVENT_FEAT=0 disables conditioning (zero vector).
_FEAT_BANK = None
if _cfg.get("feat_dim", 0) > 0 and "feat_bank" in _ckpt:
    _FEAT_BANK = _ckpt["feat_bank"].to(_DEVICE)
    _fb_ld = _ckpt["feat_bank_log_dist"]
    _FB_ORDER = torch.argsort(_fb_ld).to(_DEVICE)
    _FB_SORTED_LD = _fb_ld.sort().values.to(_DEVICE)
_FEAT_ON = os.environ.get("EVENT_FEAT", "1") == "1"
_FEAT_BW = float(os.environ.get("EVENT_FEAT_BW", "0.25"))
_FEAT_WIN = int(os.environ.get("EVENT_FEAT_WIN", "256"))
# Best-of-N candidate selection: sample K candidates per spec under the SAME
# commanded character, keep the one whose realized features (computed with
# the training-side differentiable pipeline, z-scored by the checkpoint
# stats) land closest to the command. Selection-side realization pressure;
# the gradient-side critics could not deliver this. 1 disables.
# NEGATIVE RESULT July 5: argmin-distance selection shrinks conditional
# feature variance and scores WORSE (0.698 vs 0.651 seed 42). Kept for the
# record; use EVENT_SIR instead.
_BESTOF = int(os.environ.get("EVENT_BESTOF", "1"))
if _BESTOF > 1:
    from training.train_events_polar_dm import (
        build_value_tables, detector_features, real_batch_values,
        stream_to_frames,
    )
    _TABLES = build_value_tables(_DEVICE)
    _FEAT_MU = _ckpt["feat_mu"].to(_DEVICE)
    _FEAT_SD = _ckpt["feat_sd"].to(_DEVICE)
# Sampling-importance-resampling: K candidates per spec with INDEPENDENT
# character draws, then one kept per spec by weighted draw where the weight
# is the human/synthetic density ratio from a discriminator fitted on the
# spot (18 eval-pipeline features, human refs from data/). Matches the
# realized feature DISTRIBUTION to the human one instead of pulling each
# sample toward a target, which is the mistake EVENT_BESTOF made.
# EVENT_SIR_TEMP tempers the weights (higher = closer to uniform).
_SIR_K = int(os.environ.get("EVENT_SIR", "1"))
_SIR_TEMP = float(os.environ.get("EVENT_SIR_TEMP", "1.0"))
# Reference set for the discriminator. Default is a 4000-trajectory pool
# sample DISJOINT from the eval set (human_eval_features.npy IS the eval
# human class, so fitting on it would leak the eval sample into selection).
_SIR_REF = os.environ.get("EVENT_SIR_REF", "data/human_ref_features_sir.npy")
_SIR_TREES = int(os.environ.get("EVENT_SIR_TREES", "200"))
# Iterated SIR: after the first lottery, refit the discriminator on the
# SELECTED outputs vs human and add its log-odds to every candidate's
# accumulated weight, then redraw. Boosting-style density-ratio refinement
# on the same candidate pool; costs CPU only. 1 = plain SIR.
_SIR_ITER = int(os.environ.get("EVENT_SIR_ITER", "1"))
assert not (_BESTOF > 1 and _SIR_K > 1), "EVENT_BESTOF and EVENT_SIR are exclusive"

print(f"[event_stream_polar] ckpt={_ckpt_name} epoch={_ckpt.get('epoch')} "
      f"steps={_N_STEPS} temp={_TEMP} th_temp={_TH_TEMP} order={_ORDER} "
      f"round={_ROUND} bestof={_BESTOF}")


def _decode(dt_z, s_cls, th_cls, sx, sy, angle) -> Trajectory | None:
    # PAD lives on the speed head only; truncate at the first one
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
    if _SNAP > 0:
        # emit slow steps as whole lattice steps at the heading's nearest
        # realizable direction; the integrated heading stays continuous
        slow = motion & (s > 0) & (s < _SNAP)
        dx = np.where(slow, np.round(dx), dx)
        dy = np.where(slow, np.round(dy), dy)

    dt_ms = np.exp(dt_z[:n] * _DT_STD + _DT_MEAN)
    dt_s = np.clip(dt_ms, 0.1, 1000.0) / 1000.0

    if _TICKMERGE and n >= 3:
        spd = np.hypot(dx, dy)
        mid = (~motion[1:-1]) & (spd[:-2] >= _TICKMERGE_MIN) & (spd[2:] >= _TICKMERGE_MIN)
        drop = np.zeros(n, dtype=bool)
        drop[1:-1] = mid
        if drop.any():
            dt_s = dt_s.copy()
            for i in np.flatnonzero(drop):
                dt_s[i + 1] += dt_s[i]
            keep = ~drop
            dx, dy, dt_s = dx[keep], dy[keep], dt_s[keep]
            if len(dx) < 2:
                return None

    x = np.concatenate([[sx], sx + np.cumsum(dx)])
    y = np.concatenate([[sy], sy + np.cumsum(dy)])
    if _ROUND:
        x = np.round(x)
        y = np.round(y)
    t = np.concatenate([[0.0], np.cumsum(dt_s)])
    return list(zip(x.tolist(), y.tolist(), t.tolist()))


def generate_paths(specs: list) -> list:
    results: list = [None] * len(specs)
    pending = []
    for idx, (sx, sy, ex, ey) in enumerate(specs):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            results[idx] = [(sx, sy, 0.0), (ex, ey, 0.008)]
            continue
        log_dist = math.log(dist)
        angle = math.atan2(ey - sy, ex - sx)
        log_dur = math.log(_duration.sample(log_dist))
        pending.append({
            "idx": idx, "sx": sx, "sy": sy, "angle": angle,
            "cond": [log_dist, log_dur, math.cos(angle), math.sin(angle)],
        })

    seq_len = _cfg["max_seq_len"]
    K = _BESTOF if (_BESTOF > 1 and _FEAT_BANK is not None and _FEAT_ON) else 1
    K_sir = _SIR_K if (_SIR_K > 1 and _FEAT_BANK is not None and _FEAT_ON) else 1
    sir_cands: dict = {it["idx"]: [] for it in pending} if K_sir > 1 else {}
    chunk_size = max(_EVAL_BATCH // max(K, K_sir), 1)
    for c0 in range(0, len(pending), chunk_size):
        chunk = pending[c0:c0 + chunk_size]
        cond = torch.tensor([it["cond"] for it in chunk],
                            dtype=torch.float32, device=_DEVICE)
        if K_sir > 1:
            cond = cond.repeat_interleave(K_sir, dim=0)
        feat = None
        if _FEAT_BANK is not None:
            B = cond.shape[0]
            if _FEAT_ON:
                # draw a movement character consistent with the requested
                # distance: nearest bank rows by log-distance, jittered
                pos = torch.searchsorted(_FB_SORTED_LD, cond[:, 0].contiguous())
                jit = torch.randint(-_FEAT_WIN, _FEAT_WIN + 1, (B,), device=_DEVICE)
                pos = (pos + jit).clamp(0, len(_FB_ORDER) - 1)
                feat = _FEAT_BANK[_FB_ORDER[pos]] + _FEAT_BW * torch.randn(
                    B, _FEAT_BANK.shape[1], device=_DEVICE)
            else:
                feat = torch.zeros(B, _FEAT_BANK.shape[1], device=_DEVICE)
        if K > 1:
            cond_s = cond.repeat_interleave(K, dim=0)
            feat_s = feat.repeat_interleave(K, dim=0)
        else:
            cond_s, feat_s = cond, feat
        with torch.no_grad():
            dt_z, s_tok, th_tok = _model.sample(
                cond_s, seq_len, n_steps=_N_STEPS, temperature=_TEMP,
                th_temperature=_TH_TEMP, order=_ORDER, choice_temp=_CHOICE_TEMP,
                feat=feat_s,
            )
        if K > 1:
            # realized character of each candidate, same pipeline and
            # z-scoring that built the bank; rank by distance to command
            with torch.no_grad():
                pad = s_tok >= S_PAD_CLASS
                real = (pad.cumsum(dim=1) == 0).float()
                dt_ms = torch.exp(dt_z * _DT_STD + _DT_MEAN)
                dt_s = dt_ms.clamp(0.1, 1000.0) / 1000.0
                speed, motion, tick, cos_th, sin_th = real_batch_values(
                    s_tok.clamp(max=S_PAD_CLASS), th_tok, _TABLES)
                x, y, fmask = stream_to_frames(speed, motion, cos_th, sin_th,
                                               dt_s, real, cond_s, 256)
                realized = detector_features(x, y, fmask)
                realized = ((realized - _FEAT_MU) / _FEAT_SD).clamp(-10.0, 10.0)
                score = ((realized - feat_s) ** 2).mean(dim=1)
                score = torch.where(torch.isfinite(score), score,
                                    torch.full_like(score, 1e9))
                order = score.view(-1, K).argsort(dim=1)
            order_np = order.cpu().numpy()
        dt_np = dt_z.float().cpu().numpy()
        s_np = s_tok.cpu().numpy()
        th_np = th_tok.cpu().numpy()
        for k, it in enumerate(chunk):
            if K > 1:
                # decode candidates best-first until one is valid
                for j in order_np[k]:
                    row = k * K + int(j)
                    traj = _decode(dt_np[row], s_np[row], th_np[row],
                                   it["sx"], it["sy"], it["angle"])
                    if traj is not None:
                        break
                results[it["idx"]] = traj
            elif K_sir > 1:
                for j in range(K_sir):
                    row = k * K_sir + j
                    sir_cands[it["idx"]].append(
                        _decode(dt_np[row], s_np[row], th_np[row],
                                it["sx"], it["sy"], it["angle"]))
            else:
                results[it["idx"]] = _decode(dt_np[k], s_np[k], th_np[k],
                                             it["sx"], it["sy"], it["angle"])

    if K_sir > 1:
        _sir_select(sir_cands, results)
    return results


def _sir_select(sir_cands: dict, results: list) -> None:
    """Keep one candidate per spec by a weighted draw whose weights are the
    human/synthetic density ratio from a freshly fitted discriminator."""
    from sklearn.ensemble import GradientBoostingClassifier

    from features import extract_features, resample_trajectory

    feats, owners = [], []
    for idx, cands in sir_cands.items():
        for traj in cands:
            if traj is None or len(traj) < 3:
                continue
            f = extract_features(resample_trajectory(traj))
            if f is None or not np.all(np.isfinite(f)):
                continue
            feats.append(f)
            owners.append((idx, traj))
    if not feats:
        return
    X_syn = np.asarray(feats)
    X_hum = np.load(_SIR_REF)

    def fit_logodds(X_neg):
        X = np.concatenate([X_hum, X_neg])
        y = np.concatenate([np.ones(len(X_hum)), np.zeros(len(X_neg))])
        clf = GradientBoostingClassifier(n_estimators=_SIR_TREES, max_depth=3,
                                         subsample=0.8, random_state=0)
        clf.fit(X, y)
        p = np.clip(clf.predict_proba(X_syn)[:, 1], 1e-4, 1 - 1e-4)
        return np.log(p) - np.log(1.0 - p)

    logw = fit_logodds(X_syn)

    per_spec: dict = {}
    for ci, (idx, traj) in enumerate(owners):
        per_spec.setdefault(idx, []).append((ci, traj))
    rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))

    picks: dict = {}
    for it in range(max(_SIR_ITER, 1)):
        if it > 0:
            sel = np.array(sorted(picks.values()))
            logw = logw + fit_logodds(X_syn[sel])
        ess = []
        for cands in per_spec.values():
            lw = logw[[ci for ci, _ in cands]] / _SIR_TEMP
            p_ = np.exp(lw - lw.max())
            p_ /= p_.sum()
            ess.append(1.0 / np.sum(p_ ** 2))
        ess = np.array(ess)
        print(f"[sir] iter={it + 1}/{_SIR_ITER} K={_SIR_K} temp={_SIR_TEMP} "
              f"specs={len(per_spec)} logw mean={logw.mean():+.2f} "
              f"std={logw.std():.2f} | per-spec ESS median={np.median(ess):.2f} "
              f"p10={np.percentile(ess, 10):.2f} p90={np.percentile(ess, 90):.2f} "
              f"(max {_SIR_K})", flush=True)
        for idx, cands in per_spec.items():
            g = rng.gumbel(size=len(cands))
            j = int(np.argmax(logw[[ci for ci, _ in cands]] / _SIR_TEMP + g))
            picks[idx] = cands[j][0]
            results[idx] = cands[j][1]
    # specs whose every candidate failed to decode or featurize
    for idx, cands in sir_cands.items():
        if results[idx] is None:
            results[idx] = next((t for t in cands if t is not None), None)


def generate_path(sx, sy, ex, ey) -> Trajectory | None:
    return generate_paths([(sx, sy, ex, ey)])[0]
