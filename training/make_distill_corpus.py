"""SIR distillation corpus with the locked July 6 recipe.

Samples K candidates per spec from the feature-conditioned 4M model, judges
all candidates in a block with the disjoint-reference GBM (exactly the eval
SIR discriminator), keeps one per spec by the tempered Gumbel lottery, and
saves the winner's TOKENS rather than its decoded pixels. Training on tokens
keeps the fine-tune objective identical to pretraining; integer rounding and
lattice snapping stay serving-time decode steps, so nothing gets applied
twice. The judge still scores the fully decoded trajectory, because that is
the artifact a detector would see.

Writes one shard per 2000-spec block (crash-safe, skips existing shards):
    training/distill_corpus_bNN.npz  with dt_z, s_cls, th_cls, cond, length

Run (locked recipe is the default):
    .venv/Scripts/python.exe training/make_distill_corpus.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("EVENT_CKPT", "event_polar_4m_fc_v2.pt")
os.environ.setdefault("EVENT_ORDER", "gumbel")
os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_SNAP", "2.5")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_SIR", "1")  # selection happens here, not in generate_paths

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments.event_stream_polar as esp  # noqa: E402  (loads model + ckpt)
from features import extract_features, resample_trajectory  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_BINS, TH_NULL_CLASS, TICK_CLASS,
)

K = int(os.environ.get("DISTILL_K", "16"))
SIR_TEMP = float(os.environ.get("DISTILL_SIR_TEMP", "0.7"))
N_SPECS = int(os.environ.get("DISTILL_SPECS", "20000"))
BLOCK = int(os.environ.get("DISTILL_BLOCK", "2000"))
SEED = int(os.environ.get("DISTILL_SEED", "20260706"))
OUT_DIR = Path(__file__).resolve().parent


def sanitize(dt_row, s_row, th_row):
    """Bring a sampled token row to the real-data padding convention."""
    s = np.minimum(s_row, S_PAD_CLASS).astype(np.int16)
    th = np.minimum(th_row, TH_NULL_CLASS).astype(np.int16)
    dt = dt_row.astype(np.float32).copy()
    pad = s >= S_PAD_CLASS
    L = int(np.argmax(pad)) if pad.any() else len(s)
    s[L:] = S_PAD_CLASS
    th[L:] = TH_NULL_CLASS
    dt[L:] = 0.0
    th[s <= TICK_CLASS] = TH_NULL_CLASS
    return dt, s, th, L


def run_block(bi, specs, X_hum, rng):
    shard = OUT_DIR / f"distill_corpus_b{bi:02d}.npz"
    if shard.exists():
        print(f"SKIP block {bi} (shard exists)", flush=True)
        return
    t0 = time.time()

    pending = []
    for i, (sx, sy, ex, ey) in enumerate(specs):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        log_dist = math.log(dist)
        angle = math.atan2(ey - sy, ex - sx)
        log_dur = math.log(esp._duration.sample(log_dist))
        pending.append({
            "i": i, "sx": sx, "sy": sy, "angle": angle,
            "cond": [log_dist, log_dur, math.cos(angle), math.sin(angle)],
        })

    seq_len = esp._cfg["max_seq_len"]
    chunk_size = max(esp._EVAL_BATCH // K, 1)
    cand_dt = []    # float16 token rows, one per surviving candidate
    cand_s = []
    cand_th = []
    cand_feat = []  # 18 detector features of the DECODED candidate
    cand_owner = []  # local spec index
    cand_cond = {}

    for c0 in range(0, len(pending), chunk_size):
        chunk = pending[c0:c0 + chunk_size]
        cond = torch.tensor([it["cond"] for it in chunk],
                            dtype=torch.float32, device=esp._DEVICE)
        cond = cond.repeat_interleave(K, dim=0)
        B = cond.shape[0]
        pos = torch.searchsorted(esp._FB_SORTED_LD, cond[:, 0].contiguous())
        jit = torch.randint(-esp._FEAT_WIN, esp._FEAT_WIN + 1, (B,),
                            device=esp._DEVICE)
        pos = (pos + jit).clamp(0, len(esp._FB_ORDER) - 1)
        feat = esp._FEAT_BANK[esp._FB_ORDER[pos]] + esp._FEAT_BW * torch.randn(
            B, esp._FEAT_BANK.shape[1], device=esp._DEVICE)
        with torch.no_grad():
            dt_z, s_tok, th_tok = esp._model.sample(
                cond, seq_len, n_steps=esp._N_STEPS, temperature=esp._TEMP,
                th_temperature=esp._TH_TEMP, order=esp._ORDER,
                choice_temp=esp._CHOICE_TEMP, feat=feat,
            )
        dt_np = dt_z.float().cpu().numpy()
        s_np = s_tok.cpu().numpy()
        th_np = th_tok.cpu().numpy()
        for k, it in enumerate(chunk):
            cand_cond[it["i"]] = it["cond"]
            for j in range(K):
                row = k * K + j
                traj = esp._decode(dt_np[row], s_np[row], th_np[row],
                                   it["sx"], it["sy"], it["angle"])
                if traj is None or len(traj) < 3:
                    continue
                f = extract_features(resample_trajectory(traj))
                if f is None or not np.all(np.isfinite(f)):
                    continue
                cand_dt.append(dt_np[row].astype(np.float16))
                cand_s.append(s_np[row].astype(np.int16))
                cand_th.append(th_np[row].astype(np.int16))
                cand_feat.append(f)
                cand_owner.append(it["i"])

    if not cand_feat:
        print(f"block {bi}: no valid candidates, skipping", flush=True)
        return

    # judge: fresh GBM, disjoint human reference vs this block's candidates
    from sklearn.ensemble import GradientBoostingClassifier
    X_syn = np.asarray(cand_feat)
    X = np.concatenate([X_hum, X_syn])
    y = np.concatenate([np.ones(len(X_hum)), np.zeros(len(X_syn))])
    clf = GradientBoostingClassifier(n_estimators=esp._SIR_TREES, max_depth=3,
                                     subsample=0.8, random_state=0)
    clf.fit(X, y)
    p = np.clip(clf.predict_proba(X_syn)[:, 1], 1e-4, 1 - 1e-4)
    logw = np.log(p) - np.log(1.0 - p)

    per_spec: dict = {}
    for ci, owner in enumerate(cand_owner):
        per_spec.setdefault(owner, []).append(ci)

    sel_dt, sel_s, sel_th, sel_cond, sel_len = [], [], [], [], []
    ess_all = []
    for owner, cis in per_spec.items():
        lw = logw[cis] / SIR_TEMP
        p_ = np.exp(lw - lw.max())
        p_ /= p_.sum()
        ess_all.append(1.0 / np.sum(p_ ** 2))
        g = rng.gumbel(size=len(cis))
        ci = cis[int(np.argmax(lw + g))]
        dt, s, th, L = sanitize(np.asarray(cand_dt[ci], dtype=np.float32),
                                np.asarray(cand_s[ci]), np.asarray(cand_th[ci]))
        if L < 2:
            continue
        sel_dt.append(dt)
        sel_s.append(s)
        sel_th.append(th)
        sel_cond.append(cand_cond[owner])
        sel_len.append(L)

    np.savez(shard,
             dt_z=np.stack(sel_dt),
             s_cls=np.stack(sel_s).astype(np.int16),
             th_cls=np.stack(sel_th).astype(np.int16),
             cond=np.asarray(sel_cond, dtype=np.float32),
             length=np.asarray(sel_len, dtype=np.int32))
    ess = np.asarray(ess_all)
    print(f"block {bi}: {len(sel_len)}/{len(specs)} selected | "
          f"logw mean={logw.mean():+.2f} std={logw.std():.2f} | "
          f"ESS median={np.median(ess):.2f} p10={np.percentile(ess, 10):.2f} "
          f"(max {K}) | {time.time() - t0:.0f}s", flush=True)


def main():
    print(f"[distill corpus] K={K} sir_temp={SIR_TEMP} specs={N_SPECS} "
          f"seed={SEED} ckpt={os.environ['EVENT_CKPT']}", flush=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    dists = np.load("data/human_distances.npy")
    X_hum = np.load(esp._SIR_REF)
    print(f"  judge reference: {esp._SIR_REF} ({len(X_hum)} rows)", flush=True)

    specs = []
    for d in rng.choice(dists, size=N_SPECS):
        ang = rng.uniform(-np.pi, np.pi)
        sx, sy = rng.uniform(200, 800), rng.uniform(200, 800)
        specs.append((sx, sy, sx + d * np.cos(ang), sy + d * np.sin(ang)))

    for bi in range(0, N_SPECS // BLOCK):
        run_block(bi, specs[bi * BLOCK:(bi + 1) * BLOCK], X_hum, rng)
    print("CORPUS DONE", flush=True)


if __name__ == "__main__":
    main()
