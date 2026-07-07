"""Tune the trust-region selection loop on a cached candidate pool.

The first trust run (frac 0.15, 10 rounds, GBM judge) beat the per-item SIR
baseline by 0.026 proxy / 0.0145 honest, with an oscillating trace that says
the step size overshoots. This sweeps step size, round count, judge family,
and step decay, all offline against reference half A with proxy AUC reported
on the untouched half B.

Run:
    .venv/Scripts/python.exe tune_trust.py --pool pool_s42_k16.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from selection_lab import Pool, pick_sir, pick_trust, rf_proxy_auc

CONFIGS = [
    ("f15_r10_gbm", dict(rounds=10, frac=0.15, judge="gbm")),
    ("f05_r30_gbm", dict(rounds=30, frac=0.05, judge="gbm")),
    ("f10_r20_rf", dict(rounds=20, frac=0.10, judge="rf")),
    ("f05_r30_rf", dict(rounds=30, frac=0.05, judge="rf")),
    ("f20d85_r30_gbm", dict(rounds=30, frac=0.20, judge="gbm",
                            frac_decay=0.85)),
    ("f20d85_r30_rf", dict(rounds=30, frac=0.20, judge="rf",
                           frac_decay=0.85)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--ref", default="data/human_ref_features_sir.npy")
    args = ap.parse_args()

    pool = Pool(args.pool)
    ref = np.load(args.ref)
    perm = np.random.default_rng(0).permutation(len(ref))
    half = len(ref) // 2
    ref_a, ref_b = ref[perm[:half]], ref[perm[half:]]
    prefix = args.pool.replace(".npz", "")

    sir = pick_sir(pool, ref_a)
    print(f"sir init proxy AUC vs B: "
          f"{rf_proxy_auc(pool.selected(sir), ref_b):.4f}")

    results = []
    for name, kw in CONFIGS:
        t0 = time.time()
        picks, auc = pick_trust(pool, ref_a, ref_b, sir, label=name, **kw)
        out = f"{prefix}_picks_trust_{name}.npy"
        np.save(out, pool.picks_to_full(picks))
        results.append((auc, name, out))
        print(f"{name}: proxy {auc:.4f} ({time.time() - t0:.0f}s) -> {out}",
              flush=True)

    print("\n=== trust tuning summary (proxy AUC vs B) ===")
    for auc, name, out in sorted(results):
        print(f"  {name:16s} {auc:.4f}")


if __name__ == "__main__":
    main()
