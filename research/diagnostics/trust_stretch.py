"""Stretch attempt at the last residual: MLP/LogReg near 0.54.

The confirmed recipe (33-dim RF judge trust loop, 0.504 honest RF OOB) leaves
one detector family above chance: smooth-boundary models (MLP 0.550, LogReg
0.542 on the same summary features). The ensemble judge failed because
averaging log-odds dilutes the forest. This script tries the other move the
journal called for: a stronger SINGLE smooth judge, applied as a short polish
on top of the already-converged RF picks, small decaying steps so the tree
win is not traded away.

Every round reports a proxy suite against reference half B (never fit on):
RF OOB (the guard, mirrors the honest primary), MLP 5-fold, LogReg 5-fold.
Best round = smallest worst-case deviation from 0.5 across the three. Final
claims still come only from evaluate.py replay plus external_detectors.py.

Run:
    .venv/Scripts/python.exe trust_stretch.py --pool pool_s42_k16.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from selection_lab import rf_proxy_auc
from trust33 import Pool33, build_ref33

# (name, kwargs for polish loop); judge "alt" alternates rf and mlp rounds.
CONFIGS = [
    ("mlp_f05d85_r12", dict(rounds=12, frac=0.05, judge="mlp", frac_decay=0.85)),
    ("mlp_f10d85_r12", dict(rounds=12, frac=0.10, judge="mlp", frac_decay=0.85)),
    ("lr_f05d85_r12", dict(rounds=12, frac=0.05, judge="logreg", frac_decay=0.85)),
    ("alt_f10d85_r16", dict(rounds=16, frac=0.10, judge="alt", frac_decay=0.85)),
]


def _judge_clf(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                      random_state=seed)
    if kind == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32),
                                           max_iter=400, random_state=seed))
    if kind == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000))
    raise ValueError(kind)


def fit_logodds_x(X_pos, X_neg, X_score, judge: str, seed: int) -> np.ndarray:
    clf = _judge_clf(judge, seed)
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    clf.fit(X, y)
    p = np.clip(clf.predict_proba(X_score)[:, 1], 1e-4, 1 - 1e-4)
    return np.log(p) - np.log(1.0 - p)


def _cv_auc(clf, X_sel, X_ref, seed=42) -> float:
    n = min(len(X_sel), len(X_ref))
    X = np.vstack([X_ref[:n], X_sel[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def proxy_suite(X_sel, ref_b) -> dict[str, float]:
    return {
        "rf": rf_proxy_auc(X_sel, ref_b),
        "mlp": _cv_auc(_judge_clf("mlp", 42), X_sel, ref_b),
        "lr": _cv_auc(_judge_clf("logreg", 42), X_sel, ref_b),
    }


def worst_dev(s: dict[str, float]) -> float:
    return max(abs(v - 0.5) for v in s.values())


def fmt(s: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in s.items())


def polish(pool, ref_a, ref_b, init_picks, rounds=12, frac=0.05,
           judge="mlp", frac_decay=0.85, label="polish"):
    picks = dict(init_picks)
    suite = proxy_suite(pool.selected(picks), ref_b)
    best = (worst_dev(suite), dict(picks), suite)
    print(f"  {label} r00 {fmt(suite)} worst={worst_dev(suite):.4f}")
    f = frac
    for r in range(rounds):
        j = ("rf" if r % 2 else "mlp") if judge == "alt" else judge
        logw = fit_logodds_x(ref_a, pool.selected(picks), pool.X,
                             judge=j, seed=r)
        gains = []
        for idx, rows in pool.spec_rows.items():
            k = int(np.argmax(logw[rows]))
            gains.append((logw[rows[k]] - logw[picks[idx]], idx, int(rows[k])))
        gains.sort(reverse=True)
        moved = 0
        for g, idx, ci in gains[:max(1, int(f * len(gains)))]:
            if g <= 0:
                break
            picks[idx] = ci
            moved += 1
        suite = proxy_suite(pool.selected(picks), ref_b)
        print(f"  {label} r{r + 1:02d} [{j}] moved={moved} {fmt(suite)} "
              f"worst={worst_dev(suite):.4f}", flush=True)
        if worst_dev(suite) < best[0]:
            best = (worst_dev(suite), dict(picks), suite)
        f *= frac_decay
        if moved == 0:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool_s42_k16.npz")
    ap.add_argument("--init-cfg", default="f20d85_r30_rf")
    args = ap.parse_args()

    ref = build_ref33()
    perm = np.random.default_rng(0).permutation(len(ref))
    half = len(ref) // 2
    ref_a, ref_b = ref[perm[:half]], ref[perm[half:]]
    print(f"reference: {len(ref_a)} fit rows (A), {len(ref_b)} proxy rows (B)")

    pool = Pool33(args.pool)
    prefix = args.pool.replace(".npz", "")
    init_path = f"{prefix}_picks_trust33_{args.init_cfg}.npy"
    full = np.load(init_path).astype(int)
    init = {int(i): int(ci) for i, ci in enumerate(full) if ci >= 0}
    print(f"init {init_path}")

    for name, kw in CONFIGS:
        out = f"{prefix}_picks_stretch_{name}.npy"
        if Path(out).exists():
            print(f"SKIP {out}")
            continue
        t0 = time.time()
        dev, picks, suite = polish(pool, ref_a, ref_b, init,
                                   label=name, **kw)
        np.save(out, pool.picks_to_full(picks))
        print(f"stretch_{name}: best {fmt(suite)} worst={dev:.4f} "
              f"({time.time() - t0:.0f}s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
