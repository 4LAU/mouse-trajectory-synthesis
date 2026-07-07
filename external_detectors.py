"""Stress-test the locked synthetic pool against detector families OUTSIDE
evaluate.py's built-in suite (RF OOB, RF 5-fold, GBM 5-fold on 18 features).

Selection was optimized against an RF judge, so a clean win there could just
mean we gamed one detector's inductive bias. This script throws independent
families at the same selected population: gradient boosting via xgboost,
extra-trees / hist-GBM / MLP / logistic regression on the same 18 hand-crafted
features (sklearn, but different bias than RF), and a catch22-style canonical
time-series feature set computed directly off the raw (dx, dy) signal instead
of the hand-crafted 18.

Synthetic class: one candidate per spec, selected from a saved SIR pool
(pool_s42_k16.npz) by a picks array (selection_lab.py / _pool_replay in
experiments/event_stream_polar.py). Human class: data/human_eval_features.npy
for the 18-feature detectors (the honest eval sample, loaded ONLY here) and
the same raw held-out split detector_raw.py's Raw-NN detector uses for the
raw-signal detector.

Usage:
    .venv/Scripts/python.exe external_detectors.py \
        --pool pool_s42_k16.npz \
        --picks pool_s42_k16_picks_trust_f20d85_r30_rf.npy \
        --n 2000
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from detector_raw import load_human_raw
from features import extract_features, resample_trajectory

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

try:
    import pycatch22
    HAVE_CATCH22 = True
except ImportError:
    HAVE_CATCH22 = False


SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pool_selection(pool_path: str, picks_path: str) -> tuple[np.ndarray, list]:
    """Select one candidate row per spec via a picks array (-1 = none).

    Mirrors _pool_replay in experiments/event_stream_polar.py: picks[idx] is
    a global candidate row into X/trajs, or -1 if the spec has no pick.
    """
    d = np.load(pool_path, allow_pickle=True)
    X, owner_idx, trajs = d["X"], d["owner_idx"].astype(int), d["trajs"]
    picks = np.load(picks_path).astype(int)

    feat_rows, raw_trajs = [], []
    for idx, ci in enumerate(picks):
        if ci < 0:
            continue
        assert owner_idx[ci] == idx, (
            f"pick {ci} belongs to spec {owner_idx[ci]}, not {idx}")
        feat_rows.append(X[ci])
        raw_trajs.append([tuple(p) for p in trajs[ci]])
    print(f"  pool selection: {len(feat_rows)}/{len(picks)} specs have a pick")
    return np.asarray(feat_rows, dtype=np.float64), raw_trajs


def raw_traj_to_channels(traj) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """125Hz resample then derive (speed, vx, vy), identical to features.py's
    extract_features preprocessing, so raw-signal detectors see the same
    signal the 18 hand-crafted features are built from."""
    pts = np.asarray(resample_trajectory(traj), dtype=np.float64)
    if len(pts) < 5:
        return None
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    dx, dy = np.diff(x), np.diff(y)
    dt = np.maximum(np.diff(t), 1e-6)
    vx, vy = dx / dt, dy / dt
    speed = np.hypot(dx, dy) / dt
    return speed, vx, vy


# ---------------------------------------------------------------------------
# catch22 (or fallback mini set) on raw channels
# ---------------------------------------------------------------------------

def _manual_mini_features(x: np.ndarray) -> list[float]:
    """Fallback per-channel feature set when pycatch22 is unavailable:
    autocorrelation at lags 1, 2, 5; spectral centroid; zero-crossing rate
    of the channel's acceleration (first difference)."""
    n = len(x)
    xm = x - x.mean()
    denom = float(np.sum(xm ** 2)) + 1e-12

    def acf(lag: int) -> float:
        if n <= lag:
            return 0.0
        return float(np.sum(xm[:-lag] * xm[lag:]) / denom)

    spec = np.abs(np.fft.rfft(xm))
    freqs = np.fft.rfftfreq(n)
    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))

    accel = np.diff(x)
    zcr = float(np.mean(np.diff(np.sign(accel)) != 0)) if len(accel) > 1 else 0.0

    return [acf(1), acf(2), acf(5), centroid, zcr]


def channel_features(x: np.ndarray) -> list[float]:
    if HAVE_CATCH22:
        return list(pycatch22.catch22_all(x.tolist())["values"])
    return _manual_mini_features(x)


def build_catch22_matrix(trajs: list) -> np.ndarray:
    rows = []
    for traj in trajs:
        ch = raw_traj_to_channels(traj)
        if ch is None:
            continue
        speed, vx, vy = ch
        row = channel_features(speed) + channel_features(vx) + channel_features(vy)
        if np.all(np.isfinite(row)):
            rows.append(row)
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# CV harness
# ---------------------------------------------------------------------------

def cv_auc(clf, X: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def make_xy(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """a = human (label 0), b = synthetic (label 1), balanced to min length."""
    n = min(len(a), len(b))
    X = np.vstack([a[:n], b[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stress-test synthetic pool selection against detector "
                    "families outside evaluate.py's built-in suite.",
    )
    p.add_argument("--pool", default="pool_s42_k16.npz")
    p.add_argument("--picks", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--skip", default="",
                   help="Comma-separated detector keys to skip: "
                        "xgb,extratrees,histgbm,mlp,logreg,catch22")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--train-dir", default="./training")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    rng = np.random.default_rng(args.seed)

    print(f"pycatch22 available: {HAVE_CATCH22}")
    if not HAVE_CATCH22:
        print("  pycatch22 has no Windows wheel and failed to build from "
              "source (needs MSVC Build Tools). Falling back to a manual "
              "5-feature mini set per channel (acf@1,2,5, spectral centroid, "
              "accel zero-crossing rate) -> 15 features instead of the "
              "canonical 66. Results below are NOT the full catch22 suite.")
    print(f"xgboost available: {HAVE_XGB}")

    # --- synthetic class: selected candidates from the pool -----------------
    print(f"\nLoading pool {args.pool} with picks {args.picks} ...")
    synth_feat, synth_trajs = load_pool_selection(args.pool, args.picks)
    if args.n < len(synth_feat):
        sel = rng.choice(len(synth_feat), size=args.n, replace=False)
        synth_feat = synth_feat[sel]
        synth_trajs = [synth_trajs[i] for i in sel]
    print(f"  synthetic: {len(synth_feat)} feature rows, "
          f"{len(synth_trajs)} raw trajectories")

    # --- human class: honest eval sample + raw held-out split ---------------
    human_feat = np.load(f"{args.data_dir}/human_eval_features.npy")
    print(f"  human 18-feature eval sample: {len(human_feat)} rows "
          f"(data/human_eval_features.npy, loaded only for this eval)")
    if args.n < len(human_feat):
        sel = rng.choice(len(human_feat), size=args.n, replace=False)
        human_feat = human_feat[sel]

    human_trajs = load_human_raw(len(synth_trajs), train_dir=args.train_dir,
                                 seed=args.seed)
    print(f"  human raw trajectories (Raw-NN split): {len(human_trajs)}")

    X18, y18 = make_xy(human_feat, synth_feat)
    print(f"\n18-feature detectors: n={len(y18) // 2} per class")

    results: list[tuple[str, float, int]] = []

    if "xgb" not in skip and HAVE_XGB:
        t0 = time.time()
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                learning_rate=0.05, eval_metric="logloss",
                                random_state=args.seed, n_jobs=-1)
        auc = cv_auc(clf, X18, y18, args.seed)
        results.append(("xgboost (18 feat)", auc, X18.shape[1]))
        print(f"  xgboost           {auc:.4f}  ({time.time() - t0:.1f}s)")
    elif "xgb" not in skip:
        print("  xgboost           SKIPPED (not installed)")

    if "extratrees" not in skip:
        clf = ExtraTreesClassifier(n_estimators=300, n_jobs=-1,
                                   random_state=args.seed)
        auc = cv_auc(clf, X18, y18, args.seed)
        results.append(("ExtraTrees (18 feat)", auc, X18.shape[1]))
        print(f"  ExtraTrees        {auc:.4f}")

    if "histgbm" not in skip:
        clf = HistGradientBoostingClassifier(random_state=args.seed)
        auc = cv_auc(clf, X18, y18, args.seed)
        results.append(("HistGBM (18 feat)", auc, X18.shape[1]))
        print(f"  HistGBM           {auc:.4f}")

    if "mlp" not in skip:
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                          random_state=args.seed),
        )
        auc = cv_auc(clf, X18, y18, args.seed)
        results.append(("MLP (18 feat, standardized)", auc, X18.shape[1]))
        print(f"  MLP               {auc:.4f}")

    if "logreg" not in skip:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=args.seed),
        )
        auc = cv_auc(clf, X18, y18, args.seed)
        results.append(("LogReg (18 feat, standardized)", auc, X18.shape[1]))
        print(f"  LogReg            {auc:.4f}")

    if "catch22" not in skip:
        print("\ncatch22-style raw-signal detector (speed, vx, vy channels):")
        t0 = time.time()
        human_c22 = build_catch22_matrix(human_trajs)
        synth_c22 = build_catch22_matrix(synth_trajs)
        print(f"  valid: human {len(human_c22)}/{len(human_trajs)}, "
              f"synth {len(synth_c22)}/{len(synth_trajs)} "
              f"({time.time() - t0:.1f}s to extract)")
        Xc, yc = make_xy(human_c22, synth_c22)
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                     random_state=args.seed)
        auc = cv_auc(clf, Xc, yc, args.seed)
        tag = "catch22" if HAVE_CATCH22 else "catch22-fallback"
        results.append((f"RF on {tag} (3ch x "
                        f"{Xc.shape[1] // 3} feat = {Xc.shape[1]})",
                        auc, Xc.shape[1]))
        print(f"  RF on {tag}  {auc:.4f}  n={len(yc) // 2} per class")

    print("\n=== external detector AUC table (0.5 = indistinguishable) ===")
    print(f"{'detector':40s} {'AUC':>8s} {'dims':>6s}")
    for name, auc, dims in results:
        print(f"{name:40s} {auc:8.4f} {dims:6d}")


if __name__ == "__main__":
    main()
