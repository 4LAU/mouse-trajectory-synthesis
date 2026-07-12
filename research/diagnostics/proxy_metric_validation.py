"""Validate proxy metrics for predicting N=2000 RF AUC.

Strategy: generate 2000 trajectories once, then subsample at various N
to measure which metric is most stable and best predicts N=2000 AUC.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, extract_feature_matrix, normalized_wasserstein_by_feature

RESULTS_FILE = Path("proxy_validation_results.json")

N_GENERATE = 2000
SUBSAMPLE_NS = [50, 100, 200, 500, 1000, 2000]
REPEATS_PER_N = {50: 20, 100: 15, 200: 10, 500: 5, 1000: 3, 2000: 1}
SEED = 42


def compute_metrics(human_feat: np.ndarray, synth_feat: np.ndarray, seed: int) -> dict:
    n = min(len(human_feat), len(synth_feat))
    hf = human_feat[:n]
    sf = synth_feat[:n]

    # 1. Per-feature Wasserstein distances (normalized by human std)
    w_dists = normalized_wasserstein_by_feature(hf, sf)
    w_sum = float(np.sum(w_dists))
    w_top5 = float(np.sum(sorted(w_dists, reverse=True)[:5]))
    w_mean = float(np.mean(w_dists))

    # 2. Correlation matrix distance
    h_corr = np.corrcoef(hf.T)
    s_corr = np.corrcoef(sf.T)
    corr_frob = float(np.linalg.norm(h_corr - s_corr, 'fro'))
    corr_max = float(np.max(np.abs(h_corr - s_corr)))

    # Top-10 correlation gaps (sum)
    n_feat = len(FEATURE_NAMES)
    gaps = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            gaps.append(abs(h_corr[i, j] - s_corr[i, j]))
    gaps.sort(reverse=True)
    corr_top10 = float(sum(gaps[:10]))

    # 3. RF OOB AUC
    X = np.vstack([hf, sf])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=seed)
    rf.fit(X, y)
    oob_proba = rf.oob_decision_function_[:, 1]
    rf_auc = float(roc_auc_score(y, oob_proba))

    # 4. Logistic regression AUC (leave-one-out-ish via OOB analog)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    lr_proba = cross_val_predict(lr, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    lr_auc = float(roc_auc_score(y, lr_proba))

    return {
        "n": n,
        "w_sum": w_sum,
        "w_top5": w_top5,
        "w_mean": w_mean,
        "corr_frob": corr_frob,
        "corr_max": corr_max,
        "corr_top10": corr_top10,
        "rf_auc": rf_auc,
        "lr_auc": lr_auc,
        "combo_w_corr": w_sum + corr_frob,
    }


def main():
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    human_features = np.load(data_dir / "human_eval_features.npy")
    human_distances = np.load(data_dir / "human_distances.npy")
    print(f"Loaded {len(human_features)} human features, {len(human_distances)} distances")

    # Load experiment
    ckpt = os.environ.get("CANDI_CKPT", "candi_polar_flow_best.pt")
    print(f"Using checkpoint: {ckpt}")

    import importlib
    mod = importlib.import_module("experiments.candi")
    generate_path = mod.generate_path

    # Generate all trajectories at once
    rng = np.random.default_rng(SEED)
    center_x, center_y = 960.0, 540.0
    print(f"\nGenerating {N_GENERATE} trajectories...")
    t0 = time.perf_counter()
    trajectories = []
    for i in range(N_GENERATE):
        dist = float(rng.choice(human_distances))
        angle = float(rng.uniform(0, 2 * np.pi))
        end_x = center_x + dist * np.cos(angle)
        end_y = center_y + dist * np.sin(angle)
        try:
            traj = generate_path(center_x, center_y, end_x, end_y)
            if traj is not None and len(traj) >= 2:
                trajectories.append(traj)
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    print(f"Generated {len(trajectories)} in {elapsed:.0f}s")

    # Extract features
    synth_features = extract_feature_matrix(trajectories)
    print(f"Valid features: {len(synth_features)}/{len(trajectories)}")

    # Run metrics at various subsample sizes
    results = []
    rng_sub = np.random.default_rng(123)

    for n_sub in SUBSAMPLE_NS:
        repeats = REPEATS_PER_N[n_sub]
        print(f"\n--- N={n_sub} ({repeats} repeats) ---")

        for rep in range(repeats):
            if n_sub >= len(synth_features):
                sf_sub = synth_features
                hf_sub = human_features[:len(synth_features)]
            else:
                idx_s = rng_sub.choice(len(synth_features), n_sub, replace=False)
                idx_h = rng_sub.choice(len(human_features), n_sub, replace=False)
                sf_sub = synth_features[idx_s]
                hf_sub = human_features[idx_h]

            seed_rep = SEED + rep * 7
            metrics = compute_metrics(hf_sub, sf_sub, seed_rep)
            metrics["repeat"] = rep
            results.append(metrics)
            print(f"  rep {rep}: RF={metrics['rf_auc']:.4f}  LR={metrics['lr_auc']:.4f}  "
                  f"W_sum={metrics['w_sum']:.3f}  corr_frob={metrics['corr_frob']:.3f}")

    # Save results
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} results to {RESULTS_FILE}")

    # Summary: variance by N
    print(f"\n{'='*70}")
    print("STABILITY ANALYSIS: std dev across repeats for each metric at each N")
    print(f"{'='*70}")
    header = f"{'N':>6} {'RF_AUC':>10} {'LR_AUC':>10} {'W_sum':>10} {'W_top5':>10} {'corr_frob':>10} {'combo':>10}"
    print(header)
    print("-" * len(header))

    for n_sub in SUBSAMPLE_NS:
        subset = [r for r in results if r["n"] == n_sub]
        if len(subset) < 2:
            vals = subset[0] if subset else {}
            print(f"{n_sub:>6} {'n/a':>10} {'n/a':>10} "
                  f"{vals.get('w_sum', 0):>10.4f} {vals.get('w_top5', 0):>10.4f} "
                  f"{vals.get('corr_frob', 0):>10.4f} {vals.get('combo_w_corr', 0):>10.4f}")
            continue

        for metric in ["rf_auc", "lr_auc", "w_sum", "w_top5", "corr_frob", "combo_w_corr"]:
            vals = [r[metric] for r in subset]
        line = f"{n_sub:>6}"
        for metric in ["rf_auc", "lr_auc", "w_sum", "w_top5", "corr_frob", "combo_w_corr"]:
            vals = [r[metric] for r in subset]
            line += f" {np.std(vals):>10.4f}"
        print(line)

    # Also print means
    print(f"\n{'='*70}")
    print("MEAN values for each metric at each N")
    print(f"{'='*70}")
    print(header)
    print("-" * len(header))
    for n_sub in SUBSAMPLE_NS:
        subset = [r for r in results if r["n"] == n_sub]
        line = f"{n_sub:>6}"
        for metric in ["rf_auc", "lr_auc", "w_sum", "w_top5", "corr_frob", "combo_w_corr"]:
            vals = [r[metric] for r in subset]
            line += f" {np.mean(vals):>10.4f}"
        print(line)


if __name__ == "__main__":
    main()
