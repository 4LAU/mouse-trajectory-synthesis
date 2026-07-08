"""Name the 0.54 linear residual left in the headline result.

The three-seed headline (RF OOB 0.504) says an 18-feature random forest
cannot tell the selected synthetic set from held-out humans. The July 7
stretch day disclosed a smaller, stubborn residual next to it: a logistic
regression on the same 18 features reads ~0.54 (external_detectors.py,
5-fold stratified CV, StandardScaler + LogisticRegression), and it does not
move with pool size the way the MLP residual does. This script reproduces
that number as a sanity gate, then reads out what the linear detector is
actually keying on: standardized coefficients per seed, feature-group
ablations, the direction and size of the shift on the top features, and a
mean-shift-vs-spread-deficit check via squared features and variance
ratios.

Data: data/human_eval_features.npy (2000x18, the honest eval sample) and
external_data/synth_features_seed{42,43,44}.npy (2000x18 each, the
headline-selected picks run through the identical 18-feature extractor).
Both are read-only inputs; nothing here touches selection or the headline
numbers.

Run:
    .venv/Scripts/python.exe experiments/residual_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 43, 44]
PUBLISHED_LOGREG_S42 = 0.5421  # external_detectors_trust33_f20d85_r30_rf_s42.log

FEATURE_NAMES = [
    "mean_velocity", "std_velocity", "max_velocity", "velocity_skewness",
    "mean_acceleration", "std_acceleration", "max_acceleration",
    "mean_jerk", "std_jerk",
    "path_efficiency", "max_deviation",
    "curvature_mean", "curvature_std",
    "num_direction_changes",
    "movement_duration", "time_to_peak_velocity",
    "angular_velocity_mean", "angular_velocity_std",
]

GROUPS = {
    "speed": ["mean_velocity", "std_velocity", "max_velocity", "velocity_skewness"],
    "accel_jerk": ["mean_acceleration", "std_acceleration", "max_acceleration",
                   "mean_jerk", "std_jerk"],
    "curvature": ["curvature_mean", "curvature_std",
                  "angular_velocity_mean", "angular_velocity_std"],
    "shape": ["path_efficiency", "max_deviation", "num_direction_changes"],
    "duration": ["movement_duration", "time_to_peak_velocity"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURE_NAMES)
NAME_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def make_xy(human: np.ndarray, synth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(human), len(synth))
    X = np.vstack([human[:n], synth[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def standardized_coefs(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Single fit on all rows, standardized, for coefficient readout
    (separate from the cross-validated AUC used to score the detector)."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    Xz = (X - mu) / sd
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xz, y)
    return clf.coef_.ravel()


def main() -> None:
    human = np.load(ROOT / "data" / "human_eval_features.npy")
    synth = {s: np.load(ROOT / "external_data" / f"synth_features_seed{s}.npy")
             for s in SEEDS}

    # --- 1. sanity gate: reproduce the journaled ~0.54 -----------------------
    X42, y42 = make_xy(human, synth[42])
    gate_auc = cv_auc(X42, y42, seed=42)
    gate_diff = abs(gate_auc - PUBLISHED_LOGREG_S42)
    print(f"sanity gate (seed 42 LogReg, 5-fold, 18 feat): {gate_auc:.4f} "
          f"vs published {PUBLISHED_LOGREG_S42:.4f} (diff {gate_diff:.4f})")
    if gate_diff > 0.01:
        print("GATE FAILED: reproduction outside 0.01 tolerance, stopping.")
        return
    print("GATE PASSED, proceeding.\n")

    results: dict = {
        "sanity_gate": {"seed": 42, "reproduced_auc": gate_auc,
                        "published_auc": PUBLISHED_LOGREG_S42,
                        "diff": gate_diff, "pass": True},
    }

    # --- 2. per-seed AUC + standardized coefficients --------------------------
    per_seed_auc, per_seed_coef = {}, {}
    for s in SEEDS:
        X, y = make_xy(human, synth[s])
        per_seed_auc[s] = cv_auc(X, y, seed=s)
        per_seed_coef[s] = standardized_coefs(X, y, s)
        print(f"seed {s}: CV AUC {per_seed_auc[s]:.4f}")

    ranks = {s: np.argsort(-np.abs(per_seed_coef[s])) for s in SEEDS}
    top5_by_seed = {s: [FEATURE_NAMES[i] for i in ranks[s][:5]] for s in SEEDS}
    stable_top5 = sorted(set.intersection(*[set(v) for v in top5_by_seed.values()]),
                        key=lambda n: -np.mean([abs(per_seed_coef[s][NAME_IDX[n]])
                                                for s in SEEDS]))
    print("\ntop 5 |standardized coef| per seed:")
    for s in SEEDS:
        line = ", ".join(f"{FEATURE_NAMES[i]}={per_seed_coef[s][i]:+.3f}"
                        for i in ranks[s][:5])
        print(f"  seed {s}: {line}")
    print(f"stable across all 3 seeds (intersection of top-5): {stable_top5}\n")

    results["per_seed_auc"] = per_seed_auc
    results["coefficients"] = {s: {FEATURE_NAMES[i]: float(per_seed_coef[s][i])
                                    for i in range(len(FEATURE_NAMES))}
                                for s in SEEDS}
    results["top5_by_seed"] = top5_by_seed
    results["stable_top5"] = stable_top5

    # --- 3. feature-group ablation ---------------------------------------------
    print("group ablation (drop group, refit on remaining 14 feat, CV AUC):")
    ablation = {}
    for gname, gfeat in GROUPS.items():
        keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in gfeat]
        aucs = {}
        for s in SEEDS:
            X, y = make_xy(human, synth[s])
            aucs[s] = cv_auc(X[:, keep], y, seed=s)
        mean_auc = float(np.mean(list(aucs.values())))
        chance_gap = abs(mean_auc - 0.5)
        ablation[gname] = {"per_seed": aucs, "mean": mean_auc,
                          "chance_gap": chance_gap, "dropped": gfeat}
        print(f"  drop {gname:10s} ({len(gfeat)} feat): "
              f"mean AUC {mean_auc:.4f}, |gap to chance| {chance_gap:.4f}")
    full_mean = float(np.mean(list(per_seed_auc.values())))
    best_group = min(ablation, key=lambda g: ablation[g]["chance_gap"])
    print(f"full 18-feat mean AUC: {full_mean:.4f}")
    print(f"smallest group whose removal is closest to chance: {best_group} "
          f"({ablation[best_group]['mean']:.4f})\n")
    results["ablation"] = ablation
    results["full_mean_auc"] = full_mean
    results["closest_to_chance_group"] = best_group

    # --- 4. direction on the stable top features --------------------------------
    print("direction, stable top features (human vs selected synthetic):")
    direction = {}
    report_feats = stable_top5 if stable_top5 else \
        [FEATURE_NAMES[i] for i in ranks[42][:5]]
    for name in report_feats:
        i = NAME_IDX[name]
        h_mean, h_std = float(human[:, i].mean()), float(human[:, i].std())
        h_med = float(np.median(human[:, i]))
        h_iqr = float(np.percentile(human[:, i], 75) - np.percentile(human[:, i], 25))
        h_p99 = float(np.percentile(np.abs(human[:, i]), 99))
        per_seed_dir = {}
        for s in SEEDS:
            col = synth[s][:, i]
            s_mean, s_std = float(col.mean()), float(col.std())
            s_med = float(np.median(col))
            s_iqr = float(np.percentile(col, 75) - np.percentile(col, 25))
            s_p99 = float(np.percentile(np.abs(col), 99))
            z = (s_mean - h_mean) / (h_std + 1e-12)
            med_shift = (s_med - h_med) / (h_iqr + 1e-12)
            tail_ratio = s_p99 / (h_p99 + 1e-12)
            per_seed_dir[s] = {"synth_mean": s_mean, "synth_std": s_std,
                              "z_shift": z, "median_shift_in_iqr": med_shift,
                              "p99_tail_ratio": tail_ratio}
        direction[name] = {"human_mean": h_mean, "human_std": h_std,
                          "human_median": h_med, "human_iqr": h_iqr,
                          "per_seed": per_seed_dir}
        z_avg = np.mean([per_seed_dir[s]["z_shift"] for s in SEEDS])
        med_avg = np.mean([per_seed_dir[s]["median_shift_in_iqr"] for s in SEEDS])
        tail_avg = np.mean([per_seed_dir[s]["p99_tail_ratio"] for s in SEEDS])
        print(f"  {name}: mean-based shift {z_avg:+.2f} human-std units, "
              f"median shift {med_avg:+.2f} human-IQR units (bulk match), "
              f"p99-tail ratio synth/human {tail_avg:.2f} "
              f"(seeds z={[round(per_seed_dir[s]['z_shift'], 2) for s in SEEDS]})")
    results["direction"] = direction

    # --- 5. mean shift vs spread deficit: squared features + variance ratio ----
    print("\nsecond-moment check: features + squared standardized features:")
    sq_aucs = {}
    for s in SEEDS:
        X, y = make_xy(human, synth[s])
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
        Xz = (X - mu) / sd
        Xsq = np.hstack([Xz, Xz ** 2])
        sq_aucs[s] = cv_auc(Xsq, y, seed=s)
    sq_mean = float(np.mean(list(sq_aucs.values())))
    print(f"  linear-only mean AUC: {full_mean:.4f}")
    print(f"  linear+squared mean AUC: {sq_mean:.4f} "
          f"(delta {sq_mean - full_mean:+.4f})")

    print("\n  per-feature variance ratio (synth var / human var), all seeds:")
    var_ratio = {}
    for name in FEATURE_NAMES:
        i = NAME_IDX[name]
        h_var = float(human[:, i].var()) + 1e-12
        ratios = {s: float(synth[s][:, i].var()) / h_var for s in SEEDS}
        var_ratio[name] = ratios
    # flag the features with the most extreme mean variance ratio
    mean_ratio = {n: np.mean(list(var_ratio[n].values())) for n in FEATURE_NAMES}
    extreme = sorted(FEATURE_NAMES, key=lambda n: abs(np.log(mean_ratio[n])))[-5:]
    for name in reversed(extreme):
        r = mean_ratio[name]
        print(f"    {name}: mean var ratio {r:.3f} "
              f"({'narrower' if r < 1 else 'wider'} than human)")
    results["second_moment"] = {
        "linear_only_mean_auc": full_mean,
        "linear_plus_squared_mean_auc": sq_mean,
        "delta": sq_mean - full_mean,
        "variance_ratio_by_feature": var_ratio,
        "most_extreme_variance_ratio_features": extreme,
    }

    out_path = ROOT / "experiments" / "residual_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
