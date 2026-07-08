"""Can the headline's RF judge tell M4D humans from the selected synthetics?

Standalone script under external_validation/ - does not modify or reimplement
any existing repo code. Mirrors validate_adserp.py's structure and detector code
exactly (RF-OOB / RF-5fold-CV / GBM-5fold-CV, copied verbatim from
evaluate.py's main(), same hyperparameters and random_state=seed convention).

Pipeline correspondence:
  - External side: external_data/m4d_features_2000.npy, built by
    m4d_features.py using features.extract_feature_matrix on canonically
    segmented M4D human-session movements (phase1 + phase2, bots excluded).
  - Synthetic side: external_data/synth_features_seed{42,43,44}.npy, the
    SAME cached feature matrices validate_adserp.py already produced (each
    file is the headline's 2000 selected-synthetic trajectories for that
    seed, run through features.extract_feature_matrix) - reused as-is, no
    need to regenerate since seed/pool/picks are unchanged.
  - Other human sides: data/human_eval_features.npy (headline internal human
    class) and external_data/adserp_features_2000.npy (the other external
    dataset already validated).

Blocks:
  (a) MAIN per seed (42, 43, 44): M4D-2000 (class 0) vs that seed's cached
      synth features (class 1).
  (b) CONTROL 1: M4D-2000 vs human_eval_features.npy (humans vs humans,
      seed 42 config, run once).
  (c) CONTROL 2: M4D-2000 vs adserp_features_2000.npy (external vs external
      humans, seed 42 config, run once) - tests whether the two external
      datasets resemble each other more than either resembles our internal
      human data.

Also computes per-feature Wasserstein distances (features.py's
normalized_wasserstein_by_feature, the same normalized-by-std diagnostic used
throughout the repo, e.g. evaluate.py / diag_features.py / proxy_metric_
validation.py) for MAIN seed 42 and CONTROL 1, reporting the top 5 features
each.

Run:
    .venv/Scripts/python.exe external_validation/validate_m4d.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parent.parent
EXT_DIR = Path(__file__).resolve().parent  # external_validation/ - this script's own dir, for results.json
EXT_DATA_DIR = REPO_ROOT / "external_data"  # raw datasets + cached .npy feature files
sys.path.insert(0, str(REPO_ROOT))

from features import FEATURE_NAMES, normalized_wasserstein_by_feature  # noqa: E402

SEEDS = [42, 43, 44]


def load_synth(seed: int) -> np.ndarray:
    path = EXT_DATA_DIR / f"synth_features_seed{seed}.npy"
    if not path.exists():
        raise RuntimeError(
            f"Missing {path} - expected validate_adserp.py's cached synth "
            f"features to already exist for seed {seed}.")
    return np.load(path)


def rf_oob_auc(X, y, seed: int) -> float:
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                  random_state=seed)
    clf.fit(X, y)
    oob_proba = clf.oob_decision_function_[:, 1]
    return float(roc_auc_score(y, oob_proba))


def rf_cv_auc(X, y, seed: int) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rf_cv = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                    random_state=seed)
    proba = cross_val_predict(rf_cv, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def gbm_cv_auc(X, y, seed: int) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                      random_state=seed)
    proba = cross_val_predict(gbm, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def run_suite(class0: np.ndarray, class1: np.ndarray, seed: int) -> dict:
    """class0 = label 0 (first arg), class1 = label 1 - matches evaluate.py's
    human=0, synthetic=1 convention wherever the synthetic/generated set is
    class1."""
    for name, arr in [("class0", class0), ("class1", class1)]:
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} has non-finite values, aborting")
    n_use = min(len(class0), len(class1))
    c0 = class0[:n_use]
    c1 = class1[:n_use]
    X = np.vstack([c0, c1])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    return {
        "n_class0_total": int(len(class0)),
        "n_class1_total": int(len(class1)),
        "n_use_per_class": int(n_use),
        "rf_oob_auc": rf_oob_auc(X, y, seed),
        "rf_5fold_cv_auc": rf_cv_auc(X, y, seed),
        "gbm_5fold_cv_auc": gbm_cv_auc(X, y, seed),
    }


def top5_wasserstein(class0: np.ndarray, class1: np.ndarray) -> list:
    n_use = min(len(class0), len(class1))
    wd = normalized_wasserstein_by_feature(class0[:n_use], class1[:n_use])
    order = np.argsort(wd)[::-1]
    return [
        {"feature": FEATURE_NAMES[i], "wasserstein": float(wd[i])}
        for i in order[:5]
    ]


def main() -> None:
    m4d_2000 = np.load(EXT_DATA_DIR / "m4d_features_2000.npy")
    human_eval = np.load(REPO_ROOT / "data" / "human_eval_features.npy")
    adserp_2000 = np.load(EXT_DATA_DIR / "adserp_features_2000.npy")
    print(f"Loaded m4d_features_2000.npy: {m4d_2000.shape}")
    print(f"Loaded human_eval_features.npy: {human_eval.shape}")
    print(f"Loaded adserp_features_2000.npy: {adserp_2000.shape}")

    results: dict = {
        "main_per_seed": {},
        "control_1_vs_human_eval": None,
        "control_2_vs_adserp": None,
        "wasserstein_main_seed42": None,
        "wasserstein_control_1": None,
    }

    # --- Main: M4D-2000 vs each seed's selected synthetics
    for seed in SEEDS:
        print(f"\n=== MAIN seed {seed}: M4D-2000 (class0) vs selected "
              f"synthetics (class1) ===")
        synth = load_synth(seed)
        main_result = run_suite(m4d_2000, synth, seed=seed)
        results["main_per_seed"][seed] = main_result
        print(json.dumps(main_result, indent=2))

    # --- Control 1: M4D-2000 vs human_eval (humans vs humans)
    print("\n=== CONTROL 1: M4D-2000 (class0) vs human_eval_features.npy "
          "(class1) - humans vs humans, seed 42, run once ===")
    control1 = run_suite(m4d_2000, human_eval, seed=42)
    results["control_1_vs_human_eval"] = control1
    print(json.dumps(control1, indent=2))

    # --- Control 2: M4D-2000 vs AdSERP-2000 (external vs external humans)
    print("\n=== CONTROL 2: M4D-2000 (class0) vs adserp_features_2000.npy "
          "(class1) - external vs external humans, seed 42, run once ===")
    control2 = run_suite(m4d_2000, adserp_2000, seed=42)
    results["control_2_vs_adserp"] = control2
    print(json.dumps(control2, indent=2))

    # --- Per-feature Wasserstein diagnostics
    print("\n=== Wasserstein: MAIN seed 42 (M4D vs synth42) top 5 ===")
    synth42 = load_synth(42)
    w_main = top5_wasserstein(m4d_2000, synth42)
    results["wasserstein_main_seed42"] = w_main
    print(json.dumps(w_main, indent=2))

    print("\n=== Wasserstein: CONTROL 1 (M4D vs human_eval) top 5 ===")
    w_control1 = top5_wasserstein(m4d_2000, human_eval)
    results["wasserstein_control_1"] = w_control1
    print(json.dumps(w_control1, indent=2))

    out_path = EXT_DIR / "validate_m4d_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n=== SUMMARY ===")
    for seed in SEEDS:
        r = results["main_per_seed"][seed]
        print(f"Main seed {seed} (M4D vs selected synth): "
              f"RF-OOB {r['rf_oob_auc']:.4f}  RF-CV {r['rf_5fold_cv_auc']:.4f}  "
              f"GBM-CV {r['gbm_5fold_cv_auc']:.4f}  (n={r['n_use_per_class']}/class)")
    print(f"Control 1 (M4D vs human_eval, dataset shift): "
          f"RF-OOB {control1['rf_oob_auc']:.4f}  RF-CV {control1['rf_5fold_cv_auc']:.4f}  "
          f"GBM-CV {control1['gbm_5fold_cv_auc']:.4f}")
    print(f"Control 2 (M4D vs AdSERP, external vs external): "
          f"RF-OOB {control2['rf_oob_auc']:.4f}  RF-CV {control2['rf_5fold_cv_auc']:.4f}  "
          f"GBM-CV {control2['gbm_5fold_cv_auc']:.4f}")


if __name__ == "__main__":
    main()
