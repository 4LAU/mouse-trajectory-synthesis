"""Can the headline's RF judge tell AdSERP humans from the selected synthetics?

Standalone script under external_validation/ - does not modify or reimplement
any existing repo code. Pipeline correspondence:

  - Synthetic side: for each headline seed, _extract_synth_features.py runs
    IN A SUBPROCESS with EVENT_POOL_LOAD / EVENT_POOL_PICKS set exactly as
    verify_headline.py's replay() sets them, and calls evaluate.py's own
    load_experiment / generate_synthetic_trajectories functions (imported,
    not reimplemented) to reproduce the identical 2000 selected-synthetic
    trajectories, then features.extract_feature_matrix (imported) to get
    the 18-dim feature matrix.
  - Human sides: data/human_eval_features.npy (headline human class, used
    as-is) and external_data/adserp_features_2000.npy (built by
    adserp_features.py using features.extract_feature_matrix on canonically
    segmented AdSERP movements - see that script's docstring).
  - Detector: the RF-OOB / RF-5fold-CV / GBM-5fold-CV block is copied
    verbatim from evaluate.py's main(), same hyperparameters and the same
    random_state=seed convention (evaluate.py has no linear/MLP suite; these
    three are the full same-feature detector set it runs. Its fourth
    detector, the raw-trajectory CNN in detector_raw.py, operates on raw
    point sequences rather than the 18 features and is excluded from the
    published headline itself via verify_headline.py's --no-raw-nn flag, so
    it is out of scope here too).

Controls:
  (b) AdSERP-2000 vs human_eval_features.npy (dataset-shift control, humans
      vs humans, run once with a fixed seed, not per synthetic seed)
  (c) human_eval_features.npy vs synth-seed-42 (sanity: must reproduce the
      published 0.5095 headline number through this script's own loading
      code before the main comparisons are trusted)

Run:
    .venv/Scripts/python.exe external_validation/validate_adserp.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parent.parent
EXT_DIR = Path(__file__).resolve().parent  # external_validation/ - this script's own dir, for results.json
EXT_DATA_DIR = REPO_ROOT / "external_data"  # raw datasets + cached .npy feature files
SEEDS = [42, 43, 44]
EXPECTED_HEADLINE = {42: 0.5095, 43: 0.5030, 44: 0.4993}
SANITY_TOLERANCE = 0.005


def extract_synth_features(seed: int) -> np.ndarray:
    out_path = EXT_DATA_DIR / f"synth_features_seed{seed}.npy"
    if out_path.exists():
        print(f"[seed {seed}] reusing cached {out_path.name}")
        return np.load(out_path)

    env = os.environ.copy()
    env["EVENT_POOL_LOAD"] = f"pool_s{seed}_k16.npz"
    env["EVENT_POOL_PICKS"] = (
        f"pool_s{seed}_k16_picks_trust33_f20d85_r30_rf.npy")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")

    result = subprocess.run(
        [sys.executable, str(EXT_DIR / "_extract_synth_features.py"),
         "--seed", str(seed), "--out", str(out_path)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
    )
    print(f"[seed {seed}] extraction subprocess output:")
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"seed {seed}: extraction subprocess failed "
                            f"(exit {result.returncode})")
    return np.load(out_path)


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


def main() -> None:
    human_eval = np.load(REPO_ROOT / "data" / "human_eval_features.npy")
    adserp_2000 = np.load(EXT_DATA_DIR / "adserp_features_2000.npy")
    print(f"Loaded human_eval_features.npy: {human_eval.shape}")
    print(f"Loaded adserp_features_2000.npy: {adserp_2000.shape}")

    results: dict = {"sanity_control_c": None, "control_1_dataset_shift": None,
                      "main_per_seed": {}}

    # --- Control (c): sanity - reproduce the published headline for seed 42
    print("\n=== CONTROL (c): sanity check vs published headline (seed 42) ===")
    synth_42 = extract_synth_features(42)
    sanity = run_suite(human_eval, synth_42, seed=42)
    sanity["published_expected"] = EXPECTED_HEADLINE[42]
    delta = abs(sanity["rf_oob_auc"] - EXPECTED_HEADLINE[42])
    sanity["delta_from_published"] = delta
    sanity["within_tolerance"] = delta <= SANITY_TOLERANCE
    results["sanity_control_c"] = sanity
    print(json.dumps(sanity, indent=2))

    if not sanity["within_tolerance"]:
        print("\n*** STOP: sanity control (c) does NOT reproduce the "
              f"published headline. Got RF-OOB {sanity['rf_oob_auc']:.4f}, "
              f"expected {EXPECTED_HEADLINE[42]:.4f} "
              f"(tolerance {SANITY_TOLERANCE}). "
              "Halting before running the main AdSERP comparisons - "
              "loading code does not match the published pipeline. ***")
        out_path = EXT_DIR / "validate_adserp_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(1)
    print(f"Sanity OK: RF-OOB {sanity['rf_oob_auc']:.4f} vs published "
          f"{EXPECTED_HEADLINE[42]:.4f} (delta {delta:.4f} <= "
          f"{SANITY_TOLERANCE})")

    # --- Control 1: dataset shift, AdSERP vs human_eval (humans vs humans)
    print("\n=== CONTROL 1: dataset shift - AdSERP-2000 vs human_eval "
          "(humans vs humans, seed 42, run once) ===")
    control1 = run_suite(adserp_2000, human_eval, seed=42)
    results["control_1_dataset_shift"] = control1
    print(json.dumps(control1, indent=2))

    # --- Main: AdSERP-2000 vs each seed's selected synthetics
    for seed in SEEDS:
        print(f"\n=== MAIN seed {seed}: AdSERP-2000 vs selected synthetics ===")
        synth = synth_42 if seed == 42 else extract_synth_features(seed)
        main_result = run_suite(adserp_2000, synth, seed=seed)
        main_result["headline_published_for_reference"] = EXPECTED_HEADLINE[seed]
        results["main_per_seed"][seed] = main_result
        print(json.dumps(main_result, indent=2))

    out_path = EXT_DIR / "validate_adserp_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n=== SUMMARY ===")
    print(f"Sanity (human_eval vs synth42, this code): "
          f"{sanity['rf_oob_auc']:.4f} (published {EXPECTED_HEADLINE[42]:.4f})")
    print(f"Control 1 (AdSERP vs human_eval, dataset shift): "
          f"RF-OOB {control1['rf_oob_auc']:.4f}  RF-CV {control1['rf_5fold_cv_auc']:.4f}  "
          f"GBM-CV {control1['gbm_5fold_cv_auc']:.4f}")
    for seed in SEEDS:
        r = results["main_per_seed"][seed]
        print(f"Main seed {seed} (AdSERP vs selected synth): "
              f"RF-OOB {r['rf_oob_auc']:.4f}  RF-CV {r['rf_5fold_cv_auc']:.4f}  "
              f"GBM-CV {r['gbm_5fold_cv_auc']:.4f}  "
              f"(n={r['n_use_per_class']}/class)")


if __name__ == "__main__":
    main()
