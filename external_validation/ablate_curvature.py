"""Curvature ablation: does AdSERP-vs-synth / AdSERP-vs-human separability
collapse once curvature_mean and curvature_std are dropped from the 18-dim
feature vector?

Standalone diagnostic script under external_validation/ - does not modify or
reimplement any existing repo code beyond re-reading features.py's
FEATURE_NAMES list to locate the two suspect column indices. Loads the
already-cached feature matrices used by validate_adserp.py (no re-extraction,
no m4d_* files touched):

  - external_data/adserp_features_2000.npy   (external humans)
  - data/human_eval_features.npy             (internal held-out humans)
  - external_data/synth_features_seed{42,43,44}.npy (selected synthetics)

Detector blocks (RF-OOB, RF-5fold-CV, GBM-5fold-CV) are copied verbatim from
validate_adserp.py / evaluate.py, same hyperparameters and random_state=seed
convention.

Run:
    .venv/Scripts/python.exe external_validation/ablate_curvature.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parent.parent
EXT_DIR = Path(__file__).resolve().parent  # external_validation/ - this script's own dir, for results.json
EXT_DATA_DIR = REPO_ROOT / "external_data"  # raw datasets + cached .npy feature files
SEEDS = [42, 43, 44]

sys.path.insert(0, str(REPO_ROOT))
from features import FEATURE_NAMES  # noqa: E402

DROP_NAMES = ["curvature_mean", "curvature_std"]
DROP_IDX = [FEATURE_NAMES.index(n) for n in DROP_NAMES]
KEEP_IDX = [i for i in range(len(FEATURE_NAMES)) if i not in DROP_IDX]
KEEP_NAMES = [FEATURE_NAMES[i] for i in KEEP_IDX]


def rf_oob_auc(X, y, seed: int) -> float:
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                  random_state=seed)
    clf.fit(X, y)
    oob_proba = clf.oob_decision_function_[:, 1]
    return float(roc_auc_score(y, oob_proba)), clf


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


def top5_wasserstein(class0: np.ndarray, class1: np.ndarray, names) -> list:
    dists = []
    for idx, name in enumerate(names):
        l = class0[:, idx]
        r = class1[:, idx]
        std = np.std(l)
        if std < 1e-10:
            dists.append((name, 0.0))
            continue
        dists.append((name, float(wasserstein_distance(l / std, r / std))))
    dists.sort(key=lambda t: -t[1])
    return [{"feature": n, "normalized_wasserstein": d} for n, d in dists[:5]]


def run_suite(class0: np.ndarray, class1: np.ndarray, seed: int,
              names, run_cv: bool = True) -> dict:
    for name, arr in [("class0", class0), ("class1", class1)]:
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} has non-finite values, aborting")
    n_use = min(len(class0), len(class1))
    c0 = class0[:n_use]
    c1 = class1[:n_use]
    X = np.vstack([c0, c1])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])

    oob_auc, clf = rf_oob_auc(X, y, seed)
    result = {
        "n_class0_total": int(len(class0)),
        "n_class1_total": int(len(class1)),
        "n_use_per_class": int(n_use),
        "rf_oob_auc": oob_auc,
    }
    if run_cv:
        result["rf_5fold_cv_auc"] = rf_cv_auc(X, y, seed)
        result["gbm_5fold_cv_auc"] = gbm_cv_auc(X, y, seed)

    importances = sorted(zip(names, clf.feature_importances_.tolist()),
                          key=lambda t: -t[1])[:5]
    result["rf_top5_feature_importances"] = [
        {"feature": n, "importance": v} for n, v in importances
    ]
    result["top5_wasserstein"] = top5_wasserstein(c0, c1, names)
    return result


def curvature_diagnostics(adserp, human_eval, synth42) -> dict:
    cm_idx = FEATURE_NAMES.index("curvature_mean")
    out = {}
    for label, arr in [("adserp", adserp), ("internal_human", human_eval),
                        ("synth_seed42", synth42)]:
        col = arr[:, cm_idx]
        out[label] = {
            "mean": float(np.mean(col)),
            "median": float(np.median(col)),
            "frac_gt_10": float(np.mean(col > 10)),
            "n": int(len(col)),
        }
    return out


def main() -> None:
    human_eval = np.load(REPO_ROOT / "data" / "human_eval_features.npy")
    adserp_2000 = np.load(EXT_DATA_DIR / "adserp_features_2000.npy")
    synth = {s: np.load(EXT_DATA_DIR / f"synth_features_seed{s}.npy") for s in SEEDS}

    print(f"FEATURE_NAMES ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
    print(f"Dropping indices {DROP_IDX} ({DROP_NAMES}); "
          f"keeping {len(KEEP_IDX)} features: {KEEP_NAMES}")

    adserp_16 = adserp_2000[:, KEEP_IDX]
    human_eval_16 = human_eval[:, KEEP_IDX]
    synth_16 = {s: synth[s][:, KEEP_IDX] for s in SEEDS}

    results: dict = {
        "feature_names_18": FEATURE_NAMES,
        "dropped_indices": DROP_IDX,
        "dropped_names": DROP_NAMES,
        "kept_names_16": KEEP_NAMES,
        "curvature_diagnostics_18feat": curvature_diagnostics(
            adserp_2000, human_eval, synth[42]),
        "main_per_seed_16feat": {},
        "control_b_adserp_vs_human_eval_16feat": None,
        "control_c_internal_headline_16feat_seed42": None,
    }

    # (a) AdSERP vs synth per seed, 16 features
    for seed in SEEDS:
        print(f"\n=== (a) seed {seed}: AdSERP-16feat vs synth-16feat ===")
        r = run_suite(adserp_16, synth_16[seed], seed=seed, names=KEEP_NAMES)
        results["main_per_seed_16feat"][seed] = r
        print(json.dumps({k: v for k, v in r.items()
                           if k not in ("rf_top5_feature_importances", "top5_wasserstein")},
                          indent=2))

    # (b) AdSERP vs human_eval control, seed 42, 16 features
    print("\n=== (b) CONTROL: AdSERP-16feat vs human_eval-16feat (seed 42) ===")
    control_b = run_suite(adserp_16, human_eval_16, seed=42, names=KEEP_NAMES)
    results["control_b_adserp_vs_human_eval_16feat"] = control_b
    print(json.dumps({k: v for k, v in control_b.items()
                       if k not in ("rf_top5_feature_importances", "top5_wasserstein")},
                      indent=2))

    # (c) internal headline check: human_eval vs synth42, 16 features
    print("\n=== (c) internal headline check: human_eval-16feat vs synth42-16feat ===")
    control_c = run_suite(human_eval_16, synth_16[42], seed=42, names=KEEP_NAMES)
    results["control_c_internal_headline_16feat_seed42"] = control_c
    print(json.dumps({k: v for k, v in control_c.items()
                       if k not in ("rf_top5_feature_importances", "top5_wasserstein")},
                      indent=2))

    out_path = EXT_DIR / "ablate_curvature_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n=== SUMMARY (16-feature, curvature_mean/std removed) ===")
    for seed in SEEDS:
        r = results["main_per_seed_16feat"][seed]
        cv = f"  RF-CV {r['rf_5fold_cv_auc']:.4f}  GBM-CV {r['gbm_5fold_cv_auc']:.4f}" \
            if "rf_5fold_cv_auc" in r else ""
        print(f"(a) AdSERP vs synth seed {seed}: RF-OOB {r['rf_oob_auc']:.4f}{cv}")
    print(f"(b) AdSERP vs human_eval (control): RF-OOB {control_b['rf_oob_auc']:.4f}  "
          f"RF-CV {control_b['rf_5fold_cv_auc']:.4f}  GBM-CV {control_b['gbm_5fold_cv_auc']:.4f}")
    print(f"(c) human_eval vs synth42 (headline check): RF-OOB {control_c['rf_oob_auc']:.4f}  "
          f"RF-CV {control_c['rf_5fold_cv_auc']:.4f}  GBM-CV {control_c['gbm_5fold_cv_auc']:.4f}")
    print("\ncurvature_mean diagnostics (18-feat, pre-ablation):")
    print(json.dumps(results["curvature_diagnostics_18feat"], indent=2))


if __name__ == "__main__":
    main()
