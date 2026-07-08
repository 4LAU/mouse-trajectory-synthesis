"""Statistical rigor pass on the headline RF-OOB AUC result.

For each of the three published seeds (42, 43, 44) this script:

  1. SANITY   - refits the RF-OOB classifier exactly as evaluate.py does
                (RandomForestClassifier(n_estimators=100, oob_score=True,
                n_jobs=-1, random_state=<seed>)) on the cached human vs.
                synthetic feature matrices, and checks the result reproduces
                the published AUC to ~1e-3.
  2. SWEEP    - refits the same RF with 20 different random_state values
                (1000..1019), same data, to show how much the OOB AUC moves
                under detector-seed randomness alone.
  3. BOOTSTRAP- percentile bootstrap (10,000 resamples, numpy
                default_rng(12345)) over the OOB (score, label) pairs from
                the sanity fit, reporting the 2.5/97.5 percentiles.
  4. COMBINED - 3-seed mean/range and a chance-coverage check.

The same three statistics are also computed for one external comparison
(AdSERP-2000 vs. synth seed 42) as a sanity-check control that the pipeline
can still detect a large, real distributional gap (expected AUC ~0.948).

This script does not modify any existing repo file. It only reads cached
.npy feature matrices and writes external_validation/headline_statistics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
HUMAN_FEATURES_PATH = REPO_ROOT / "data" / "human_eval_features.npy"
SYNTH_FEATURES_PATHS = {
    42: REPO_ROOT / "external_data" / "synth_features_seed42.npy",
    43: REPO_ROOT / "external_data" / "synth_features_seed43.npy",
    44: REPO_ROOT / "external_data" / "synth_features_seed44.npy",
}
ADSERP_FEATURES_PATH = REPO_ROOT / "external_data" / "adserp_features_2000.npy"

PUBLISHED_AUC = {42: 0.5095, 43: 0.5030, 44: 0.4993}
ADSERP_EXPECTED_AUC = 0.948

SWEEP_SEEDS = list(range(1000, 1020))  # 20 detector-randomness seeds
N_BOOTSTRAP = 10_000
BOOTSTRAP_RNG_SEED = 12345
SANITY_TOLERANCE = 1e-3


def fit_rf_oob(X: np.ndarray, y: np.ndarray, random_state: int):
    """Exact RF-OOB config copied from evaluate.py's step 8a."""
    clf = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X, y)
    oob_proba = clf.oob_decision_function_[:, 1]
    auc = roc_auc_score(y, oob_proba)
    return auc, oob_proba


def build_xy(human_features: np.ndarray, other_features: np.ndarray):
    n_use = min(len(human_features), len(other_features))
    h = human_features[:n_use]
    o = other_features[:n_use]
    X = np.vstack([h, o])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    return X, y


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, n_boot: int, rng_seed: int):
    rng = np.random.default_rng(rng_seed)
    n = len(y)
    boot_aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        s_b = scores[idx]
        # Guard against a degenerate resample with a single class present.
        if len(np.unique(y_b)) < 2:
            boot_aucs[b] = np.nan
            continue
        boot_aucs[b] = roc_auc_score(y_b, s_b)
    valid = boot_aucs[~np.isnan(boot_aucs)]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return {
        "n_boot": n_boot,
        "n_valid": int(len(valid)),
        "n_degenerate_dropped": int(n_boot - len(valid)),
        "mean": float(np.mean(valid)),
        "sd": float(np.std(valid, ddof=1)),
        "ci_2_5": float(lo),
        "ci_97_5": float(hi),
        "covers_0_50": bool(lo <= 0.50 <= hi),
    }


def run_case(label: str, human_features: np.ndarray, other_features: np.ndarray,
             sanity_seed: int, published_auc: float | None):
    X, y = build_xy(human_features, other_features)

    # 1. SANITY
    sanity_auc, oob_proba = fit_rf_oob(X, y, random_state=sanity_seed)
    sanity_result = {
        "random_state": sanity_seed,
        "auc": float(sanity_auc),
        "published_auc": published_auc,
        "abs_diff_vs_published": (
            float(abs(sanity_auc - published_auc)) if published_auc is not None else None
        ),
        "matches_within_tol": (
            bool(abs(sanity_auc - published_auc) <= SANITY_TOLERANCE)
            if published_auc is not None else None
        ),
        "tolerance": SANITY_TOLERANCE,
    }

    # 2. DETECTOR-SEED SWEEP
    sweep_aucs = []
    for rs in SWEEP_SEEDS:
        auc_rs, _ = fit_rf_oob(X, y, random_state=rs)
        sweep_aucs.append(float(auc_rs))
    sweep_aucs_arr = np.array(sweep_aucs)
    sweep_result = {
        "random_states": SWEEP_SEEDS,
        "aucs": sweep_aucs,
        "mean": float(np.mean(sweep_aucs_arr)),
        "sd": float(np.std(sweep_aucs_arr, ddof=1)),
        "min": float(np.min(sweep_aucs_arr)),
        "max": float(np.max(sweep_aucs_arr)),
    }

    # 3. BOOTSTRAP CI (on the OOB scores from the sanity fit)
    bootstrap_result = bootstrap_ci(y, oob_proba, N_BOOTSTRAP, BOOTSTRAP_RNG_SEED)

    return {
        "label": label,
        "n_per_class": int(len(y) // 2),
        "sanity": sanity_result,
        "sweep": sweep_result,
        "bootstrap": bootstrap_result,
    }


def main() -> None:
    print(f"Loading human features from {HUMAN_FEATURES_PATH}")
    human_features = np.load(HUMAN_FEATURES_PATH)
    print(f"  shape = {human_features.shape}")

    results: dict = {"seeds": {}, "adserp_context": None, "combined": None}

    # --- Headline: 3 seeds vs human ---
    seed_aucs_sanity = {}
    for seed in (42, 43, 44):
        print(f"\n=== Seed {seed} ===")
        synth_features = np.load(SYNTH_FEATURES_PATHS[seed])
        print(f"  synth shape = {synth_features.shape}")
        case = run_case(
            label=f"human_vs_synth_seed{seed}",
            human_features=human_features,
            other_features=synth_features,
            sanity_seed=seed,  # evaluate.py uses random_state=args.seed (the synth seed)
            published_auc=PUBLISHED_AUC[seed],
        )
        results["seeds"][str(seed)] = case
        seed_aucs_sanity[seed] = case["sanity"]["auc"]

        print(f"  sanity AUC        = {case['sanity']['auc']:.4f} "
              f"(published {PUBLISHED_AUC[seed]:.4f}, "
              f"match={case['sanity']['matches_within_tol']})")
        print(f"  sweep mean +/- SD = {case['sweep']['mean']:.4f} +/- {case['sweep']['sd']:.4f} "
              f"[{case['sweep']['min']:.4f}, {case['sweep']['max']:.4f}]")
        print(f"  bootstrap 95% CI  = [{case['bootstrap']['ci_2_5']:.4f}, "
              f"{case['bootstrap']['ci_97_5']:.4f}] covers 0.50 = "
              f"{case['bootstrap']['covers_0_50']}")

    # --- Combined 3-seed summary ---
    vals = np.array(list(seed_aucs_sanity.values()))
    results["combined"] = {
        "seed_level_sanity_aucs": {str(k): float(v) for k, v in seed_aucs_sanity.items()},
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "range": float(np.max(vals) - np.min(vals)),
        "all_seed_bootstrap_cis_cover_0_50": bool(all(
            results["seeds"][str(s)]["bootstrap"]["covers_0_50"] for s in (42, 43, 44)
        )),
    }
    print(f"\n=== Combined 3-seed ===")
    print(f"  mean={results['combined']['mean']:.4f}  sd={results['combined']['sd']:.4f}  "
          f"range=[{results['combined']['min']:.4f}, {results['combined']['max']:.4f}]")
    print(f"  all seed CIs cover 0.50 = {results['combined']['all_seed_bootstrap_cis_cover_0_50']}")

    # --- AdSERP external-comparison context row ---
    print(f"\n=== AdSERP-2000 vs synth seed42 (context) ===")
    adserp_features = np.load(ADSERP_FEATURES_PATH)
    print(f"  adserp shape = {adserp_features.shape}")
    synth42 = np.load(SYNTH_FEATURES_PATHS[42])
    adserp_case = run_case(
        label="adserp2000_vs_synth_seed42",
        human_features=adserp_features,
        other_features=synth42,
        sanity_seed=42,
        published_auc=ADSERP_EXPECTED_AUC,
    )
    results["adserp_context"] = adserp_case
    print(f"  sanity AUC        = {adserp_case['sanity']['auc']:.4f} "
          f"(expected ~{ADSERP_EXPECTED_AUC:.3f})")
    print(f"  sweep mean +/- SD = {adserp_case['sweep']['mean']:.4f} +/- {adserp_case['sweep']['sd']:.4f} "
          f"[{adserp_case['sweep']['min']:.4f}, {adserp_case['sweep']['max']:.4f}]")
    print(f"  bootstrap 95% CI  = [{adserp_case['bootstrap']['ci_2_5']:.4f}, "
          f"{adserp_case['bootstrap']['ci_97_5']:.4f}]")

    out_path = Path(__file__).resolve().parent / "headline_statistics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
