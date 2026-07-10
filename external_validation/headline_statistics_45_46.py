"""Statistical rigor pass on seeds 45/46 (T2 out-of-sample check).

Companion to headline_statistics.py, which hardcodes seeds 42/43/44 and has
no CLI flags for other seeds. Rather than edit that committed script, this is
a thin wrapper that reimports its exact fit_rf_oob / build_xy / bootstrap_ci
functions (not reimplemented) and runs the same three checks (sanity,
20-seed detector sweep, 10k-resample bootstrap) for seeds 45 and 46 against
the published honest-replay RF OOB AUCs (0.5148, 0.5190).

Inputs: external_data/synth_features_seed{45,46}.npy, regenerated the same
way validate_adserp.py builds them for seeds 42-44 (offline pool replay via
EVENT_POOL_LOAD / EVENT_POOL_PICKS with the trust33 f20d85_r30_rf picks,
CPU only, then the 18 evaluate.py features per trajectory).

Does not modify headline_statistics.py or headline_statistics.json. Writes
external_validation/headline_statistics_45_46.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from headline_statistics import (
    HUMAN_FEATURES_PATH,
    N_BOOTSTRAP,
    BOOTSTRAP_RNG_SEED,
    bootstrap_ci,
    run_case,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTH_FEATURES_PATHS = {
    45: REPO_ROOT / "external_data" / "synth_features_seed45.npy",
    46: REPO_ROOT / "external_data" / "synth_features_seed46.npy",
}
PUBLISHED_AUC = {45: 0.5148, 46: 0.5190}


def main() -> None:
    print(f"Loading human features from {HUMAN_FEATURES_PATH}")
    human_features = np.load(HUMAN_FEATURES_PATH)
    print(f"  shape = {human_features.shape}")

    results: dict = {"seeds": {}, "combined": None}
    seed_aucs_sanity = {}

    for seed in (45, 46):
        print(f"\n=== Seed {seed} ===")
        synth_features = np.load(SYNTH_FEATURES_PATHS[seed])
        print(f"  synth shape = {synth_features.shape}")
        case = run_case(
            label=f"human_vs_synth_seed{seed}",
            human_features=human_features,
            other_features=synth_features,
            sanity_seed=seed,
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

    vals = np.array(list(seed_aucs_sanity.values()))
    results["combined"] = {
        "seed_level_sanity_aucs": {str(k): float(v) for k, v in seed_aucs_sanity.items()},
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "range": float(np.max(vals) - np.min(vals)),
        "all_seed_bootstrap_cis_cover_0_50": bool(all(
            results["seeds"][str(s)]["bootstrap"]["covers_0_50"] for s in (45, 46)
        )),
    }
    print("\n=== Combined seeds 45/46 ===")
    print(f"  mean={results['combined']['mean']:.4f}  sd={results['combined']['sd']:.4f}  "
          f"range=[{results['combined']['min']:.4f}, {results['combined']['max']:.4f}]")
    print(f"  all seed CIs cover 0.50 = {results['combined']['all_seed_bootstrap_cis_cover_0_50']}")

    out_path = Path(__file__).resolve().parent / "headline_statistics_45_46.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
