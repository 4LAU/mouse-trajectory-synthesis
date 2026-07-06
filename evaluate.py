"""Adversarial evaluator for mouse-trajectory generators.

Trains a Random Forest classifier to distinguish real human mouse trajectories
from synthetically generated ones.  Reports out-of-bag (OOB) AUC - lower is
better, meaning the classifier struggles to tell them apart.

Usage:
    python evaluate.py --experiment experiments.ddpm_arclen
    python evaluate.py --experiment experiments.my_model --n-synthetic 5000 --seed 123
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from features import FEATURE_NAMES, extract_feature_matrix, normalized_wasserstein_by_feature


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarial evaluation: RF classifier distinguishes human vs. generated trajectories.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Dotted module path exporting generate_path(sx, sy, ex, ey). "
        "Example: experiments.ddpm_arclen",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing human_eval_features.npy and human_distances.npy (default: ./data)",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=2000,
        help="Number of synthetic trajectories to generate (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-raw-nn",
        action="store_true",
        help="Skip the raw-trajectory neural detector",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_experiment(module_path: str):
    """Dynamically import an experiment module and return it."""
    module = importlib.import_module(module_path)
    if not hasattr(module, "generate_path"):
        print(f"ERROR: module '{module_path}' has no generate_path function", file=sys.stderr)
        sys.exit(1)
    return module


def generate_synthetic_trajectories(
    module,
    distances: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> list:
    """Generate *n* synthetic trajectories using the experiment's generator.

    Each trajectory starts at screen centre (960, 540).  The travel distance is
    sampled from *distances* (the empirical human distribution) and the angle is
    uniformly random.  If the module exports generate_paths (batched), it is
    used; otherwise trajectories are generated one at a time.
    """
    center_x, center_y = 960.0, 540.0

    specs = []
    for _ in range(n):
        dist = float(rng.choice(distances))
        angle = float(rng.uniform(0, 2 * np.pi))
        end_x = center_x + dist * np.cos(angle)
        end_y = center_y + dist * np.sin(angle)
        specs.append((center_x, center_y, end_x, end_y))

    if hasattr(module, "generate_paths"):
        print("  Using batched generation")
        raw = module.generate_paths(specs)
        return [t for t in raw if t is not None and len(t) >= 2]

    generate_path = module.generate_path
    trajectories: list = []
    for i, (sx, sy, ex, ey) in enumerate(specs):
        try:
            traj = generate_path(sx, sy, ex, ey)
        except Exception as exc:  # noqa: BLE001
            if i == 0:
                print(f"  WARNING: generate_path raised {type(exc).__name__}: {exc}")
            continue
        if traj is not None and len(traj) >= 2:
            trajectories.append(traj)

    return trajectories


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_wasserstein_diagnostics(human_features: np.ndarray, synth_features: np.ndarray) -> None:
    """Print per-feature normalised Wasserstein distances, sorted descending."""
    w_dists = normalized_wasserstein_by_feature(human_features, synth_features)
    ranked = sorted(zip(FEATURE_NAMES, w_dists), key=lambda x: x[1], reverse=True)

    print("\n--- Per-feature Wasserstein distances (normalised) ---")
    for name, d in ranked:
        flag = "  *** HIGH ***" if d > 0.3 else ""
        print(f"  {name:30s} {d:.4f}{flag}")
    print()


def print_correlation_gaps(human_features: np.ndarray, synth_features: np.ndarray, top_k: int = 10) -> None:
    """Print feature pairs with largest correlation structure differences."""
    n = len(FEATURE_NAMES)
    h_corr = np.corrcoef(human_features.T)
    s_corr = np.corrcoef(synth_features.T)
    gaps = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(h_corr[i, j] - s_corr[i, j])
            gaps.append((FEATURE_NAMES[i], FEATURE_NAMES[j],
                         h_corr[i, j], s_corr[i, j], diff))
    gaps.sort(key=lambda x: x[4], reverse=True)
    print(f"--- Top {top_k} correlation structure gaps ---")
    for a, b, hr, sr, d in gaps[:top_k]:
        print(f"  {a:25s} × {b:25s}  human={hr:+.3f}  synth={sr:+.3f}  gap={d:.3f}")
    print()


def print_feature_importances(clf: RandomForestClassifier, top_k: int = 8) -> None:
    """Print the top-k RF feature importances."""
    importances = clf.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)

    print(f"--- RF feature importances (top {top_k}) ---")
    for name, imp in ranked[:top_k]:
        print(f"  {name:30s} {imp:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    # 1. Load precomputed human data
    data_dir = args.data_dir.rstrip("/")
    human_features = np.load(f"{data_dir}/human_eval_features.npy")
    human_distances = np.load(f"{data_dir}/human_distances.npy")
    print(f"Loaded {len(human_features)} human feature vectors, "
          f"{len(human_distances)} distance samples")

    # 2. Import experiment
    module = load_experiment(args.experiment)
    print(f"Experiment: {args.experiment}")

    # 3. Generate synthetic trajectories
    n = args.n_synthetic
    print(f"Generating {n} synthetic trajectories …")
    t0 = time.perf_counter()
    trajectories = generate_synthetic_trajectories(module, human_distances, n, rng)
    elapsed = time.perf_counter() - t0
    print(f"  Generated {len(trajectories)} trajectories in {elapsed:.1f}s")

    # 4. Extract features
    synth_features = extract_feature_matrix(trajectories)
    valid_ratio = len(synth_features) / max(n, 1)
    print(f"  Valid feature vectors: {len(synth_features)}/{n} ({valid_ratio:.0%})")

    # 5. Penalise if too few valid features
    if valid_ratio < 0.80:
        print(f"\nToo few valid trajectories ({valid_ratio:.0%} < 80%). Penalising.")
        print("val_auc: 0.9990")
        return

    # 6. Balance sample sizes
    n_human = len(human_features)
    n_synth = len(synth_features)
    n_use = min(n_human, n_synth)
    if n_human != n_synth:
        print(f"  Balancing: using {n_use} samples from each class")

    human_balanced = human_features[:n_use]
    synth_balanced = synth_features[:n_use]

    # 7. Wasserstein diagnostics
    print_wasserstein_diagnostics(human_balanced, synth_balanced)
    print_correlation_gaps(human_balanced, synth_balanced)

    # 8. Train adversarial classifiers
    X = np.vstack([human_balanced, synth_balanced])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])

    # 8a. Random Forest with OOB
    clf = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        n_jobs=-1,
        random_state=args.seed,
    )
    clf.fit(X, y)

    oob_proba = clf.oob_decision_function_[:, 1]
    auc_rf_oob = roc_auc_score(y, oob_proba)

    # 8b. Held-out AUC via 5-fold cross-validation (RF)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    rf_cv = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=args.seed,
    )
    cv_proba = cross_val_predict(rf_cv, X, y, cv=cv, method="predict_proba")[:, 1]
    auc_rf_cv = roc_auc_score(y, cv_proba)

    # 8c. Gradient Boosting (second classifier family)
    gbm = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, random_state=args.seed,
    )
    gbm_cv_proba = cross_val_predict(gbm, X, y, cv=cv, method="predict_proba")[:, 1]
    auc_gbm_cv = roc_auc_score(y, gbm_cv_proba)

    # 9. Feature importances
    print_feature_importances(clf)

    # 10. Raw-trajectory neural detector (held-out, never tuned against)
    auc_raw_nn = None
    if not args.no_raw_nn:
        try:
            from detector_raw import raw_nn_auc
            t0 = time.perf_counter()
            auc_raw_nn = raw_nn_auc(trajectories)
            print(f"  Raw-NN detector trained in {time.perf_counter() - t0:.1f}s")
        except FileNotFoundError as exc:
            print(f"  Raw-NN detector skipped (missing data: {exc})")

    # 11. Final results
    print(f"val_auc: {auc_rf_oob:.4f}")
    print(f"  RF OOB AUC:      {auc_rf_oob:.4f}")
    print(f"  RF 5-fold CV:    {auc_rf_cv:.4f}")
    print(f"  GBM 5-fold CV:   {auc_gbm_cv:.4f}")
    if auc_raw_nn is not None:
        print(f"  Raw-NN 3-fold:   {auc_raw_nn:.4f}")


if __name__ == "__main__":
    main()
