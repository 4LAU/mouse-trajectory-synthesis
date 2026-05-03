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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_experiment(module_path: str):
    """Dynamically import an experiment module and return its generate_path function."""
    module = importlib.import_module(module_path)
    if not hasattr(module, "generate_path"):
        print(f"ERROR: module '{module_path}' has no generate_path function", file=sys.stderr)
        sys.exit(1)
    return module.generate_path


def generate_synthetic_trajectories(
    generate_path,
    distances: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> list:
    """Generate *n* synthetic trajectories using the experiment's generator.

    Each trajectory starts at screen centre (960, 540).  The travel distance is
    sampled from *distances* (the empirical human distribution) and the angle is
    uniformly random.
    """
    center_x, center_y = 960.0, 540.0
    trajectories: list = []

    for i in range(n):
        dist = float(rng.choice(distances))
        angle = float(rng.uniform(0, 2 * np.pi))
        end_x = center_x + dist * np.cos(angle)
        end_y = center_y + dist * np.sin(angle)

        try:
            traj = generate_path(center_x, center_y, end_x, end_y)
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
    generate_path = load_experiment(args.experiment)
    print(f"Experiment: {args.experiment}")

    # 3. Generate synthetic trajectories
    n = args.n_synthetic
    print(f"Generating {n} synthetic trajectories …")
    t0 = time.perf_counter()
    trajectories = generate_synthetic_trajectories(generate_path, human_distances, n, rng)
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

    # 8. Train adversarial classifier
    X = np.vstack([human_balanced, synth_balanced])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])

    clf = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        n_jobs=-1,
        random_state=args.seed,
    )
    clf.fit(X, y)

    # 9. Compute OOB AUC
    oob_proba = clf.oob_decision_function_[:, 1]
    auc = roc_auc_score(y, oob_proba)

    # 10. Feature importances
    print_feature_importances(clf)

    # 11. Final result
    print(f"val_auc: {auc:.4f}")


if __name__ == "__main__":
    main()
