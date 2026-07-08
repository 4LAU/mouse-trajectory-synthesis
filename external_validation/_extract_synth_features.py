"""Subprocess helper invoked by validate_adserp.py, once per seed.

Reproduces the EXACT synthetic-trajectory generation path evaluate.py uses
(same functions, imported not reimplemented), under the pool-replay env vars
(EVENT_POOL_LOAD / EVENT_POOL_PICKS) that verify_headline.py sets for the
published replay. Running this in its own process (like verify_headline.py's
replay() does via subprocess) avoids any event_stream_polar module-level
state leaking between seeds, since that module reads its env knobs once at
import time.

Saves the resulting (N, 18) feature matrix - built purely from
features.extract_feature_matrix, unmodified - to --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluate import generate_synthetic_trajectories, load_experiment  # noqa: E402
from features import extract_feature_matrix  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--n-synthetic", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    human_distances = np.load(Path(args.data_dir) / "human_distances.npy")

    module = load_experiment("experiments.event_stream_polar")
    trajectories = generate_synthetic_trajectories(
        module, human_distances, args.n_synthetic, rng)
    print(f"generated {len(trajectories)}/{args.n_synthetic} trajectories",
          flush=True)

    feats = extract_feature_matrix(trajectories)
    print(f"valid features: {len(feats)}/{args.n_synthetic} "
          f"({len(feats) / args.n_synthetic:.0%})", flush=True)

    np.save(args.out, feats)
    print(f"saved {args.out} {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
