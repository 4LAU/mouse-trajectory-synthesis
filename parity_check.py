"""Compare sequential generate_path vs batched generate_paths distributions.

Same endpoint specs through both code paths, then per-feature Wasserstein
distances between the two synthetic sets. Near-zero distances mean the
batched path is distribution-identical and any AUC gap vs old logs comes
from something else.
"""
from __future__ import annotations

import time

import numpy as np

from features import FEATURE_NAMES, extract_feature_matrix, normalized_wasserstein_by_feature

N = 300
rng = np.random.default_rng(42)
distances = np.load("data/human_distances.npy")

specs = []
for _ in range(N):
    dist = float(rng.choice(distances))
    angle = float(rng.uniform(0, 2 * np.pi))
    specs.append((960.0, 540.0, 960.0 + dist * np.cos(angle), 540.0 + dist * np.sin(angle)))

import experiments.candi as candi

t0 = time.time()
batched = candi.generate_paths(specs)
print(f"batched: {time.time() - t0:.1f}s")

t0 = time.time()
sequential = [candi.generate_path(*s) for s in specs]
print(f"sequential: {time.time() - t0:.1f}s")

fb = extract_feature_matrix([t for t in batched if t is not None and len(t) >= 2])
fs = extract_feature_matrix([t for t in sequential if t is not None and len(t) >= 2])
print(f"valid: batched {len(fb)}, sequential {len(fs)}")

w = normalized_wasserstein_by_feature(fs, fb)
print("\nPer-feature W distance, sequential vs batched (same specs):")
for name, d in sorted(zip(FEATURE_NAMES, w), key=lambda x: -x[1]):
    flag = "  <-- CHECK" if d > 0.15 else ""
    print(f"  {name:30s} {d:.4f}{flag}")
