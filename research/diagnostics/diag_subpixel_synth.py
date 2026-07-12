"""WS4 synth-side diagnostic: sub-pixel step fraction of generated output.

Human reference (3000 held-out trajectories, after the standard 125Hz resample):
35% of steps move less than 1 px, 4.8 to 6.0% are exact zeros, mean distance to
the nearest integer 0.148 px. Measure the same numbers on the best config.
"""
from __future__ import annotations

import os

os.environ.setdefault("CANDI_CKPT", "candi_polar_flow_best.pt")
os.environ.setdefault("CANDI_STEPS", "200")
os.environ.setdefault("CANDI_CFG", "0")
os.environ.setdefault("CANDI_CORRECT", "rotate")
os.environ.setdefault("CANDI_SPEED_SKEW", "0")
os.environ.setdefault("CANDI_PERP_SCALE", "0.85")
os.environ.setdefault("CANDI_GUIDE", "0.15")

import numpy as np

from features import resample_trajectory

N = 1000
rng = np.random.default_rng(42)
distances = np.load("data/human_distances.npy")

specs = []
for _ in range(N):
    dist = float(rng.choice(distances))
    angle = float(rng.uniform(0, 2 * np.pi))
    specs.append((960.0, 540.0, 960.0 + dist * np.cos(angle), 540.0 + dist * np.sin(angle)))

import experiments.candi as candi

trajs = candi.generate_paths(specs)

sub_px = zero = total = 0
near_int = []
for t in trajs:
    if t is None or len(t) < 2:
        continue
    r = resample_trajectory(t)
    pts = np.asarray(r, dtype=np.float64)
    if len(pts) < 2:
        continue
    d = np.diff(pts[:, :2], axis=0)
    step = np.hypot(d[:, 0], d[:, 1])
    total += len(step)
    sub_px += int((step < 1.0).sum())
    zero += int((step == 0.0).sum())
    near_int.append(np.abs(d - np.round(d)).ravel())

near = np.concatenate(near_int)
print(f"steps analyzed: {total}")
print(f"sub-pixel (<1px) fraction: {sub_px / total:.3f}   (human 0.35)")
print(f"exact-zero fraction:       {zero / total:.4f}  (human 0.048-0.060)")
print(f"mean dist to integer:      {near.mean():.3f}   (human 0.148, uniform 0.25)")
