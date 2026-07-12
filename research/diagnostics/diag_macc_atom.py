"""Exact-zero mean_acceleration atom: synth vs human.

diag_synth_boundary.py showed synth p75(mean_acc) = -4e-12: a point mass at
exactly zero, which exists when the first and last resampled steps are both
stationary (v0 = v1 = 0, telescoping). If humans lack that atom the RF can
split on it. Measures the atom and the near-zero mass on both classes.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("EVENT_CKPT", "event_polar_4m_fc_v2.pt")
os.environ.setdefault("EVENT_ORDER", "gumbel")
os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_SNAP", "2.5")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("EVENT_SIR", "1")

import numpy as np

from features import FEATURE_NAMES, extract_features, resample_trajectory

i_macc = FEATURE_NAMES.index("mean_acceleration")
i_mjerk = FEATURE_NAMES.index("mean_jerk")

hum = np.load("data/human_eval_features.npy")
h_macc = hum[:, i_macc]
h_mjerk = hum[:, i_mjerk]

from experiments.event_stream_polar import generate_paths

N = 400
rng = np.random.default_rng(456)
dists = np.load("data/human_distances.npy")
specs = []
for d in rng.choice(dists, size=N):
    ang = rng.uniform(-np.pi, np.pi)
    sx, sy = rng.uniform(200, 800), rng.uniform(200, 800)
    specs.append((sx, sy, sx + d * np.cos(ang), sy + d * np.sin(ang)))
print(f"Generating {N} trajectories on CPU...", flush=True)
trajs = generate_paths(specs)

s_macc, s_mjerk, both_ends_still = [], [], []
for traj in trajs:
    if traj is None or len(traj) < 5:
        continue
    r = resample_trajectory(traj)
    f = extract_features(r)
    if f is None or not np.all(np.isfinite(f)):
        continue
    s_macc.append(f[i_macc])
    s_mjerk.append(f[i_mjerk])
    pts = np.asarray(r)
    dt = np.maximum(np.diff(pts[:, 2]), 1e-6)
    sp = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])) / dt
    both_ends_still.append(sp[0] < 1e-9 and sp[-1] < 1e-9)

s_macc = np.array(s_macc)
s_mjerk = np.array(s_mjerk)
print(f"n synth = {len(s_macc)}, n human = {len(h_macc)}")

for eps in [1e-9, 1e-3, 1.0, 10.0]:
    print(f"|mean_acc| < {eps:g}:  synth {np.mean(np.abs(s_macc) < eps):.4f}   "
          f"human {np.mean(np.abs(h_macc) < eps):.4f}")
for eps in [1e-9, 1e-3, 1.0, 100.0]:
    print(f"|mean_jerk| < {eps:g}: synth {np.mean(np.abs(s_mjerk) < eps):.4f}   "
          f"human {np.mean(np.abs(h_mjerk) < eps):.4f}")
print(f"synth both-ends-stationary fraction: {np.mean(both_ends_still):.4f}")
