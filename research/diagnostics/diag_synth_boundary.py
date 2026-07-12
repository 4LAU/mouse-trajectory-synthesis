"""Measure boundary speeds of generated trajectories (best T3 config, SIR off).

Companion to diag_boundary_speed.py: human pool segments start mid-flight
(median first-step speed 250 px/s, 90% > 10 px/s). Does the generator
reproduce that, or does it emit rest-to-rest ramps?
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

N = 300
rng = np.random.default_rng(123)
dists = np.load("data/human_distances.npy")
sample = rng.choice(dists, size=N)

from experiments.event_stream_polar import generate_paths

specs = []
for d in sample:
    ang = rng.uniform(-np.pi, np.pi)
    sx, sy = rng.uniform(200, 800), rng.uniform(200, 800)
    specs.append((sx, sy, sx + d * np.cos(ang), sy + d * np.sin(ang)))

print(f"Generating {N} trajectories on CPU...", flush=True)
trajs = generate_paths(specs)

i_macc = FEATURE_NAMES.index("mean_acceleration")
i_ttp = FEATURE_NAMES.index("time_to_peak_velocity")
i_vsk = FEATURE_NAMES.index("velocity_skewness")

v0s, v1s, maccs, ttps, vsks = [], [], [], [], []
for traj in trajs:
    if traj is None or len(traj) < 5:
        continue
    r = resample_trajectory(traj)
    f = extract_features(r)
    if f is None or not np.all(np.isfinite(f)):
        continue
    pts = np.asarray(r)
    if len(pts) < 5:
        continue
    dt = np.maximum(np.diff(pts[:, 2]), 1e-6)
    sp = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])) / dt
    v0s.append(sp[0])
    v1s.append(sp[-1])
    maccs.append(f[i_macc])
    ttps.append(f[i_ttp])
    vsks.append(f[i_vsk])

v0s, v1s, maccs = np.array(v0s), np.array(v1s), np.array(maccs)
print(f"n = {len(v0s)} valid synthetic trajectories")

print("\n--- synthetic boundary speeds (px/s, after 125Hz resample) ---")
for name, v in [("first-step", v0s), ("last-step", v1s)]:
    q = np.percentile(v, [50, 75, 90, 95, 99])
    print(f"{name}: median={q[0]:.1f} p75={q[1]:.1f} p90={q[2]:.1f} "
          f"p95={q[3]:.1f} p99={q[4]:.1f}")
    for th in [10, 50, 200, 1000]:
        print(f"   fraction > {th} px/s: {np.mean(v > th):.3f}")

print("\n--- mean_acc / shape features, synth vs human eval ---")
hum = np.load("data/human_eval_features.npy")
for nm, arr in [("mean_acceleration", maccs),
                ("time_to_peak_velocity", np.array(ttps)),
                ("velocity_skewness", np.array(vsks))]:
    i = FEATURE_NAMES.index(nm)
    h = hum[:, i]
    print(f"{nm}:")
    print(f"  synth p5/25/50/75/95: "
          + " ".join(f"{np.percentile(arr, p):.3g}" for p in [5, 25, 50, 75, 95]))
    print(f"  human p5/25/50/75/95: "
          + " ".join(f"{np.percentile(h, p):.3g}" for p in [5, 25, 50, 75, 95]))
print(f"\nsynth |mean_acc| p99: {np.percentile(np.abs(maccs), 99):.4g}   "
      f"human |mean_acc| p99: {np.percentile(np.abs(hum[:, i_macc]), 99):.4g}")
