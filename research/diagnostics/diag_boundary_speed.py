"""Boundary-speed diagnostic: do human eval segments start/end mid-flight?

mean_acceleration telescopes to (v_end - v_start) / duration after the 125Hz
resample, so rest-to-rest generation pins it near zero while any mid-flight
segment cut gives it a value proportional to the cut speed. This script
measures boundary speeds in the human pool and tests whether the human
+1.000 mean_acc correlations are driven by a heavy tail of mid-flight cuts.
"""
from __future__ import annotations

import numpy as np

from features import FEATURE_NAMES, extract_features, resample_trajectory

RNG = np.random.default_rng(7)
N_SAMPLE = 3000

offsets = np.load("training/full_pool_offsets.npy")
flat = np.load("training/pool_flat_i16.npy", mmap_mode="r")
t = np.load("training/pool_t_rel_f32.npy", mmap_mode="r")
n_pool = len(offsets) - 1
idx = RNG.choice(n_pool, size=N_SAMPLE, replace=False)

i_macc = FEATURE_NAMES.index("mean_acceleration")
i_sacc = FEATURE_NAMES.index("std_acceleration")
i_mvel = FEATURE_NAMES.index("mean_velocity")
i_dur = FEATURE_NAMES.index("movement_duration")

v0s, v1s, maccs, saccs, mvels, durs, tele = [], [], [], [], [], [], []
for i in idx:
    s, e = int(offsets[i]), int(offsets[i + 1])
    if e - s < 5:
        continue
    xy = flat[s:e].astype(np.float64)
    ts = t[s:e].astype(np.float64)
    traj = list(zip(xy[:, 0].tolist(), xy[:, 1].tolist(), ts.tolist()))
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
    saccs.append(f[i_sacc])
    mvels.append(f[i_mvel])
    durs.append(f[i_dur])
    tele.append((sp[-1] - sp[0]) / max(pts[-1, 2] - pts[0, 2], 1e-6))

v0s, v1s = np.array(v0s), np.array(v1s)
maccs, saccs, mvels = np.array(maccs), np.array(saccs), np.array(mvels)
tele = np.array(tele)
n = len(v0s)
print(f"n = {n} valid human pool segments")

print("\n--- boundary speeds (px/s, after 125Hz resample) ---")
for name, v in [("first-step", v0s), ("last-step", v1s)]:
    q = np.percentile(v, [50, 75, 90, 95, 99])
    print(f"{name}: median={q[0]:.1f} p75={q[1]:.1f} p90={q[2]:.1f} "
          f"p95={q[3]:.1f} p99={q[4]:.1f}")
    for th in [10, 50, 200, 1000]:
        print(f"   fraction > {th} px/s: {np.mean(v > th):.3f}")

print("\n--- telescoping identity check ---")
print(f"corr(mean_acc, (v_end-v_start)/dur) = {np.corrcoef(maccs, tele)[0,1]:.4f}")

print("\n--- outlier domination test on mean_acc x std_acc ---")
print(f"Pearson full:      {np.corrcoef(maccs, saccs)[0,1]:.4f}")
from scipy.stats import spearmanr
print(f"Spearman full:     {spearmanr(maccs, saccs).statistic:.4f}")
cap = np.abs(maccs) < np.percentile(np.abs(maccs), 99)
print(f"Pearson |macc|<p99: {np.corrcoef(maccs[cap], saccs[cap])[0,1]:.4f}")
print(f"top-5 |mean_acc| rows: {np.sort(np.abs(maccs))[-5:]}")

print("\n--- who has big |mean_acc|? ---")
big = np.abs(maccs) > np.percentile(np.abs(maccs), 95)
print(f"big |mean_acc| group: mean v0={v0s[big].mean():.1f} v1={v1s[big].mean():.1f} "
      f"dur={np.array(durs)[big].mean():.3f}")
print(f"rest         group: mean v0={v0s[~big].mean():.1f} v1={v1s[~big].mean():.1f} "
      f"dur={np.array(durs)[~big].mean():.3f}")
