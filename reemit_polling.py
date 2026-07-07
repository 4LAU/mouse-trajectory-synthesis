"""Event-level realism: re-emit generated trajectories on the polling clock.

A raw-event diagnostic showed the one place the generated data is trivially
separable from human before any resampling: event timing. Human hardware
reports motion on a polling grid, 95.3 percent of held-out deltas are exactly
8 ms (125 Hz USB polling) with a small dropped-frame tail at 16 and 24 ms.
The model emits deltas spread over 1 to 11 ms. Every detector in the eval
suite works on a 125 Hz resample so none of them can see this, but a
one-feature detector on raw dt would.

The fix is presentation, not kinematics, and two failed attempts pinned down
the constraints. Drawing deltas iid from the POOLED human dt distribution
mixes clock signatures from different devices into every trajectory and a
histogram detector still wins (0.98). Transplanting a whole donor dt sequence
fixes the mixing but puts timestamps off the 8 ms grid, and any off-grid node
smooths the path under the downstream 125 Hz resample: angular velocity drops
15 percent and the honest RF jumps to 0.58. A strict 8 ms grid, by contrast,
is an exact no-op for every feature, because the feature pipeline's 125 Hz
resample reads the path at those same instants.

So the working version keeps every node on the 8 ms clock and takes only the
DROPPED-FRAME pattern from a human donor: sample a donor trajectory recorded
on the 8 ms clock, quantize its deltas to clock multiples, tile them from a
random offset, and dilate the generated path's time axis by the sub-1-percent
factor that lands the duration on the clock (human durations are naturally
clock multiples; synthetic ones were continuous). Human sessions on 6 or 7 ms
clocks (a few percent) are not imitated; that residual is disclosed.
This script re-emits the selected set, then reports:

  1. raw-dt detector (RF on per-trajectory dt histogram) before and after,
  2. the 18-feature RF OOB detector vs the honest eval sample before and
     after, to show the kinematic story does not move.

Coordinate granularity stays out of scope: the human dataset stores float32
normalized positions, so its raw pixel lattice is a storage artifact that
cannot be compared honestly (originals were integer, generated is integer).

Run:
    .venv/Scripts/python.exe reemit_polling.py \
        --pool pool_s42_k16.npz \
        --picks pool_s42_k16_picks_trust33_f20d85_r30_rf.npy
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from detector_raw import load_human_raw
from features import FEATURE_NAMES, extract_features, resample_trajectory

DT_EDGES_MS = np.array([0, 2, 4, 6, 7.5, 8.5, 10, 14, 17, 25, 40, 1e9])


CLOCK = 0.008


def human_dt_sequences(human_trajs, min_len=20) -> list[np.ndarray]:
    """Donor clocks: dt sequences of human trajectories recorded on the 8 ms
    clock, quantized to clock multiples (so drops read as 16 or 24 ms)."""
    seqs = []
    for t in human_trajs:
        dts = np.diff(np.asarray(t, dtype=np.float64)[:, 2])
        dts = dts[dts > 0]
        if len(dts) >= min_len and abs(np.median(dts) - CLOCK) < 0.0004:
            seqs.append(np.maximum(np.round(dts / CLOCK), 1) * CLOCK)
    return seqs


def reemit(traj: np.ndarray, donor_seqs: list[np.ndarray], rng) -> np.ndarray:
    """Re-emit the path on the 8 ms clock with one donor's drop pattern.

    Every output timestamp is a clock multiple, so the downstream 125 Hz
    resample reads the original path unchanged except across dropped frames.
    The time axis is dilated by the small factor that puts the total duration
    on the clock; endpoint position is exact."""
    t = traj[:, 2] - traj[0, 2]
    total = t[-1]
    donor = donor_seqs[int(rng.integers(len(donor_seqs)))]
    k = int(rng.integers(len(donor)))
    ts = [0.0]
    while ts[-1] < max(total, CLOCK):
        ts.append(ts[-1] + float(donor[k % len(donor)]))
        k += 1
    # Round the duration to the nearest achievable tick, not always up.
    if len(ts) > 2 and abs(ts[-2] - total) < abs(ts[-1] - total):
        ts.pop()
    ts = np.asarray(ts)
    scale = ts[-1] / total
    x = np.interp(ts / scale, t, traj[:, 0])
    y = np.interp(ts / scale, t, traj[:, 1])
    return np.column_stack([x, y, ts])


def dt_hist_matrix(trajs) -> np.ndarray:
    rows = []
    for t in trajs:
        dts = np.diff(np.asarray(t, dtype=np.float64)[:, 2]) * 1e3
        dts = dts[dts > 0]
        h = np.histogram(dts, bins=DT_EDGES_MS)[0].astype(np.float64)
        h /= max(h.sum(), 1)
        rows.append(np.concatenate([h, [np.median(dts), dts.std()]]))
    return np.asarray(rows)


def rf_oob(a: np.ndarray, b: np.ndarray, seeds=(42, 43, 44)) -> float:
    n = min(len(a), len(b))
    X = np.vstack([a[:n], b[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    aucs = []
    for s in seeds:
        clf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                     n_jobs=-1, random_state=s)
        clf.fit(X, y)
        aucs.append(roc_auc_score(y, clf.oob_decision_function_[:, 1]))
    return float(np.mean(aucs))


def feat18(trajs) -> np.ndarray:
    rows = []
    for t in trajs:
        f = extract_features(resample_trajectory(np.asarray(t, dtype=np.float64)))
        if f is not None and np.all(np.isfinite(f)):
            rows.append(f)
    return np.asarray(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool_s42_k16.npz")
    ap.add_argument("--picks", required=True)
    ap.add_argument("--out", default=None,
                    help="save re-emitted trajectories npz here")
    args = ap.parse_args()

    d = np.load(args.pool, allow_pickle=True)
    picks = np.load(args.picks).astype(int)
    gen = [np.asarray(d["trajs"][ci], dtype=np.float64)
           for ci in picks if ci >= 0]
    print(f"selected set: {len(gen)} trajectories")

    hum = [np.asarray(t, dtype=np.float64) for t in load_human_raw(2000)]
    # Donor clocks come from a DIFFERENT random draw of the held-out split
    # than the detector's human class (seed 7 vs 42), so the detector never
    # compares a trajectory against its own transplanted clock. The two
    # draws of 2000 from 208k overlap by under 1 percent in expectation.
    donors = human_dt_sequences(
        [np.asarray(t, dtype=np.float64) for t in load_human_raw(2000, seed=7)])
    print(f"human donor clocks: {len(donors)} dt sequences (disjoint draw)")

    rng = np.random.default_rng(0)
    gen_re = [reemit(t, donors, rng) for t in gen]

    H = dt_hist_matrix(hum)
    print(f"raw-dt detector (RF OOB on dt histogram, n={len(gen)}/class):")
    print(f"  before re-emission: {rf_oob(H, dt_hist_matrix(gen)):.4f}")
    print(f"  after re-emission:  {rf_oob(H, dt_hist_matrix(gen_re)):.4f}")

    human_eval = np.load("data/human_eval_features.npy")
    print("18-feature RF OOB vs honest eval sample:")
    F0, F1 = feat18(gen), feat18(gen_re)
    print(f"  before re-emission: {rf_oob(human_eval, F0):.4f} (n={len(F0)})")
    print(f"  after re-emission:  {rf_oob(human_eval, F1):.4f} (n={len(F1)})")
    dmax = np.max(np.abs(np.median(F0, 0) - np.median(F1, 0))
                  / (np.abs(np.median(F0, 0)) + 1e-9))
    print(f"  max relative median feature shift: {dmax:.4f}")

    if args.out:
        np.savez_compressed(
            args.out, trajs=np.asarray([t.astype(np.float32) for t in gen_re],
                                       dtype=object))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
