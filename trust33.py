"""Widen the trust-region judge to close the raw-signal gap.

The external detector suite showed the trust-selected set is at chance for
every feature-based family (0.485-0.536) but a raw-signal detector (per-channel
autocorrelations, spectral centroid, accel zero-crossings on speed/vx/vy)
still scores 0.583. The selection judge only ever saw the 18 hand-crafted
features, so it had no way to care. Fix: give the judge those raw features
too (18 + 15 = 33 dims) and rerun the same trust loop offline.

Reference honesty: the 33-dim human reference is rebuilt from the raw pool
with the eval sample's indices (rng seed 42, n=2000, the draw
regenerate_human_features.py makes) explicitly excluded, so nothing the
honest evaluator uses as its human class ever touches selection.

Run:
    .venv/Scripts/python.exe trust33.py --pool pool_s42_k16.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from external_detectors import _manual_mini_features, raw_traj_to_channels
from features import extract_features, resample_trajectory
from selection_lab import Pool, pick_trust, rf_proxy_auc

CONFIGS = [
    ("f20d85_r30_rf", dict(rounds=30, frac=0.20, judge="rf", frac_decay=0.85)),
    ("f05_r30_rf", dict(rounds=30, frac=0.05, judge="rf")),
]


def raw15(traj) -> np.ndarray:
    ch = raw_traj_to_channels(traj)
    if ch is None:
        return np.zeros(15)
    speed, vx, vy = ch
    row = np.asarray(_manual_mini_features(speed)
                     + _manual_mini_features(vx)
                     + _manual_mini_features(vy), dtype=np.float64)
    return np.where(np.isfinite(row), row, 0.0)


class Pool33(Pool):
    """Pool whose feature matrix is the 18 hand-crafted features plus the
    15 raw-signal features, cached per pool file."""

    def __init__(self, path):
        super().__init__(path)
        cache = Path(path.replace(".npz", "_raw15.npy"))
        if cache.exists():
            R = np.load(cache)
        else:
            d = np.load(path, allow_pickle=True)
            trajs = d["trajs"]
            t0 = time.time()
            R = np.asarray([raw15(np.asarray(tr, dtype=np.float64))
                            for tr in trajs])
            np.save(cache, R)
            print(f"  raw15 for {len(R):,} candidates in "
                  f"{time.time() - t0:.0f}s -> {cache}")
        assert len(R) == len(self.X), (len(R), len(self.X))
        self.X = np.hstack([self.X, R])


def build_ref33(train_dir="./training", n_ref=4000, out="data/human_ref33.npy",
                eval_seed=42, n_eval=2000, ref_seed=7) -> np.ndarray:
    if Path(out).exists():
        return np.load(out)
    offsets = np.load(f"{train_dir}/full_pool_offsets.npy")
    flat = np.load(f"{train_dir}/pool_flat_i16.npy", mmap_mode="r")
    t = np.load(f"{train_dir}/pool_t_rel_f32.npy", mmap_mode="r")
    n_pool = len(offsets) - 1
    eval_idx = np.random.default_rng(eval_seed).choice(
        n_pool, size=n_eval, replace=False)
    avail = np.setdiff1d(np.arange(n_pool), eval_idx)
    idx = np.random.default_rng(ref_seed).choice(
        avail, size=n_ref + 500, replace=False)
    t0, rows = time.time(), []
    for i in idx:
        s, e = int(offsets[i]), int(offsets[i + 1])
        traj = np.column_stack([flat[s:e].astype(np.float64),
                                t[s:e].astype(np.float64)])
        f18 = extract_features(resample_trajectory(traj))
        if f18 is None or np.any(~np.isfinite(f18)):
            continue
        rows.append(np.concatenate([f18, raw15(traj)]))
        if len(rows) >= n_ref:
            break
    R = np.asarray(rows)
    np.save(out, R)
    print(f"built {out}: {R.shape} in {time.time() - t0:.0f}s "
          f"(eval indices excluded, ref seed {ref_seed})")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool_s42_k16.npz")
    ap.add_argument("--init-picks", default=None,
                    help="full picks .npy to start from "
                         "(default <pool>_picks_sir.npy)")
    args = ap.parse_args()

    ref = build_ref33()
    perm = np.random.default_rng(0).permutation(len(ref))
    half = len(ref) // 2
    ref_a, ref_b = ref[perm[:half]], ref[perm[half:]]
    print(f"reference: {len(ref_a)} fit rows (A), {len(ref_b)} proxy rows (B), "
          f"{ref.shape[1]} dims")

    pool = Pool33(args.pool)
    prefix = args.pool.replace(".npz", "")
    init_path = args.init_picks or f"{prefix}_picks_sir.npy"
    full = np.load(init_path).astype(int)
    init = {int(i): int(ci) for i, ci in enumerate(full) if ci >= 0}
    print(f"init {init_path}: proxy33 vs B = "
          f"{rf_proxy_auc(pool.selected(init), ref_b):.4f}")

    for name, kw in CONFIGS:
        t0 = time.time()
        picks, auc = pick_trust(pool, ref_a, ref_b, init,
                                label=f"trust33_{name}", **kw)
        out = f"{prefix}_picks_trust33_{name}.npy"
        np.save(out, pool.picks_to_full(picks))
        print(f"trust33_{name}: proxy33 {auc:.4f} "
              f"({time.time() - t0:.0f}s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
