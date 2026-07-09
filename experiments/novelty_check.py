"""Are the picks copies? A nearest-neighbor novelty check.

The headline (RF OOB 0.504, three seeds) shows a Random Forest cannot tell
the selected synthetic movements from held-out humans. A separate and more
basic question sits underneath that number: is the generator actually
producing new paths, or is the "selection" step just picking out training
trajectories the 6M-parameter model memorized and can reproduce verbatim?
If the latter were true, indistinguishability would be trivial and would
say nothing about generalization.

This script answers with a nearest-neighbor distance check in raw (x, y)
coordinate space, no kinematic features involved. For each of the 2000
headline-selected synthetic movements (seed 42, and 43/44 for confirmation),
it finds the closest trajectory in a large sample of the actual training
corpus and records the distance. The control repeats the identical search
for the 2000 held-out human evaluation movements against the same corpus
sample. If the generator were copying, synthetic-to-corpus distances would
cluster near zero, far below the human-to-corpus distances (real recordings
are not exact duplicates of each other). If synthetic distances land in the
same range as human distances, or higher, the generator is not memorizing.

Data and provenance:
  - training/full_pool_offsets.npy, pool_flat_i16.npy, pool_t_rel_f32.npy:
    the full 4.16M-trajectory training pool built by
    training/prepare_training_data.py. This is the actual corpus the
    generator trained on, across all five datasets.
  - The held-out human evaluation sample (data/human_eval_features.npy) is
    reproduced here from raw coordinates using the exact recipe in
    regenerate_human_features.py: np.random.default_rng(42).choice(n_pool,
    2000, replace=False) against the same full pool. This script recomputes
    18-feature vectors for those same indices and checks them against the
    cached file byte-for-byte (allclose) before trusting anything downstream,
    so the human raw trajectories used here are provably the same 2000
    movements the headline evaluates against.
  - Synthetic trajectories come straight from the cached candidate pools
    (pool_s{seed}_k16.npz, "trajs" array) and the winning pick index per
    spec (pool_s{seed}_k16_picks_trust33_f20d85_r30_rf.npy), the same
    files verify_headline.py replays. No sampler or checkpoint is touched.

Method: each trajectory is resampled to 64 points by arc length and
translated so it starts at the origin. Distance between two trajectories is
the RMS per-point Euclidean distance on this 64-point representation (which
is a monotonic rescaling of ordinary Euclidean distance on the flattened
128-dim vector, so nearest-neighbor search can use fast matrix operations).
A scale-normalized variant (each path divided by its own resampled end-point
distance from the origin) is also computed as a secondary check, since two
paths of very different length are never going to look close in raw pixels
regardless of shape.

The training corpus is subsampled uniformly at random to 200,000
trajectories (out of ~4.16M) for tractability; the SAME subsample is used
for both the synthetic and the human nearest-neighbor search, and the 2000
held-out human evaluation indices are excluded from it so the human control
cannot trivially match itself.

Sanity gate: 10 trajectories already in the corpus subsample are queried
against the full subsample including themselves. Distance must be exactly
0 and the nearest index must be the trajectory's own position -- otherwise
the resampling/indexing pipeline itself is broken and nothing downstream
can be trusted.

Run:
    .venv/Scripts/python.exe experiments/novelty_check.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 43, 44]
N_POINTS = 64
CORPUS_SIZE = 200_000
CORPUS_SEED = 123
EVAL_SEED = 42          # matches regenerate_human_features.py default
N_EVAL = 2000
N_DUP_CHECK = 10
CHUNK = 4000            # corpus rows per NN-search batch


# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------

def load_full_pool():
    offsets = np.load(ROOT / "training" / "full_pool_offsets.npy")
    flat = np.load(ROOT / "training" / "pool_flat_i16.npy", mmap_mode="r")
    t = np.load(ROOT / "training" / "pool_t_rel_f32.npy", mmap_mode="r")
    n_pool = len(offsets) - 1
    return offsets, flat, t, n_pool


def pool_xy(offsets, flat, idx: int) -> np.ndarray:
    s, e = int(offsets[idx]), int(offsets[idx + 1])
    return flat[s:e].astype(np.float64)


def pool_traj_for_features(offsets, flat, t, idx: int) -> list:
    s, e = int(offsets[idx]), int(offsets[idx + 1])
    xy = flat[s:e].astype(np.float64)
    ts = t[s:e].astype(np.float64)
    return [(float(xy[j, 0]), float(xy[j, 1]), float(ts[j])) for j in range(len(xy))]


# ---------------------------------------------------------------------------
# Resampling / normalization
# ---------------------------------------------------------------------------

def resample_arclength(xy: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    if len(xy) < 2:
        return np.repeat(xy[:1], n, axis=0)
    d = np.diff(xy, axis=0)
    seglen = np.hypot(d[:, 0], d[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(xy[:1], n, axis=0)
    target = np.linspace(0.0, total, n)
    x = np.interp(target, cum, xy[:, 0])
    y = np.interp(target, cum, xy[:, 1])
    return np.stack([x, y], axis=1)


def normalize(resampled: np.ndarray, scale: bool) -> np.ndarray:
    out = resampled - resampled[0]
    if scale:
        span = float(np.linalg.norm(out[-1]))
        if span < 1e-9:
            span = 1.0
        out = out / span
    return out


def build_vectors(xy_list, scale: bool) -> np.ndarray:
    vecs = np.empty((len(xy_list), N_POINTS * 2), dtype=np.float32)
    for i, xy in enumerate(xy_list):
        r = normalize(resample_arclength(xy), scale=scale)
        vecs[i] = r.reshape(-1)
    return vecs


# ---------------------------------------------------------------------------
# Nearest-neighbor search (chunked, exact)
# ---------------------------------------------------------------------------

def nn_search(query: np.ndarray, corpus: np.ndarray, chunk: int = CHUNK):
    """Exact nearest neighbor of each query row in corpus by Euclidean distance
    on the flattened vector. Returns (rms_per_point_dist, corpus_index)."""
    q = query.astype(np.float64)
    n_q = len(q)
    q_norm2 = (q ** 2).sum(axis=1)
    best_d2 = np.full(n_q, np.inf, dtype=np.float64)
    best_idx = np.full(n_q, -1, dtype=np.int64)
    for c0 in range(0, len(corpus), chunk):
        c = corpus[c0:c0 + chunk].astype(np.float64)
        c_norm2 = (c ** 2).sum(axis=1)
        cross = q @ c.T
        d2 = q_norm2[:, None] + c_norm2[None, :] - 2.0 * cross
        np.maximum(d2, 0.0, out=d2)
        local_idx = np.argmin(d2, axis=1)
        local_min = d2[np.arange(n_q), local_idx]
        better = local_min < best_d2
        best_d2[better] = local_min[better]
        best_idx[better] = local_idx[better] + c0
    rms = np.sqrt(best_d2 / (query.shape[1] // 2))
    return rms, best_idx


NEAR_DUP_THRESHOLDS = (1.0, 2.0, 5.0)
EXACT_TOL = 1e-6        # float-exact collision in the 64-point representation


def summarize(dist: np.ndarray) -> dict:
    out = {
        "n": int(len(dist)),
        "median": float(np.median(dist)),
        "p1": float(np.percentile(dist, 1)),
        "p99": float(np.percentile(dist, 99)),
        "min": float(np.min(dist)),
        "max": float(np.max(dist)),
    }
    for thr in NEAR_DUP_THRESHOLDS:
        out[f"frac_below_{thr:g}px"] = float(np.mean(dist < thr))
    # Collision counts: exact in float terms, and below a tenth of a pixel.
    # Both classes are expected to have some. The 64-point origin-translated
    # representation of a short, nearly straight movement has almost no
    # degrees of freedom, so distinct movements (human or synthetic) can
    # collide in it by coincidence; the characterization below verifies that
    # is what the collisions actually are.
    out["n_exact_collisions"] = int(np.sum(dist < EXACT_TOL))
    out["n_below_0.1px"] = int(np.sum(dist < 0.1))
    return out


def characterize_traj(xy: np.ndarray) -> dict:
    """Raw-trajectory shape summary: how many recorded points, how long the
    path is, and how straight it is (1.0 = perfectly straight)."""
    d = np.diff(xy, axis=0)
    seglen = np.hypot(d[:, 0], d[:, 1])
    path_len = float(seglen.sum())
    straight = float(np.hypot(xy[-1, 0] - xy[0, 0], xy[-1, 1] - xy[0, 1]))
    return {
        "n_points": int(len(xy)),
        "path_length_px": round(path_len, 2),
        "straight_dist_px": round(straight, 2),
        "path_efficiency": round(straight / max(path_len, 1e-9), 4),
    }


def collision_stats(chars: list) -> dict:
    """Aggregate shape stats over all exact-collision trajectories."""
    if not chars:
        return {"count": 0}
    n_pts = [c["n_points"] for c in chars]
    plen = [c["path_length_px"] for c in chars]
    eff = [c["path_efficiency"] for c in chars]
    return {
        "count": len(chars),
        "n_points_median": float(np.median(n_pts)),
        "n_points_max": int(max(n_pts)),
        "path_length_median_px": float(np.median(plen)),
        "path_length_max_px": float(max(plen)),
        "path_efficiency_median": float(np.median(eff)),
        "path_efficiency_min": float(min(eff)),
    }


def main() -> None:
    t_start = time.perf_counter()
    results: dict = {}

    print("Loading full training pool...")
    offsets, flat, t_arr, n_pool = load_full_pool()
    print(f"  n_pool = {n_pool}")
    results["n_pool"] = int(n_pool)

    # --- Reproduce the held-out human evaluation sample from raw coords ---
    print("Reconstructing the held-out human evaluation sample (raw coords)...")
    eval_rng = np.random.default_rng(EVAL_SEED)
    eval_indices = eval_rng.choice(n_pool, size=N_EVAL, replace=False)

    from features import extract_feature_matrix
    check_trajs = [pool_traj_for_features(offsets, flat, t_arr, int(i))
                   for i in eval_indices]
    recomputed_feats = extract_feature_matrix(check_trajs)
    cached_feats = np.load(ROOT / "data" / "human_eval_features.npy")
    feats_match = (recomputed_feats.shape == cached_feats.shape
                   and bool(np.allclose(recomputed_feats, cached_feats, atol=1e-6)))
    print(f"  reproduced feature match vs data/human_eval_features.npy: {feats_match}")
    results["human_eval_reproduction_matches_cache"] = feats_match
    if not feats_match:
        raise RuntimeError(
            "Reconstructed human eval sample does not match the cached "
            "features. Stopping rather than compare against the wrong rows.")

    human_xy = [pool_xy(offsets, flat, int(i)) for i in eval_indices]

    # --- Training corpus subsample, excluding the eval indices ---
    print(f"Building a {CORPUS_SIZE}-trajectory training corpus subsample "
          f"(seed {CORPUS_SEED}, eval indices excluded)...")
    mask = np.ones(n_pool, dtype=bool)
    mask[eval_indices] = False
    remaining = np.flatnonzero(mask)
    corpus_rng = np.random.default_rng(CORPUS_SEED)
    corpus_indices = corpus_rng.choice(remaining, size=CORPUS_SIZE, replace=False)
    results["corpus_size"] = int(CORPUS_SIZE)
    results["corpus_seed"] = CORPUS_SEED
    results["corpus_excludes_eval_indices"] = True

    t0 = time.perf_counter()
    corpus_xy = [pool_xy(offsets, flat, int(i)) for i in corpus_indices]
    print(f"  loaded corpus raw coords in {time.perf_counter() - t0:.1f}s")

    for scale in (False, True):
        tag = "scaled" if scale else "unscaled"
        print(f"\n=== {tag} variant ===")
        t0 = time.perf_counter()
        corpus_vecs = build_vectors(corpus_xy, scale=scale)
        print(f"  corpus vectors built in {time.perf_counter() - t0:.1f}s")

        # Sanity gate: 10 corpus members queried against the corpus that
        # contains them must come back at distance 0, matching their own row.
        dup_pos = np.arange(N_DUP_CHECK)
        dup_dist, dup_idx = nn_search(corpus_vecs[dup_pos], corpus_vecs)
        # atol accounts for float32 vector storage plus the norm-trick's
        # subtractive cancellation; real distances below are two to four
        # orders of magnitude larger, so 1e-3 still means "found itself".
        gate_ok = bool(np.allclose(dup_dist, 0.0, atol=1e-3)
                       and np.array_equal(dup_idx, dup_pos))
        print(f"  sanity gate (10 exact duplicates find themselves at 0): "
              f"{gate_ok} (max dist {dup_dist.max():.6f})")
        results[f"sanity_gate_{tag}"] = {
            "ok": gate_ok,
            "max_dist": float(dup_dist.max()),
        }
        if not gate_ok:
            raise RuntimeError(f"Sanity gate failed for {tag} variant; "
                               "pipeline is not trustworthy.")

        human_vecs = build_vectors(human_xy, scale=scale)
        human_dist, human_idx = nn_search(human_vecs, corpus_vecs)
        human_summary = summarize(human_dist)
        print(f"  human NN distances: median={human_summary['median']:.2f} "
              f"p1={human_summary['p1']:.2f} p99={human_summary['p99']:.2f} "
              f"min={human_summary['min']:.4f}")
        results[f"human_{tag}"] = human_summary
        human_p1 = human_summary["p1"]
        h_arg = int(np.argmin(human_dist))
        human_min_pair = {
            "dist": human_summary["min"],
            "eval_idx": int(eval_indices[h_arg]),
            "corpus_idx": int(corpus_indices[int(human_idx[h_arg])]),
        }
        results[f"human_closest_pair_{tag}"] = human_min_pair

        if not scale:
            results["human_closest_pair_char_unscaled"] = {
                "human": characterize_traj(human_xy[h_arg]),
                "corpus": characterize_traj(corpus_xy[int(human_idx[h_arg])]),
            }
            h_coll = np.flatnonzero(human_dist < EXACT_TOL)
            results["human_exact_collision_stats_unscaled"] = collision_stats(
                [characterize_traj(human_xy[int(i)]) for i in h_coll])
            print(f"  human exact collisions (<{EXACT_TOL:g}): {len(h_coll)}, "
                  f"below 0.1px: {int(np.sum(human_dist < 0.1))}")

        for seed in SEEDS:
            pool_path = ROOT / f"pool_s{seed}_k16.npz"
            picks_path = ROOT / f"pool_s{seed}_k16_picks_trust33_f20d85_r30_rf.npy"
            d = np.load(pool_path, allow_pickle=True)
            trajs = d["trajs"]
            picks = np.load(picks_path).astype(int)
            synth_xy = []
            valid_specs = []
            for spec_idx, ci in enumerate(picks):
                if ci < 0:
                    continue
                traj = np.asarray(trajs[ci], dtype=np.float64)[:, :2]
                synth_xy.append(traj)
                valid_specs.append(spec_idx)
            synth_vecs = build_vectors(synth_xy, scale=scale)
            synth_dist, synth_idx = nn_search(synth_vecs, corpus_vecs)
            synth_summary = summarize(synth_dist)
            frac_closer = float(np.mean(synth_dist < human_p1))
            print(f"  seed {seed} synthetic NN distances "
                  f"(n={synth_summary['n']}): median={synth_summary['median']:.2f} "
                  f"p1={synth_summary['p1']:.2f} p99={synth_summary['p99']:.2f} "
                  f"min={synth_summary['min']:.4f} | frac < human p1: {frac_closer:.4f}")
            results[f"synthetic_seed{seed}_{tag}"] = synth_summary
            results[f"synthetic_seed{seed}_{tag}"]["frac_closer_than_human_p1"] = frac_closer
            argmin = int(np.argmin(synth_dist))
            results[f"synthetic_seed{seed}_closest_pair_{tag}"] = {
                "dist": synth_summary["min"],
                "spec_idx": int(valid_specs[argmin]),
                "corpus_idx": int(corpus_indices[int(synth_idx[argmin])]),
            }
            if not scale:
                results[f"synthetic_seed{seed}_closest_pair_char_unscaled"] = {
                    "synthetic": characterize_traj(synth_xy[argmin]),
                    "corpus": characterize_traj(
                        corpus_xy[int(synth_idx[argmin])]),
                }
                s_coll = np.flatnonzero(synth_dist < EXACT_TOL)
                results[f"synthetic_seed{seed}_exact_collision_stats_unscaled"] = (
                    collision_stats(
                        [characterize_traj(synth_xy[int(i)]) for i in s_coll]))
                print(f"    seed {seed} exact collisions (<{EXACT_TOL:g}): "
                      f"{len(s_coll)}, below 0.1px: "
                      f"{int(np.sum(synth_dist < 0.1))}")

    results["elapsed_s"] = time.perf_counter() - t_start
    out_path = ROOT / "experiments" / "novelty_check_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {out_path}")
    print(f"Total runtime: {results['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
