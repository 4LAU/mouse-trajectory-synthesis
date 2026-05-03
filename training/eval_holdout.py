"""Holdout Discriminator Suite.

Provides 4 independent discriminators that are harder than the 18-feature RF
baseline.  Importable as a module (call ``run_holdout_eval``) or runnable
directly (``python3 -m training.eval_holdout``) to produce baseline numbers.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.spatial import KDTree
from scipy.stats import entropy
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

from features import extract_features, resample_trajectory

Trajectory = List[Tuple[float, float, float]]

# ---------------------------------------------------------------------------
# Paths -- adjust POOL_DIR to where your pool .npy files live
# ---------------------------------------------------------------------------
POOL_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data loaders (lazy, cached)
# ---------------------------------------------------------------------------
_cache: dict = {}


def _load_pool():
    """Return mmap'd pool arrays + offsets + meta (cached)."""
    if "pool" not in _cache:
        offsets = np.load(POOL_DIR / "full_pool_offsets.npy")
        meta = np.load(POOL_DIR / "full_pool_meta.npy")
        assert len(offsets) == len(meta) + 1, (
            f"pool data mismatch: {len(offsets)} offsets vs {len(meta)} meta rows"
        )
        _cache["pool"] = {
            "flat": np.load(POOL_DIR / "pool_flat_i16.npy", mmap_mode="r"),
            "t": np.load(POOL_DIR / "pool_t_rel_f32.npy", mmap_mode="r"),
            "offsets": offsets,
            "meta": meta,
        }
    return _cache["pool"]




def _load_human_distances() -> np.ndarray:
    if "hd" not in _cache:
        _cache["hd"] = np.load(POOL_DIR / "human_distances.npy")
    return _cache["hd"]


def _pool_trajectory(idx: int) -> Trajectory:
    """Extract a single trajectory from the mmap'd pool by index."""
    pool = _load_pool()
    lo, hi = int(pool["offsets"][idx]), int(pool["offsets"][idx + 1])
    xy = pool["flat"][lo:hi]
    t = pool["t"][lo:hi]
    return [(float(xy[i, 0]), float(xy[i, 1]), float(t[i])) for i in range(hi - lo)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two (possibly unnormalised) histograms."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        return 1.0
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))


def _traj_to_arrays(traj: Trajectory):
    """Return (x, y, t, speed, dt, ds, dx, dy) arrays from a trajectory."""
    pts = np.asarray(traj, dtype=np.float64)
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.maximum(np.diff(t), 1e-6)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    speed = ds / dt
    return x, y, t, speed, dt, ds, dx, dy


def _compute_distributional_stats(trajectories: List[Trajectory]):
    """Compute raw distributional arrays (velocity, accel, delay, curvature)."""
    all_vel, all_acc, all_delay, all_curv = [], [], [], []
    for traj in trajectories:
        if len(traj) < 5:
            continue
        _x, _y, _t, speed, dt, _ds, dx, dy = _traj_to_arrays(traj)
        all_vel.append(speed)
        all_delay.append(dt)

        dv = np.diff(speed)
        dt2 = np.maximum(dt[:-1], 1e-6)
        acc = np.abs(dv / dt2)
        if len(acc) > 0:
            all_acc.append(acc)

        # curvature
        vx = dx / dt
        vy = dy / dt
        if len(vx) > 1:
            ax = np.diff(vx) / dt2
            ay = np.diff(vy) / dt2
            sp = np.maximum(speed[:-1], 1e-6)
            cross = np.abs(vx[:-1] * ay - vy[:-1] * ax)
            curv = np.clip(cross / (sp ** 3), 0, 1e6)  # 0 ok: not fed into log10
            all_curv.append(curv)

    vel = np.concatenate(all_vel) if all_vel else np.array([1.0])
    acc = np.concatenate(all_acc) if all_acc else np.array([1.0])
    delay = np.concatenate(all_delay) if all_delay else np.array([0.01])
    curv = np.concatenate(all_curv) if all_curv else np.array([1e-5])
    return vel, acc, delay, curv


# ---------------------------------------------------------------------------
# Novel 10-feature extraction (Discriminator B)
# ---------------------------------------------------------------------------

NOVEL_FEATURE_NAMES = [
    "pause_count",
    "overshoot_magnitude",
    "submovements",
    "velocity_autocorrelation_lag1",
    "tangential_jerk_rms",
    "perpendicular_deviation_rms",
    "click_approach_deceleration",
    "curvature_entropy",
    "speed_symmetry",
    "micro_correction_count",
]


def extract_novel_features(traj: Trajectory) -> np.ndarray | None:
    """Extract the 10 novel features from a raw trajectory.  Returns None if too short."""
    if len(traj) < 5:
        return None
    pts = np.asarray(traj, dtype=np.float64)
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.maximum(np.diff(t), 1e-6)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    speed = ds / dt
    n = len(speed)

    # 1. pause_count: segments with speed < 5 px/s for > 50ms
    paused = speed < 5.0
    pause_count = 0
    in_pause = False
    pause_start = 0.0
    for i in range(n):
        if paused[i]:
            if not in_pause:
                in_pause = True
                pause_start = t[i]
        else:
            if in_pause:
                if t[i] - pause_start > 0.05:
                    pause_count += 1
                in_pause = False
    if in_pause and t[-1] - pause_start > 0.05:
        pause_count += 1

    # 2. overshoot_magnitude: max distance past endpoint along start->end axis
    sx, sy = x[0], y[0]
    ex, ey = x[-1], y[-1]
    line_len = math.hypot(ex - sx, ey - sy)
    if line_len > 1e-6:
        ux, uy = (ex - sx) / line_len, (ey - sy) / line_len
        proj = (x - sx) * ux + (y - sy) * uy
        overshoot = float(np.max(proj) - line_len)
        overshoot = max(overshoot, 0.0)
    else:
        overshoot = 0.0

    # 3. submovements: velocity peaks with prominence > 10% of max speed
    max_speed = float(np.max(speed)) if n > 0 else 1.0
    threshold = 0.1 * max_speed
    submovements = 0
    if n >= 3:
        for i in range(1, n - 1):
            if speed[i] > speed[i - 1] and speed[i] > speed[i + 1]:
                # Check prominence: min drop on either side
                left_min = float(np.min(speed[:i])) if i > 0 else 0.0
                right_min = float(np.min(speed[i + 1:])) if i < n - 1 else 0.0
                prominence = speed[i] - max(left_min, right_min)
                if prominence > threshold:
                    submovements += 1

    # 4. velocity_autocorrelation_lag1
    if n > 1:
        std_s = np.std(speed)
        if std_s > 1e-10:
            vel_ac = float(np.corrcoef(speed[:-1], speed[1:])[0, 1])
            if np.isnan(vel_ac):
                vel_ac = 0.0
        else:
            vel_ac = 1.0
    else:
        vel_ac = 0.0

    # 5. tangential_jerk_rms: jerk projected onto movement direction
    if n >= 3:
        vx = dx / dt
        vy = dy / dt
        ax_arr = np.diff(vx) / np.maximum(dt[:-1], 1e-6)
        ay_arr = np.diff(vy) / np.maximum(dt[:-1], 1e-6)
        if len(ax_arr) >= 2:
            jx = np.diff(ax_arr) / np.maximum(dt[:-2], 1e-6)
            jy = np.diff(ay_arr) / np.maximum(dt[:-2], 1e-6)
            # tangent direction at each jerk point
            sp_j = np.maximum(speed[:-2], 1e-6)
            tx_hat = vx[:-2] / sp_j
            ty_hat = vy[:-2] / sp_j
            tang_jerk = jx * tx_hat + jy * ty_hat
            tang_jerk_rms = float(np.sqrt(np.mean(tang_jerk ** 2)))
        else:
            tang_jerk_rms = 0.0
    else:
        tang_jerk_rms = 0.0

    # 6. perpendicular_deviation_rms (normalised by distance)
    if line_len > 1e-6:
        perp = np.abs(uy * (x - sx) - ux * (y - sy))
        perp_rms = float(np.sqrt(np.mean(perp ** 2))) / line_len
    else:
        perp_rms = 0.0

    # 7. click_approach_deceleration: mean speed last 10% / peak speed
    last_10_start = max(1, int(n * 0.9))
    if max_speed > 1e-6:
        click_decel = float(np.mean(speed[last_10_start:])) / max_speed
    else:
        click_decel = 1.0

    # 8. curvature_entropy: Shannon entropy of curvature histogram (10 bins)
    if n >= 3:
        vx = dx / dt
        vy = dy / dt
        dt2 = np.maximum(dt[:-1], 1e-6)
        ax_arr = np.diff(vx) / dt2
        ay_arr = np.diff(vy) / dt2
        sp_mid = np.maximum(speed[:-1], 1e-6)
        cross = np.abs(vx[:-1] * ay_arr - vy[:-1] * ax_arr)
        curv = np.clip(cross / (sp_mid ** 3), 1e-10, 1e6)  # 1e-10: next line calls np.log10, so 0 would be -inf
        log_curv = np.log10(curv)
        hist, _ = np.histogram(log_curv, bins=10)
        hist = hist.astype(np.float64)
        h_sum = hist.sum()
        if h_sum > 0:
            hist_p = hist / h_sum
            curv_ent = float(entropy(hist_p, base=2))
        else:
            curv_ent = 0.0
    else:
        curv_ent = 0.0

    # 9. speed_symmetry: argmax on raw speed / total duration index
    if n > 0:
        speed_sym = float(np.argmax(speed)) / max(n - 1, 1)
    else:
        speed_sym = 0.5

    # 10. micro_correction_count: direction sign changes in last 20% via cross product
    last_20_start = max(1, int(n * 0.8))
    dx_tail = dx[last_20_start:]
    dy_tail = dy[last_20_start:]
    if len(dx_tail) >= 2:
        cross_prod = dx_tail[:-1] * dy_tail[1:] - dy_tail[:-1] * dx_tail[1:]
        signs = np.sign(cross_prod)
        micro_corr = int(np.sum(np.diff(signs) != 0))
    else:
        micro_corr = 0

    return np.array([
        pause_count,
        overshoot,
        submovements,
        vel_ac,
        tang_jerk_rms,
        perp_rms,
        click_decel,
        curv_ent,
        speed_sym,
        micro_corr,
    ], dtype=np.float64)


def _extract_novel_matrix(trajectories: List[Trajectory]) -> np.ndarray:
    """Extract novel feature matrix, skipping bad trajectories."""
    rows = []
    for traj in trajectories:
        f = extract_novel_features(traj)
        if f is not None and not np.any(np.isnan(f)):
            rows.append(f)
    return np.array(rows, dtype=np.float64) if rows else np.empty((0, 10))


def _extract_aligned_features(
    trajectories: List[Trajectory],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract both original (18) and novel (10) features, row-aligned.

    Only keeps rows where BOTH feature sets succeed, guaranteeing row k
    in the original matrix and row k in the novel matrix correspond to the
    same source trajectory.
    """
    orig_rows = []
    novel_rows = []
    for traj in trajectories:
        f_orig = extract_features(resample_trajectory(traj))
        f_novel = extract_novel_features(traj)
        if (f_orig is not None and not np.any(np.isnan(f_orig))
                and f_novel is not None and not np.any(np.isnan(f_novel))):
            orig_rows.append(f_orig)
            novel_rows.append(f_novel)
    orig = np.array(orig_rows, dtype=np.float64) if orig_rows else np.empty((0, 18))
    novel = np.array(novel_rows, dtype=np.float64) if novel_rows else np.empty((0, 10))
    return orig, novel


# ---------------------------------------------------------------------------
# Discrete Frechet distance  /  DTW distance
#
# Benchmark (500 calls, trajectory length 50, numpy 1.x):
#   _discrete_frechet: Python loops 0.63 s → numpy col-loop 0.30 s (2.1×)
#   _dtw_distance:     Python loops 0.67 s → numpy col-loop 0.30 s (2.2×)
# Bottleneck is the DP column loop (left-cell dependency prevents full 2-D
# vectorization); numba/Cython would be needed for 5×+ improvement.
# ---------------------------------------------------------------------------

def _pairwise_dist(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    diff = P[:, None, :] - Q[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    """Discrete Fréchet distance between 2D point sequences P and Q.
    P, Q: (n, 2) and (m, 2) arrays.

    The pairwise distance matrix is pre-computed with numpy broadcasting,
    eliminating n×m individual sqrt calls.  The DP table is then filled with
    two nested Python loops (still O(n×m) iterations, but with a reduced
    constant factor).  For 5×+ speedup, numba or Cython would be needed.
    """
    if len(P) == 0 or len(Q) == 0:
        raise ValueError("_discrete_frechet requires non-empty trajectories")
    n, m = len(P), len(Q)

    # Full pairwise Euclidean distance matrix in one numpy call
    dist = _pairwise_dist(P, Q)  # (n, m)

    ca = np.empty((n, m), dtype=np.float64)
    ca[:, 0] = np.maximum.accumulate(dist[:, 0])   # first column
    ca[0, :] = np.maximum.accumulate(dist[0, :])   # first row

    # ca[i,j] = max(dist[i,j], min(ca[i-1,j], ca[i,j-1], ca[i-1,j-1]))
    # Left dependency (ca[i,j-1]) forces one Python loop per column.
    for j in range(1, m):
        prev_col = ca[:, j - 1]
        for i in range(1, n):
            ca[i, j] = max(min(ca[i - 1, j], prev_col[i], ca[i - 1, j - 1]),
                           dist[i, j])

    return float(ca[n - 1, m - 1])


def _dtw_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Simple DTW distance between 2D sequences.

    The pairwise distance matrix is pre-computed with numpy broadcasting,
    eliminating n×m individual sqrt calls.  The DP table is then filled with
    two nested Python loops (still O(n×m) iterations, but with a reduced
    constant factor).  For 5×+ speedup, numba or Cython would be needed.
    """
    if len(P) == 0 or len(Q) == 0:
        raise ValueError("_dtw_distance requires non-empty trajectories")
    n, m = len(P), len(Q)

    # Full pairwise Euclidean distance matrix in one numpy call
    dist = _pairwise_dist(P, Q)  # (n, m)

    # DTW cost table with sentinel border (size (n+1) × (m+1))
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    # dtw[i,j] = dist[i-1,j-1] + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    # Left dependency (dtw[i,j-1]) forces one Python loop per column.
    for j in range(1, m + 1):
        prev_col = dtw[:, j - 1]
        for i in range(1, n + 1):
            dtw[i, j] = dist[i - 1, j - 1] + min(dtw[i - 1, j], prev_col[i], prev_col[i - 1])

    return float(dtw[n, m])


# ---------------------------------------------------------------------------
# Discriminator A -- Raw Distributional JSD
# ---------------------------------------------------------------------------

def _disc_a(human_trajs: List[Trajectory], synth_trajs: List[Trajectory], verbose: bool) -> dict:
    t0 = time.time()
    h_vel, h_acc, h_delay, h_curv = _compute_distributional_stats(human_trajs)
    s_vel, s_acc, s_delay, s_curv = _compute_distributional_stats(synth_trajs)

    # Velocity: 50 bins log-spaced 1..10000
    vel_bins = np.logspace(0, 4, 51)
    h_vel_hist, _ = np.histogram(h_vel, bins=vel_bins)
    s_vel_hist, _ = np.histogram(s_vel, bins=vel_bins)
    vel_jsd = _jsd(h_vel_hist, s_vel_hist)

    # Acceleration: 50 bins log-spaced 1..100000
    acc_bins = np.logspace(0, 5, 51)
    h_acc_hist, _ = np.histogram(h_acc, bins=acc_bins)
    s_acc_hist, _ = np.histogram(s_acc, bins=acc_bins)
    acc_jsd = _jsd(h_acc_hist, s_acc_hist)

    # Delay: 50 bins linear 0..0.1
    delay_bins = np.linspace(0, 0.1, 51)
    h_del_hist, _ = np.histogram(h_delay, bins=delay_bins)
    s_del_hist, _ = np.histogram(s_delay, bins=delay_bins)
    delay_jsd = _jsd(h_del_hist, s_del_hist)

    # Curvature: 30 bins log-spaced 1e-6..1.0
    curv_bins = np.logspace(-6, 0, 31)
    h_cur_hist, _ = np.histogram(h_curv, bins=curv_bins)
    s_cur_hist, _ = np.histogram(s_curv, bins=curv_bins)
    curv_jsd = _jsd(h_cur_hist, s_cur_hist)

    mean_jsd = (vel_jsd + acc_jsd + delay_jsd + curv_jsd) / 4.0

    if verbose:
        print(f"Discriminator A (Raw Distributional)  [{time.time() - t0:.1f}s]:")
        print(f"  velocity_jsd:     {vel_jsd:.4f}")
        print(f"  acceleration_jsd: {acc_jsd:.4f}")
        print(f"  delay_jsd:        {delay_jsd:.4f}")
        print(f"  curvature_jsd:    {curv_jsd:.4f}")
        print(f"  mean_jsd:         {mean_jsd:.4f}")

    return {
        "velocity_jsd": vel_jsd,
        "acceleration_jsd": acc_jsd,
        "delay_jsd": delay_jsd,
        "curvature_jsd": curv_jsd,
        "mean_jsd": mean_jsd,
    }


# ---------------------------------------------------------------------------
# Discriminator B -- Novel Feature RF
# ---------------------------------------------------------------------------

def _disc_b(human_novel: np.ndarray, synth_novel: np.ndarray, verbose: bool) -> dict:
    t0 = time.time()
    n_h, n_s = len(human_novel), len(synth_novel)
    X = np.vstack([human_novel, synth_novel])
    y = np.concatenate([np.ones(n_h), np.zeros(n_s)])

    # Replace any remaining nans/infs
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    auc = float(np.mean(scores))

    if verbose:
        print(f"\nDiscriminator B (Novel Features RF)  [{time.time() - t0:.1f}s]:")
        print(f"  holdout_auc: {auc:.4f}  (per-fold: {', '.join(f'{s:.3f}' for s in scores)})")

    return {"holdout_auc": auc}


# ---------------------------------------------------------------------------
# Discriminator C -- Union GBM
# ---------------------------------------------------------------------------

def _disc_c(human_orig: np.ndarray, human_novel: np.ndarray,
            synth_orig: np.ndarray, synth_novel: np.ndarray, verbose: bool) -> dict:
    t0 = time.time()
    # Rows are guaranteed aligned by _extract_aligned_features
    assert len(human_orig) == len(human_novel), "human feature matrices not aligned"
    assert len(synth_orig) == len(synth_novel), "synth feature matrices not aligned"
    n_h = len(human_orig)
    n_s = len(synth_orig)
    h_all = np.hstack([human_orig, human_novel])
    s_all = np.hstack([synth_orig, synth_novel])

    X = np.vstack([h_all, s_all])
    y = np.concatenate([np.ones(n_h), np.zeros(n_s)])
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    auc = float(np.mean(scores))

    if verbose:
        print(f"\nDiscriminator C (Union GBM)  [{time.time() - t0:.1f}s]:")
        print(f"  union_auc: {auc:.4f}  (per-fold: {', '.join(f'{s:.3f}' for s in scores)})")

    return {"union_auc": auc}


# ---------------------------------------------------------------------------
# Discriminator D -- Novelty Check
# ---------------------------------------------------------------------------

def _build_pool_summary(max_pool: int = 500_000) -> tuple:
    """Build KD-tree summary stats from pool.  Subsample to keep memory sane."""
    pool = _load_pool()
    offsets = pool["offsets"]
    meta = pool["meta"]
    N = len(meta)

    # Subsample
    rng = np.random.default_rng(42)
    if N > max_pool:
        indices = rng.choice(N, max_pool, replace=False)
        indices.sort()
    else:
        indices = np.arange(N)

    flat = pool["flat"]
    t_arr = pool["t"]

    summaries = np.empty((len(indices), 5), dtype=np.float64)
    for k, idx in enumerate(indices):
        lo, hi = int(offsets[idx]), int(offsets[idx + 1])
        n_pts = hi - lo
        if n_pts < 2:
            summaries[k] = [1.0, 0.0, 0.0, 0.0, 0.0]
            continue
        sx, sy = float(flat[lo, 0]), float(flat[lo, 1])
        ex, ey = float(flat[hi - 1, 0]), float(flat[hi - 1, 1])
        t0_val, t1_val = float(t_arr[lo]), float(t_arr[hi - 1])
        d_straight = math.hypot(ex - sx, ey - sy)
        # path_efficiency
        xy_seg = flat[lo:hi].astype(np.float64)
        ds = np.sqrt(np.sum(np.diff(xy_seg, axis=0) ** 2, axis=1))
        d_travel = float(np.sum(ds))
        path_eff = d_straight / max(d_travel, 1e-6)
        duration = t1_val - t0_val
        # peak velocity fraction
        dt_seg = np.maximum(np.diff(t_arr[lo:hi].astype(np.float64)), 1e-6)
        speed_seg = ds / dt_seg
        peak_v = float(np.max(speed_seg))
        mean_v = float(np.mean(speed_seg))
        peak_frac = mean_v / max(peak_v, 1e-6)
        summaries[k] = [path_eff, duration, peak_frac, ex, ey]
    return summaries, indices


def _traj_summary(traj: Trajectory) -> np.ndarray:
    """Same 5 summary stats as pool."""
    pts = np.asarray(traj, dtype=np.float64)
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    d_straight = math.hypot(x[-1] - x[0], y[-1] - y[0])
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    d_travel = float(np.sum(ds))
    path_eff = d_straight / max(d_travel, 1e-6)
    duration = t[-1] - t[0]
    dt = np.maximum(np.diff(t), 1e-6)
    speed = ds / dt
    peak_v = float(np.max(speed))
    mean_v = float(np.mean(speed))
    peak_frac = mean_v / max(peak_v, 1e-6)
    return np.array([path_eff, duration, peak_frac, x[-1], y[-1]], dtype=np.float64)


def _disc_d(synth_trajs: List[Trajectory], generate_fn: Callable, stochastic: bool,
            verbose: bool) -> dict:
    t0 = time.time()
    if verbose:
        print("\nDiscriminator D (Novelty):  building pool KD-tree...", end=" ", flush=True)

    summaries, pool_indices = _build_pool_summary(max_pool=500_000)
    tree = KDTree(summaries)
    if verbose:
        print(f"done ({len(summaries)} pool entries)")

    pool = _load_pool()
    offsets = pool["offsets"]
    flat = pool["flat"]

    # For each synthetic trajectory, find 20 nearest pool trajectories and compute Frechet
    frechet_dists = []
    _bench_calls = 0
    _bench_elapsed = 0.0
    for i, traj in enumerate(synth_trajs):
        if len(traj) < 3:
            continue
        q = _traj_summary(traj)
        _, nn_idx = tree.query(q, k=20)
        if np.ndim(nn_idx) == 0:
            nn_idx = [int(nn_idx)]

        synth_xy = np.asarray(traj, dtype=np.float64)[:, :2]
        min_frechet = float("inf")
        for ni in nn_idx:
            pool_real_idx = int(pool_indices[ni])
            lo, hi = int(offsets[pool_real_idx]), int(offsets[pool_real_idx + 1])
            pool_xy = flat[lo:hi].astype(np.float64)
            # Subsample long trajectories for speed (cap at 200 points)
            if len(synth_xy) > 200:
                s_sub = synth_xy[np.linspace(0, len(synth_xy) - 1, 200, dtype=int)]
            else:
                s_sub = synth_xy
            if len(pool_xy) > 200:
                p_sub = pool_xy[np.linspace(0, len(pool_xy) - 1, 200, dtype=int)]
            else:
                p_sub = pool_xy
            _t_call = time.perf_counter()
            fd = _discrete_frechet(s_sub, p_sub)
            _bench_elapsed += time.perf_counter() - _t_call
            _bench_calls += 1
            if fd < min_frechet:
                min_frechet = fd
        frechet_dists.append(min_frechet)

        if verbose and (i + 1) % 200 == 0:
            print(f"  Frechet: {i + 1}/{len(synth_trajs)}", flush=True)

    frechet_arr = np.array(frechet_dists) if frechet_dists else np.array([0.0])
    mean_frechet = float(np.mean(frechet_arr))
    near_dup_pct = float(np.mean(frechet_arr < 2.0) * 100.0)

    result = {
        "mean_frechet_nn": mean_frechet,
        "near_duplicate_pct": near_dup_pct,
    }

    if verbose:
        print(f"  mean_frechet_nn:    {mean_frechet:.1f} px")
        print(f"  near_duplicate_pct: {near_dup_pct:.1f}%")

    # Stochastic repeat test
    if stochastic:
        if verbose:
            print("  Running stochastic repeat test (100 identical queries)...", flush=True)
        repeat_trajs = []
        for _ in range(100):
            rt = generate_fn(500.0, 400.0, 800.0, 600.0)
            repeat_trajs.append(np.asarray(rt, dtype=np.float64)[:, :2])

        # Pairwise DTW on subsampled versions
        n_rep = len(repeat_trajs)
        dtw_vals = []
        for i in range(min(n_rep, 50)):  # cap pairwise comparisons
            ri = repeat_trajs[i]
            if len(ri) > 100:
                ri = ri[np.linspace(0, len(ri) - 1, 100, dtype=int)]
            for j in range(i + 1, min(n_rep, 50)):
                rj = repeat_trajs[j]
                if len(rj) > 100:
                    rj = rj[np.linspace(0, len(rj) - 1, 100, dtype=int)]
                _t_call = time.perf_counter()
                dtw_vals.append(_dtw_distance(ri, rj))
                _bench_elapsed += time.perf_counter() - _t_call
                _bench_calls += 1
        dtw_arr = np.array(dtw_vals) if dtw_vals else np.array([0.0])
        repeat_dup_pct = float(np.mean(dtw_arr < 5.0) * 100.0)
        result["repeat_duplicate_pct"] = repeat_dup_pct
        if verbose:
            print(f"  repeat_duplicate_pct: {repeat_dup_pct:.1f}%")

    _bench_avg_ms = (_bench_elapsed / _bench_calls * 1000) if _bench_calls else 0.0
    if verbose:
        print(f"[benchmark] Frechet/DTW: {_bench_calls} calls in {_bench_elapsed:.2f}s "
              f"(avg {_bench_avg_ms:.2f}ms per call)")

    if verbose:
        print(f"  [{time.time() - t0:.1f}s total]")

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _sample_human_trajs(n: int, rng: np.random.Generator) -> List[Trajectory]:
    """Sample n human trajectories from the pool."""
    pool = _load_pool()
    N = len(pool["offsets"]) - 1
    indices = rng.choice(N, min(n, N), replace=False)
    trajs = []
    for idx in indices:
        trajs.append(_pool_trajectory(int(idx)))
    return trajs


def _generate_synthetic(generate_fn: Callable, n: int, verbose: bool) -> List[Trajectory]:
    """Generate n synthetic trajectories using human distance distribution."""
    distances = _load_human_distances()
    rng = np.random.default_rng(123)
    # Sample distances
    dist_indices = rng.choice(len(distances), n, replace=True)

    trajs = []
    for i, di in enumerate(dist_indices):
        d = float(distances[di])
        angle = rng.uniform(0, 2 * math.pi)
        sx, sy = 960.0, 540.0  # center of 1920x1080
        ex = sx + d * math.cos(angle)
        ey = sy + d * math.sin(angle)
        traj = generate_fn(sx, sy, ex, ey)
        trajs.append(traj)
        if verbose and (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{n} trajectories", flush=True)
    return trajs


def run_holdout(
    generate_fn: Callable,
    n_synthetic: int = 2000,
    stochastic: bool = True,
    verbose: bool = True,
    skip_disc_d: bool = False,
) -> dict:
    """Run all 4 discriminators.  Returns dict with all metrics.

    Args:
        skip_disc_d: If True, skip Discriminator D (novelty/Frechet) for speed.
    """
    rng = np.random.default_rng(42)
    results: Dict[str, float] = {}

    if verbose:
        print("=== Holdout Assessment ===\n")

    # --- Generate data ---
    if verbose:
        print(f"Sampling {n_synthetic} human trajectories from pool...")
    human_trajs = _sample_human_trajs(n_synthetic, rng)

    if verbose:
        print(f"Generating {n_synthetic} synthetic trajectories...")
    synth_trajs = _generate_synthetic(generate_fn, n_synthetic, verbose)

    # --- Discriminator A ---
    if verbose:
        print()
    a_results = _disc_a(human_trajs, synth_trajs, verbose)
    results.update({f"disc_a_{k}": v for k, v in a_results.items()})

    # --- Feature extraction for B and C (row-aligned) ---
    if verbose:
        print("\nExtracting aligned features (human)...", flush=True)
    human_orig, human_novel = _extract_aligned_features(human_trajs)
    if verbose:
        print(f"  {len(human_orig)} valid rows (both original + novel)")
        print("Extracting aligned features (synthetic)...", flush=True)
    synth_orig, synth_novel = _extract_aligned_features(synth_trajs)
    if verbose:
        print(f"  {len(synth_orig)} valid rows (both original + novel)")

    # --- Discriminator B ---
    b_results = _disc_b(human_novel, synth_novel, verbose)
    results.update({f"disc_b_{k}": v for k, v in b_results.items()})

    # --- Discriminator C ---
    c_results = _disc_c(human_orig, human_novel, synth_orig, synth_novel, verbose)
    results.update({f"disc_c_{k}": v for k, v in c_results.items()})

    # --- Discriminator D ---
    if skip_disc_d:
        if verbose:
            print("\nDiscriminator D (Novelty):  SKIPPED (skip_disc_d=True)")
    else:
        d_results = _disc_d(synth_trajs, generate_fn, stochastic, verbose)
        results.update({f"disc_d_{k}": v for k, v in d_results.items()})

    # --- Summary ---
    if verbose:
        print("\n=== BASELINE VALUES ===")
        print(f"disc_a_mean_jsd={results['disc_a_mean_jsd']:.4f}")
        print(f"disc_b_auc={results['disc_b_holdout_auc']:.4f}")
        print(f"disc_c_auc={results['disc_c_union_auc']:.4f}")
        if "disc_d_mean_frechet_nn" in results:
            print(f"disc_d_mean_frechet={results['disc_d_mean_frechet_nn']:.1f}")
            print(f"disc_d_near_dup_pct={results['disc_d_near_duplicate_pct']:.1f}")
            if "disc_d_repeat_duplicate_pct" in results:
                print(f"disc_d_repeat_dup_pct={results['disc_d_repeat_duplicate_pct']:.1f}")

    return results


# Keep backward-compatible alias
run_holdout_eval = run_holdout


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # To run standalone, you need a generate_path function.
    # Example: from your experiment module
    #   from experiment import generate_path
    #   run_holdout(generate_path, n_synthetic=2000, stochastic=True, verbose=True)
    print("Usage: import run_holdout from this module and pass a generate_fn callable.")
    print("  generate_fn(sx, sy, ex, ey) -> List[Tuple[x, y, t]]")
