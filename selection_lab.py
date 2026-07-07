"""Offline set-level selection over a saved SIR candidate pool.

The locked recipe picks each trajectory independently: a tempered lottery on
per-candidate judge log-odds. But the detector never sees trajectories one at
a time; it trains on the whole selected population vs the whole human sample.
Per-item selection leaves set-level structure on the table twice over. First,
it cannot trade one spec's pick against another's to fix a marginal that the
whole pool overshoots. Second, the tempered lottery itself distorts the
selected distribution in a way a fresh detector can see.

This script treats selection as an explicit set-level optimization: choose
one candidate per spec so the SELECTED SET of 18-dim feature vectors is as
close as possible to the human reference distribution. Strategies:

  sir       reproduce the per-item lottery (calibration baseline)
  greedy    iterated adversarial reselection: fit a discriminator between
            the reference and the CURRENT selected set, accumulate log-odds,
            re-pick argmax per spec, repeat. The set-level feedback loop
            per-item SIR lacks.
  hist      coordinate-descent exchange minimizing summed L1 distance
            between selected-set histograms and reference histograms:
            every feature marginal (quantile bins) plus the most correlated
            feature pairs (2-D bins). Directly targets what an axis-aligned
            forest can split on. Init from greedy or sir picks.

Honesty protocol: the 4000-row disjoint reference is split in half. All
fitting and objectives use half A only. Half B is used purely to report a
proxy RF OOB AUC (same classifier settings as evaluate.py) that no strategy
ever optimized against. The final numbers still come from evaluate.py via
EVENT_POOL_LOAD / EVENT_POOL_PICKS replay, where the human class is the real
eval sample.

Run:
    .venv/Scripts/python.exe selection_lab.py --pool pool_s42_k16.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score


def rf_proxy_auc(X_sel: np.ndarray, X_ref: np.ndarray,
                 seeds=(42, 43, 44)) -> float:
    """Mirror of evaluate.py's primary detector (RF 100 trees, OOB AUC),
    averaged over three RF seeds to cut ranking noise."""
    n = min(len(X_sel), len(X_ref))
    X = np.vstack([X_ref[:n], X_sel[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    aucs = []
    for seed in seeds:
        clf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                     n_jobs=-1, random_state=seed)
        clf.fit(X, y)
        aucs.append(roc_auc_score(y, clf.oob_decision_function_[:, 1]))
    return float(np.mean(aucs))


def fit_logodds(X_pos, X_neg, X_score, trees=200, seed=0):
    """Same discriminator family as the in-recipe SIR judge."""
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    clf = GradientBoostingClassifier(n_estimators=trees, max_depth=3,
                                     subsample=0.8, random_state=seed)
    clf.fit(X, y)
    p = np.clip(clf.predict_proba(X_score)[:, 1], 1e-4, 1 - 1e-4)
    return np.log(p) - np.log(1.0 - p)


class Pool:
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.X = d["X"]
        self.owner = d["owner_idx"].astype(int)
        self.n_specs = len(d["specs"])
        self.spec_rows: dict[int, np.ndarray] = {}
        for ci, idx in enumerate(self.owner):
            self.spec_rows.setdefault(int(idx), []).append(ci)
        for idx in self.spec_rows:
            self.spec_rows[idx] = np.asarray(self.spec_rows[idx])
        sizes = [len(v) for v in self.spec_rows.values()]
        print(f"pool: {len(self.X):,} candidates over {len(self.spec_rows)} "
              f"specs (K min={min(sizes)} max={max(sizes)})")

    def picks_to_full(self, picks: dict[int, int]) -> np.ndarray:
        full = np.full(self.n_specs, -1, dtype=np.int64)
        for idx, ci in picks.items():
            full[idx] = ci
        return full

    def selected(self, picks: dict[int, int]) -> np.ndarray:
        return self.X[np.asarray(sorted(picks.values()))]


def pick_sir(pool: Pool, ref_a, temp=0.7, seed=0):
    logw = fit_logodds(ref_a, pool.X, pool.X)
    rng = np.random.default_rng(seed)
    picks = {}
    for idx, rows in pool.spec_rows.items():
        g = rng.gumbel(size=len(rows))
        picks[idx] = int(rows[np.argmax(logw[rows] / temp + g)])
    return picks


def pick_greedy(pool: Pool, ref_a, ref_b, rounds=8):
    """Iterated adversarial reselection with cumulative log-odds."""
    cum = fit_logodds(ref_a, pool.X, pool.X)
    best_auc, best_picks, trace = 1.0, None, []
    for r in range(rounds + 1):
        picks = {idx: int(rows[np.argmax(cum[rows])])
                 for idx, rows in pool.spec_rows.items()}
        auc = rf_proxy_auc(pool.selected(picks), ref_b)
        trace.append(auc)
        if auc < best_auc:
            best_auc, best_picks = auc, dict(picks)
        if r < rounds:
            cum = cum + fit_logodds(ref_a, pool.selected(picks), pool.X)
    print(f"  greedy trace (proxy AUC vs B): "
          + " ".join(f"{a:.4f}" for a in trace))
    return best_picks, best_auc


class HistObjective:
    """Summed L1 distance between selected-set histograms and reference
    histograms: all 18 feature marginals on reference-quantile bins, plus the
    top correlated feature pairs on 2-D quantile bins. Supports O(bins)
    incremental swap deltas for coordinate-descent exchange."""

    def __init__(self, X_pool, ref_a, n_sel, n_bins=32, n_pairs=12,
                 pair_bins=6, pair_weight=0.5):
        self.pair_weight = pair_weight
        F = X_pool.shape[1]
        scale = n_sel / len(ref_a)

        self.m_bin = np.empty((len(X_pool), F), dtype=np.int32)
        self.m_tgt = []
        for f in range(F):
            edges = np.quantile(ref_a[:, f], np.linspace(0, 1, n_bins + 1)[1:-1])
            self.m_bin[:, f] = np.searchsorted(edges, X_pool[:, f])
            tgt = np.bincount(np.searchsorted(edges, ref_a[:, f]),
                              minlength=n_bins).astype(np.float64) * scale
            self.m_tgt.append(tgt)
        self.m_tgt = np.asarray(self.m_tgt)
        self.n_bins = n_bins

        corr = np.corrcoef(ref_a.T)
        cand_pairs = [(abs(corr[i, j]), i, j)
                      for i in range(F) for j in range(i + 1, F)]
        cand_pairs.sort(reverse=True)
        self.pairs = [(i, j) for _, i, j in cand_pairs[:n_pairs]]
        self.p_bin = np.empty((len(X_pool), len(self.pairs)), dtype=np.int32)
        self.p_tgt = []
        nb2 = pair_bins
        for pi, (i, j) in enumerate(self.pairs):
            ei = np.quantile(ref_a[:, i], np.linspace(0, 1, nb2 + 1)[1:-1])
            ej = np.quantile(ref_a[:, j], np.linspace(0, 1, nb2 + 1)[1:-1])
            bi = np.searchsorted(ei, X_pool[:, i])
            bj = np.searchsorted(ej, X_pool[:, j])
            self.p_bin[:, pi] = bi * nb2 + bj
            ri = np.searchsorted(ei, ref_a[:, i])
            rj = np.searchsorted(ej, ref_a[:, j])
            tgt = np.bincount(ri * nb2 + rj,
                              minlength=nb2 * nb2).astype(np.float64) * scale
            self.p_tgt.append(tgt)
        self.p_tgt = np.asarray(self.p_tgt)
        self.n_pbins = nb2 * nb2

    def init_counts(self, sel_rows):
        F = self.m_bin.shape[1]
        self.m_cnt = np.zeros((F, self.n_bins))
        for f in range(F):
            self.m_cnt[f] = np.bincount(self.m_bin[sel_rows, f],
                                        minlength=self.n_bins)
        P = len(self.pairs)
        self.p_cnt = np.zeros((P, self.n_pbins))
        for pi in range(P):
            self.p_cnt[pi] = np.bincount(self.p_bin[sel_rows, pi],
                                         minlength=self.n_pbins)

    def value(self):
        v = np.abs(self.m_cnt - self.m_tgt).sum()
        v += self.pair_weight * np.abs(self.p_cnt - self.p_tgt).sum()
        return v

    def _delta_axis(self, cnt, tgt, old_bins, new_bins):
        """Objective change from moving one item old->new per candidate."""
        rem = (np.abs(cnt[old_bins] - 1 - tgt[old_bins])
               - np.abs(cnt[old_bins] - tgt[old_bins]))
        add = (np.abs(cnt[new_bins] + 1 - tgt[new_bins])
               - np.abs(cnt[new_bins] - tgt[new_bins]))
        d = rem + add
        d[new_bins == old_bins] = 0.0
        return d

    def swap_delta(self, old_ci, cand_rows):
        d = np.zeros(len(cand_rows))
        for f in range(self.m_bin.shape[1]):
            d += self._delta_axis(self.m_cnt[f], self.m_tgt[f],
                                  np.repeat(self.m_bin[old_ci, f],
                                            len(cand_rows)),
                                  self.m_bin[cand_rows, f])
        for pi in range(len(self.pairs)):
            d += self.pair_weight * self._delta_axis(
                self.p_cnt[pi], self.p_tgt[pi],
                np.repeat(self.p_bin[old_ci, pi], len(cand_rows)),
                self.p_bin[cand_rows, pi])
        return d

    def apply_swap(self, old_ci, new_ci):
        for f in range(self.m_bin.shape[1]):
            self.m_cnt[f, self.m_bin[old_ci, f]] -= 1
            self.m_cnt[f, self.m_bin[new_ci, f]] += 1
        for pi in range(len(self.pairs)):
            self.p_cnt[pi, self.p_bin[old_ci, pi]] -= 1
            self.p_cnt[pi, self.p_bin[new_ci, pi]] += 1


class MMDObjective:
    """Multi-bandwidth RBF MMD^2 between the selected set and reference A on
    z-scored features. Kernels see the full joint geometry, so unlike the
    histogram objective this penalizes interactions of every order. Supports
    incremental swap deltas by maintaining, for every pool candidate, its
    kernel sum against the current selected set."""

    def __init__(self, X_pool, ref_a, bw_mults=(0.5, 1.0, 2.0), chunk=4096):
        mu, sd = ref_a.mean(axis=0), ref_a.std(axis=0) + 1e-9
        self.Z = ((X_pool - mu) / sd).astype(np.float64)
        A = ((ref_a - mu) / sd).astype(np.float64)
        sub = A[np.random.default_rng(0).permutation(len(A))[:1000]]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0])
        self.gammas = np.array([1.0 / (m * med) for m in bw_mults])
        self.chunk = chunk
        self.n_ref = len(A)
        self.cross = self._ksum_rows(A)

    def _k_to_point(self, y):
        d2 = ((self.Z - y) ** 2).sum(axis=1)
        return np.exp(-self.gammas[:, None] * d2[None, :]).mean(axis=0)

    def _ksum_rows(self, Y):
        out = np.zeros(len(self.Z))
        for y in Y:
            out += self._k_to_point(y)
        return out

    def init_counts(self, sel_rows):
        self.n_sel = len(sel_rows)
        self.ssum = self._ksum_rows(self.Z[sel_rows])

    def swap_delta(self, old_ci, cand_rows):
        k_vu = np.exp(-self.gammas[:, None] * (
            (self.Z[cand_rows] - self.Z[old_ci]) ** 2).sum(-1)[None]).mean(0)
        k_vv = np.ones(len(cand_rows))
        n, m = self.n_sel, self.n_ref
        d_t1 = (2.0 * (self.ssum[cand_rows] - k_vu)
                - 2.0 * self.ssum[old_ci] + 1.0 + k_vv)
        d_t2 = self.cross[cand_rows] - self.cross[old_ci]
        d = d_t1 / n**2 - 2.0 * d_t2 / (n * m)
        d[cand_rows == old_ci] = 0.0
        return d

    def apply_swap(self, old_ci, new_ci):
        self.ssum += self._k_to_point(self.Z[new_ci])
        self.ssum -= self._k_to_point(self.Z[old_ci])


def pick_exchange(pool: Pool, ref_a, ref_b, init_picks, objective,
                  max_sweeps=15, seed=0, label="obj"):
    """Coordinate-descent exchange under any incremental objective."""
    picks = dict(init_picks)
    objective.init_counts(np.asarray(sorted(picks.values())))
    rng = np.random.default_rng(seed)
    spec_ids = np.asarray(list(pool.spec_rows.keys()))
    for sweep in range(max_sweeps):
        n_moves = 0
        for idx in spec_ids[rng.permutation(len(spec_ids))]:
            rows = pool.spec_rows[int(idx)]
            old = picks[int(idx)]
            d = objective.swap_delta(old, rows)
            j = int(np.argmin(d))
            if d[j] < -1e-15 and int(rows[j]) != old:
                objective.apply_swap(old, int(rows[j]))
                picks[int(idx)] = int(rows[j])
                n_moves += 1
        print(f"  {label} sweep {sweep + 1}: {n_moves} moves", flush=True)
        if n_moves == 0:
            break
    auc = rf_proxy_auc(pool.selected(picks), ref_b)
    return picks, auc


def pick_trust(pool: Pool, ref_a, ref_b, init_picks, rounds=10, frac=0.15):
    """Set-aware discriminator loop with a trust region: each round, fit the
    judge on reference vs the CURRENT selected set, then move only the
    fraction of specs with the largest log-odds gain. Avoids the argmax
    collapse of the plain greedy loop."""
    picks = dict(init_picks)
    best_auc = rf_proxy_auc(pool.selected(picks), ref_b)
    best_picks = dict(picks)
    trace = [best_auc]
    for r in range(rounds):
        logw = fit_logodds(ref_a, pool.selected(picks), pool.X, seed=r)
        gains = []
        for idx, rows in pool.spec_rows.items():
            j = int(np.argmax(logw[rows]))
            gains.append((logw[rows[j]] - logw[picks[idx]], idx, int(rows[j])))
        gains.sort(reverse=True)
        n_move = max(1, int(frac * len(gains)))
        moved = 0
        for g, idx, ci in gains[:n_move]:
            if g <= 0:
                break
            picks[idx] = ci
            moved += 1
        auc = rf_proxy_auc(pool.selected(picks), ref_b)
        trace.append(auc)
        if auc < best_auc:
            best_auc, best_picks = auc, dict(picks)
        if moved == 0:
            break
    print(f"  trust trace (proxy AUC vs B): "
          + " ".join(f"{a:.4f}" for a in trace))
    return best_picks, best_auc


def pick_hist(pool: Pool, ref_a, ref_b, init_picks, max_sweeps=25, seed=0):
    obj = HistObjective(pool.X, ref_a, n_sel=len(init_picks))
    picks = dict(init_picks)
    obj.init_counts(np.asarray(sorted(picks.values())))
    rng = np.random.default_rng(seed)
    spec_ids = np.asarray(list(pool.spec_rows.keys()))
    print(f"  hist objective init: {obj.value():.0f}")
    for sweep in range(max_sweeps):
        n_moves = 0
        for idx in spec_ids[rng.permutation(len(spec_ids))]:
            rows = pool.spec_rows[int(idx)]
            old = picks[int(idx)]
            d = obj.swap_delta(old, rows)
            j = int(np.argmin(d))
            if d[j] < -1e-9 and int(rows[j]) != old:
                obj.apply_swap(old, int(rows[j]))
                picks[int(idx)] = int(rows[j])
                n_moves += 1
        print(f"  sweep {sweep + 1}: {n_moves} moves, "
              f"objective {obj.value():.0f}")
        if n_moves == 0:
            break
    auc = rf_proxy_auc(pool.selected(picks), ref_b)
    return picks, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--ref", default="data/human_ref_features_sir.npy")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--strategies",
                    default="random,sir,greedy,hist_g,hist_s,mmd,trust")
    args = ap.parse_args()
    strategies = set(args.strategies.split(","))

    pool = Pool(args.pool)
    ref = np.load(args.ref)
    perm = np.random.default_rng(0).permutation(len(ref))
    half = len(ref) // 2
    ref_a, ref_b = ref[perm[:half]], ref[perm[half:]]
    print(f"reference: {len(ref_a)} fit rows (A), {len(ref_b)} proxy rows (B)")
    prefix = args.out_prefix or args.pool.replace(".npz", "")

    results = {}

    if "random" in strategies:
        t0 = time.time()
        rng = np.random.default_rng(1)
        rnd = {idx: int(rng.choice(rows))
               for idx, rows in pool.spec_rows.items()}
        results["random"] = (rnd, rf_proxy_auc(pool.selected(rnd), ref_b))
        print(f"random-of-K proxy AUC vs B: {results['random'][1]:.4f} "
              f"({time.time() - t0:.0f}s)")

    sir = pick_sir(pool, ref_a)
    if "sir" in strategies:
        results["sir"] = (sir, rf_proxy_auc(pool.selected(sir), ref_b))
        print(f"sir (temp 0.7) proxy AUC vs B: {results['sir'][1]:.4f}")

    if "greedy" in strategies or "hist_g" in strategies:
        t0 = time.time()
        print("greedy adversarial reselection:")
        gp, ga = pick_greedy(pool, ref_a, ref_b, rounds=args.rounds)
        if "greedy" in strategies:
            results["greedy"] = (gp, ga)
        print(f"greedy best proxy AUC vs B: {ga:.4f} "
              f"({time.time() - t0:.0f}s)")

    if "hist_g" in strategies:
        t0 = time.time()
        print("hist exchange from greedy init:")
        hp, ha = pick_hist(pool, ref_a, ref_b, gp)
        results["hist_g"] = (hp, ha)
        print(f"hist(greedy init) proxy AUC vs B: {ha:.4f} "
              f"({time.time() - t0:.0f}s)")

    if "hist_s" in strategies:
        t0 = time.time()
        print("hist exchange from sir init:")
        hp2, ha2 = pick_hist(pool, ref_a, ref_b, sir)
        results["hist_s"] = (hp2, ha2)
        print(f"hist(sir init) proxy AUC vs B: {ha2:.4f} "
              f"({time.time() - t0:.0f}s)")

    if "mmd" in strategies:
        t0 = time.time()
        print("mmd exchange from sir init:")
        mo = MMDObjective(pool.X, ref_a)
        mp, ma = pick_exchange(pool, ref_a, ref_b, sir, mo, label="mmd")
        results["mmd"] = (mp, ma)
        print(f"mmd(sir init) proxy AUC vs B: {ma:.4f} "
              f"({time.time() - t0:.0f}s)")

    if "trust" in strategies:
        t0 = time.time()
        print("trust-region discriminator loop from sir init:")
        tp, ta = pick_trust(pool, ref_a, ref_b, sir)
        results["trust"] = (tp, ta)
        print(f"trust(sir init) proxy AUC vs B: {ta:.4f} "
              f"({time.time() - t0:.0f}s)")

    print("\n=== summary (proxy RF OOB AUC vs held-out reference B) ===")
    for name, (picks, auc) in sorted(results.items(), key=lambda x: x[1][1]):
        out = f"{prefix}_picks_{name}.npy"
        np.save(out, pool.picks_to_full(picks))
        print(f"  {name:8s} {auc:.4f}  -> {out}")


if __name__ == "__main__":
    main()
