"""
RL fine-tuning for ZIMT with feature-matching reward, RLOO baseline, and KL penalty.

Improvements over GRPO:
1. Per-feature z-score reward (continuous, directional) instead of RF AUC (binary, noisy)
2. RLOO baseline (per-trajectory variance reduction)
3. KL penalty from frozen reference model (prevents drift)

Run: python -m training.train_zimt_rl [--checkpoint PATH] [--n-iters 300]
"""
from __future__ import annotations

import argparse
import copy
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import FEATURE_NAMES, extract_feature_matrix
from models.zimt import ZIMTModel, sample_step

TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAINING_DIR.parent / "data"

_stop_requested = False


def _handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[RL] Graceful stop requested...")


class DurationModel:
    def __init__(self, n_bins=60):
        conditions = np.load(DATA_DIR / "train_conditions.npy")
        log_dist = conditions[:, 0]
        log_dur = conditions[:, 1]
        self._n_bins = n_bins
        self._d_edges = np.linspace(log_dist.min(), log_dist.max(), n_bins + 1)
        self._dur_mean = np.zeros(n_bins)
        self._dur_std = np.zeros(n_bins)
        for b in range(n_bins):
            m = (log_dist >= self._d_edges[b]) & (log_dist < self._d_edges[b + 1])
            if m.sum() >= 3:
                self._dur_mean[b] = log_dur[m].mean()
                self._dur_std[b] = log_dur[m].std()
            else:
                self._dur_mean[b] = np.median(log_dur)
                self._dur_std[b] = 0.12
        self._rng = np.random.default_rng()

    def sample(self, log_dist):
        b = int(np.clip(np.searchsorted(self._d_edges[1:], log_dist), 0, self._n_bins - 1))
        std = max(float(self._dur_std[b]), 0.05)
        return float(np.clip(math.exp(self._rng.normal(self._dur_mean[b], std * 0.7)), 0.05, 4.0))


def bivariate_log_prob(dx, dy, mu, sigma, rho, log_pi):
    dx_ = dx.unsqueeze(-1) - mu[:, :, :, 0]
    dy_ = dy.unsqueeze(-1) - mu[:, :, :, 1]
    sx = sigma[:, :, :, 0]
    sy = sigma[:, :, :, 1]
    z = (dx_ / sx) ** 2 + (dy_ / sy) ** 2 - 2 * rho * dx_ * dy_ / (sx * sy)
    denom = (1 - rho ** 2).clamp(min=1e-8)
    log_norm = -math.log(2 * math.pi) - torch.log(sx) - torch.log(sy) - 0.5 * torch.log(denom)
    log_exp = -0.5 * z / denom
    return torch.logsumexp(log_pi + log_norm + log_exp, dim=-1)


@torch.no_grad()
def generate_batch(model, condition, cos_a, sin_a, n_target, input_dim,
                   batch_size, device, temperature=1.0, gate_bias=-1.0):
    B = batch_size
    cond = condition.expand(B, -1)
    input_buf = torch.zeros(B, n_target, input_dim, device=device)
    all_dxdy = torch.zeros(B, n_target, 2, device=device)
    all_stall = torch.zeros(B, n_target, device=device)
    cum_dx = torch.zeros(B, device=device)
    cum_dy = torch.zeros(B, device=device)

    for step in range(n_target):
        if step > 0:
            input_buf[:, step, 0] = all_dxdy[:, step - 1, 0]
            input_buf[:, step, 1] = all_dxdy[:, step - 1, 1]
            input_buf[:, step, 2] = all_stall[:, step - 1]
        input_buf[:, step, 3] = cos_a - cum_dx
        input_buf[:, step, 4] = sin_a - cum_dy
        input_buf[:, step, 5] = 1.0 - step / n_target

        params = model(input_buf[:, :step + 1], cond)
        gate_logit = params["gate_logit"][:, -1]
        stall_prob = torch.sigmoid(gate_logit + gate_bias)
        is_stall = torch.bernoulli(stall_prob)

        pi = params["pi"][:, -1]
        mu = params["mu"][:, -1]
        sigma = params["sigma"][:, -1]
        rho = params["rho"][:, -1]

        if temperature != 1.0:
            logit_pi = params["logit_pi"][:, -1]
            pi = torch.softmax(logit_pi / temperature, dim=-1)
            sigma = sigma * temperature

        comp_idx = torch.multinomial(pi, 1).squeeze(-1)
        sel_mu = mu[torch.arange(B), comp_idx]
        sel_sigma = sigma[torch.arange(B), comp_idx]
        sel_rho = rho[torch.arange(B), comp_idx]

        z1 = torch.randn(B, device=device)
        z2 = torch.randn(B, device=device)
        dx = sel_mu[:, 0] + sel_sigma[:, 0] * z1
        dy = sel_mu[:, 1] + sel_sigma[:, 1] * (
            sel_rho * z1 + torch.sqrt((1 - sel_rho ** 2).clamp(min=1e-8)) * z2
        )
        dx = dx * (1 - is_stall)
        dy = dy * (1 - is_stall)

        all_dxdy[:, step, 0] = dx
        all_dxdy[:, step, 1] = dy
        all_stall[:, step] = is_stall
        cum_dx = cum_dx + dx
        cum_dy = cum_dy + dy

    return all_dxdy, all_stall


def build_trajectory(actions_dx, actions_dy, start_x, start_y, end_x, end_y, total_dist, hz=125.0):
    positions_x = [start_x]
    positions_y = [start_y]
    cx, cy = start_x, start_y
    for ddx, ddy in zip(actions_dx, actions_dy):
        cx += ddx * total_dist
        cy += ddy * total_dist
        positions_x.append(cx)
        positions_y.append(cy)

    n = len(positions_x)
    step_mags = []
    for i in range(1, n):
        dx = positions_x[i] - positions_x[i - 1]
        dy = positions_y[i] - positions_y[i - 1]
        step_mags.append(math.hypot(dx, dy))

    err_x = end_x - positions_x[-1]
    err_y = end_y - positions_y[-1]
    if err_x * err_x + err_y * err_y > 0.01:
        moving = [m > 0.3 for m in step_mags]
        total_moving = sum(m for m, mv in zip(step_mags, moving) if mv)
        if total_moving > 0.1:
            cum_cx, cum_cy = 0.0, 0.0
            for i in range(len(step_mags)):
                if moving[i]:
                    w = step_mags[i] / total_moving
                    cum_cx += err_x * w
                    cum_cy += err_y * w
                positions_x[i + 1] += cum_cx
                positions_y[i + 1] += cum_cy

    dt = 1.0 / hz
    result = [(float(positions_x[i]), float(positions_y[i]), i * dt) for i in range(n)]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, result[-1][2])
    return result


def compute_trajectory_log_prob(model, actions_dxdy, actions_stall, condition,
                                cos_a, sin_a, n_target, device, gate_bias=-1.0):
    T = n_target
    feat = torch.zeros(1, T, 6, device=device)
    cum_dx, cum_dy = 0.0, 0.0
    for t in range(T):
        if t > 0:
            feat[0, t, 0] = actions_dxdy[t - 1, 0]
            feat[0, t, 1] = actions_dxdy[t - 1, 1]
            feat[0, t, 2] = actions_stall[t - 1]
            cum_dx += actions_dxdy[t - 1, 0].item()
            cum_dy += actions_dxdy[t - 1, 1].item()
        feat[0, t, 3] = cos_a - cum_dx
        feat[0, t, 4] = sin_a - cum_dy
        feat[0, t, 5] = 1.0 - t / T

    params = model(feat, condition)
    gate_logit = params["gate_logit"][0]
    mu = params["mu"]
    sigma = params["sigma"]
    rho = params["rho"]
    log_pi = torch.log_softmax(params["logit_pi"], dim=-1)

    total_lp = torch.tensor(0.0, device=device)
    for t in range(T):
        is_stall = actions_stall[t].item() > 0.5
        gate_lp = torch.nn.functional.logsigmoid(
            (gate_logit[t] + gate_bias) if is_stall else -(gate_logit[t] + gate_bias)
        )
        if is_stall:
            total_lp = total_lp + gate_lp
        else:
            dx_val = actions_dxdy[t, 0].unsqueeze(0).unsqueeze(0)
            dy_val = actions_dxdy[t, 1].unsqueeze(0).unsqueeze(0)
            mdn_lp = bivariate_log_prob(
                dx_val, dy_val,
                mu[:, t:t+1], sigma[:, t:t+1], rho[:, t:t+1], log_pi[:, t:t+1],
            )
            total_lp = total_lp + gate_lp + mdn_lp.squeeze()

    return total_lp


def main():
    parser = argparse.ArgumentParser(description="RL fine-tune ZIMT with feature-matching reward")
    parser.add_argument("--checkpoint", default=str(DATA_DIR / "zimt_best.pt"))
    parser.add_argument("--n-iters", type=int, default=300)
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--n-per-query", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gate-bias", type=float, default=-1.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    device = torch.device(args.device)
    print(f"[RL] Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ZIMTModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Frozen reference model for KL penalty
    ref_model = ZIMTModel(**cfg).to(device)
    ref_model.load_state_dict(ckpt["model_state_dict"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print(f"[RL] ZIMT ({cfg['d_model']}d, {cfg['n_layers']}L, {cfg['n_components']}K)")
    print(f"[RL] KL beta={args.beta}, LR={args.lr}")

    # Load human data
    human_features = np.load(DATA_DIR / "human_eval_features.npy")
    human_distances = np.load(DATA_DIR / "human_distances.npy")

    # Precompute human feature statistics for reward
    h_mean = human_features.mean(axis=0)
    h_std = human_features.std(axis=0)
    h_std = np.maximum(h_std, 1e-8)
    print(f"[RL] Human: {len(human_features)} features, {len(human_distances)} distances")

    duration_model = DurationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    n_total = args.n_queries * args.n_per_query
    hz = 125.0
    cx, cy = 960.0, 540.0
    rng = np.random.default_rng(42)

    best_auc = 1.0
    print(f"\n[RL] Starting: {args.n_iters} iters, {n_total} trajs/iter, beta={args.beta}")

    for iteration in range(args.n_iters):
        if _stop_requested:
            break

        t0 = time.time()
        model.eval()

        # --- Generate trajectories ---
        all_trajs = []
        all_dxdy = []
        all_stall = []
        all_conds = []
        all_cos = []
        all_sin = []
        all_nt = []
        query_ids = []

        for q in range(args.n_queries):
            dist = float(rng.choice(human_distances))
            angle = float(rng.uniform(0, 2 * np.pi))
            ex = cx + dist * np.cos(angle)
            ey = cy + dist * np.sin(angle)
            td = math.hypot(ex - cx, ey - cy)

            if td < 1.0:
                for _ in range(args.n_per_query):
                    all_trajs.append([(cx, cy, 0.0), (ex, ey, 0.008)])
                    all_dxdy.append(torch.zeros(1, 2, device=device))
                    all_stall.append(torch.zeros(1, device=device))
                    all_conds.append(torch.zeros(1, 4, device=device))
                    all_cos.append(0.0)
                    all_sin.append(0.0)
                    all_nt.append(1)
                    query_ids.append(q)
                continue

            ld = math.log(td)
            ca = (ex - cx) / td
            sa = (ey - cy) / td
            dur = duration_model.sample(ld)
            ldur = math.log(max(dur, 0.01))
            cond = torch.tensor([[ld, ldur, ca, sa]], device=device, dtype=torch.float32)
            nt = max(5, min(int(round(dur * hz)), cfg["max_seq_len"] - 2))

            batch_dxdy, batch_stall = generate_batch(
                model, cond, ca, sa, nt, cfg["input_dim"],
                args.n_per_query, device, args.temperature, args.gate_bias,
            )

            for b in range(args.n_per_query):
                adx = batch_dxdy[b, :, 0].tolist()
                ady = batch_dxdy[b, :, 1].tolist()
                traj = build_trajectory(adx, ady, cx, cy, ex, ey, td, hz)
                all_trajs.append(traj)
                all_dxdy.append(batch_dxdy[b])
                all_stall.append(batch_stall[b])
                all_conds.append(cond)
                all_cos.append(ca)
                all_sin.append(sa)
                all_nt.append(nt)
                query_ids.append(q)

        gen_time = time.time() - t0

        # --- Per-feature z-score reward ---
        synth_feats = extract_feature_matrix(all_trajs)
        n_valid = len(synth_feats)
        if n_valid < n_total * 0.5:
            print(f"  Iter {iteration+1}: too few valid ({n_valid}/{n_total}), skip")
            continue

        z_scores = np.abs((synth_feats - h_mean) / h_std)
        raw_rewards = -z_scores.sum(axis=1)  # (n_valid,)
        raw_rewards = np.clip(raw_rewards, -50, 0)

        # --- Compute log probs for KL (both models, no grad) ---
        ref_lps = np.zeros(n_valid)
        cur_lps = np.zeros(n_valid)
        with torch.no_grad():
            for i in range(n_valid):
                if all_nt[i] < 2:
                    continue
                ref_lps[i] = compute_trajectory_log_prob(
                    ref_model, all_dxdy[i], all_stall[i], all_conds[i],
                    all_cos[i], all_sin[i], all_nt[i], device, args.gate_bias,
                ).item()
                cur_lps[i] = compute_trajectory_log_prob(
                    model, all_dxdy[i], all_stall[i], all_conds[i],
                    all_cos[i], all_sin[i], all_nt[i], device, args.gate_bias,
                ).item()

        # Per-trajectory KL and modified rewards
        kls = np.array([(cur_lps[i] - ref_lps[i]) / max(all_nt[i], 1)
                        for i in range(n_valid)])
        modified_rewards = raw_rewards - args.beta * kls

        # --- RLOO advantages ---
        rewards_by_q = {}
        indices_by_q = {}
        for i in range(n_valid):
            q = query_ids[i]
            rewards_by_q.setdefault(q, []).append(modified_rewards[i])
            indices_by_q.setdefault(q, []).append(i)

        advantages = np.zeros(n_valid)
        for q in rewards_by_q:
            r = np.array(rewards_by_q[q])
            idx = indices_by_q[q]
            K = len(r)
            if K < 2:
                continue
            for j in range(K):
                baseline = (r.sum() - r[j]) / (K - 1)
                advantages[idx[j]] = r[j] - baseline

        adv_std = advantages.std()
        if adv_std > 1e-6:
            advantages = advantages / adv_std

        # --- REINFORCE loss ---
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        n_counted = 0

        for i in range(min(n_valid, n_total)):
            if all_nt[i] < 2:
                continue

            cur_lp = compute_trajectory_log_prob(
                model, all_dxdy[i], all_stall[i], all_conds[i],
                all_cos[i], all_sin[i], all_nt[i], device, args.gate_bias,
            )

            lp_norm = cur_lp / max(all_nt[i], 1)
            adv = torch.tensor(advantages[i], device=device, dtype=torch.float32)
            total_loss = total_loss - adv * lp_norm
            n_counted += 1

        if n_counted > 0:
            total_loss = total_loss / n_counted
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        else:
            grad_norm = 0.0

        iter_time = time.time() - t0
        mean_reward = raw_rewards.mean()
        mean_kl = float(np.abs(kls).mean())

        print(
            f"  Iter {iteration+1:4d} | "
            f"loss {total_loss.item():+.4f} | "
            f"reward {mean_reward:.2f} | "
            f"kl {mean_kl:.3f} | "
            f"grad {grad_norm:.3f} | "
            f"{iter_time:.1f}s"
        )

        # --- Periodic evaluation ---
        if (iteration + 1) % args.eval_every == 0:
            model.eval()
            eval_trajs = []
            eval_rng = np.random.default_rng(42)
            for _ in range(200):
                dist = float(eval_rng.choice(human_distances))
                angle = float(eval_rng.uniform(0, 2 * np.pi))
                ex_ = cx + dist * np.cos(angle)
                ey_ = cy + dist * np.sin(angle)
                td_ = math.hypot(ex_ - cx, ey_ - cy)
                if td_ < 1.0:
                    eval_trajs.append([(cx, cy, 0.0), (ex_, ey_, 0.008)])
                    continue
                ld_ = math.log(td_)
                ca_ = (ex_ - cx) / td_
                sa_ = (ey_ - cy) / td_
                dur_ = duration_model.sample(ld_)
                ldur_ = math.log(max(dur_, 0.01))
                cond_ = torch.tensor([[ld_, ldur_, ca_, sa_]], device=device, dtype=torch.float32)
                nt_ = max(5, min(int(round(dur_ * hz)), cfg["max_seq_len"] - 2))
                bd, bs = generate_batch(
                    model, cond_, ca_, sa_, nt_, cfg["input_dim"], 1, device,
                    args.temperature, args.gate_bias,
                )
                adx_ = bd[0, :, 0].tolist()
                ady_ = bd[0, :, 1].tolist()
                eval_trajs.append(build_trajectory(adx_, ady_, cx, cy, ex_, ey_, td_, hz))

            ef = extract_feature_matrix(eval_trajs)
            n_ev = min(len(human_features), len(ef))
            eX = np.vstack([human_features[:n_ev], ef[:n_ev]])
            ey_lab = np.concatenate([np.zeros(n_ev), np.ones(n_ev)])
            erf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
            erf.fit(eX, ey_lab)
            eval_auc = roc_auc_score(ey_lab, erf.oob_decision_function_[:, 1])

            # Feature diagnostics
            from features import normalized_wasserstein_by_feature
            w_dists = normalized_wasserstein_by_feature(human_features[:n_ev], ef[:n_ev])
            top3 = sorted(zip(FEATURE_NAMES, w_dists), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"{n}={d:.3f}" for n, d in top3)

            is_best = eval_auc < best_auc
            if is_best:
                best_auc = eval_auc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "phase": "rl",
                    "rl_iter": iteration + 1,
                    "rl_auc": eval_auc,
                }, TRAINING_DIR / "zimt_rl_best.pt")

            marker = " *BEST*" if is_best else ""
            print(f"  >>> EVAL {iteration+1}: AUC {eval_auc:.4f} (best {best_auc:.4f}){marker}")
            print(f"      Top gaps: {top_str}")

        if (iteration + 1) % 25 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "phase": "rl",
                "rl_iter": iteration + 1,
            }, TRAINING_DIR / "zimt_rl_latest.pt")

    print(f"\n[RL] Done. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
