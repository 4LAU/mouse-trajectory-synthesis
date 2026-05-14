"""
GRPO fine-tuning for ZIMT.

Uses the RF classifier (the evaluation metric) as the RL reward signal.
Generates trajectories, scores them with RF, and uses Group Relative
Policy Optimization to update the model toward more human-like outputs.

Run: python -m training.train_zimt_grpo [--checkpoint PATH] [--n-iters 500]
"""
from __future__ import annotations

import argparse
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

from features import extract_feature_matrix
from models.zimt import ZIMTModel, sample_step

TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAINING_DIR.parent / "data"

_stop_requested = False


def _handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[GRPO] Graceful stop requested, finishing iteration...")


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
    """Compute log p(dx, dy) under the MDN mixture.

    dx, dy: (B, T) — generated displacements
    mu: (B, T, M, 2)
    sigma: (B, T, M, 2)
    rho: (B, T, M)
    log_pi: (B, T, M)

    Returns: (B, T) log probabilities
    """
    dx_ = dx.unsqueeze(-1) - mu[:, :, :, 0]  # (B, T, M)
    dy_ = dy.unsqueeze(-1) - mu[:, :, :, 1]
    sx = sigma[:, :, :, 0]
    sy = sigma[:, :, :, 1]

    z = (dx_ / sx) ** 2 + (dy_ / sy) ** 2 - 2 * rho * dx_ * dy_ / (sx * sy)
    denom = (1 - rho ** 2).clamp(min=1e-8)

    log_norm = -math.log(2 * math.pi) - torch.log(sx) - torch.log(sy) - 0.5 * torch.log(denom)
    log_exp = -0.5 * z / denom
    log_comp = log_norm + log_exp  # (B, T, M)

    return torch.logsumexp(log_pi + log_comp, dim=-1)  # (B, T)


@torch.no_grad()
def generate_batch(model, condition, cos_a, sin_a, n_target, input_dim,
                   batch_size, device, temperature=1.0, gate_bias=-1.0):
    """Generate batch_size trajectories in parallel for one query.

    All trajectories share the same condition and n_target.
    Returns: dxdy (B, T, 2), stall (B, T)
    """
    B = batch_size
    cond = condition.expand(B, -1)  # (B, 4)
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

        # Batch sampling from MDN
        gate_logit = params["gate_logit"][:, -1]  # (B,)
        stall_prob = torch.sigmoid(gate_logit + gate_bias)
        is_stall = torch.bernoulli(stall_prob)

        pi = params["pi"][:, -1]  # (B, M)
        mu = params["mu"][:, -1]  # (B, M, 2)
        sigma = params["sigma"][:, -1]  # (B, M, 2)
        rho = params["rho"][:, -1]  # (B, M)

        if temperature != 1.0:
            logit_pi = params["logit_pi"][:, -1]
            pi = torch.softmax(logit_pi / temperature, dim=-1)
            sigma = sigma * temperature

        comp_idx = torch.multinomial(pi, 1).squeeze(-1)  # (B,)
        sel_mu = mu[torch.arange(B), comp_idx]  # (B, 2)
        sel_sigma = sigma[torch.arange(B), comp_idx]  # (B, 2)
        sel_rho = rho[torch.arange(B), comp_idx]  # (B,)

        z1 = torch.randn(B, device=device)
        z2 = torch.randn(B, device=device)
        dx = sel_mu[:, 0] + sel_sigma[:, 0] * z1
        dy = sel_mu[:, 1] + sel_sigma[:, 1] * (
            sel_rho * z1 + torch.sqrt((1 - sel_rho ** 2).clamp(min=1e-8)) * z2
        )

        # Apply stall mask
        dx = dx * (1 - is_stall)
        dy = dy * (1 - is_stall)

        all_dxdy[:, step, 0] = dx
        all_dxdy[:, step, 1] = dy
        all_stall[:, step] = is_stall
        cum_dx = cum_dx + dx
        cum_dy = cum_dy + dy

    return all_dxdy, all_stall


def build_trajectory(actions_dx, actions_dy, start_x, start_y, end_x, end_y, total_dist, hz=125.0):
    """Build pixel-space trajectory with magnitude-weighted endpoint correction."""
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


def build_input_features_from_actions(actions_dxdy, actions_stall, cos_a, sin_a, n_target, device):
    """Build input features tensor from generated actions for log prob computation."""
    T = n_target
    feat = torch.zeros(1, T, 6, device=device)

    cum_dx = torch.zeros(1, device=device)
    cum_dy = torch.zeros(1, device=device)

    for t in range(T):
        if t > 0:
            feat[0, t, 0] = actions_dxdy[t - 1, 0]
            feat[0, t, 1] = actions_dxdy[t - 1, 1]
            feat[0, t, 2] = actions_stall[t - 1]
            cum_dx = cum_dx + actions_dxdy[t - 1, 0]
            cum_dy = cum_dy + actions_dxdy[t - 1, 1]
        feat[0, t, 3] = cos_a - cum_dx
        feat[0, t, 4] = sin_a - cum_dy
        feat[0, t, 5] = 1.0 - t / T

    return feat


def compute_trajectory_log_prob(model, actions_dxdy, actions_stall, condition,
                                cos_a, sin_a, n_target, device, gate_bias=-1.0):
    """Compute total log prob of a generated trajectory under the current model."""
    feat = build_input_features_from_actions(
        actions_dxdy, actions_stall, cos_a, sin_a, n_target, device,
    )
    params = model(feat, condition)

    gate_logit = params["gate_logit"][0]  # (T,)
    mu = params["mu"]  # (1, T, M, 2)
    sigma = params["sigma"]
    rho = params["rho"]
    logit_pi = params["logit_pi"]
    log_pi = torch.log_softmax(logit_pi, dim=-1)

    total_log_prob = torch.tensor(0.0, device=device)

    for t in range(n_target):
        is_stall = actions_stall[t].item() > 0.5
        gate_log_prob = torch.nn.functional.logsigmoid(
            (gate_logit[t] + gate_bias) if is_stall else -(gate_logit[t] + gate_bias)
        )

        if is_stall:
            total_log_prob = total_log_prob + gate_log_prob
        else:
            dx_val = actions_dxdy[t, 0].unsqueeze(0).unsqueeze(0)  # (1, 1)
            dy_val = actions_dxdy[t, 1].unsqueeze(0).unsqueeze(0)
            mdn_log_prob = bivariate_log_prob(
                dx_val, dy_val,
                mu[:, t:t+1], sigma[:, t:t+1], rho[:, t:t+1], log_pi[:, t:t+1],
            )
            total_log_prob = total_log_prob + gate_log_prob + mdn_log_prob.squeeze()

    return total_log_prob


def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tune ZIMT")
    parser.add_argument("--checkpoint", default=str(DATA_DIR / "zimt_best.pt"))
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--n-queries", type=int, default=25)
    parser.add_argument("--n-per-query", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gate-bias", type=float, default=-1.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    device = torch.device(args.device)
    print(f"[GRPO] Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ZIMTModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[GRPO] Loaded ZIMT ({cfg['d_model']}d, {cfg['n_layers']}L, {cfg['n_components']}K)")

    # Load human features for RF training
    human_features = np.load(DATA_DIR / "human_eval_features.npy")
    human_distances = np.load(DATA_DIR / "human_distances.npy")
    print(f"[GRPO] Human: {len(human_features)} features, {len(human_distances)} distances")

    duration_model = DurationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    n_total = args.n_queries * args.n_per_query
    hz = 125.0
    center_x, center_y = 960.0, 540.0
    rng = np.random.default_rng(42)

    best_auc = 1.0
    print(f"\n[GRPO] Starting: {args.n_iters} iterations, {n_total} trajectories/iter")
    print(f"  LR={args.lr}, temp={args.temperature}, gate_bias={args.gate_bias}")

    for iteration in range(args.n_iters):
        if _stop_requested:
            print("[GRPO] Stopping early.")
            break

        t0 = time.time()
        model.eval()

        # --- Generate trajectories (batched per query) ---
        all_trajectories = []
        all_actions_dxdy = []
        all_actions_stall = []
        all_conditions = []
        all_cos_a = []
        all_sin_a = []
        all_n_targets = []
        query_indices = []

        for q in range(args.n_queries):
            dist = float(rng.choice(human_distances))
            angle = float(rng.uniform(0, 2 * np.pi))
            end_x = center_x + dist * np.cos(angle)
            end_y = center_y + dist * np.sin(angle)

            total_dist = math.hypot(end_x - center_x, end_y - center_y)
            if total_dist < 1.0:
                for _ in range(args.n_per_query):
                    all_trajectories.append([(center_x, center_y, 0.0), (end_x, end_y, 0.008)])
                    all_actions_dxdy.append(torch.zeros(1, 2, device=device))
                    all_actions_stall.append(torch.zeros(1, device=device))
                    all_conditions.append(torch.zeros(1, 4, device=device))
                    all_cos_a.append(0.0)
                    all_sin_a.append(0.0)
                    all_n_targets.append(1)
                    query_indices.append(q)
                continue

            log_dist = math.log(total_dist)
            cos_a = (end_x - center_x) / total_dist
            sin_a = (end_y - center_y) / total_dist
            total_duration = duration_model.sample(log_dist)
            log_dur = math.log(max(total_duration, 0.01))
            condition = torch.tensor([[log_dist, log_dur, cos_a, sin_a]],
                                     device=device, dtype=torch.float32)

            n_target = max(5, int(round(total_duration * hz)))
            n_target = min(n_target, cfg["max_seq_len"] - 2)

            batch_dxdy, batch_stall = generate_batch(
                model, condition, cos_a, sin_a, n_target,
                cfg["input_dim"], args.n_per_query, device,
                temperature=args.temperature, gate_bias=args.gate_bias,
            )

            for b in range(args.n_per_query):
                adx = batch_dxdy[b, :, 0].tolist()
                ady = batch_dxdy[b, :, 1].tolist()
                traj = build_trajectory(adx, ady, center_x, center_y, end_x, end_y, total_dist, hz)
                all_trajectories.append(traj)
                all_actions_dxdy.append(batch_dxdy[b])
                all_actions_stall.append(batch_stall[b])
                all_conditions.append(condition)
                all_cos_a.append(cos_a)
                all_sin_a.append(sin_a)
                all_n_targets.append(n_target)
                query_indices.append(q)

        gen_time = time.time() - t0

        # --- Compute rewards via RF ---
        synth_features = extract_feature_matrix(all_trajectories)
        n_valid = len(synth_features)

        if n_valid < n_total * 0.5:
            print(f"  Iter {iteration+1}: too few valid trajectories ({n_valid}/{n_total}), skipping")
            continue

        n_use = min(len(human_features), n_valid)
        X = np.vstack([human_features[:n_use], synth_features[:n_use]])
        y = np.concatenate([np.zeros(n_use), np.ones(n_use)])

        rf = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=-1, random_state=iteration)
        rf.fit(X, y)

        synth_scores = rf.predict_proba(synth_features)[:, 1]
        rewards = -synth_scores  # higher reward for lower P(synthetic)

        oob_auc = roc_auc_score(y, rf.oob_decision_function_[:, 1])

        # --- Compute within-group advantages ---
        rewards_by_query = {}
        indices_by_query = {}
        for i, q in enumerate(query_indices):
            if i >= n_valid:
                break
            rewards_by_query.setdefault(q, []).append(rewards[i])
            indices_by_query.setdefault(q, []).append(i)

        advantages = np.zeros(n_valid)
        for q in rewards_by_query:
            r = np.array(rewards_by_query[q])
            idx = indices_by_query[q]
            if len(r) > 1 and r.std() > 1e-6:
                adv = (r - r.mean()) / r.std()
            else:
                adv = np.zeros_like(r)
            for j, i in enumerate(idx):
                advantages[i] = adv[j]

        # --- Compute log probs and GRPO loss ---
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        n_counted = 0

        for i in range(min(n_valid, n_total)):
            if all_n_targets[i] < 2:
                continue

            log_prob = compute_trajectory_log_prob(
                model,
                all_actions_dxdy[i],
                all_actions_stall[i],
                all_conditions[i],
                all_cos_a[i],
                all_sin_a[i],
                all_n_targets[i],
                device,
                gate_bias=args.gate_bias,
            )

            # Normalize by trajectory length to prevent gradient explosion
            log_prob_normalized = log_prob / max(all_n_targets[i], 1)

            adv = torch.tensor(advantages[i], device=device, dtype=torch.float32)
            total_loss = total_loss - adv * log_prob_normalized
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

        # --- Logging ---
        mean_reward = rewards.mean()
        print(
            f"  Iter {iteration+1:4d} | "
            f"loss {total_loss.item():+.4f} | "
            f"reward {mean_reward:.3f} | "
            f"rf_auc {oob_auc:.4f} | "
            f"grad {grad_norm:.3f} | "
            f"gen {gen_time:.1f}s | "
            f"total {iter_time:.1f}s"
        )

        # --- Periodic evaluation ---
        if (iteration + 1) % args.eval_every == 0:
            model.eval()
            eval_trajs = []
            eval_rng = np.random.default_rng(42)
            for _ in range(200):
                dist = float(eval_rng.choice(human_distances))
                angle = float(eval_rng.uniform(0, 2 * np.pi))
                ex = center_x + dist * np.cos(angle)
                ey = center_y + dist * np.sin(angle)
                td = math.hypot(ex - center_x, ey - center_y)
                if td < 1.0:
                    eval_trajs.append([(center_x, center_y, 0.0), (ex, ey, 0.008)])
                    continue
                ld = math.log(td)
                ca = (ex - center_x) / td
                sa = (ey - center_y) / td
                dur = duration_model.sample(ld)
                ldur = math.log(max(dur, 0.01))
                cond = torch.tensor([[ld, ldur, ca, sa]], device=device, dtype=torch.float32)
                nt = max(5, min(int(round(dur * hz)), cfg["max_seq_len"] - 2))
                batch_dxdy, batch_stall = generate_batch(
                    model, cond, ca, sa, nt,
                    cfg["input_dim"], 1, device,
                    temperature=args.temperature, gate_bias=args.gate_bias,
                )
                adx = batch_dxdy[0, :, 0].tolist()
                ady = batch_dxdy[0, :, 1].tolist()
                eval_trajs.append(build_trajectory(adx, ady, center_x, center_y, ex, ey, td, hz))

            eval_feat = extract_feature_matrix(eval_trajs)
            n_eval = min(len(human_features), len(eval_feat))
            eX = np.vstack([human_features[:n_eval], eval_feat[:n_eval]])
            ey_labels = np.concatenate([np.zeros(n_eval), np.ones(n_eval)])
            eval_rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
            eval_rf.fit(eX, ey_labels)
            eval_auc = roc_auc_score(ey_labels, eval_rf.oob_decision_function_[:, 1])

            is_best = eval_auc < best_auc
            if is_best:
                best_auc = eval_auc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": ckpt.get("epoch", 0),
                    "val_loss": ckpt.get("val_loss", 0),
                    "phase": "grpo",
                    "grpo_iter": iteration + 1,
                    "grpo_auc": eval_auc,
                }, TRAINING_DIR / "zimt_grpo_best.pt")

            marker = " *BEST*" if is_best else ""
            print(f"  >>> EVAL iter {iteration+1}: AUC {eval_auc:.4f} (best {best_auc:.4f}){marker}")

        # Save latest checkpoint every iteration
        if (iteration + 1) % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": ckpt.get("epoch", 0),
                "val_loss": ckpt.get("val_loss", 0),
                "phase": "grpo",
                "grpo_iter": iteration + 1,
            }, TRAINING_DIR / "zimt_grpo_latest.pt")

    print(f"\n[GRPO] Done. Best eval AUC: {best_auc:.4f}")
    print(f"  Best checkpoint: {TRAINING_DIR / 'zimt_grpo_best.pt'}")


if __name__ == "__main__":
    main()
