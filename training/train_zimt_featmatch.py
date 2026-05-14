"""
Differentiable feature-matching fine-tuning for ZIMT.

Generates trajectories with reparameterized sampling (Gumbel-Softmax for
component selection, straight-through for stalls) and directly backpropagates
through differentiable feature computation. No REINFORCE needed.

Loss = Σ_j (batch_mean(f_j) - human_mean_j)² / human_std_j²

Run: python -m training.train_zimt_featmatch [--checkpoint PATH] [--n-iters 200]
"""
from __future__ import annotations

import argparse
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import FEATURE_NAMES, extract_feature_matrix
from models.zimt import ZIMTModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAINING_DIR.parent / "data"

_stop_requested = False


def _handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[FM] Graceful stop requested...")


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


@torch.no_grad()
def _generate_reference(model, condition, cos_a, sin_a, n_target, input_dim,
                        device, temperature=1.0, gate_bias=-1.0):
    """Generate a single reference trajectory without gradient (fast)."""
    input_buf = torch.zeros(1, n_target, input_dim, device=device)
    cond = condition
    dxs, dys = [], []
    cum_dx, cum_dy = 0.0, 0.0

    for step in range(n_target):
        if step > 0:
            input_buf[0, step, 0] = dxs[-1]
            input_buf[0, step, 1] = dys[-1]
            input_buf[0, step, 2] = 1.0 if (abs(dxs[-1]) + abs(dys[-1])) < 1e-6 else 0.0
        input_buf[0, step, 3] = cos_a - cum_dx
        input_buf[0, step, 4] = sin_a - cum_dy
        input_buf[0, step, 5] = 1.0 - step / n_target

        from models.zimt import sample_step
        params = model(input_buf[:, :step + 1], cond)
        dx, dy, _ = sample_step(params, temperature=temperature, gate_bias=gate_bias)
        dxs.append(dx)
        dys.append(dy)
        cum_dx += dx
        cum_dy += dy

    return dxs, dys


def differentiable_generate(model, condition, cos_a, sin_a, n_target, input_dim,
                            batch_size, device, temperature=1.0, gate_bias=-1.0,
                            gumbel_tau=0.5):
    """Two-phase generation: reference trajectory (no grad) + parallel reparameterized sampling.

    Phase 1: Generate B reference trajectories autoregressively (no grad).
    Phase 2: Teacher-force from references, single forward pass, reparameterized sample.
    """
    B = batch_size

    # Phase 1: generate reference trajectories (no gradient, fast)
    ref_dx = torch.zeros(B, n_target, device=device)
    ref_dy = torch.zeros(B, n_target, device=device)
    for b in range(B):
        dxs, dys = _generate_reference(
            model, condition, cos_a, sin_a, n_target, input_dim,
            device, temperature, gate_bias,
        )
        ref_dx[b] = torch.tensor(dxs, device=device)
        ref_dy[b] = torch.tensor(dys, device=device)

    # Phase 2: build teacher-forced input from references, single forward pass
    input_buf = torch.zeros(B, n_target, input_dim, device=device)
    input_buf[:, 1:, 0] = ref_dx[:, :-1]
    input_buf[:, 1:, 1] = ref_dy[:, :-1]
    stall_ref = ((ref_dx.abs() + ref_dy.abs()) < 1e-6).float()
    input_buf[:, 1:, 2] = stall_ref[:, :-1]
    cum_ref = torch.zeros_like(ref_dx)
    cum_ref[:, 1:] = torch.cumsum(ref_dx[:, :-1], dim=1)
    cum_ref_y = torch.zeros_like(ref_dy)
    cum_ref_y[:, 1:] = torch.cumsum(ref_dy[:, :-1], dim=1)
    input_buf[:, :, 3] = cos_a - cum_ref
    input_buf[:, :, 4] = sin_a - cum_ref_y
    t_frac = torch.arange(n_target, device=device).float() / n_target
    input_buf[:, :, 5] = 1.0 - t_frac.unsqueeze(0)

    cond = condition.expand(B, -1)
    params = model(input_buf, cond)  # single forward pass (B, T, ...)

    gate_logit = params["gate_logit"]       # (B, T)
    logit_pi = params["logit_pi"]           # (B, T, M)
    mu = params["mu"]                       # (B, T, M, 2)
    sigma = params["sigma"]                 # (B, T, M, 2)
    rho = params["rho"]                     # (B, T, M)

    if temperature != 1.0:
        logit_pi = logit_pi / temperature
        sigma = sigma * temperature

    # Gumbel-Softmax component selection at all timesteps
    B_, T_, M_ = logit_pi.shape
    comp_weight = F.gumbel_softmax(
        logit_pi.reshape(B_ * T_, M_), tau=gumbel_tau, hard=True
    ).reshape(B_, T_, M_)

    sel_mu_x = (comp_weight * mu[:, :, :, 0]).sum(dim=-1)     # (B, T)
    sel_mu_y = (comp_weight * mu[:, :, :, 1]).sum(dim=-1)
    sel_sigma_x = (comp_weight * sigma[:, :, :, 0]).sum(dim=-1)
    sel_sigma_y = (comp_weight * sigma[:, :, :, 1]).sum(dim=-1)
    sel_rho = (comp_weight * rho).sum(dim=-1)

    z1 = torch.randn(B, n_target, device=device)
    z2 = torch.randn(B, n_target, device=device)
    dx_seq = sel_mu_x + sel_sigma_x * z1
    dy_seq = sel_mu_y + sel_sigma_y * (
        sel_rho * z1 + torch.sqrt((1 - sel_rho ** 2).clamp(min=1e-8)) * z2
    )

    stall_prob = torch.sigmoid(gate_logit + gate_bias)
    stall_hard = (stall_prob > 0.5).float()
    stall_st = stall_hard - stall_prob.detach() + stall_prob
    dx_seq = dx_seq * (1 - stall_st)
    dy_seq = dy_seq * (1 - stall_st)

    return dx_seq, dy_seq  # (B, T)


def differentiable_features(dx_seq, dy_seq, total_dist, hz=125.0):
    """Compute differentiable trajectory features from displacement sequences.

    dx_seq, dy_seq: (B, T) normalized displacements
    total_dist: scalar pixel distance
    Returns: dict of feature tensors, each (B,)
    """
    B, T = dx_seq.shape
    dt = 1.0 / hz

    # Pixel displacements
    px_dx = dx_seq * total_dist
    px_dy = dy_seq * total_dist

    # Speed
    ds = torch.sqrt(px_dx ** 2 + px_dy ** 2 + 1e-8)
    speed = ds / dt  # (B, T)

    # Acceleration
    dv = speed[:, 1:] - speed[:, :-1]
    acc = dv / dt  # (B, T-1)

    # Jerk
    if T > 2:
        da = acc[:, 1:] - acc[:, :-1]
        jerk = da / dt  # (B, T-2)
    else:
        jerk = torch.zeros(B, 1, device=dx_seq.device)

    # Direction angles
    angles = torch.atan2(px_dy, px_dx)  # (B, T)
    angle_diff = angles[:, 1:] - angles[:, :-1]
    # Wrap to [-pi, pi]
    angle_diff = torch.remainder(angle_diff + math.pi, 2 * math.pi) - math.pi
    omega = (angle_diff / dt).clamp(-50.0, 50.0)  # angular velocity (B, T-1)

    # Path geometry
    cum_x = torch.cumsum(px_dx, dim=1)
    cum_y = torch.cumsum(px_dy, dim=1)
    total_path = ds.sum(dim=1)  # (B,)
    final_dist = torch.sqrt(cum_x[:, -1] ** 2 + cum_y[:, -1] ** 2 + 1e-8)
    path_efficiency = final_dist / total_path.clamp(min=1e-6)

    # Velocity skewness (third standardized moment)
    speed_mean = speed.mean(dim=1, keepdim=True)
    speed_std = speed.std(dim=1, keepdim=True).clamp(min=1e-8)
    speed_z = (speed - speed_mean) / speed_std
    vel_skew = (speed_z ** 3).mean(dim=1)

    # Time to peak velocity (soft argmax)
    t_fracs = torch.linspace(0, 1, T, device=dx_seq.device).unsqueeze(0).expand(B, -1)
    soft_peak_weight = F.softmax(speed * 5.0, dim=1)  # temperature-scaled softmax
    time_to_peak = (soft_peak_weight * t_fracs).sum(dim=1)

    # Curvature: |vx*ay - vy*ax| / speed^3
    vx = px_dx / dt
    vy = px_dy / dt
    if T > 1:
        ax = (vx[:, 1:] - vx[:, :-1]) / dt
        ay = (vy[:, 1:] - vy[:, :-1]) / dt
        cross = torch.abs(vx[:, :-1] * ay - vy[:, :-1] * ax)
        speed_mid = speed[:, :-1].clamp(min=10.0)
        curvature = cross / (speed_mid ** 3)
        curvature = curvature.clamp(max=1.0)
    else:
        curvature = torch.zeros(B, 1, device=dx_seq.device)

    return {
        "mean_velocity": speed.mean(dim=1),
        "std_velocity": speed.std(dim=1),
        "max_velocity": speed.max(dim=1).values,
        "velocity_skewness": vel_skew,
        "mean_acceleration": acc.mean(dim=1) if T > 1 else torch.zeros(B, device=dx_seq.device),
        "std_acceleration": acc.std(dim=1) if T > 1 else torch.zeros(B, device=dx_seq.device),
        "max_acceleration": acc.abs().max(dim=1).values if T > 1 else torch.zeros(B, device=dx_seq.device),
        "mean_jerk": jerk.mean(dim=1),
        "std_jerk": jerk.std(dim=1),
        "path_efficiency": path_efficiency,
        "curvature_mean": curvature.mean(dim=1),
        "curvature_std": curvature.std(dim=1),
        "time_to_peak_velocity": time_to_peak,
        "angular_velocity_mean": omega.abs().mean(dim=1) if T > 1 else torch.zeros(B, device=dx_seq.device),
        "angular_velocity_std": omega.std(dim=1) if T > 1 else torch.zeros(B, device=dx_seq.device),
    }


def build_trajectory_np(dx_seq, dy_seq, start_x, start_y, end_x, end_y,
                         total_dist, hz=125.0):
    """Build trajectory for evaluation (non-differentiable)."""
    T = len(dx_seq)
    px = [start_x]
    py = [start_y]
    cx, cy = start_x, start_y
    for t in range(T):
        cx += dx_seq[t] * total_dist
        cy += dy_seq[t] * total_dist
        px.append(cx)
        py.append(cy)

    n = len(px)
    step_mags = [math.hypot(px[i] - px[i-1], py[i] - py[i-1]) for i in range(1, n)]
    err_x = end_x - px[-1]
    err_y = end_y - py[-1]
    if err_x**2 + err_y**2 > 0.01:
        moving = [m > 0.3 for m in step_mags]
        total_mov = sum(m for m, mv in zip(step_mags, moving) if mv)
        if total_mov > 0.1:
            ccx, ccy = 0.0, 0.0
            for i in range(len(step_mags)):
                if moving[i]:
                    w = step_mags[i] / total_mov
                    ccx += err_x * w
                    ccy += err_y * w
                px[i+1] += ccx
                py[i+1] += ccy

    dt = 1.0 / hz
    result = [(float(px[i]), float(py[i]), i * dt) for i in range(n)]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, result[-1][2])
    return result


def main():
    parser = argparse.ArgumentParser(description="Feature-matching fine-tune ZIMT")
    parser.add_argument("--checkpoint", default=str(DATA_DIR / "zimt_best.pt"))
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--n-queries", type=int, default=5)
    parser.add_argument("--n-per-query", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--nll-weight", type=float, default=0.1,
                        help="Weight of NLL loss to prevent collapse")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gate-bias", type=float, default=-1.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    device = torch.device(args.device)
    print(f"[FM] Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ZIMTModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[FM] ZIMT ({cfg['d_model']}d, {cfg['n_layers']}L, {cfg['n_components']}K)")

    # Also load NLL training data for regularization
    _N_NLL = 1000
    nll_dxdy = torch.from_numpy(np.load(TRAINING_DIR / "zimt_dxdy.npy")[:_N_NLL, :cfg["max_seq_len"]]).float().to(device)
    nll_lengths = np.load(TRAINING_DIR / "zimt_lengths.npy")[:_N_NLL]
    nll_conds = torch.from_numpy(np.load(TRAINING_DIR / "zimt_conditions.npy")[:_N_NLL].copy()).float().to(device)
    nll_endpoints = torch.from_numpy(np.load(TRAINING_DIR / "zimt_endpoints.npy")[:_N_NLL].copy()).float().to(device)
    nll_masks = torch.zeros(_N_NLL, cfg["max_seq_len"], dtype=torch.bool, device=device)
    for i in range(_N_NLL):
        nll_masks[i, :min(int(nll_lengths[i]), cfg["max_seq_len"])] = True
    nll_stalls = (nll_dxdy[:, :, 0].abs() + nll_dxdy[:, :, 1].abs()) < 1e-8

    human_features = np.load(DATA_DIR / "human_eval_features.npy")
    human_distances = np.load(DATA_DIR / "human_distances.npy")
    h_mean = torch.from_numpy(human_features.mean(axis=0)).float().to(device)
    h_std = torch.from_numpy(np.maximum(human_features.std(axis=0), 1e-8)).float().to(device)

    # Feature name to index mapping for the differentiable features
    diff_feature_names = [
        "mean_velocity", "std_velocity", "max_velocity", "velocity_skewness",
        "mean_acceleration", "std_acceleration", "max_acceleration",
        "mean_jerk", "std_jerk", "path_efficiency",
        "curvature_mean", "curvature_std",
        "time_to_peak_velocity",
        "angular_velocity_mean", "angular_velocity_std",
    ]
    # Map to indices in the 18-feature vector
    feat_indices = {name: FEATURE_NAMES.index(name) for name in diff_feature_names if name in FEATURE_NAMES}

    duration_model = DurationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    hz = 125.0
    cx, cy = 960.0, 540.0
    rng = np.random.default_rng(42)
    best_auc = 1.0
    n_total = args.n_queries * args.n_per_query

    print(f"\n[FM] Starting: {args.n_iters} iters, {n_total} trajs/iter")
    print(f"  LR={args.lr}, nll_weight={args.nll_weight}")

    for iteration in range(args.n_iters):
        if _stop_requested:
            break

        t0 = time.time()
        model.train()

        # --- Feature-matching loss via differentiable generation ---
        all_feat_vals = {name: [] for name in diff_feature_names}

        for q in range(args.n_queries):
            dist = float(rng.choice(human_distances))
            angle = float(rng.uniform(0, 2 * np.pi))
            ex = cx + dist * np.cos(angle)
            ey = cy + dist * np.sin(angle)
            td = math.hypot(ex - cx, ey - cy)
            if td < 1.0:
                continue

            ld = math.log(td)
            ca = (ex - cx) / td
            sa = (ey - cy) / td
            dur = duration_model.sample(ld)
            ldur = math.log(max(dur, 0.01))
            cond = torch.tensor([[ld, ldur, ca, sa]], device=device, dtype=torch.float32)
            nt = max(5, min(int(round(dur * hz)), cfg["max_seq_len"] - 2))

            dx_seq, dy_seq = differentiable_generate(
                model, cond, ca, sa, nt, cfg["input_dim"],
                args.n_per_query, device, args.temperature, args.gate_bias,
            )

            feats = differentiable_features(dx_seq, dy_seq, td, hz)
            for name in diff_feature_names:
                if name in feats:
                    all_feat_vals[name].append(feats[name])

        # Compute feature-matching loss (L1 for stable gradients)
        fm_loss = torch.tensor(0.0, device=device)
        n_feats = 0
        for name in diff_feature_names:
            if name not in feat_indices or not all_feat_vals[name]:
                continue
            vals = torch.cat(all_feat_vals[name])  # (N,)
            idx = feat_indices[name]
            target_mean = h_mean[idx]
            target_std = h_std[idx]
            batch_mean = vals.mean()
            fm_loss = fm_loss + ((batch_mean - target_mean) / target_std).abs()
            n_feats += 1

        if n_feats > 0:
            fm_loss = fm_loss / n_feats

        # --- NLL regularization (prevents collapse) ---
        nll_loss = torch.tensor(0.0, device=device)
        if args.nll_weight > 0:
            nll_idx = rng.integers(0, _N_NLL, size=16)
            nll_batch = nll_dxdy[nll_idx]
            nll_mask = nll_masks[nll_idx]
            nll_cond = nll_conds[nll_idx]
            nll_stall_batch = nll_stalls[nll_idx]

            # Build input features for teacher forcing
            nll_bs = len(nll_idx)
            nll_ep = nll_endpoints[nll_idx]
            nll_input = torch.zeros(nll_bs, cfg["max_seq_len"], cfg["input_dim"], device=device)
            nll_input[:, 1:, 0] = nll_batch[:, :-1, 0]
            nll_input[:, 1:, 1] = nll_batch[:, :-1, 1]
            nll_input[:, 1:, 2] = nll_stall_batch[:, :-1].float()
            total_disp = nll_ep[:, 2:4] - nll_ep[:, 0:2]
            td = torch.sqrt((total_disp ** 2).sum(-1, keepdim=True)).clamp(min=1e-6)
            cum_shifted = torch.zeros_like(nll_batch)
            cum_shifted[:, 1:] = torch.cumsum(nll_batch[:, :-1], dim=1)
            remaining = total_disp.unsqueeze(1) - cum_shifted
            remaining_norm = remaining / td.unsqueeze(1)
            nll_input[:, :, 3] = remaining_norm[:, :, 0]
            nll_input[:, :, 4] = remaining_norm[:, :, 1]
            lengths_f = nll_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            t_idx = torch.arange(cfg["max_seq_len"], device=device).unsqueeze(0).float()
            nll_input[:, :, 5] = (1.0 - t_idx / lengths_f).clamp(0, 1)

            from models.zimt import zimt_loss
            params = model(nll_input, nll_cond)
            nll_loss, _, _ = zimt_loss(params, nll_batch, nll_stall_batch, nll_mask)

        total_loss = fm_loss + args.nll_weight * nll_loss

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()

        iter_time = time.time() - t0
        print(
            f"  Iter {iteration+1:4d} | "
            f"fm {fm_loss.item():.4f} | "
            f"nll {nll_loss.item():.4f} | "
            f"grad {grad_norm:.3f} | "
            f"{iter_time:.1f}s"
        )

        # --- Periodic evaluation ---
        if (iteration + 1) % args.eval_every == 0:
            model.eval()
            eval_trajs = []
            eval_rng = np.random.default_rng(42)
            with torch.no_grad():
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

                    dx_s, dy_s = differentiable_generate(
                        model, cond_, ca_, sa_, nt_, cfg["input_dim"],
                        1, device, args.temperature, args.gate_bias,
                    )
                    adx = dx_s[0].cpu().tolist()
                    ady = dy_s[0].cpu().tolist()
                    eval_trajs.append(build_trajectory_np(adx, ady, cx, cy, ex_, ey_, td_, hz))

            ef = extract_feature_matrix(eval_trajs)
            n_ev = min(len(human_features), len(ef))
            eX = np.vstack([human_features[:n_ev], ef[:n_ev]])
            ey_lab = np.concatenate([np.zeros(n_ev), np.ones(n_ev)])
            erf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
            erf.fit(eX, ey_lab)
            eval_auc = roc_auc_score(ey_lab, erf.oob_decision_function_[:, 1])

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
                    "phase": "featmatch",
                    "fm_iter": iteration + 1,
                    "fm_auc": eval_auc,
                }, TRAINING_DIR / "zimt_fm_best.pt")

            marker = " *BEST*" if is_best else ""
            print(f"  >>> EVAL {iteration+1}: AUC {eval_auc:.4f} (best {best_auc:.4f}){marker}")
            print(f"      Top gaps: {top_str}")

        if (iteration + 1) % 25 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "phase": "featmatch",
                "fm_iter": iteration + 1,
            }, TRAINING_DIR / "zimt_fm_latest.pt")

    print(f"\n[FM] Done. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
