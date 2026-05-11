"""
Train ZIMT (Zero-Inflated Mouse Trajectory Generator).

Phased training for rapid iteration:
  Phase 1:  50K traj, max_len=64,  5 epochs  (~15 min)
  Phase 2: 200K traj, max_len=128, 20 epochs (~2-3 hrs)
  Phase 3: 500K traj, max_len=256, 40 epochs (~6-8 hrs)

Run with: python -m training.train_zimt [--phase 1|2|3] [--resume PATH]
"""
from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.zimt import ZIMTModel, zimt_loss, jerk_loss

TRAINING_DIR = Path(__file__).resolve().parent
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

PHASE_CONFIG = {
    1: {"max_traj": 50_000,  "max_len": 64,  "n_epochs": 5,  "batch_size": 256, "lr": 3e-4},
    2: {"max_traj": 200_000, "max_len": 128, "n_epochs": 20, "batch_size": 128, "lr": 3e-4},
    3: {"max_traj": 500_000, "max_len": 256, "n_epochs": 40, "batch_size": 64,  "lr": 1e-4},
}

_stop_requested = False


def _handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[ZIMT] Graceful stop requested, finishing epoch...")


class ZIMTDataset(Dataset):
    def __init__(self, dxdy, stall, lengths, conditions, endpoints, max_len):
        self.dxdy = dxdy
        self.stall = stall
        self.lengths = lengths
        self.conditions = conditions
        self.endpoints = endpoints
        self.max_len = max_len

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        seq_len = min(int(self.lengths[idx]), self.max_len)
        dxdy = self.dxdy[idx, :seq_len].copy()
        stall = self.stall[idx, :seq_len].copy()
        cond = self.conditions[idx].copy()
        ep = self.endpoints[idx].copy()
        return dxdy, stall, seq_len, cond, ep


def collate_fn(batch):
    max_len = max(b[2] for b in batch)
    B = len(batch)

    dxdy = torch.zeros(B, max_len, 2)
    stall = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    cond = torch.zeros(B, 4)
    ep = torch.zeros(B, 4)

    for i, (d, s, l, c, e) in enumerate(batch):
        dxdy[i, :l] = torch.from_numpy(d)
        stall[i, :l] = torch.from_numpy(s.astype(np.float32))
        mask[i, :l] = True
        cond[i] = torch.from_numpy(c)
        ep[i] = torch.from_numpy(e)

    return dxdy, stall, mask, cond, ep


def build_input_features(dxdy, stall, mask, endpoints, conditions):
    """
    Build per-step input features: (dx_prev, dy_prev, stall_prev,
    remaining_dx, remaining_dy, remaining_frac).

    dxdy:      (B, T, 2) target displacements
    stall:     (B, T) target stall flags
    mask:      (B, T) valid mask
    endpoints: (B, 4) [start_x, start_y, end_x, end_y]
    conditions: (B, 4) [log_dist, log_dur, cos_a, sin_a]
    Returns:   (B, T, 6) input features
    """
    B, T, _ = dxdy.shape
    device = dxdy.device

    # Previous displacement (shifted right, zero for first step)
    dx_prev = torch.zeros(B, T, 2, device=device)
    dx_prev[:, 1:] = dxdy[:, :-1]

    stall_prev = torch.zeros(B, T, 1, device=device)
    stall_prev[:, 1:] = stall[:, :-1].unsqueeze(-1)

    # Endpoint conditioning: remaining displacement BEFORE current step
    cumsum_shifted = torch.zeros_like(dxdy)
    cumsum_shifted[:, 1:] = torch.cumsum(dxdy[:, :-1], dim=1)
    total_disp = endpoints[:, 2:4] - endpoints[:, 0:2]  # (B, 2)
    remaining = total_disp.unsqueeze(1) - cumsum_shifted  # (B, T, 2)

    total_dist = torch.sqrt((total_disp ** 2).sum(dim=-1, keepdim=True)).clamp(min=1e-6)  # (B, 1)
    remaining_norm = remaining / total_dist.unsqueeze(1)  # (B, T, 2)

    # Progress fraction: step/length (0 at start, approaches 1 at end)
    lengths = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
    t_idx = torch.arange(T, device=device).unsqueeze(0).float()  # (1, T)
    remaining_frac = 1.0 - t_idx / lengths.clamp(min=1.0)  # (B, T)
    remaining_frac = remaining_frac.clamp(0.0, 1.0).unsqueeze(-1)  # (B, T, 1)

    return torch.cat([dx_prev, stall_prev, remaining_norm, remaining_frac], dim=-1)


@torch.no_grad()
def sample_from_params(params):
    """Sample (dx, dy) and stall from model output for scheduled sampling."""
    gate_logit = params["gate_logit"]
    stall_probs = torch.sigmoid(gate_logit)
    sampled_stall = torch.bernoulli(stall_probs)

    pi = params["pi"]
    mu = params["mu"]
    sigma = params["sigma"]
    rho = params["rho"]
    B, T, M = pi.shape

    comp_idx = torch.multinomial(pi.reshape(B * T, M), 1).reshape(B, T)
    ci = comp_idx.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, 2)
    sel_mu = mu.gather(2, ci).squeeze(2)
    sel_sigma = sigma.gather(2, ci).squeeze(2)
    sel_rho = rho.gather(2, comp_idx.unsqueeze(-1)).squeeze(2)

    z1 = torch.randn(B, T, device=gate_logit.device)
    z2 = torch.randn(B, T, device=gate_logit.device)
    dx = sel_mu[:, :, 0] + sel_sigma[:, :, 0] * z1
    dy = sel_mu[:, :, 1] + sel_sigma[:, :, 1] * (
        sel_rho * z1 + torch.sqrt((1 - sel_rho ** 2).clamp(min=1e-8)) * z2
    )
    sampled_dxdy = torch.stack([dx, dy], dim=-1)
    sampled_dxdy = sampled_dxdy * (1 - sampled_stall.unsqueeze(-1))
    return sampled_dxdy, sampled_stall


def train_epoch(model, loader, optimizer, gate_pw, max_len, jerk_lambda=0.0, ss_prob=0.0):
    model.train()
    total_loss = 0.0
    total_gate = 0.0
    total_mdn = 0.0
    total_jerk = 0.0
    n_batches = 0

    for dxdy, stall, mask, cond, ep in loader:
        dxdy = dxdy[:, :max_len].to(DEVICE)
        stall = stall[:, :max_len].to(DEVICE)
        mask = mask[:, :max_len].to(DEVICE)
        cond = cond.to(DEVICE)
        ep = ep.to(DEVICE)

        input_feat = build_input_features(dxdy, stall, mask, ep, cond)

        if ss_prob > 0:
            with torch.no_grad():
                params_tf = model(input_feat, cond)
                s_dxdy, s_stall = sample_from_params(params_tf)
            B, T, _ = dxdy.shape
            use_model = torch.bernoulli(
                torch.full((B, T), ss_prob, device=DEVICE),
            ).bool() & mask
            use_model[:, 0] = False
            mixed_dxdy = torch.where(use_model.unsqueeze(-1), s_dxdy, dxdy)
            mixed_stall = torch.where(use_model, s_stall, stall.float())
            input_feat = build_input_features(mixed_dxdy, mixed_stall, mask, ep, cond)

        params = model(input_feat, cond)
        loss, gate_l, mdn_l = zimt_loss(params, dxdy, stall, mask, gate_pos_weight=gate_pw)

        if jerk_lambda > 0:
            jl = jerk_loss(dxdy, mask)
            loss = loss + jerk_lambda * jl
            total_jerk += jl.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_gate += gate_l.item()
        total_mdn += mdn_l.item()
        n_batches += 1

    return total_loss / n_batches, total_gate / n_batches, total_mdn / n_batches, total_jerk / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, gate_pw, max_len):
    model.eval()
    total_loss = 0.0
    total_gate = 0.0
    total_mdn = 0.0
    n_stall_correct = 0
    n_stall_total = 0
    n_motion_correct = 0
    n_motion_total = 0
    predicted_stalls = 0
    total_steps = 0
    n_batches = 0

    for dxdy, stall, mask, cond, ep in loader:
        dxdy = dxdy[:, :max_len].to(DEVICE)
        stall = stall[:, :max_len].to(DEVICE)
        mask = mask[:, :max_len].to(DEVICE)
        cond = cond.to(DEVICE)
        ep = ep.to(DEVICE)

        input_feat = build_input_features(dxdy, stall, mask, ep, cond)
        params = model(input_feat, cond)

        loss, gate_l, mdn_l = zimt_loss(params, dxdy, stall, mask, gate_pos_weight=gate_pw)

        gate_pred = (torch.sigmoid(params["gate_logit"]) > 0.5).float()
        valid = mask.float()

        stall_mask = stall.bool() & mask
        motion_mask = (~stall.bool()) & mask

        n_stall_correct += ((gate_pred == 1) & stall_mask).sum().item()
        n_stall_total += stall_mask.sum().item()
        n_motion_correct += ((gate_pred == 0) & motion_mask).sum().item()
        n_motion_total += motion_mask.sum().item()
        predicted_stalls += (gate_pred * valid).sum().item()
        total_steps += valid.sum().item()

        total_loss += loss.item()
        total_gate += gate_l.item()
        total_mdn += mdn_l.item()
        n_batches += 1

    stall_recall = n_stall_correct / max(n_stall_total, 1)
    motion_recall = n_motion_correct / max(n_motion_total, 1)
    pred_stall_rate = predicted_stalls / max(total_steps, 1)
    true_stall_rate = n_stall_total / max(total_steps, 1)

    return {
        "loss": total_loss / n_batches,
        "gate_loss": total_gate / n_batches,
        "mdn_loss": total_mdn / n_batches,
        "stall_recall": stall_recall,
        "motion_recall": motion_recall,
        "pred_stall_rate": pred_stall_rate,
        "true_stall_rate": true_stall_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ZIMT")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gate-pos-weight", type=float, default=5.0)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--jerk-lambda", type=float, default=0.0)
    parser.add_argument("--ss-max", type=float, default=0.3,
                        help="Max scheduled sampling probability")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)

    cfg = PHASE_CONFIG[args.phase]
    print(f"[ZIMT] Phase {args.phase}: {cfg['max_traj']} traj, "
          f"max_len={cfg['max_len']}, {cfg['n_epochs']} epochs, "
          f"batch={cfg['batch_size']}, lr={cfg['lr']}")
    print(f"  Device: {DEVICE}")

    # Load data
    print("Loading ZIMT data...")
    t0 = time.time()
    all_dxdy = np.load(TRAINING_DIR / "zimt_dxdy.npy", mmap_mode="r")
    all_stall = np.load(TRAINING_DIR / "zimt_stall.npy", mmap_mode="r")
    all_lengths = np.load(TRAINING_DIR / "zimt_lengths.npy")
    all_conditions = np.load(TRAINING_DIR / "zimt_conditions.npy", mmap_mode="r")
    all_endpoints = np.load(TRAINING_DIR / "zimt_endpoints.npy", mmap_mode="r")

    # Filter: length >= 5 and length <= current max_len * 2 (some padding ok)
    valid = (all_lengths >= 5) & (all_lengths <= cfg["max_len"] * 2)
    valid_idx = np.where(valid)[0]
    print(f"  {len(valid_idx)} valid of {len(all_lengths)} "
          f"(filtered to len 5-{cfg['max_len']*2})")

    # Subsample
    rng = np.random.default_rng(42)
    max_traj = min(cfg["max_traj"], len(valid_idx))
    sub_idx = valid_idx if len(valid_idx) <= max_traj else valid_idx[
        rng.choice(len(valid_idx), max_traj, replace=False)
    ]
    print(f"  Using {len(sub_idx)} trajectories")

    # Need to load into memory for DataLoader
    dxdy = np.array(all_dxdy[sub_idx])
    stall = np.array(all_stall[sub_idx])
    lengths = all_lengths[sub_idx]
    conditions = np.array(all_conditions[sub_idx])
    endpoints = np.array(all_endpoints[sub_idx])

    # Train/val split
    n_val = max(2000, len(sub_idx) // 20)
    perm = rng.permutation(len(sub_idx))

    train_ds = ZIMTDataset(
        dxdy[perm[n_val:]], stall[perm[n_val:]], lengths[perm[n_val:]],
        conditions[perm[n_val:]], endpoints[perm[n_val:]], cfg["max_len"],
    )
    val_ds = ZIMTDataset(
        dxdy[perm[:n_val]], stall[perm[:n_val]], lengths[perm[:n_val]],
        conditions[perm[:n_val]], endpoints[perm[:n_val]], cfg["max_len"],
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"  Data loaded in {time.time()-t0:.1f}s")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Model
    model_cfg = {
        "input_dim": 6, "d_model": args.d_model, "n_heads": 4,
        "n_layers": args.n_layers, "d_ff": args.d_model * 4,
        "max_seq_len": 256, "cond_dim": 4,
        "n_components": args.n_components, "dropout": 0.1,
    }

    model = ZIMTModel(**model_cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params ({args.d_model}d, {args.n_layers}L, "
          f"{args.n_components}K MDN)")

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed from {args.resume} (epoch {start_epoch})")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["n_epochs"],
    )

    n_epochs = cfg["n_epochs"]
    ss_max = args.ss_max
    if ss_max > 0:
        print(f"  Scheduled sampling: 0 -> {ss_max:.2f} over {n_epochs} epochs")

    print(f"\nTraining ({n_epochs} epochs)...")
    best_val_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + n_epochs):
        if _stop_requested:
            print("[ZIMT] Stopping early.")
            break

        ss_prob = ss_max * min((epoch - start_epoch) / max(n_epochs - 1, 1), 1.0)

        epoch_t0 = time.time()
        train_loss, train_gate, train_mdn, train_jerk = train_epoch(
            model, train_loader, optimizer, args.gate_pos_weight, cfg["max_len"],
            jerk_lambda=args.jerk_lambda, ss_prob=ss_prob,
        )
        scheduler.step()

        val = validate(model, val_loader, args.gate_pos_weight, cfg["max_len"])
        epoch_time = time.time() - epoch_t0

        is_best = val["loss"] < best_val_loss
        if is_best:
            best_val_loss = val["loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model_cfg,
                "gate_pos_weight": args.gate_pos_weight,
                "epoch": epoch + 1,
                "val_loss": val["loss"],
                "phase": args.phase,
            }, TRAINING_DIR / "zimt_best.pt")

        marker = " *BEST*" if is_best else ""
        jerk_str = f" jerk {train_jerk:.4f}" if args.jerk_lambda > 0 else ""
        ss_str = f" ss={ss_prob:.2f}" if ss_max > 0 else ""
        print(
            f"  Ep {epoch+1:3d} | "
            f"train {train_loss:.4f} (gate {train_gate:.4f} mdn {train_mdn:.4f}{jerk_str}) | "
            f"val {val['loss']:.4f} (gate {val['gate_loss']:.4f} mdn {val['mdn_loss']:.4f}) | "
            f"stall_rec {val['stall_recall']:.2f} mot_rec {val['motion_recall']:.2f} | "
            f"pred_stall {val['pred_stall_rate']:.3f} true {val['true_stall_rate']:.3f} | "
            f"{epoch_time:.0f}s{ss_str}{marker}"
        )

        # Save latest checkpoint every epoch
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model_cfg,
            "gate_pos_weight": args.gate_pos_weight,
            "epoch": epoch + 1,
            "val_loss": val["loss"],
            "phase": args.phase,
        }, TRAINING_DIR / "zimt_latest.pt")

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {TRAINING_DIR / 'zimt_best.pt'}")


if __name__ == "__main__":
    main()
