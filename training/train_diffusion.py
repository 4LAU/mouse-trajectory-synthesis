"""
Train trajectory diffusion model.

Uses the same preprocessed data as ZIMT (zimt_dxdy.npy etc.).
Predicts x_0 directly with MSE loss, masked for valid timesteps.

Run: python -m training.train_diffusion [--epochs 100] [--batch-size 128]
"""
from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.traj_diffusion import TrajectoryDiffusionModel

TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAINING_DIR.parent / "data"

_stop_requested = False


def _handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[Diffusion] Graceful stop requested...")


def main():
    parser = argparse.ArgumentParser(description="Train trajectory diffusion model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--n-diff-steps", type=int, default=1000)
    parser.add_argument("--cond-drop", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    device = torch.device(args.device)

    print("[Diffusion] Loading data...")
    all_dxdy = np.load(TRAINING_DIR / "zimt_dxdy.npy")
    all_lengths = np.load(TRAINING_DIR / "zimt_lengths.npy")
    all_conditions = np.load(TRAINING_DIR / "zimt_conditions.npy")

    T = args.max_len  # cap sequence length

    # Filter to sequences that fit in max_len
    valid_mask = all_lengths <= T
    all_dxdy = all_dxdy[valid_mask, :T]
    all_lengths = all_lengths[valid_mask]
    all_conditions = all_conditions[valid_mask]

    # Subsample if too many
    N = len(all_lengths)
    if N > args.max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=args.max_samples, replace=False)
        all_dxdy = all_dxdy[idx]
        all_lengths = all_lengths[idx]
        all_conditions = all_conditions[idx]
        N = args.max_samples

    # Compute data scale
    sample_idx = np.random.default_rng(42).choice(N, size=min(10000, N), replace=False)
    sample_vals = np.concatenate([all_dxdy[i, :all_lengths[i]].flatten() for i in sample_idx])
    data_std = float(np.std(sample_vals[sample_vals != 0]))
    data_scale = 1.0 / max(data_std, 1e-6)
    print(f"[Diffusion] N={N}, T={T}, std={data_std:.5f}, scale={data_scale:.1f}")

    # Pre-scale and convert to tensors
    all_dxdy = (all_dxdy * data_scale).astype(np.float32)

    # Build masks (fixed size)
    masks = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        masks[i, :all_lengths[i]] = 1.0

    print("[Diffusion] Converting to tensors...")
    t_dxdy = torch.from_numpy(all_dxdy)
    t_masks = torch.from_numpy(masks)
    t_conds = torch.from_numpy(all_conditions.copy()).float()

    # Train/val split
    n_val = min(5000, N // 10)
    perm = np.random.default_rng(42).permutation(N)
    vi, ti = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(t_dxdy[ti], t_masks[ti], t_conds[ti])
    val_ds = TensorDataset(t_dxdy[vi], t_masks[vi], t_conds[vi])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True)

    del all_dxdy, masks, all_conditions, t_dxdy, t_masks, t_conds

    # Build model
    model = TrajectoryDiffusionModel(
        d_model=args.d_model, n_heads=4, n_layers=args.n_layers,
        d_ff=args.d_model * 4, max_seq_len=T, cond_dim=4,
        n_diff_steps=args.n_diff_steps, dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Diffusion] Model: {args.d_model}d, {args.n_layers}L, {n_params/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[Diffusion] Resumed from epoch {start_epoch}")

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    use_amp = device.type == "cuda"

    print(f"\n[Diffusion] Training: {args.epochs} epochs, bs={args.batch_size}, amp={use_amp}")

    for epoch in range(start_epoch, args.epochs):
        if _stop_requested:
            break

        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_0, mask, cond in train_loader:
            x_0 = x_0.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            B = x_0.shape[0]

            if args.cond_drop > 0:
                drop = torch.rand(B, device=device) < args.cond_drop
                cond = cond * (~drop).unsqueeze(-1).float()

            t = torch.randint(0, args.n_diff_steps, (B,), device=device)
            noise = torch.randn_like(x_0)
            x_t = model.q_sample(x_0, t, noise)

            with torch.amp.autocast("cuda", enabled=use_amp):
                x_0_pred = model(x_t, t, cond, mask.bool())
                diff = (x_0_pred - x_0) ** 2
                loss = (diff.sum(-1) * mask).sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # Validation (sample a few batches, not the whole set)
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for x_0, mask, cond in val_loader:
                x_0 = x_0.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                cond = cond.to(device, non_blocking=True)
                t = torch.randint(0, args.n_diff_steps, (x_0.shape[0],), device=device)
                x_t = model.q_sample(x_0, t, torch.randn_like(x_0))
                with torch.amp.autocast("cuda", enabled=use_amp):
                    x_0_pred = model(x_t, t, cond, mask.bool())
                    diff = (x_0_pred - x_0) ** 2
                    loss = (diff.sum(-1) * mask).sum() / mask.sum().clamp(min=1)
                val_loss_sum += loss.item()
                val_n += 1

        val_loss = val_loss_sum / max(val_n, 1)
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "d_model": args.d_model, "n_heads": 4,
                    "n_layers": args.n_layers, "d_ff": args.d_model * 4,
                    "max_seq_len": T, "cond_dim": 4,
                    "n_diff_steps": args.n_diff_steps,
                },
                "data_scale": data_scale, "data_std": data_std,
                "epoch": epoch + 1, "best_val_loss": best_val_loss,
            }, TRAINING_DIR / "diffusion_best.pt")

        marker = " *BEST*" if is_best else ""
        print(
            f"  Epoch {epoch+1:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.0f}s{marker}"
        )

        if (epoch + 1) % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "d_model": args.d_model, "n_heads": 4,
                    "n_layers": args.n_layers, "d_ff": args.d_model * 4,
                    "max_seq_len": T, "cond_dim": 4,
                    "n_diff_steps": args.n_diff_steps,
                },
                "data_scale": data_scale, "data_std": data_std,
                "epoch": epoch + 1, "best_val_loss": best_val_loss,
            }, TRAINING_DIR / "diffusion_latest.pt")

    print(f"\n[Diffusion] Done. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
