"""Train chunk-level diffusion model for mouse trajectory generation."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.chunk_diffusion import ChunkDiffusionModel

CHUNK_SIZE = 25
STALL_LOGIT_POS = 3.0
STALL_LOGIT_NEG = -3.0


def load_data(data_dir: Path, max_chunks: int | None = None):
    chunk_dxdy = np.load(data_dir / "chunk_dxdy.npy")
    chunk_stall = np.load(data_dir / "chunk_stall.npy")
    ctx_dxdy = np.load(data_dir / "chunk_context_dxdy.npy")
    ctx_stall = np.load(data_dir / "chunk_context_stall.npy")
    global_cond = np.load(data_dir / "chunk_global_cond.npy")
    local_cond = np.load(data_dir / "chunk_local_cond.npy")
    chunk_lengths = np.load(data_dir / "chunk_lengths.npy")

    if max_chunks and max_chunks < len(chunk_dxdy):
        idx = np.random.default_rng(42).choice(len(chunk_dxdy), max_chunks, replace=False)
        chunk_dxdy = chunk_dxdy[idx]
        chunk_stall = chunk_stall[idx]
        ctx_dxdy = ctx_dxdy[idx]
        ctx_stall = ctx_stall[idx]
        global_cond = global_cond[idx]
        local_cond = local_cond[idx]
        chunk_lengths = chunk_lengths[idx]

    # Build chunk_x0: (N, 25, 3) — dx, dy, stall_logit
    stall_logit = np.where(chunk_stall > 0.5, STALL_LOGIT_POS, STALL_LOGIT_NEG)
    chunk_x0 = np.concatenate([chunk_dxdy, stall_logit[..., None]], axis=-1)

    # Build context: (N, 5, 3)
    ctx_stall_logit = np.where(ctx_stall > 0.5, STALL_LOGIT_POS, STALL_LOGIT_NEG)
    context = np.concatenate([ctx_dxdy, ctx_stall_logit[..., None]], axis=-1)

    # Build mask: (N, 25)
    mask = np.zeros((len(chunk_lengths), CHUNK_SIZE), dtype=np.float32)
    for i, cl in enumerate(chunk_lengths):
        mask[i, :cl] = 1.0

    return (
        torch.from_numpy(chunk_x0).float(),
        torch.from_numpy(context).float(),
        torch.from_numpy(global_cond).float(),
        torch.from_numpy(local_cond).float(),
        torch.from_numpy(mask).float(),
    )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    print("Loading data...")
    chunk_x0, context, global_cond, local_cond, mask = load_data(
        data_dir, max_chunks=args.max_chunks
    )
    N = len(chunk_x0)
    print(f"  {N:,} chunks loaded")

    # Data scale for normalization
    dxdy_vals = chunk_x0[:, :, :2][mask.unsqueeze(-1).expand_as(chunk_x0[:, :, :2]) > 0]
    data_std = float(dxdy_vals.std())
    data_scale = 1.0 / max(data_std, 1e-6)
    print(f"  Data std: {data_std:.6f}, scale: {data_scale:.2f}")

    # Scale dxdy channels (not stall logit)
    chunk_x0[:, :, :2] *= data_scale
    context[:, :, :2] *= data_scale
    # Scale local_cond remaining/cumulative displacements
    local_cond[:, 0] *= data_scale  # rem_dx
    local_cond[:, 1] *= data_scale  # rem_dy
    local_cond[:, 4] *= data_scale  # cum_dx
    local_cond[:, 5] *= data_scale  # cum_dy

    # Train/val split
    n_val = min(N // 10, 50000)
    perm = torch.randperm(N)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = TensorDataset(
        chunk_x0[train_idx], context[train_idx],
        global_cond[train_idx], local_cond[train_idx], mask[train_idx],
    )
    val_ds = TensorDataset(
        chunk_x0[val_idx], context[val_idx],
        global_cond[val_idx], local_cond[val_idx], mask[val_idx],
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=0, pin_memory=True)

    model = ChunkDiffusionModel(
        n_diff_steps=args.n_diff_steps,
        cond_dropout=args.cond_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    save_path = data_dir / "chunk_diffusion_best.pt"

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_dl:
            bx0, bctx, bgc, blc, bmask = [b.to(device) for b in batch]

            t = torch.randint(0, args.n_diff_steps, (bx0.shape[0],), device=device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                x_0_pred = model(bx0, t, bctx, bgc, blc)
                diff = (x_0_pred - bx0) ** 2  # (B, 25, 3)
                loss = (diff * bmask.unsqueeze(-1)).sum() / bmask.sum() / 3

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_dl:
                bx0, bctx, bgc, blc, bmask = [b.to(device) for b in batch]
                t = torch.randint(0, args.n_diff_steps, (bx0.shape[0],), device=device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    x_0_pred = model(bx0, t, bctx, bgc, blc)
                    diff = (x_0_pred - bx0) ** 2
                    loss = (diff * bmask.unsqueeze(-1)).sum() / bmask.sum() / 3
                val_loss += loss.item()
                n_val_batches += 1
        val_loss /= max(n_val_batches, 1)

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch + 1:3d}/{args.epochs} | "
              f"train {train_loss:.6f} | val {val_loss:.6f} | "
              f"lr {lr:.2e} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "n_diff_steps": args.n_diff_steps,
                    "cond_dropout": args.cond_dropout,
                },
                "data_scale": data_scale,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, save_path)
            print(f"    -> Saved best (val {val_loss:.6f})")

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-diff-steps", type=int, default=200)
    parser.add_argument("--cond-dropout", type=float, default=0.1)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()
    train(args)
