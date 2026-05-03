"""
Train VQ-VAE v2: normalized + clipped data, k-means init, codebook reset.

Fixes from v1:
1. Normalize (dx, dy) by clipping to P0.5-P99.5 range then z-score normalizing
2. K-means initialization of codebook from encoded data
3. Periodic codebook reset for dead entries
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.vqvae import MotionVQVAE

GEN_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def kmeans_init(data: torch.Tensor, n_clusters: int, n_iter: int = 20) -> torch.Tensor:
    """Simple k-means for codebook initialization."""
    n = data.shape[0]
    indices = torch.randperm(n)[:n_clusters]
    centers = data[indices].clone()

    for _ in range(n_iter):
        dists = torch.cdist(data[:50000], centers)
        assignments = dists.argmin(dim=1)
        for k in range(n_clusters):
            mask = assignments == k
            if mask.sum() > 0:
                centers[k] = data[:50000][mask].mean(dim=0)

    return centers


def main():
    print("Device:", DEVICE)

    # Load data
    print("Loading displacement data...")
    dxdy = np.load(GEN_DIR / "vqvae_dxdy.npy")
    speeds = np.load(GEN_DIR / "vqvae_speeds.npy")

    # Filter stalls
    non_stall = speeds >= 1.0
    dxdy_motion = dxdy[non_stall].copy()
    print(f"  Non-stall: {len(dxdy_motion)} ({100*len(dxdy_motion)/len(dxdy):.1f}%)")

    # Clip to P0.5-P99.5 range
    clip_lo = np.percentile(dxdy_motion, 0.5, axis=0)
    clip_hi = np.percentile(dxdy_motion, 99.5, axis=0)
    print(f"  Clip: dx=[{clip_lo[0]:.1f}, {clip_hi[0]:.1f}], dy=[{clip_lo[1]:.1f}, {clip_hi[1]:.1f}]")
    dxdy_motion = np.clip(dxdy_motion, clip_lo, clip_hi)

    # Z-score normalize
    data_mean = dxdy_motion.mean(axis=0)
    data_std = dxdy_motion.std(axis=0)
    print(f"  Norm: mean=[{data_mean[0]:.3f}, {data_mean[1]:.3f}], std=[{data_std[0]:.3f}, {data_std[1]:.3f}]")
    dxdy_norm = ((dxdy_motion - data_mean) / data_std).astype(np.float32)

    # Save normalization params
    norm_params = np.array([
        data_mean[0], data_mean[1], data_std[0], data_std[1],
        clip_lo[0], clip_lo[1], clip_hi[0], clip_hi[1],
    ], dtype=np.float32)
    np.save(GEN_DIR / "vqvae_norm_params.npy", norm_params)

    # Subsample
    max_train = 5_000_000
    rng = np.random.default_rng(42)
    if len(dxdy_norm) > max_train:
        idx = rng.choice(len(dxdy_norm), max_train, replace=False)
        dxdy_norm = dxdy_norm[idx]
        print(f"  Subsampled to {max_train}")

    # Split
    n_val = 250_000
    dxdy_train = torch.tensor(dxdy_norm[:-n_val], dtype=torch.float32)
    dxdy_val = torch.tensor(dxdy_norm[-n_val:], dtype=torch.float32)
    print(f"  Train: {len(dxdy_train)}, Val: {len(dxdy_val)}")

    # Model
    model = MotionVQVAE(n_codes=1024, code_dim=64, beta=0.25).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    # K-means init for codebook
    print("  K-means codebook initialization...")
    encoded_sample = model.encoder(dxdy_train[:100000].to(DEVICE)).detach()
    init_centers = kmeans_init(encoded_sample, 1024, n_iter=30)
    model.vq.embedding.weight.data.copy_(init_centers)
    model.vq.ema_embed_sum.data.copy_(init_centers)
    model.vq.ema_cluster_size.data.fill_(1.0)
    print("  K-means init done")

    # Data loaders
    train_loader = DataLoader(TensorDataset(dxdy_train), batch_size=4096, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(dxdy_val), batch_size=4096, shuffle=False)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    # Training
    best_val_loss = float("inf")
    print("\nTraining VQ-VAE v2...")
    t0 = time.time()

    for epoch in range(80):
        model.train()
        train_recon = 0.0
        train_vq = 0.0
        n_batches = 0

        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            recon, vq_loss, indices = model(batch)
            recon_loss = F.mse_loss(recon, batch)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_recon += recon_loss.item()
            train_vq += vq_loss.item()
            n_batches += 1

        scheduler.step()

        # Codebook reset for dead entries every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.train(False)
            with torch.no_grad():
                sample = dxdy_train[:200000].to(DEVICE)
                _, _, sample_idx = model(sample)
                usage = torch.bincount(sample_idx, minlength=model.n_codes)
                dead = (usage == 0).nonzero().squeeze(-1)
                if len(dead) > 0:
                    encoded = model.encoder(sample[:len(dead) * 2])
                    rand_idx = torch.randperm(len(encoded))[:len(dead)]
                    model.vq.embedding.weight.data[dead] = encoded[rand_idx].detach()
                    model.vq.ema_embed_sum.data[dead] = encoded[rand_idx].detach()
                    model.vq.ema_cluster_size.data[dead] = 1.0
            model.train(True)

        # Validate
        model.train(False)
        val_recon = 0.0
        val_vq = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(DEVICE)
                recon, vq_loss, _ = model(batch)
                val_recon += F.mse_loss(recon, batch).item()
                val_vq += vq_loss.item()
                n_val_batches += 1

        train_recon /= n_batches
        train_vq /= n_batches
        val_recon /= max(n_val_batches, 1)
        val_vq /= max(n_val_batches, 1)
        val_total = val_recon + val_vq

        # Codebook utilization
        with torch.no_grad():
            sample = dxdy_train[:50000].to(DEVICE)
            _, _, si = model(sample)
            n_used = len(torch.unique(si))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep {epoch+1:3d}: recon={train_recon:.4f} vq={train_vq:.4f} | "
                  f"val={val_recon:.4f}+{val_vq:.4f}={val_total:.4f} | "
                  f"codebook={n_used}/1024 ({100*n_used/1024:.0f}%)")

        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"n_codes": 1024, "code_dim": 64},
                "epoch": epoch + 1,
                "val_recon": val_recon,
                "val_vq": val_vq,
                "codebook_usage": n_used,
                "norm_mean": data_mean.tolist(),
                "norm_std": data_std.tolist(),
                "clip_lo": clip_lo.tolist(),
                "clip_hi": clip_hi.tolist(),
            }, GEN_DIR / "vqvae_v2_best.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"n_codes": 1024, "code_dim": 64},
        "epoch": 80,
        "norm_mean": data_mean.tolist(),
        "norm_std": data_std.tolist(),
        "clip_lo": clip_lo.tolist(),
        "clip_hi": clip_hi.tolist(),
    }, GEN_DIR / "vqvae_v2_last.pt")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Time: {time.time()-t0:.1f}s")

    # Reconstruction quality in original space
    print("\nReconstruction quality (original space):")
    model.train(False)
    with torch.no_grad():
        sample_norm = dxdy_val[:10000].to(DEVICE)
        recon_norm, _, indices = model(sample_norm)

        orig = sample_norm.cpu().numpy() * data_std + data_mean
        recon = recon_norm.cpu().numpy() * data_std + data_mean
        err = orig - recon
        print(f"  MSE: dx={np.mean(err[:,0]**2):.4f}, dy={np.mean(err[:,1]**2):.4f}")
        print(f"  MAE: dx={np.mean(np.abs(err[:,0])):.4f}, dy={np.mean(np.abs(err[:,1])):.4f}")

        orig_speed = np.sqrt(orig[:,0]**2 + orig[:,1]**2) * 125.0
        recon_speed = np.sqrt(recon[:,0]**2 + recon[:,1]**2) * 125.0
        print(f"  Speed MAE: {np.abs(orig_speed-recon_speed).mean():.1f} px/s (mean: {orig_speed.mean():.1f})")
        print(f"  Unique codes (10K sample): {len(torch.unique(indices))}")


if __name__ == "__main__":
    main()
