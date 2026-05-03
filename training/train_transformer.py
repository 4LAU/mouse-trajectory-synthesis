"""
Train autoregressive transformer on tokenized mouse trajectories.

Input: token sequences (0=stall, 1-1024=motion) with conditions.
Output: next-token prediction model.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.trajectory_transformer import TrajectoryTransformer
from models.vqvae import MotionVQVAE

GEN_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def build_codebook_dxdy(vqvae_ckpt_path: Path) -> torch.Tensor:
    """Build a (1025, 2) lookup table of (dx, dy) per token index in position space.

    Token 0 is the stall token -> (0, 0).
    Tokens 1-1024 map to VQ-VAE codebook indices 0-1023, decoded and unnormalized.
    Returns tensor on CPU; shape (1025, 2).
    """
    ckpt = torch.load(vqvae_ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    norm_mean = torch.tensor(ckpt["norm_mean"], dtype=torch.float32)  # (2,)
    norm_std = torch.tensor(ckpt["norm_std"], dtype=torch.float32)    # (2,)

    model = MotionVQVAE(n_codes=cfg["n_codes"], code_dim=cfg["code_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)

    with torch.no_grad():
        # Decode all 1024 codebook indices -> normalized (dx, dy)
        all_indices = torch.arange(cfg["n_codes"], dtype=torch.long)
        dxdy_norm = model.decode(all_indices)           # (1024, 2) in z-score space
        dxdy_pos = dxdy_norm * norm_std + norm_mean     # (1024, 2) in position space

    # Build vocab-sized lookup: index 0 = stall (0,0), indices 1-1024 = codebook 0-1023
    codebook_dxdy = torch.zeros(1025, 2, dtype=torch.float32)
    codebook_dxdy[1:] = dxdy_pos
    return codebook_dxdy


def compute_endpoint_info(
    input_tokens: torch.Tensor,   # (B, T) token indices (already on DEVICE)
    endpoints: torch.Tensor,      # (B, 4) = (start_x, start_y, end_x, end_y) on DEVICE
    codebook_dxdy: torch.Tensor,  # (1025, 2) on DEVICE
) -> torch.Tensor:
    """Compute per-step endpoint conditioning tensor (B, T, 3).

    endpoint_info[:, t, :] = [remaining_dx / total_dist,
                               remaining_dy / total_dist,
                               1 - t / (T - 1)]

    This satisfies the conditioning contract: endpoint_info[t] describes the
    remaining distance after consuming token t (used to predict token t+1).

    COORDINATE SPACE ASSUMPTION:
    endpoints are in distance-normalized space (origin-translated, divided by
    total Euclidean distance); codebook_dxdy is in raw pixel space. The
    coordinate-space sanity check runs once at startup in main().
    """
    B, T = input_tokens.shape
    start_xy = endpoints[:, :2]               # (B, 2)
    end_xy = endpoints[:, 2:]                  # (B, 2)
    total_dist = (end_xy - start_xy).norm(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1)

    # Cumulative displacement: sum of per-token (dx, dy) from step 0..t
    dxdy = codebook_dxdy[input_tokens]         # (B, T, 2)
    cum_disp = dxdy.cumsum(dim=1)              # (B, T, 2) - position after each token

    # Absolute position after each token (in normalized coordinate space)
    cum_pos = start_xy.unsqueeze(1) + cum_disp  # (B, T, 2)

    # Remaining displacement: from current position to endpoint
    remaining = end_xy.unsqueeze(1) - cum_pos   # (B, T, 2)
    remaining_norm = remaining / total_dist.unsqueeze(1)  # (B, T, 2)

    # Remaining fraction of sequence
    t_idx = torch.arange(T, device=input_tokens.device, dtype=torch.float32)
    remaining_frac = 1.0 - t_idx / max(T, 1)              # (T,)
    remaining_frac = remaining_frac.unsqueeze(0).expand(B, -1).unsqueeze(2)  # (B, T, 1)

    endpoint_info = torch.cat([remaining_norm, remaining_frac], dim=2)  # (B, T, 3)
    return endpoint_info


class TrajectoryTokenDataset(Dataset):
    def __init__(self, tokens, lengths, conditions, endpoints):
        self.tokens = tokens
        self.lengths = lengths
        self.conditions = conditions
        self.endpoints = endpoints

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        length = int(self.lengths[idx])
        tokens = self.tokens[idx, :length].astype(np.int64)
        cond = self.conditions[idx]
        ep = self.endpoints[idx]
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "condition": torch.tensor(cond, dtype=torch.float32),
            "endpoint": torch.tensor(ep, dtype=torch.float32),
            "length": length,
        }


def collate_fn(batch):
    max_len = max(b["length"] for b in batch)
    B = len(batch)
    tokens = torch.zeros(B, max_len, dtype=torch.long)
    conditions = torch.stack([b["condition"] for b in batch])
    endpoints = torch.stack([b["endpoint"] for b in batch])
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b["length"]
        tokens[i, :L] = b["tokens"]
        mask[i, :L] = True
    return tokens, conditions, endpoints, mask


def main():
    print("Device:", DEVICE)

    # Load codebook displacement lookup table from VQ-VAE checkpoint
    print("Building codebook displacement table...")
    codebook_dxdy = build_codebook_dxdy(GEN_DIR / "vqvae_v2_best.pt").to(DEVICE)
    print(f"  Codebook dxdy: {codebook_dxdy.shape}, "
          f"mean|dx|={codebook_dxdy[1:, 0].abs().mean():.4f}, "
          f"mean|dy|={codebook_dxdy[1:, 1].abs().mean():.4f}")

    # Coordinate space sanity check (runs once at startup)
    mean_step = codebook_dxdy[1:].norm(dim=1).mean().item()
    if mean_step > 5.0:
        raise ValueError(
            f"Coordinate space mismatch detected: codebook displacements have "
            f"mean step magnitude {mean_step:.3f} (pixel space), but endpoints "
            f"are in distance-normalized space where total_dist ≈ 1.0. "
            f"Cumulative pixel displacements cannot be compared to normalized "
            f"endpoint coordinates. Either convert codebook_dxdy to normalized "
            f"space (divide each trajectory's steps by its total distance) or "
            f"store endpoints in pixel space. Fix the mismatch before training."
        )

    # Load tokenized data
    print("Loading tokenized data...")
    all_tokens = np.load(GEN_DIR / "vqvae_token_seqs.npy")
    all_lengths = np.load(GEN_DIR / "vqvae_seq_lens.npy")
    all_conditions = np.load(GEN_DIR / "vqvae_seq_conditions.npy")
    all_endpoints = np.load(GEN_DIR / "vqvae_seq_endpoints.npy")
    print(f"  {len(all_tokens)} trajectories, mean len {all_lengths.mean():.1f}")

    # Filter
    valid = (all_lengths >= 5) & (all_lengths <= 256)
    tokens = all_tokens[valid]
    lengths = all_lengths[valid]
    conditions = all_conditions[valid]
    endpoints = all_endpoints[valid]
    print(f"  After filter: {len(tokens)}")

    # Subsample for speed (100K is enough for first pass)
    max_train = 200_000
    rng = np.random.default_rng(42)
    if len(tokens) > max_train:
        sub_idx = rng.choice(len(tokens), max_train, replace=False)
        tokens = tokens[sub_idx]
        lengths = lengths[sub_idx]
        conditions = conditions[sub_idx]
        endpoints = endpoints[sub_idx]
        print(f"  Subsampled to {max_train}")

    # Train/val split
    n_val = max(5000, len(tokens) // 20)
    perm = rng.permutation(len(tokens))

    train_dataset = TrajectoryTokenDataset(
        tokens[perm[n_val:]], lengths[perm[n_val:]],
        conditions[perm[n_val:]], endpoints[perm[n_val:]],
    )
    val_dataset = TrajectoryTokenDataset(
        tokens[perm[:n_val]], lengths[perm[:n_val]],
        conditions[perm[:n_val]], endpoints[perm[:n_val]],
    )
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Model
    model = TrajectoryTransformer(
        vocab_size=1025, d_model=256, n_heads=4, n_layers=4,
        d_ff=1024, max_seq_len=256, cond_dim=4, dropout=0.1,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    # Unweighted CE (stall weight was 3x -> caused 55% stall over-generation)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    n_epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"\nTraining ({n_epochs} epochs)...")
    t0 = time.time()
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for tokens_batch, cond_batch, ep_batch, mask_batch in train_loader:
            tokens_batch = tokens_batch.to(DEVICE)
            cond_batch = cond_batch.to(DEVICE)
            ep_batch = ep_batch.to(DEVICE)
            mask_batch = mask_batch.to(DEVICE)

            input_tokens = tokens_batch[:, :-1]
            target_tokens = tokens_batch[:, 1:].clone()
            target_mask = mask_batch[:, 1:]
            target_tokens[~target_mask] = -1

            endpoint_info = compute_endpoint_info(input_tokens, ep_batch, codebook_dxdy)
            logits = model(input_tokens, cond_batch, endpoint_info)
            loss = criterion(logits.reshape(-1, 1025), target_tokens.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.train(False)
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        n_val_batches = 0

        with torch.no_grad():
            for tokens_batch, cond_batch, ep_batch, mask_batch in val_loader:
                tokens_batch = tokens_batch.to(DEVICE)
                cond_batch = cond_batch.to(DEVICE)
                ep_batch = ep_batch.to(DEVICE)
                mask_batch = mask_batch.to(DEVICE)

                input_tokens = tokens_batch[:, :-1]
                target_tokens = tokens_batch[:, 1:].clone()
                target_mask = mask_batch[:, 1:]
                target_tokens[~target_mask] = -1

                endpoint_info = compute_endpoint_info(input_tokens, ep_batch, codebook_dxdy)
                logits = model(input_tokens, cond_batch, endpoint_info)
                loss = criterion(logits.reshape(-1, 1025), target_tokens.reshape(-1))
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                correct = (preds == tokens_batch[:, 1:]) & target_mask
                val_correct += correct.sum().item()
                val_total += target_mask.sum().item()
                n_val_batches += 1

        train_loss /= max(n_batches, 1)
        val_loss /= max(n_val_batches, 1)
        val_acc = val_correct / max(val_total, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  ep {epoch+1:3d}: train={train_loss:.4f} | "
                  f"val={val_loss:.4f} acc={val_acc:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": 1025, "d_model": 256, "n_heads": 4,
                    "n_layers": 4, "d_ff": 1024, "max_seq_len": 256, "cond_dim": 4,
                },
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, GEN_DIR / "trajectory_transformer_best.pt")

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
