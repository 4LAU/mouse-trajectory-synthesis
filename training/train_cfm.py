"""Conditional Flow Matching training loop (joint position + timing).

OT-CFM (Optimal Transport Conditional Flow Matching):
  x_t = (1-t)*x_0 + t*x_1,  u_t = x_1 - x_0
  loss = MSE(v_theta(x_t, t, c), u_t)

Generates (192, 3): positions (x, y) + normalized timing.

Usage:
  python3 -m training.train_cfm
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.temporal_unet import TemporalUNet

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_GEN_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 256
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 500
STEPS_PER_EPOCH = 1024  # gradient steps (batches) per epoch; 1024*200=204800 total
EPOCHS = 200
VAL_EVERY = 5
SAVE_EVERY = 10
VAL_SAMPLES = 1024
CHECKPOINT_PATH = _GEN_DIR / "cfm_model.pt"
BEST_CHECKPOINT_PATH = _GEN_DIR / "cfm_model_best.pt"


# ---------------------------------------------------------------------------
# Dataset (mmap-backed)
# ---------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """Memory-mapped dataset for trajectory positions + timestamps + conditions."""

    def __init__(self, split: str = "train"):
        self.positions = np.load(_GEN_DIR / f"{split}_positions.npy", mmap_mode="r")   # (N, 192, 2)
        self.timestamps = np.load(_GEN_DIR / f"{split}_timestamps.npy", mmap_mode="r") # (N, 192)
        self.conditions = np.load(_GEN_DIR / f"{split}_conditions.npy", mmap_mode="r") # (N, 4)
        self.n_real = np.load(_GEN_DIR / f"{split}_n_real.npy", mmap_mode="r")         # (N,)
        self.n = len(self.positions)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        pos = np.array(self.positions[idx], dtype=np.float32)    # (192, 2)
        ts = np.array(self.timestamps[idx], dtype=np.float32)    # (192,)
        cond = np.array(self.conditions[idx], dtype=np.float32)  # (4,)
        n_real = int(self.n_real[idx])

        # Normalize timestamps to [0, 1] using actual trajectory duration
        duration = ts[n_real - 1] if n_real > 1 else 1.0
        if duration < 1e-6:
            duration = 1.0
        t_norm = ts / max(duration, 1e-6)
        t_norm = np.clip(t_norm, 0.0, 1.0)

        # Combine: (192, 3) = [x, y, t_norm]
        x = np.concatenate([pos, t_norm[:, None]], axis=1)  # (192, 3)
        return torch.from_numpy(x), torch.from_numpy(cond)


class RandomSubsetSampler:
    """Samples `num_samples` random indices from dataset each iteration."""

    def __init__(self, dataset_len: int, num_samples: int):
        self.dataset_len = dataset_len
        self.num_samples = num_samples

    def __iter__(self):
        return iter(np.random.randint(0, self.dataset_len, self.num_samples).tolist())

    def __len__(self):
        return self.num_samples


# ---------------------------------------------------------------------------
# LR schedule: cosine with warmup
# ---------------------------------------------------------------------------
def get_lr(step: int, total_steps: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
_CFM_CONFIG = {"n_points": 192, "in_channels": 3, "cond_dim": 4}


def _save_checkpoint(model, epoch, val_loss, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": _CFM_CONFIG,
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("[CFM] Using CUDA backend")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[CFM] Using MPS backend")
    else:
        device = torch.device("cpu")
        print("[CFM] Using CPU backend")

    # Model
    model = TemporalUNet().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Data
    train_ds = TrajectoryDataset("train")
    val_ds = TrajectoryDataset("val")
    print(f"[CFM] Train: {len(train_ds):,} samples, Val: {len(val_ds):,} samples")

    # STEPS_PER_EPOCH = number of gradient steps (batches) per epoch
    batches_per_epoch = STEPS_PER_EPOCH
    samples_per_epoch = batches_per_epoch * BATCH_SIZE
    total_steps = batches_per_epoch * EPOCHS

    print(f"[CFM] {batches_per_epoch} batches/epoch x {BATCH_SIZE} = {samples_per_epoch:,} samples/epoch")
    print(f"[CFM] Total steps: {total_steps:,}")

    best_val_loss = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Random subset sampler for this epoch
        sampler = RandomSubsetSampler(len(train_ds), samples_per_epoch)
        loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        for batch_data, conditions in loader:
            global_step += 1

            # LR schedule
            lr = get_lr(global_step, batches_per_epoch * EPOCHS, LR, WARMUP_STEPS)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch_data = batch_data.to(device)  # (B, 192, 3)
            conditions = conditions.to(device)  # (B, 4)

            B = batch_data.shape[0]

            # OT-CFM: sample noise, time, interpolate
            x_1 = batch_data
            x_0 = torch.randn_like(x_1)
            t = torch.rand(B, 1, 1, device=device)
            x_t = (1 - t) * x_0 + t * x_1
            u_t = x_1 - x_0  # ground truth velocity

            # Forward
            t_scalar = t.squeeze(-1).squeeze(-1)  # (B,)
            v_theta = model(x_t, t_scalar, conditions)

            # Loss
            loss = F.mse_loss(v_theta, u_t)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if global_step % 500 == 0:
                elapsed = time.time() - t0
                print(
                    f"  step {global_step:6d} | loss {loss.item():.6f} | "
                    f"lr {lr:.2e} | {elapsed:.0f}s",
                    flush=True,
                )

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | train_loss {avg_loss:.6f} | "
            f"{elapsed:.0f}s elapsed",
            flush=True,
        )

        # Validation
        if epoch % VAL_EVERY == 0:
            val_loss = validate(model, val_ds, device)
            print(f"  val_loss {val_loss:.6f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, epoch, val_loss, BEST_CHECKPOINT_PATH)
                print(f"  ** new best -> {BEST_CHECKPOINT_PATH.name}", flush=True)

        # Periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            _save_checkpoint(model, epoch, avg_loss, CHECKPOINT_PATH)
            print(f"  checkpoint -> {CHECKPOINT_PATH.name}", flush=True)

    # Final save
    _save_checkpoint(model, EPOCHS, avg_loss, CHECKPOINT_PATH)
    total_time = time.time() - t0
    print(f"\n[CFM] Training complete. {total_time:.0f}s total.")
    print(f"[CFM] Best val loss: {best_val_loss:.6f}")
    print(f"[CFM] Checkpoints: {CHECKPOINT_PATH.name}, {BEST_CHECKPOINT_PATH.name}")


def validate(model, val_ds, device) -> float:
    """Compute mean flow matching loss on a subset of validation data."""
    model.train(False)
    indices = np.random.randint(0, len(val_ds), VAL_SAMPLES)
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        # Process in batches
        for start in range(0, VAL_SAMPLES, BATCH_SIZE):
            end = min(start + BATCH_SIZE, VAL_SAMPLES)
            batch_idx = indices[start:end]

            batch_data = torch.stack([val_ds[i][0] for i in batch_idx]).to(device)
            conditions = torch.stack([val_ds[i][1] for i in batch_idx]).to(device)

            B = batch_data.shape[0]
            x_1 = batch_data
            x_0 = torch.randn_like(x_1)
            t = torch.rand(B, 1, 1, device=device)
            x_t = (1 - t) * x_0 + t * x_1
            u_t = x_1 - x_0

            t_scalar = t.squeeze(-1).squeeze(-1)
            v_theta = model(x_t, t_scalar, conditions)
            loss = F.mse_loss(v_theta, u_t)
            total_loss += loss.item() * B
            n += B

    model.train(True)
    return total_loss / max(1, n)


if __name__ == "__main__":
    train()
