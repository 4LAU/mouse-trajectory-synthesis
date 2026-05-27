"""Train SoundStorm masked bidirectional transformer for VQ-VAE token sequences."""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.soundstorm import SoundStormTransformer


class MaskedTokenDataset(Dataset):

    def __init__(self, seqs, lens, conds, max_seq_len: int):
        self.seqs = seqs
        self.lens = lens
        self.conds = conds
        self.max_seq_len = max_seq_len
        self.mask_id = SoundStormTransformer.MASK_TOKEN_ID

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx].astype(np.int64)
        length = int(self.lens[idx])
        cond = self.conds[idx]

        T = min(length, self.max_seq_len)

        r = np.random.random()
        mask_ratio = math.cos(r * math.pi / 2)
        n_mask = max(1, int(mask_ratio * T))

        mask_idx = np.random.choice(T, n_mask, replace=False)

        target = np.full(self.max_seq_len, -100, dtype=np.int64)
        target[mask_idx] = seq[mask_idx]
        seq[mask_idx] = self.mask_id

        tokens = np.full(self.max_seq_len, 0, dtype=np.int64)
        tokens[:T] = seq[:T]

        padding_mask = np.zeros(self.max_seq_len, dtype=np.float32)
        padding_mask[:T] = 1.0

        return (
            torch.from_numpy(tokens),
            torch.from_numpy(target),
            torch.from_numpy(padding_mask).bool(),
            torch.from_numpy(cond.copy()),
        )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    print("Loading data...")
    seqs = np.load(data_dir / "vqvae_token_seqs.npy")
    lens = np.load(data_dir / "vqvae_seq_lens.npy")
    conds = np.load(data_dir / "vqvae_seq_conditions.npy")

    # Keep as int16 in memory; convert per-sample in __getitem__

    if args.max_samples and args.max_samples < len(seqs):
        idx = np.random.default_rng(42).choice(len(seqs), args.max_samples, replace=False)
        seqs = seqs[idx]
        lens = lens[idx]
        conds = conds[idx]

    N = len(seqs)
    print(f"  {N:,} sequences loaded")

    n_val = min(N // 10, 50000)
    perm = np.random.default_rng(42).permutation(N)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = MaskedTokenDataset(seqs[train_idx], lens[train_idx], conds[train_idx], args.max_seq_len)
    val_ds = MaskedTokenDataset(seqs[val_idx], lens[val_idx], conds[val_idx], args.max_seq_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True,
                          persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=2, pin_memory=True,
                        persistent_workers=True)

    model = SoundStormTransformer(
        vocab_size=1025,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        cond_dim=4,
        cond_dropout=args.cond_dropout,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    save_path = data_dir / "soundstorm_best.pt"
    start_epoch = 0

    if args.resume and save_path.exists():
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_loss = ckpt["val_loss"]
        start_epoch = ckpt["epoch"]
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(start_epoch):
                scheduler.step()
        print(f"  Resumed from epoch {start_epoch} (val {best_val_loss:.4f})")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        n_batches = 0

        for tokens, targets, pad_mask, cond in train_dl:
            tokens = tokens.to(device)
            targets = targets.to(device)
            pad_mask = pad_mask.to(device)
            cond = cond.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(tokens, cond, pad_mask)
                loss = criterion(logits.view(-1, 1025), targets.view(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

            with torch.no_grad():
                mask_pos = targets != -100
                if mask_pos.any():
                    preds = logits.argmax(dim=-1)
                    total_correct += (preds[mask_pos] == targets[mask_pos]).sum().item()
                    total_masked += mask_pos.sum().item()

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        train_acc = total_correct / max(total_masked, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_masked = 0
        n_val_batches = 0
        with torch.no_grad():
            for tokens, targets, pad_mask, cond in val_dl:
                tokens = tokens.to(device)
                targets = targets.to(device)
                pad_mask = pad_mask.to(device)
                cond = cond.to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(tokens, cond, pad_mask)
                    loss = criterion(logits.view(-1, 1025), targets.view(-1))

                val_loss += loss.item()
                n_val_batches += 1

                mask_pos = targets != -100
                if mask_pos.any():
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds[mask_pos] == targets[mask_pos]).sum().item()
                    val_masked += mask_pos.sum().item()

        val_loss /= max(n_val_batches, 1)
        val_acc = val_correct / max(val_masked, 1)

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch + 1:3d}/{args.epochs} | "
              f"train {train_loss:.4f} ({train_acc:.1%}) | "
              f"val {val_loss:.4f} ({val_acc:.1%}) | "
              f"lr {lr:.2e} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": {
                    "vocab_size": 1025,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "d_ff": args.d_ff,
                    "max_seq_len": args.max_seq_len,
                    "cond_dim": 4,
                    "cond_dropout": args.cond_dropout,
                    "dropout": args.dropout,
                },
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, save_path)
            print(f"    -> Saved best (val {val_loss:.4f}, acc {val_acc:.1%})")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--cond-dropout", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
