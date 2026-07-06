"""Train the WS7 event-stream model.

Losses:
- Flow matching MSE on z-scored log(dt_ms). Real positions weight 1.0, PAD
  tail weight 0.1 (target 0 in z-space so sampling drives the tail cleanly).
- Cross-entropy on masked dx/dy tokens. Real positions weight 1.0, PAD-target
  positions downweighted so the tail does not dominate (median 43 real events
  in a 256 slot sequence).

Crash-safe: saves a _latest checkpoint every epoch, --auto-resume picks it up.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.event_stream import (
    MASK_TOKEN, PAD_CLASS, VOCAB_MAX, EventStreamModel, disp_to_class,
)


class EventDataset(Dataset):
    def __init__(self, dx, dy, dt, lengths, conditions, max_len, dt_mean, dt_std):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.lengths = lengths
        self.conditions = conditions
        self.max_len = max_len
        self.dt_mean = dt_mean
        self.dt_std = dt_std

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        L = min(int(self.lengths[idx]), self.max_len)

        dt_z = np.zeros(self.max_len, dtype=np.float32)
        dt_ms = self.dt[idx, :L].astype(np.float32)
        dt_z[:L] = (np.log(np.maximum(dt_ms, 0.05)) - self.dt_mean) / self.dt_std

        dx_cls = np.full(self.max_len, PAD_CLASS, dtype=np.int64)
        dy_cls = np.full(self.max_len, PAD_CLASS, dtype=np.int64)
        dx_cls[:L] = self.dx[idx, :L].astype(np.int64) + VOCAB_MAX
        dy_cls[:L] = self.dy[idx, :L].astype(np.int64) + VOCAB_MAX

        real = np.zeros(self.max_len, dtype=np.float32)
        real[:L] = 1.0

        return (
            torch.from_numpy(dt_z),
            torch.from_numpy(dx_cls),
            torch.from_numpy(dy_cls),
            torch.from_numpy(real),
            torch.from_numpy(self.conditions[idx].copy()),
        )


def run_epoch(model, dl, device, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    tot_flow, tot_ce, nb = 0.0, 0.0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for dt_z, dx_cls, dy_cls, real, cond in dl:
            dt_z = dt_z.to(device, non_blocking=True)
            dx_cls = dx_cls.to(device, non_blocking=True)
            dy_cls = dy_cls.to(device, non_blocking=True)
            real = real.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            B = dt_z.shape[0]

            t_cont = torch.rand(B, device=device)
            t_int = (t_cont * (args.n_steps - 1)).long()
            dt_noisy, _, velocity = model.q_flow(dt_z, t_cont)
            dx_masked, dx_mask = model.q_mask(dx_cls, t_int)
            dy_masked, dy_mask = model.q_mask(dy_cls, t_int)
            t_for_model = t_cont * (args.n_steps - 1)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                v_pred, dx_logits, dy_logits = model(
                    dt_noisy, dx_masked, dy_masked, t_for_model, cond,
                )

                w_flow = real + (1.0 - real) * args.pad_flow_weight
                flow_loss = ((v_pred - velocity) ** 2 * w_flow).sum() / w_flow.sum().clamp(1)

                w_tok = real + (1.0 - real) * args.pad_ce_weight
                ce_dx = F.cross_entropy(
                    dx_logits.reshape(-1, dx_logits.shape[-1]), dx_cls.reshape(-1),
                    reduction="none",
                ).view(B, -1)
                ce_dy = F.cross_entropy(
                    dy_logits.reshape(-1, dy_logits.shape[-1]), dy_cls.reshape(-1),
                    reduction="none",
                ).view(B, -1)
                wx = dx_mask.float() * w_tok
                wy = dy_mask.float() * w_tok
                ce_loss = 0.5 * (
                    (ce_dx * wx).sum() / wx.sum().clamp(1)
                    + (ce_dy * wy).sum() / wy.sum().clamp(1)
                )

                loss = flow_loss + args.disc_weight * ce_loss

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            tot_flow += flow_loss.item()
            tot_ce += ce_loss.item()
            nb += 1

    return tot_flow / max(nb, 1), tot_ce / max(nb, 1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    print("Loading event data...", flush=True)
    dx = np.load(data_dir / "events_dx.npy", mmap_mode="r")
    dy = np.load(data_dir / "events_dy.npy", mmap_mode="r")
    dt = np.load(data_dir / "events_dt.npy", mmap_mode="r")
    lengths = np.load(data_dir / "events_len.npy")
    conditions = np.load(data_dir / "events_cond.npy")
    N = len(lengths)
    print(f"  {N:,} trajectories", flush=True)

    if args.max_samples and args.max_samples < N:
        idx = np.sort(np.random.default_rng(42).choice(N, args.max_samples, replace=False))
        dx, dy, dt = dx[idx], dy[idx], dt[idx]
        lengths, conditions = lengths[idx], conditions[idx]
        N = args.max_samples
        print(f"  Subsampled to {N:,}", flush=True)

    # dt normalization stats from a sample of real positions
    n_stat = min(N, 100_000)
    valid = np.arange(dt.shape[1])[None, :] < lengths[:n_stat, None]
    dt_vals = np.asarray(dt[:n_stat], dtype=np.float32)[valid]
    log_dt = np.log(np.maximum(dt_vals, 0.05))
    dt_mean, dt_std = float(log_dt.mean()), float(max(log_dt.std(), 1e-3))
    print(f"  log-dt stats: mean={dt_mean:.4f}, std={dt_std:.4f}", flush=True)

    n_val = min(N // 10, 30000)
    perm = np.random.default_rng(42).permutation(N)
    tr_idx, va_idx = np.sort(perm[n_val:]), np.sort(perm[:n_val])

    train_ds = EventDataset(dx[tr_idx], dy[tr_idx], dt[tr_idx], lengths[tr_idx],
                            conditions[tr_idx], args.max_seq_len, dt_mean, dt_std)
    val_ds = EventDataset(dx[va_idx], dy[va_idx], dt[va_idx], lengths[va_idx],
                          conditions[va_idx], args.max_seq_len, dt_mean, dt_std)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    model = EventStreamModel(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, max_seq_len=args.max_seq_len, cond_dim=4,
        n_diffusion_steps=args.n_steps, cond_dropout=args.cond_dropout,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}", flush=True)

    save_path = data_dir / args.save_name
    latest_path = save_path.with_stem(save_path.stem + "_latest")

    start_epoch = 0
    best_val = float("inf")
    load_from = args.load_from
    if args.auto_resume and latest_path.exists():
        load_from = str(latest_path)
    if load_from:
        ckpt = torch.load(load_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"  Resumed from {load_from} (epoch {start_epoch})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    for _ in range(start_epoch):
        scheduler.step()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        tf, tc = run_epoch(model, train_dl, device, args, optimizer, scaler)
        scheduler.step()
        vf, vc = run_epoch(model, val_dl, device, args)
        val_total = vf + args.disc_weight * vc
        elapsed = time.time() - t0

        print(f"  Epoch {epoch + 1:3d}/{total_epochs} | "
              f"train flow={tf:.4f} ce={tc:.4f} | "
              f"val flow={vf:.4f} ce={vc:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s", flush=True)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": {
                "d_model": args.d_model, "n_heads": args.n_heads,
                "n_layers": args.n_layers, "d_ff": args.d_ff,
                "max_seq_len": args.max_seq_len, "cond_dim": 4,
                "n_diffusion_steps": args.n_steps,
                "cond_dropout": args.cond_dropout, "dropout": args.dropout,
            },
            "dt_mean": dt_mean, "dt_std": dt_std,
            "epoch": epoch + 1, "best_val": best_val,
            "val_flow": vf, "val_ce": vc,
        }
        torch.save(ckpt, latest_path)
        # Versioned per-epoch checkpoint for post-hoc eval sweeps
        torch.save(ckpt, save_path.with_stem(save_path.stem + f"_ep{epoch + 1}"))

        if val_total < best_val:
            best_val = val_total
            ckpt["best_val"] = best_val
            torch.save(ckpt, save_path)
            print(f"    -> Saved best (flow={vf:.4f}, ce={vc:.4f})", flush=True)

        if args.cooldown > 0:
            torch.cuda.empty_cache()
            time.sleep(args.cooldown)

    print(f"\nDone. Best val: {best_val:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--disc-weight", type=float, default=1.0)
    parser.add_argument("--pad-ce-weight", type=float, default=0.15)
    parser.add_argument("--pad-flow-weight", type=float, default=0.1)
    parser.add_argument("--cond-dropout", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-name", default="event_stream_best.pt")
    parser.add_argument("--load-from", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cooldown", type=int, default=0)
    args = parser.parse_args()
    train(args)
