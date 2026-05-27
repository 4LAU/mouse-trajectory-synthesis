"""Train CANDI hybrid discrete-continuous diffusion model."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.candi import CANDIModel


def _to_polar(dxdy_raw, stall_raw, T):
    dx, dy = dxdy_raw[:T, 0].astype(np.float64), dxdy_raw[:T, 1].astype(np.float64)
    speed = np.sqrt(dx ** 2 + dy ** 2)
    heading = np.arctan2(dy, dx)
    for i in range(T):
        if stall_raw[i] or speed[i] < 1e-10:
            heading[i] = heading[i - 1] if i > 0 else 0.0
    dh = np.empty(T, dtype=np.float64)
    dh[0] = heading[0]
    dh[1:] = np.diff(heading)
    dh[1:] = (dh[1:] + np.pi) % (2 * np.pi) - np.pi
    return speed.astype(np.float32), dh.astype(np.float32)


class CANDIDataset(Dataset):
    def __init__(self, dxdy, stall, lengths, conditions, max_len, data_scale, polar=False):
        self.dxdy = dxdy
        self.stall = stall
        self.lengths = lengths
        self.conditions = conditions
        self.max_len = max_len
        self.data_scale = data_scale
        self.polar = polar

    def __len__(self):
        return len(self.dxdy)

    def __getitem__(self, idx):
        L = int(self.lengths[idx])
        T = min(L, self.max_len)

        out = np.zeros((self.max_len, 2), dtype=np.float32)
        if self.polar:
            spd, dh = _to_polar(self.dxdy[idx], self.stall[idx], T)
            out[:T, 0] = spd * self.data_scale[0]
            out[:T, 1] = dh * self.data_scale[1]
        else:
            out[:T] = self.dxdy[idx, :T] * self.data_scale

        st = np.zeros(self.max_len, dtype=np.float32)
        st[:T] = self.stall[idx, :T].astype(np.float32)

        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:T] = 1.0

        return (
            torch.from_numpy(out),
            torch.from_numpy(st),
            torch.from_numpy(mask).bool(),
            torch.from_numpy(self.conditions[idx].copy()),
        )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    print("Loading data...")
    dxdy = np.load(data_dir / "zimt_dxdy.npy", mmap_mode="r")
    stall = np.load(data_dir / "zimt_stall.npy", mmap_mode="r")
    lengths = np.load(data_dir / "zimt_lengths.npy")
    conditions = np.load(data_dir / "zimt_conditions.npy")
    N = len(dxdy)

    if args.max_samples and args.max_samples < N:
        idx = np.random.default_rng(42).choice(N, args.max_samples, replace=False)
        lengths = lengths[idx]
        conditions = conditions[idx]
        dxdy_sel = dxdy[idx]
        stall_sel = stall[idx]
    else:
        dxdy_sel = dxdy
        stall_sel = stall
        idx = None

    N = len(lengths)
    print(f"  {N:,} trajectories")

    if args.polar:
        all_spd, all_dh = [], []
        for i in range(min(N, 50000)):
            L = int(lengths[i])
            spd, dh = _to_polar(dxdy_sel[i], stall_sel[i], L)
            all_spd.append(spd)
            all_dh.append(dh)
        spd_std = float(np.std(np.concatenate(all_spd)))
        dh_std = float(np.std(np.concatenate(all_dh)))
        data_std = np.array([spd_std, dh_std], dtype=np.float32)
        data_scale = np.array([1.0 / spd_std, 1.0 / dh_std], dtype=np.float32)
        print(f"  [polar] spd_std={spd_std:.6f}, dh_std={dh_std:.6f}")
    else:
        all_dx = []
        for i in range(min(N, 50000)):
            L = int(lengths[i])
            all_dx.append(dxdy_sel[i, :L].flatten())
        data_std = float(np.std(np.concatenate(all_dx)))
        data_scale = 1.0 / data_std
        print(f"  data_std={data_std:.6f}, data_scale={data_scale:.1f}")

    n_val = min(N // 10, 30000)
    perm = np.random.default_rng(42).permutation(N)

    train_ds = CANDIDataset(
        dxdy_sel[perm[n_val:]], stall_sel[perm[n_val:]],
        lengths[perm[n_val:]], conditions[perm[n_val:]],
        args.max_seq_len, data_scale, polar=args.polar,
    )
    val_ds = CANDIDataset(
        dxdy_sel[perm[:n_val]], stall_sel[perm[:n_val]],
        lengths[perm[:n_val]], conditions[perm[:n_val]],
        args.max_seq_len, data_scale, polar=args.polar,
    )

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    model = CANDIModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        cond_dim=4,
        n_diffusion_steps=args.n_steps,
        cond_dropout=args.cond_dropout,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    bce = nn.BCEWithLogitsLoss(reduction="none")
    best_val = float("inf")
    save_name = "candi_polar_best.pt" if args.polar else "candi_best.pt"
    save_path = data_dir / save_name

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        tot_cont, tot_disc, nb = 0.0, 0.0, 0

        for dxdy_b, stall_b, pad_b, cond_b in train_dl:
            dxdy_b = dxdy_b.to(device)
            stall_b = stall_b.to(device)
            pad_b = pad_b.to(device)
            cond_b = cond_b.to(device)

            B = dxdy_b.shape[0]
            t = torch.randint(0, args.n_steps, (B,), device=device)

            dxdy_noisy, noise = model.q_continuous(dxdy_b, t)
            stall_masked, disc_mask = model.q_discrete(stall_b, t)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                dxdy_pred, stall_logit = model(
                    dxdy_noisy, stall_masked, disc_mask.float(), t, cond_b, pad_b,
                )

                pad_f = pad_b.float().unsqueeze(-1)
                cont_loss = ((dxdy_pred - dxdy_b) ** 2 * pad_f).sum() / pad_f.sum().clamp(1)

                disc_target = stall_b
                disc_loss_raw = bce(stall_logit, disc_target)
                disc_weight = disc_mask.float() * pad_b.float()
                disc_loss = (disc_loss_raw * disc_weight).sum() / disc_weight.sum().clamp(1)

                loss = cont_loss + args.disc_weight * disc_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            tot_cont += cont_loss.item()
            tot_disc += disc_loss.item()
            nb += 1

        scheduler.step()

        model.eval()
        v_cont, v_disc, vnb = 0.0, 0.0, 0
        with torch.no_grad():
            for dxdy_b, stall_b, pad_b, cond_b in val_dl:
                dxdy_b = dxdy_b.to(device)
                stall_b = stall_b.to(device)
                pad_b = pad_b.to(device)
                cond_b = cond_b.to(device)

                B = dxdy_b.shape[0]
                t = torch.randint(0, args.n_steps, (B,), device=device)
                dxdy_noisy, _ = model.q_continuous(dxdy_b, t)
                stall_masked, disc_mask = model.q_discrete(stall_b, t)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    dxdy_pred, stall_logit = model(
                        dxdy_noisy, stall_masked, disc_mask.float(), t, cond_b, pad_b,
                    )
                    pad_f = pad_b.float().unsqueeze(-1)
                    cl = ((dxdy_pred - dxdy_b) ** 2 * pad_f).sum() / pad_f.sum().clamp(1)
                    dl_raw = bce(stall_logit, stall_b)
                    dw = disc_mask.float() * pad_b.float()
                    dl = (dl_raw * dw).sum() / dw.sum().clamp(1)

                v_cont += cl.item()
                v_disc += dl.item()
                vnb += 1

        tc = tot_cont / max(nb, 1)
        td = tot_disc / max(nb, 1)
        vc = v_cont / max(vnb, 1)
        vd = v_disc / max(vnb, 1)
        val_total = vc + args.disc_weight * vd
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"train cont={tc:.4f} disc={td:.4f} | "
              f"val cont={vc:.4f} disc={vd:.4f} | "
              f"lr {lr:.2e} | {elapsed:.0f}s")

        if val_total < best_val:
            best_val = val_total
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "d_ff": args.d_ff,
                    "max_seq_len": args.max_seq_len,
                    "cond_dim": 4,
                    "n_diffusion_steps": args.n_steps,
                    "cond_dropout": args.cond_dropout,
                    "dropout": args.dropout,
                },
                "data_scale": data_scale,
                "data_std": data_std,
                "polar": args.polar,
                "epoch": epoch + 1,
                "val_cont": vc,
                "val_disc": vd,
            }
            torch.save(ckpt, save_path)
            print(f"    -> Saved best (cont={vc:.4f}, disc={vd:.4f})")

    print(f"\nDone. Best val: {best_val:.4f}")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--disc-weight", type=float, default=1.0)
    parser.add_argument("--cond-dropout", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--polar", action="store_true", help="Use (speed, delta_heading) instead of (dx, dy)")
    args = parser.parse_args()
    train(args)
