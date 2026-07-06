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


def _pearson_r(x, y):
    x_c = x - x.mean()
    y_c = y - y.mean()
    num = (x_c * y_c).sum()
    den = (x_c.pow(2).sum() * y_c.pow(2).sum()).sqrt().clamp(min=1e-8)
    return num / den


def _path_loss(dxdy_pred, dxdy_noisy, pad_mask, t_or_tcont, model, pred_type,
               data_scale, target_pe_mean=0.838, target_pe_std=0.229):
    """Loss to match human path_efficiency distribution during denoising.

    Only applied at low noise levels where x0 prediction is accurate.
    Works with both DDPM (t is int) and flow matching (t is float).
    """
    dev = dxdy_pred.device

    if pred_type == "flow":
        low_noise = t_or_tcont < 0.3
    else:
        ab = model.alpha_bar[t_or_tcont]
        low_noise = ab > 0.5

    if low_noise.sum() < 8:
        return torch.tensor(0.0, device=dev)

    if pred_type == "flow":
        t_ln = t_or_tcont[low_noise].view(-1, 1, 1)
        x0 = dxdy_noisy[low_noise] - t_ln * dxdy_pred[low_noise]
    elif pred_type == "x0":
        x0 = dxdy_pred[low_noise]
    else:
        return torch.tensor(0.0, device=dev)

    mask = pad_mask[low_noise].float()
    spd_s = float(data_scale[0])
    dh_s = float(data_scale[1])

    speed = torch.clamp(x0[:, :, 0] / spd_s, min=0) * mask
    dh = x0[:, :, 1] / dh_s * mask

    heading = torch.cumsum(dh, dim=1)
    vx = speed * torch.cos(heading)
    vy = speed * torch.sin(heading)
    cx = torch.cumsum(vx, dim=1)
    cy = torch.cumsum(vy, dim=1)

    lengths = mask.sum(dim=1).clamp(min=1).long()
    B = x0.shape[0]
    last_idx = (lengths - 1).clamp(min=0)
    end_x = cx[torch.arange(B, device=dev), last_idx]
    end_y = cy[torch.arange(B, device=dev), last_idx]

    d_straight = torch.sqrt(end_x ** 2 + end_y ** 2 + 1e-8)
    d_traveled = speed.sum(dim=1).clamp(min=1e-8)
    pe = d_straight / d_traveled

    pe_mean = pe.mean()
    pe_std = pe.std().clamp(min=1e-4)
    loss = (pe_mean - target_pe_mean) ** 2 + (pe_std - target_pe_std) ** 2
    return loss


def _corr_loss(dxdy_pred, dxdy_noisy, pad_mask, t, model, pred_type):
    ab = model.alpha_bar[t]
    low_noise = ab > 0.5
    if low_noise.sum() < 8:
        return torch.tensor(0.0, device=dxdy_pred.device)

    if pred_type == "x0":
        x0 = dxdy_pred[low_noise]
    elif pred_type == "eps":
        s_ab = model.sqrt_ab[t[low_noise]].view(-1, 1, 1)
        s_1mab = model.sqrt_1mab[t[low_noise]].view(-1, 1, 1)
        x0 = (dxdy_noisy[low_noise] - s_1mab * dxdy_pred[low_noise]) / s_ab.clamp(min=1e-8)
    else:
        s_ab = model.sqrt_ab[t[low_noise]].view(-1, 1, 1)
        s_1mab = model.sqrt_1mab[t[low_noise]].view(-1, 1, 1)
        x0 = s_ab * dxdy_noisy[low_noise] - s_1mab * dxdy_pred[low_noise]

    mask = pad_mask[low_noise].float()
    speed = x0[:, :, 0]

    lengths = mask.sum(dim=1).clamp(min=1)
    mean_spd = (speed * mask).sum(dim=1) / lengths

    spd_diff = torch.diff(speed, dim=1)
    mask_diff = mask[:, 1:] * mask[:, :-1]
    len_diff = mask_diff.sum(dim=1).clamp(min=1)
    mean_acc = (spd_diff.abs() * mask_diff).sum(dim=1) / len_diff
    max_acc = torch.where(mask_diff.bool(), spd_diff.abs(), torch.zeros_like(spd_diff)).amax(dim=1)

    loss = (1.0 - _pearson_r(mean_spd, mean_acc)) + (1.0 - _pearson_r(mean_acc, max_acc))
    return loss * 0.5


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
    def __init__(self, dxdy, stall, lengths, conditions, max_len, data_scale,
                 polar=False, speed_aug=0.0, heading_flip=False,
                 heading_scale=0.0, heading_noise=0.0, spd=None, dh=None):
        self.dxdy = dxdy
        self.stall = stall
        self.lengths = lengths
        self.conditions = conditions
        self.max_len = max_len
        self.data_scale = data_scale
        self.polar = polar
        self.speed_aug = speed_aug
        self.heading_flip = heading_flip
        self.heading_scale = heading_scale
        self.heading_noise = heading_noise
        self.spd = spd
        self.dh = dh

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        L = int(self.lengths[idx])
        T = min(L, self.max_len)

        out = np.zeros((self.max_len, 2), dtype=np.float32)
        if self.polar:
            if self.spd is not None:
                out[:T, 0] = self.spd[idx, :T] * self.data_scale[0]
                out[:T, 1] = self.dh[idx, :T] * self.data_scale[1]
            else:
                spd, dh = _to_polar(self.dxdy[idx], self.stall[idx], T)
                out[:T, 0] = spd * self.data_scale[0]
                out[:T, 1] = dh * self.data_scale[1]
        else:
            out[:T] = self.dxdy[idx, :T] * self.data_scale

        st = np.zeros(self.max_len, dtype=np.float32)
        st[:T] = self.stall[idx, :T].astype(np.float32)

        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:T] = 1.0

        cond = self.conditions[idx].copy()
        if self.speed_aug > 0 and self.polar:
            log_k = np.random.uniform(-self.speed_aug, self.speed_aug)
            k = float(np.exp(log_k))
            out[:T, 0] *= k
            cond[0] += log_k
        if self.heading_scale > 0 and self.polar and T > 1:
            log_k = np.random.uniform(-self.heading_scale, self.heading_scale)
            k = float(np.exp(log_k))
            out[1:T, 1] *= k
        if self.heading_noise > 0 and self.polar and T > 2:
            dt_ou = 1.0
            theta_ou = 5.0
            x_ou = 0.0
            for i in range(1, T):
                x_ou += -theta_ou * x_ou * dt_ou + self.heading_noise * np.sqrt(dt_ou) * np.random.standard_normal()
                out[i, 1] += x_ou * self.data_scale[1]
        if self.heading_flip and self.polar and np.random.random() < 0.5:
            out[:T, 1] = -out[:T, 1]
            cond[2] = -cond[2]

        return (
            torch.from_numpy(out),
            torch.from_numpy(st),
            torch.from_numpy(mask).bool(),
            torch.from_numpy(cond),
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

    spd_all = dh_all = None
    if args.polar and (data_dir / "zimt_polar_spd.npy").exists():
        spd_all = np.load(data_dir / "zimt_polar_spd.npy", mmap_mode="r")
        dh_all = np.load(data_dir / "zimt_polar_dh.npy", mmap_mode="r")
        print("  Using precomputed polar arrays")

    if args.max_samples and args.max_samples < N:
        idx = np.random.default_rng(42).choice(N, args.max_samples, replace=False)
        lengths = lengths[idx]
        conditions = conditions[idx]
        stall_sel = stall[idx]
        dxdy_sel = None if spd_all is not None else dxdy[idx]
        spd_sel = spd_all[idx] if spd_all is not None else None
        dh_sel = dh_all[idx] if dh_all is not None else None
    else:
        dxdy_sel = dxdy
        stall_sel = stall
        spd_sel = spd_all
        dh_sel = dh_all
        idx = None

    N = len(lengths)
    print(f"  {N:,} trajectories")

    if args.polar and spd_sel is not None:
        n_scale = min(N, 50000)
        valid = np.arange(spd_sel.shape[1])[None, :] < lengths[:n_scale, None]
        spd_std = float(np.asarray(spd_sel[:n_scale])[valid].std())
        dh_std = float(np.asarray(dh_sel[:n_scale])[valid].std())
        data_std = np.array([spd_std, dh_std], dtype=np.float32)
        data_scale = np.array([1.0 / spd_std, 1.0 / dh_std], dtype=np.float32)
        print(f"  [polar] spd_std={spd_std:.6f}, dh_std={dh_std:.6f}")
    elif args.polar:
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

    tr_idx, va_idx = perm[n_val:], perm[:n_val]

    def _take(arr, sub):
        return None if arr is None else arr[sub]

    train_ds = CANDIDataset(
        _take(dxdy_sel, tr_idx), stall_sel[tr_idx],
        lengths[tr_idx], conditions[tr_idx],
        args.max_seq_len, data_scale, polar=args.polar,
        speed_aug=args.speed_aug, heading_flip=args.heading_flip,
        heading_scale=args.heading_scale, heading_noise=args.heading_noise,
        spd=_take(spd_sel, tr_idx), dh=_take(dh_sel, tr_idx),
    )
    val_ds = CANDIDataset(
        _take(dxdy_sel, va_idx), stall_sel[va_idx],
        lengths[va_idx], conditions[va_idx],
        args.max_seq_len, data_scale, polar=args.polar,
        spd=_take(spd_sel, va_idx), dh=_take(dh_sel, va_idx),
    )

    n_workers = args.num_workers
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=True, drop_last=True,
        persistent_workers=n_workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=n_workers > 0,
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

    start_epoch = 0
    if args.load_from:
        ckpt_resume = torch.load(args.load_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_resume["model_state_dict"])
        if not args.reset_schedule:
            start_epoch = ckpt_resume.get("epoch", 0)
        print(f"  Resumed from {args.load_from} (epoch {start_epoch})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_epochs = start_epoch + args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    for _ in range(start_epoch):
        scheduler.step()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    bce = nn.BCEWithLogitsLoss(reduction="none")
    best_val = float("inf")
    if args.save_name:
        save_name = args.save_name
    else:
        save_name = "candi_polar_best.pt" if args.polar else "candi_best.pt"
    save_path = data_dir / save_name

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()
        tot_cont, tot_disc, nb = 0.0, 0.0, 0

        for dxdy_b, stall_b, pad_b, cond_b in train_dl:
            dxdy_b = dxdy_b.to(device)
            stall_b = stall_b.to(device)
            pad_b = pad_b.to(device)
            cond_b = cond_b.to(device)

            B = dxdy_b.shape[0]

            if args.pred_type == "flow":
                t_cont = torch.rand(B, device=device)
                t_int = (t_cont * (args.n_steps - 1)).long()
                dxdy_noisy, noise, velocity = model.q_flow(dxdy_b, t_cont)
                stall_masked, disc_mask = model.q_discrete(stall_b, t_int)
                t_for_model = t_cont * (args.n_steps - 1)
            else:
                t_int = torch.randint(0, args.n_steps, (B,), device=device)
                dxdy_noisy, noise = model.q_continuous(dxdy_b, t_int)
                stall_masked, disc_mask = model.q_discrete(stall_b, t_int)
                t_for_model = t_int

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                dxdy_pred, stall_logit = model(
                    dxdy_noisy, stall_masked, disc_mask.float(), t_for_model, cond_b, pad_b,
                )

                pad_f = pad_b.float().unsqueeze(-1)
                if args.pred_type == "flow":
                    cont_target = velocity
                elif args.pred_type == "x0":
                    cont_target = dxdy_b
                elif args.pred_type == "eps":
                    cont_target = noise
                else:
                    s_ab = model.sqrt_ab[t_int].view(-1, 1, 1)
                    s_1mab = model.sqrt_1mab[t_int].view(-1, 1, 1)
                    cont_target = s_ab * noise - s_1mab * dxdy_b
                cont_loss = ((dxdy_pred - cont_target) ** 2 * pad_f).sum() / pad_f.sum().clamp(1)

                disc_target = stall_b
                disc_loss_raw = bce(stall_logit, disc_target)
                disc_weight = disc_mask.float() * pad_b.float()
                disc_loss = (disc_loss_raw * disc_weight).sum() / disc_weight.sum().clamp(1)

                loss = cont_loss + args.disc_weight * disc_loss

                if args.corr_weight > 0 and args.pred_type != "flow":
                    cl = _corr_loss(dxdy_pred, dxdy_noisy, pad_b, t_int, model, args.pred_type)
                    loss = loss + args.corr_weight * cl

                if args.path_weight > 0:
                    t_for_path = t_cont if args.pred_type == "flow" else t_int
                    pl = _path_loss(dxdy_pred, dxdy_noisy, pad_b, t_for_path,
                                    model, args.pred_type, data_scale)
                    loss = loss + args.path_weight * pl

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

                if args.pred_type == "flow":
                    vt_cont = torch.rand(B, device=device)
                    vt_int = (vt_cont * (args.n_steps - 1)).long()
                    dxdy_noisy, val_noise, val_velocity = model.q_flow(dxdy_b, vt_cont)
                    stall_masked, disc_mask = model.q_discrete(stall_b, vt_int)
                    vt_for_model = vt_cont * (args.n_steps - 1)
                else:
                    vt_int = torch.randint(0, args.n_steps, (B,), device=device)
                    dxdy_noisy, val_noise = model.q_continuous(dxdy_b, vt_int)
                    stall_masked, disc_mask = model.q_discrete(stall_b, vt_int)
                    vt_for_model = vt_int

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    dxdy_pred, stall_logit = model(
                        dxdy_noisy, stall_masked, disc_mask.float(), vt_for_model, cond_b, pad_b,
                    )
                    pad_f = pad_b.float().unsqueeze(-1)
                    if args.pred_type == "flow":
                        val_target = val_velocity
                    elif args.pred_type == "x0":
                        val_target = dxdy_b
                    elif args.pred_type == "eps":
                        val_target = val_noise
                    else:
                        s_ab = model.sqrt_ab[vt_int].view(-1, 1, 1)
                        s_1mab = model.sqrt_1mab[vt_int].view(-1, 1, 1)
                        val_target = s_ab * val_noise - s_1mab * dxdy_b
                    cl = ((dxdy_pred - val_target) ** 2 * pad_f).sum() / pad_f.sum().clamp(1)
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

        print(f"  Epoch {epoch+1:3d}/{total_epochs} | "
              f"train cont={tc:.4f} disc={td:.4f} | "
              f"val cont={vc:.4f} disc={vd:.4f} | "
              f"lr {lr:.2e} | {elapsed:.0f}s", flush=True)

        if args.cooldown > 0:
            torch.cuda.empty_cache()
            time.sleep(args.cooldown)

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
            "pred_type": args.pred_type,
            "epoch": epoch + 1,
            "val_cont": vc,
            "val_disc": vd,
        }
        latest_path = save_path.with_stem(save_path.stem + "_latest")
        torch.save(ckpt, latest_path)

        if val_total < best_val:
            best_val = val_total
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
    parser.add_argument("--pred-type", default="x0", choices=["x0", "eps", "v", "flow"],
                        help="Prediction target: x0 (clean data), eps (noise), v (velocity), flow (flow matching)")
    parser.add_argument("--corr-weight", type=float, default=0.0,
                        help="Weight for correlation-matching auxiliary loss (0=disabled)")
    parser.add_argument("--save-name", default=None,
                        help="Override checkpoint filename (default: auto based on polar/cartesian)")
    parser.add_argument("--load-from", default=None,
                        help="Load model weights from this checkpoint to resume training")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers (reduce to 0-1 for stability)")
    parser.add_argument("--cooldown", type=int, default=0,
                        help="Seconds to pause between epochs (GPU cooldown)")
    parser.add_argument("--speed-aug", type=float, default=0.0,
                        help="Speed augmentation range (log-scale). E.g. 0.5 means scale speed by exp(U(-0.5,0.5))=[0.6,1.6]")
    parser.add_argument("--heading-flip", action="store_true",
                        help="Randomly flip delta_heading sign (mirror paths) with 50%% probability")
    parser.add_argument("--path-weight", type=float, default=0.0,
                        help="Weight for path_efficiency distribution matching loss (0=disabled)")
    parser.add_argument("--heading-scale", type=float, default=0.0,
                        help="Per-trajectory curvature scaling range (log-scale). E.g. 0.3 means scale dh by exp(U(-0.3,0.3))=[0.74,1.35]")
    parser.add_argument("--heading-noise", type=float, default=0.0,
                        help="OU-process noise amplitude on delta_heading during training. E.g. 0.1 adds structured curvature perturbation")
    parser.add_argument("--reset-schedule", action="store_true",
                        help="Start LR schedule from epoch 0 when fine-tuning (ignore checkpoint epoch)")
    args = parser.parse_args()
    train(args)
