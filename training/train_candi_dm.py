"""WS3: distribution-matching fine-tune of the CANDI polar flow model.

Backprop through a short Euler unroll of the flow ODE, compute differentiable
kinematic features on the generated batch, and minimize a multi-bandwidth RBF
MMD against the same features on real human batches. The standard flow-matching
loss on real data stays on as a regularizer so the model cannot drift far from
the pretrained solution.

The hypothesis: the ZIMT feature-matching failure was exposure bias, an
autoregressive disease. This model has no autoregressive loop, so matching
generated-batch statistics directly should be stable.

Run (short, crash-resumable):
    .venv/Scripts/python.exe training/train_candi_dm.py --steps 600 \
        --save-name candi_dm_v1.pt
Resume after a crash with the same command; it picks up the _latest file.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.candi import CANDIModel  # noqa: E402


def masked_mean(x, mask, dim=1):
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)


def masked_std(x, mask, dim=1):
    m = masked_mean(x, mask, dim).unsqueeze(dim)
    var = ((x - m) ** 2 * mask).sum(dim) / mask.sum(dim).clamp(min=1)
    return (var + 1e-8).sqrt()


def batch_features(speed, dh, mask):
    """Differentiable per-trajectory kinematic features.

    speed: (B, T) raw px per sample, zeros at stalls. dh: (B, T) radians.
    mask: (B, T) float validity. Returns (B, F).
    """
    B, T = speed.shape
    dev = speed.device

    mean_spd = masked_mean(speed, mask)
    std_spd = masked_std(speed, mask)
    max_spd = (speed * mask).amax(dim=1)

    acc = torch.diff(speed, dim=1)
    m_acc = mask[:, 1:] * mask[:, :-1]
    mean_acc = masked_mean(acc, m_acc)
    mean_abs_acc = masked_mean(acc.abs(), m_acc)
    std_acc = masked_std(acc, m_acc)
    max_abs_acc = (acc.abs() * m_acc).amax(dim=1)

    jerk = torch.diff(acc, dim=1)
    m_jerk = m_acc[:, 1:] * m_acc[:, :-1]
    mean_abs_jerk = masked_mean(jerk.abs(), m_jerk)
    std_jerk = masked_std(jerk, m_jerk)

    mean_abs_dh = masked_mean(dh.abs(), mask)
    std_dh = masked_std(dh, mask)

    stall_frac = masked_mean(torch.sigmoid((0.5 - speed) * 4.0), mask)

    heading = torch.cumsum(dh, dim=1)
    vx = speed * torch.cos(heading) * mask
    vy = speed * torch.sin(heading) * mask
    end_x = vx.sum(dim=1)
    end_y = vy.sum(dim=1)
    d_straight = (end_x ** 2 + end_y ** 2 + 1e-8).sqrt()
    d_traveled = (speed * mask).sum(dim=1).clamp(min=1e-8)
    path_eff = d_straight / d_traveled

    return torch.stack([
        mean_spd, std_spd, max_spd,
        mean_acc, mean_abs_acc, std_acc, max_abs_acc,
        mean_abs_jerk, std_jerk,
        mean_abs_dh, std_dh,
        stall_frac, path_eff,
    ], dim=1)


def mmd_rbf(x, y, bandwidths=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Multi-bandwidth RBF MMD^2 between feature sets x (B,F) and y (B2,F)."""
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2
    loss = x.new_zeros(())
    for bw in bandwidths:
        loss = loss + (torch.exp(-xx / (2 * bw)).mean()
                       + torch.exp(-yy / (2 * bw)).mean()
                       - 2 * torch.exp(-xy / (2 * bw)).mean())
    return loss


def unroll_flow(model, cond, pad_mask, T, n_steps, grad_steps, generator=None):
    """Euler unroll of the flow ODE with gradients on the last grad_steps steps.

    Returns (spd_norm, dh_norm, stall_prob), all (B, T), differentiable.
    """
    B = cond.shape[0]
    dev = cond.device
    xt = torch.randn(B, T, 2, device=dev, generator=generator)
    stall_s = torch.full((B, T), model.STALL_MASK, device=dev)
    mflag = torch.ones(B, T, device=dev)
    dt = 1.0 / n_steps

    sl = None
    for i in range(n_steps):
        t_cont = 1.0 - i * dt
        t_scaled = torch.full((B,), t_cont * (model.n_steps - 1), device=dev)
        use_grad = i >= n_steps - grad_steps
        with torch.set_grad_enabled(use_grad):
            v_pred, sl = model(xt, stall_s, mflag, t_scaled, cond, pad_mask)
            xt = xt - dt * v_pred
        if not use_grad:
            xt = xt.detach()

    stall_p = torch.sigmoid(sl)
    return xt[:, :, 0], xt[:, :, 1], stall_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="training")
    ap.add_argument("--load-from", default="training/candi_polar_flow_best.pt")
    ap.add_argument("--save-name", default="candi_dm_v1.pt")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--unroll", type=int, default=8)
    ap.add_argument("--grad-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--fm-weight", type=float, default=1.0)
    ap.add_argument("--mmd-weight", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=128,
                    help="Unroll length; trajectories longer than this are skipped")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.load_from, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    data_scale = np.asarray(ckpt["data_scale"], dtype=np.float32)
    model = CANDIModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # dropout off; gradients still flow
    T = min(cfg["max_seq_len"], args.seq_len)

    spd_all = np.load(data_dir / "zimt_polar_spd.npy", mmap_mode="r")
    dh_all = np.load(data_dir / "zimt_polar_dh.npy", mmap_mode="r")
    stall_all = np.load(data_dir / "zimt_stall.npy", mmap_mode="r")
    lengths = np.load(data_dir / "zimt_lengths.npy")
    conditions = np.load(data_dir / "zimt_conditions.npy")
    usable = np.where((lengths >= 8) & (lengths <= T))[0]
    print(f"{len(usable):,} usable trajectories, T={T}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    save_path = data_dir / args.save_name
    latest_path = save_path.with_name(save_path.stem + "_latest.pt")
    start_step = 0
    if latest_path.exists():
        rs = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(rs["model_state_dict"])
        opt.load_state_dict(rs["opt_state_dict"])
        start_step = rs["step"]
        print(f"Resumed from {latest_path} at step {start_step}", flush=True)

    spd_scale, dh_scale = float(data_scale[0]), float(data_scale[1])

    def real_batch(B):
        idx = rng.choice(usable, B, replace=False)
        spd = np.zeros((B, T), dtype=np.float32)
        dh = np.zeros((B, T), dtype=np.float32)
        st = np.zeros((B, T), dtype=np.float32)
        msk = np.zeros((B, T), dtype=np.float32)
        for j, i in enumerate(idx):
            L = int(lengths[i])
            spd[j, :L] = spd_all[i, :L]
            dh[j, :L] = dh_all[i, :L]
            st[j, :L] = stall_all[i, :L]
            msk[j, :L] = 1.0
        cond = conditions[idx].astype(np.float32)
        return (torch.from_numpy(spd).to(device), torch.from_numpy(dh).to(device),
                torch.from_numpy(st).to(device), torch.from_numpy(msk).to(device),
                torch.from_numpy(cond).to(device))

    t0 = time.time()
    run_mmd, run_fm, nb = 0.0, 0.0, 0
    for step in range(start_step, args.steps):
        spd_r, dh_r, st_r, mask_r, cond_r = real_batch(args.batch_size)
        pad_r = mask_r.bool()

        # Generated branch: unroll conditioned on the real batch's conditions.
        spd_g_n, dh_g_n, stall_p = unroll_flow(
            model, cond_r, pad_r, T, args.unroll, args.grad_steps)
        alive = 1.0 - stall_p
        spd_g = torch.clamp(spd_g_n / spd_scale, min=0) * alive * mask_r
        dh_g = dh_g_n / dh_scale * alive * mask_r

        feat_g = batch_features(spd_g, dh_g, mask_r)
        with torch.no_grad():
            spd_r_eff = spd_r * (1.0 - st_r) * mask_r
            feat_r = batch_features(spd_r_eff, dh_r * mask_r, mask_r)
            f_mean = feat_r.mean(dim=0)
            f_std = feat_r.std(dim=0).clamp(min=1e-6)
        mmd = mmd_rbf((feat_g - f_mean) / f_std, (feat_r - f_mean) / f_std)

        # FM regularizer on the same real batch (normalized channels).
        x0 = torch.stack([spd_r * spd_scale, dh_r * dh_scale], dim=-1)
        B = x0.shape[0]
        t_cont = torch.rand(B, device=device)
        t_int = (t_cont * (cfg["n_diffusion_steps"] - 1)).long()
        x_noisy, _, velocity = model.q_flow(x0, t_cont)
        st_masked, disc_mask = model.q_discrete(st_r, t_int)
        v_pred, sl_pred = model(
            x_noisy, st_masked, disc_mask.float(),
            t_cont * (cfg["n_diffusion_steps"] - 1), cond_r, pad_r)
        pad_f = mask_r.unsqueeze(-1)
        fm_cont = ((v_pred - velocity) ** 2 * pad_f).sum() / pad_f.sum().clamp(1)
        dw = disc_mask.float() * mask_r
        fm_disc = (bce(sl_pred, st_r) * dw).sum() / dw.sum().clamp(1)
        fm = fm_cont + fm_disc

        loss = args.mmd_weight * mmd + args.fm_weight * fm
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        run_mmd += mmd.item()
        run_fm += fm.item()
        nb += 1
        if (step + 1) % args.log_every == 0:
            print(f"step {step+1:4d}/{args.steps} | mmd {run_mmd/nb:.4f} | "
                  f"fm {run_fm/nb:.4f} | {(time.time()-t0)/nb:.2f}s/step", flush=True)
            run_mmd, run_fm, nb = 0.0, 0.0, 0
            t0 = time.time()

        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            out = {
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "config": cfg,
                "data_scale": data_scale,
                "data_std": ckpt.get("data_std"),
                "polar": True,
                "pred_type": "flow",
                "step": step + 1,
            }
            torch.save(out, latest_path)
            torch.save(out, save_path)
            print(f"  saved at step {step+1}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
