"""CANDI hybrid discrete-continuous diffusion experiment.

Supports both Cartesian (dx,dy) and polar (speed, delta_heading) checkpoints.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments._common import DurationModel, Trajectory, get_device
from features import extract_features, resample_trajectory
from models.candi import CANDIModel

torch.manual_seed(42)

_DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
_TRAIN_DIR = Path(os.environ.get("TRAIN_DIR", "./training"))
_DEVICE = get_device()
_HZ = 125.0

_ckpt_name = os.environ.get("CANDI_CKPT", "candi_best.pt")
_ckpt_path = _TRAIN_DIR / _ckpt_name
_ckpt = torch.load(_ckpt_path, map_location=_DEVICE, weights_only=False)
_cfg = _ckpt["config"]
_data_scale = _ckpt["data_scale"]
_POLAR = _ckpt.get("polar", False)
_PRED_TYPE = _ckpt.get("pred_type", "x0")

_model = CANDIModel(**_cfg).to(_DEVICE)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()

_DUR_STD = float(os.environ.get("CANDI_DUR_STD", "0.7"))
_duration = DurationModel(_TRAIN_DIR, std_mult=_DUR_STD)

_N_SAMPLE_STEPS = int(os.environ.get("CANDI_STEPS", "50"))
_ETA = float(os.environ.get("CANDI_ETA", "0.0"))
_CFG = float(os.environ.get("CANDI_CFG", "2.0"))
_N_CANDIDATES = int(os.environ.get("CANDI_CANDIDATES", "1"))
_GUIDE = float(os.environ.get("CANDI_GUIDE", "0.0"))
_CORRECT = os.environ.get("CANDI_CORRECT", "additive")
_SMOOTH_DH = float(os.environ.get("CANDI_SMOOTH_DH", "0.0"))
_SMOOTH_POS = int(os.environ.get("CANDI_SMOOTH_POS", "0"))
_JITTER = float(os.environ.get("CANDI_JITTER", "0.0"))
_SPEED_JITTER = float(os.environ.get("CANDI_SPEED_JITTER", "0.0"))
_OU_SIGMA = float(os.environ.get("CANDI_OU_SIGMA", "0.0"))
_OU_THETA = float(os.environ.get("CANDI_OU_THETA", "5.0"))
_DH_OU_SIGMA = float(os.environ.get("CANDI_DH_OU_SIGMA", "0.0"))
_DH_OU_THETA = float(os.environ.get("CANDI_DH_OU_THETA", "3.0"))
_SHARPEN = float(os.environ.get("CANDI_SHARPEN", "0.0"))
_FEAT_GUIDE = float(os.environ.get("CANDI_FEAT_GUIDE", "0.0"))
_FEAT_EFF_TARGET = float(os.environ.get("CANDI_FEAT_EFF_TARGET", "0.84"))
_ACC_SCALE = float(os.environ.get("CANDI_ACC_SCALE", "0.0"))
_ACC_MODE = os.environ.get("CANDI_ACC_MODE", "speed")
_PERP_SCALE = float(os.environ.get("CANDI_PERP_SCALE", "1.0"))
_RESIDUAL_VEL = float(os.environ.get("CANDI_RESIDUAL_VEL", "0.0"))
_RESIDUAL_FRAC = float(os.environ.get("CANDI_RESIDUAL_FRAC", "0.25"))
_RESIDUAL_PROB = float(os.environ.get("CANDI_RESIDUAL_PROB", "1.0"))
_SPEED_SKEW = float(os.environ.get("CANDI_SPEED_SKEW", "0.0"))
_SPEED_SKEW_SCALE = float(os.environ.get("CANDI_SPEED_SKEW_SCALE", "0.0"))
_SPEED_REPLACE = os.environ.get("CANDI_SPEED_REPLACE", "")
_MIN_SPEED = float(os.environ.get("CANDI_MIN_SPEED", "0.0"))
_SPEED_RAMP = float(os.environ.get("CANDI_SPEED_RAMP", "0.0"))
_DH_AMP = float(os.environ.get("CANDI_DH_AMP", "0.0"))
_PERP_HP = float(os.environ.get("CANDI_PERP_HP", "1.0"))
_PERP_HP_WIN = int(os.environ.get("CANDI_PERP_HP_WIN", "21"))
_FLOW_NOISE = float(os.environ.get("CANDI_FLOW_NOISE", "0.0"))
_SCORE_MODE = os.environ.get("CANDI_SCORE_MODE", "target")
_SCORE_WEIGHTS_STR = os.environ.get("CANDI_SCORE_WEIGHTS", "")
_score_weights = None
if _SCORE_WEIGHTS_STR:
    _score_weights = np.array([float(x) for x in _SCORE_WEIGHTS_STR.split(",")])
_current_log_dist = 0.0

_human_features = np.load(_DATA_DIR / "human_eval_features.npy")
_human_feat_std = _human_features.std(axis=0)
_human_feat_std = np.maximum(_human_feat_std, 1e-8)
_human_feat_mean = _human_features.mean(axis=0)

_n_params = sum(p.numel() for p in _model.parameters())
_mode = "polar" if _POLAR else "cartesian"
print(f"[candi] {_n_params:,} params, mode={_mode}, "
      f"steps={_N_SAMPLE_STEPS}, eta={_ETA}, cfg={_CFG}, "
      f"candidates={_N_CANDIDATES}, guide={_GUIDE}, correct={_CORRECT}")


def _decode_cartesian(raw_np, stall_np):
    dxdy_np = raw_np / _data_scale
    dxdy_np[stall_np > 0.5] = 0.0
    return np.cumsum(dxdy_np[:, 0]), np.cumsum(dxdy_np[:, 1])


def _decode_polar(raw_np, stall_np):
    spd_scale, dh_scale = float(_data_scale[0]), float(_data_scale[1])
    speed = np.maximum(raw_np[:, 0] / spd_scale, 0.0)
    dheading = raw_np[:, 1] / dh_scale
    speed[stall_np > 0.5] = 0.0
    dheading[stall_np > 0.5] = 0.0
    if _SHARPEN > 0:
        moving = stall_np < 0.5
        s_mov = speed[moving]
        if len(s_mov) > 0:
            s_max = np.max(s_mov)
            if s_max > 1e-6:
                s_norm = s_mov / s_max
                p = 1.0 + _SHARPEN * math.log1p(s_max * spd_scale)
                speed[moving] = np.power(np.maximum(s_norm, 0.0), p) * s_max
    if _ACC_SCALE > 0:
        moving = stall_np < 0.5
        s_mov = speed[moving]
        if len(s_mov) > 2:
            s_mean = np.mean(s_mov)
            if s_mean > 1e-6:
                if _ACC_MODE == "dist":
                    k = 1.0 + _ACC_SCALE * _current_log_dist
                else:
                    s_max = np.max(s_mov)
                    k = 1.0 + _ACC_SCALE * math.log1p(s_max * spd_scale)
                speed[moving] = np.maximum(s_mean + k * (s_mov - s_mean), 0.0)
    if _OU_SIGMA > 0:
        rng_ou = np.random.default_rng()
        ou = np.zeros(len(speed))
        dt_ou = 1.0 / _HZ
        x_ou = 0.0
        for i in range(len(speed)):
            ou[i] = x_ou
            x_ou += -_OU_THETA * x_ou * dt_ou + _OU_SIGMA * math.sqrt(dt_ou) * rng_ou.standard_normal()
        moving = stall_np < 0.5
        speed[moving] *= np.exp(ou[moving])
        speed = np.maximum(speed, 0.0)
    if _SPEED_JITTER > 0:
        rng_s = np.random.default_rng()
        moving = stall_np < 0.5
        speed[moving] *= (1.0 + rng_s.normal(0, _SPEED_JITTER, size=moving.sum()))
        speed = np.maximum(speed, 0.0)
    if _SMOOTH_DH > 0:
        alpha = _SMOOTH_DH
        for i in range(1, len(dheading)):
            if stall_np[i] < 0.5:
                dheading[i] = alpha * dheading[i - 1] + (1 - alpha) * dheading[i]
    if _DH_AMP > 0:
        moving = stall_np < 0.5
        dheading[moving] *= (1.0 + _DH_AMP)
    if _JITTER > 0:
        rng = np.random.default_rng()
        noise = rng.normal(0, _JITTER, size=dheading.shape) * speed
        noise[stall_np > 0.5] = 0.0
        dheading = dheading + noise
    if _DH_OU_SIGMA > 0:
        rng_dh = np.random.default_rng()
        dt_dh = 1.0 / _HZ
        x_dh = 0.0
        for i in range(len(dheading)):
            if stall_np[i] < 0.5:
                dheading[i] += x_dh
            x_dh += -_DH_OU_THETA * x_dh * dt_dh + _DH_OU_SIGMA * math.sqrt(dt_dh) * rng_dh.standard_normal()
    if _SPEED_SKEW > 0:
        moving = stall_np < 0.5
        s_mov = speed[moving].copy()
        if len(s_mov) > 5:
            T = len(s_mov)
            t_orig = np.linspace(0, 1, T)
            k = _SPEED_SKEW
            if _SPEED_SKEW_SCALE > 0:
                s_max = np.max(s_mov)
                k = _SPEED_SKEW * (1.0 + _SPEED_SKEW_SCALE * math.log1p(s_max * spd_scale))
            t_warped = np.power(t_orig, 1.0 / (1.0 + k))
            speed[moving] = np.interp(t_warped, t_orig, s_mov)
    if _SPEED_REPLACE:
        moving = stall_np < 0.5
        n_mov = int(moving.sum())
        if n_mov > 5:
            total_dist = np.sum(speed[moving])
            t_norm = np.linspace(1e-6, 1.0 - 1e-6, n_mov)
            if _SPEED_REPLACE == "beta":
                from scipy.stats import beta as beta_dist
                a_beta, b_beta = 2.5, 4.5
                profile = beta_dist.pdf(t_norm, a_beta, b_beta)
            elif _SPEED_REPLACE == "asym_mj":
                peak_t = 0.35
                rise = np.where(t_norm <= peak_t,
                                np.sin(np.pi * t_norm / (2 * peak_t)) ** 2, 0)
                fall = np.where(t_norm > peak_t,
                                np.cos(np.pi * (t_norm - peak_t) / (2 * (1 - peak_t))) ** 2, 0)
                profile = rise + fall
            else:
                profile = np.ones(n_mov)
            profile = profile / np.sum(profile) * total_dist
            speed[moving] = profile
    if _SPEED_RAMP > 0:
        moving = stall_np < 0.5
        s_mov = speed[moving].copy()
        if len(s_mov) > 5:
            s_max = np.max(s_mov)
            ramp = _SPEED_RAMP * s_max * np.linspace(0, 1, len(s_mov))
            speed[moving] = s_mov + ramp
    if _MIN_SPEED > 0:
        moving = stall_np < 0.5
        s_mov = speed[moving]
        if len(s_mov) > 0:
            s_mean = np.mean(s_mov) if np.mean(s_mov) > 1e-8 else 1e-8
            floor = _MIN_SPEED * s_mean
            speed[moving] = np.maximum(s_mov, floor)
    if _RESIDUAL_VEL > 0 and np.random.random() < _RESIDUAL_PROB:
        moving = stall_np < 0.5
        s_mov = speed[moving].copy()
        if len(s_mov) > 5:
            s_mean = np.mean(s_mov)
            target_residual = _RESIDUAL_VEL * s_mean
            n_ramp = max(1, int(len(s_mov) * _RESIDUAL_FRAC))
            envelope = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n_ramp)))
            s_mov[-n_ramp:] += target_residual * envelope
            speed[moving] = s_mov
    heading = np.cumsum(dheading)
    cum_x = np.cumsum(speed * np.cos(heading))
    cum_y = np.cumsum(speed * np.sin(heading))
    if _SMOOTH_POS > 0 and len(cum_x) > _SMOOTH_POS:
        from scipy.signal import savgol_filter
        wl = _SMOOTH_POS if _SMOOTH_POS % 2 == 1 else _SMOOTH_POS + 1
        if wl >= 5 and wl < len(cum_x):
            cum_x = savgol_filter(cum_x, wl, 3)
            cum_y = savgol_filter(cum_y, wl, 3)
    return cum_x, cum_y


def _build_trajectory(cum_x, cum_y, stall_np, seq_len, total_dist, dx, dy,
                      start_x, start_y, end_x, end_y):
    target_dx = dx / total_dist if total_dist > 0 else 0.0
    target_dy = dy / total_dist if total_dist > 0 else 0.0

    if _CORRECT == "rotate":
        raw_mag = math.hypot(cum_x[-1], cum_y[-1])
        if raw_mag > 1e-8:
            tgt_mag = math.hypot(target_dx, target_dy)
            scale = tgt_mag / raw_mag
            raw_ang = math.atan2(cum_y[-1], cum_x[-1])
            tgt_ang = math.atan2(target_dy, target_dx)
            rot = tgt_ang - raw_ang
            cos_r, sin_r = math.cos(rot), math.sin(rot)
            rx = (cum_x * cos_r - cum_y * sin_r) * scale
            ry = (cum_x * sin_r + cum_y * cos_r) * scale
            cum_x, cum_y = rx, ry
    else:
        err_x = target_dx - cum_x[-1]
        err_y = target_dy - cum_y[-1]
        if err_x * err_x + err_y * err_y > 1e-8:
            moving = stall_np < 0.5
            if moving.sum() > 0:
                magnitudes = np.sqrt(np.diff(np.concatenate([[0], cum_x])) ** 2 +
                                     np.diff(np.concatenate([[0], cum_y])) ** 2)
                mag_moving = magnitudes * moving
                total_mag = mag_moving.sum()
                weights = (mag_moving / total_mag if total_mag > 1e-8
                           else moving.astype(np.float64) / moving.sum())
                cum_x = cum_x + err_x * np.cumsum(weights)
                cum_y = cum_y + err_y * np.cumsum(weights)
            else:
                frac = np.linspace(0, 1, seq_len)
                cum_x = cum_x + err_x * frac
                cum_y = cum_y + err_y * frac

    if _PERP_SCALE != 1.0 or _PERP_HP != 1.0:
        tgt_mag = math.hypot(target_dx, target_dy)
        if tgt_mag > 1e-8:
            dx_n = target_dx / tgt_mag
            dy_n = target_dy / tgt_mag
            par = cum_x * dx_n + cum_y * dy_n
            perp_x = cum_x - par * dx_n
            perp_y = cum_y - par * dy_n
            if _PERP_HP != 1.0 and len(perp_x) >= _PERP_HP_WIN:
                w = _PERP_HP_WIN
                kernel = np.ones(w) / w
                px_low = np.convolve(perp_x, kernel, mode='same')
                py_low = np.convolve(perp_y, kernel, mode='same')
                px_high = perp_x - px_low
                py_high = perp_y - py_low
                perp_x = _PERP_SCALE * px_low + _PERP_HP * px_high
                perp_y = _PERP_SCALE * py_low + _PERP_HP * py_high
            else:
                perp_x = _PERP_SCALE * perp_x
                perp_y = _PERP_SCALE * perp_y
            cum_x = par * dx_n + perp_x
            cum_y = par * dy_n + perp_y

    out_x = cum_x * total_dist + start_x
    out_y = cum_y * total_dist + start_y

    dt = 1.0 / _HZ
    result: Trajectory = [(start_x, start_y, 0.0)]
    for i in range(seq_len):
        result.append((float(out_x[i]), float(out_y[i]), (i + 1) * dt))
    result[-1] = (end_x, end_y, result[-1][2])
    return result


_target_idx = 0
_score_target = _human_features[0]


def _get_target_features():
    global _target_idx
    target = _human_features[_target_idx % len(_human_features)]
    _target_idx += 1
    return target


def _score_trajectory(traj: Trajectory) -> float:
    if len(traj) < 5:
        return -1e6
    resampled = resample_trajectory(traj)
    feats = extract_features(resampled)
    if feats is None:
        return -1e6
    if _SCORE_MODE == "marginal":
        z = (feats - _human_feat_mean) / _human_feat_std
    else:
        z = (feats - _score_target) / _human_feat_std
    z2 = z ** 2
    if _score_weights is not None and len(_score_weights) == len(z2):
        z2 = z2 * _score_weights
    return -float(np.sum(z2))


def _sample_guided_polar(cond, seq_len, target_cos, target_sin,
                         n_steps, eta, cfg_scale, guide):
    B = cond.shape[0]
    dev = cond.device
    spd_s = float(_data_scale[0])
    dh_s = float(_data_scale[1])
    tgt_angles = np.atleast_1d(np.arctan2(target_sin, target_cos)).astype(np.float64)

    xt = torch.randn(B, seq_len, 2, device=dev)
    stall_s = torch.full((B, seq_len), _model.STALL_MASK, device=dev)
    mflag = torch.ones(B, seq_len, device=dev)

    step_size = _model.n_steps // n_steps
    times = list(range(_model.n_steps - 1, -1, -step_size))
    if times[-1] != 0:
        times.append(0)

    for i, tv in enumerate(times):
        t = torch.full((B,), tv, dtype=torch.long, device=dev)
        raw, sl = _model(xt, stall_s, mflag, t, cond)

        if cfg_scale > 0:
            raw_u, sl_u = _model(xt, stall_s, mflag, t, torch.zeros_like(cond))
            raw = raw_u + cfg_scale * (raw - raw_u)
            sl = sl_u + cfg_scale * (sl - sl_u)

        if _PRED_TYPE == "x0":
            dp = raw
        elif _PRED_TYPE == "eps":
            dp = (xt - _model.sqrt_1mab[tv] * raw) / _model.sqrt_ab[tv].clamp(min=1e-8)
        else:
            dp = _model.sqrt_ab[tv] * xt - _model.sqrt_1mab[tv] * raw

        frac = 1.0 - tv / _model.n_steps
        if frac > 0.3:
            conf = torch.abs(sl)
            thresh = max(0.5, 3.0 * (1.0 - frac))
            reveal = (conf > thresh) & (mflag > 0.5)
            stall_s = torch.where(reveal, (torch.sigmoid(sl) > 0.5).float(), stall_s)
            mflag = torch.where(reveal, torch.zeros_like(mflag), mflag)

        if frac > 0.3 and guide > 0:
            with torch.no_grad():
                dp = dp.clone()
                for b in range(B):
                    spd = torch.clamp(dp[b, :, 0] / spd_s, min=0)
                    dh = dp[b, :, 1] / dh_s
                    active_stall = (stall_s[b] > 0.5) & (mflag[b] < 0.5)
                    spd_eff = spd * (~active_stall).float()
                    dh_eff = dh * (~active_stall).float()
                    heading = torch.cumsum(dh_eff, dim=0)
                    cx = torch.cumsum(spd_eff * torch.cos(heading), dim=0)
                    cy = torch.cumsum(spd_eff * torch.sin(heading), dim=0)
                    raw_mag = math.hypot(cx[-1].item(), cy[-1].item())
                    if raw_mag > 1e-6:
                        raw_ang = math.atan2(cy[-1].item(), cx[-1].item())
                        tgt_ang = float(tgt_angles[b]) if tgt_angles.size > 1 else float(tgt_angles[0])
                        rot = (tgt_ang - raw_ang) * guide * frac
                        dp[b, 0, 1] += rot * dh_s

        if frac > 0.5 and _FEAT_GUIDE > 0:
            with torch.enable_grad():
                dp_g = dp.clone().detach().requires_grad_(True)
                active = (~((stall_s[0] > 0.5) & (mflag[0] < 0.5))).float()
                spd_g = torch.clamp(dp_g[0, :, 0] / spd_s, min=0) * active
                dh_g = dp_g[0, :, 1] / dh_s * active
                heading_g = torch.cumsum(dh_g, dim=0)
                vx_g = spd_g * torch.cos(heading_g)
                vy_g = spd_g * torch.sin(heading_g)
                cx_g = torch.cumsum(vx_g, dim=0)
                cy_g = torch.cumsum(vy_g, dim=0)
                endpoint = torch.sqrt(cx_g[-1] ** 2 + cy_g[-1] ** 2 + 1e-8)
                total_path = torch.sum(spd_g) / _HZ
                eff = endpoint / (total_path + 1e-8)
                loss = (eff - _FEAT_EFF_TARGET) ** 2
                loss.backward()
            with torch.no_grad():
                grad = dp_g.grad
                grad_norm = grad.norm()
                if grad_norm > 1.0:
                    grad = grad / grad_norm
                dp = dp - _FEAT_GUIDE * frac * grad

        if tv > 0:
            nt = times[i + 1] if i + 1 < len(times) else 0
            ab_t = _model.alpha_bar[tv]
            ab_n = _model.alpha_bar[nt] if nt > 0 else torch.ones(1, device=dev)
            eps = (xt - ab_t.sqrt() * dp) / _model.sqrt_1mab[tv].clamp(min=1e-8)
            if eta > 0 and nt > 0:
                sig = eta * ((1 - ab_n) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_n).sqrt()
                noise = torch.randn_like(xt) * sig
            else:
                sig = torch.zeros(1, device=dev)
                noise = 0.0
            dir_coef = (1 - ab_n - sig ** 2).clamp(min=0).sqrt()
            xt = ab_n.sqrt() * dp + dir_coef * eps + noise
        else:
            xt = dp

    sp = torch.sigmoid(sl)
    final_stall = torch.where(mflag > 0.5, (sp > 0.5).float(), stall_s)
    out = xt.clone()
    out[final_stall > 0.5] = 0.0
    return out, final_stall


def _sample_guided_flow(cond, seq_len, target_cos, target_sin,
                        n_steps, cfg_scale, guide):
    B = cond.shape[0]
    dev = cond.device
    spd_s = float(_data_scale[0])
    dh_s = float(_data_scale[1])
    tgt_angles = np.atleast_1d(np.arctan2(target_sin, target_cos)).astype(np.float64)

    xt = torch.randn(B, seq_len, 2, device=dev)
    stall_s = torch.full((B, seq_len), _model.STALL_MASK, device=dev)
    mflag = torch.ones(B, seq_len, device=dev)

    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_cont = 1.0 - i * dt
        t_scaled = torch.full((B,), t_cont * (_model.n_steps - 1), device=dev)
        v_pred, sl = _model(xt, stall_s, mflag, t_scaled, cond)

        if cfg_scale > 0:
            v_u, sl_u = _model(xt, stall_s, mflag, t_scaled, torch.zeros_like(cond))
            v_pred = v_u + cfg_scale * (v_pred - v_u)
            sl = sl_u + cfg_scale * (sl - sl_u)

        dp = xt - t_cont * v_pred

        frac = 1.0 - t_cont
        if frac > 0.3:
            conf = torch.abs(sl)
            thresh = max(0.5, 3.0 * (1.0 - frac))
            reveal = (conf > thresh) & (mflag > 0.5)
            stall_s = torch.where(reveal, (torch.sigmoid(sl) > 0.5).float(), stall_s)
            mflag = torch.where(reveal, torch.zeros_like(mflag), mflag)

        if frac > 0.3 and guide > 0:
            dp = dp.clone()
            for b in range(B):
                spd = torch.clamp(dp[b, :, 0] / spd_s, min=0)
                dh = dp[b, :, 1] / dh_s
                active_stall = (stall_s[b] > 0.5) & (mflag[b] < 0.5)
                spd_eff = spd * (~active_stall).float()
                dh_eff = dh * (~active_stall).float()
                heading = torch.cumsum(dh_eff, dim=0)
                cx = torch.cumsum(spd_eff * torch.cos(heading), dim=0)
                cy = torch.cumsum(spd_eff * torch.sin(heading), dim=0)
                raw_mag = math.hypot(cx[-1].item(), cy[-1].item())
                if raw_mag > 1e-6:
                    raw_ang = math.atan2(cy[-1].item(), cx[-1].item())
                    tgt_ang = float(tgt_angles[b]) if tgt_angles.size > 1 else float(tgt_angles[0])
                    rot = (tgt_ang - raw_ang) * guide * frac
                    dp[b, 0, 1] += rot * dh_s

        if frac > 0.5 and _FEAT_GUIDE > 0:
            with torch.enable_grad():
                dp_g = dp.clone().detach().requires_grad_(True)
                active = (~((stall_s[0] > 0.5) & (mflag[0] < 0.5))).float()
                spd_g = torch.clamp(dp_g[0, :, 0] / spd_s, min=0) * active
                dh_g = dp_g[0, :, 1] / dh_s * active
                heading_g = torch.cumsum(dh_g, dim=0)
                vx_g = spd_g * torch.cos(heading_g)
                vy_g = spd_g * torch.sin(heading_g)
                cx_g = torch.cumsum(vx_g, dim=0)
                cy_g = torch.cumsum(vy_g, dim=0)
                endpoint = torch.sqrt(cx_g[-1] ** 2 + cy_g[-1] ** 2 + 1e-8)
                total_path = torch.sum(spd_g) / _HZ
                eff = endpoint / (total_path + 1e-8)
                loss = (eff - _FEAT_EFF_TARGET) ** 2
                loss.backward()
            with torch.no_grad():
                grad = dp_g.grad
                grad_norm = grad.norm()
                if grad_norm > 1.0:
                    grad = grad / grad_norm
                dp = dp - _FEAT_GUIDE * frac * grad

        if t_cont > 1e-6:
            v_guided = (xt - dp) / t_cont
        else:
            v_guided = v_pred
        xt = xt - dt * v_guided
        if _FLOW_NOISE > 0 and t_cont > 0.1:
            xt = xt + _FLOW_NOISE * math.sqrt(dt) * t_cont * torch.randn_like(xt)

    sp = torch.sigmoid(sl)
    final_stall = torch.where(mflag > 0.5, (sp > 0.5).float(), stall_s)
    out = xt.clone()
    out[final_stall > 0.5] = 0.0
    return out, final_stall


def _generate_single(cond, seq_len, total_dist, dx, dy,
                     start_x, start_y, end_x, end_y):
    global _current_log_dist
    _current_log_dist = math.log(max(total_dist, 1.0))
    decode = _decode_polar if _POLAR else _decode_cartesian
    angle = math.atan2(dy, dx)
    with torch.no_grad():
        if _PRED_TYPE == "flow":
            if _GUIDE > 0 and _POLAR:
                raw, stall = _sample_guided_flow(
                    cond, seq_len,
                    math.cos(angle), math.sin(angle),
                    n_steps=_N_SAMPLE_STEPS, cfg_scale=_CFG,
                    guide=_GUIDE,
                )
            else:
                raw, stall = _model.flow_sample(
                    cond, seq_len,
                    n_steps=_N_SAMPLE_STEPS, cfg_scale=_CFG,
                )
        elif _GUIDE > 0 and _POLAR:
            raw, stall = _sample_guided_polar(
                cond, seq_len,
                math.cos(angle), math.sin(angle),
                n_steps=_N_SAMPLE_STEPS, eta=_ETA, cfg_scale=_CFG,
                guide=_GUIDE,
            )
        else:
            raw, stall = _model.sample(
                cond, seq_len,
                n_steps=_N_SAMPLE_STEPS, eta=_ETA, cfg_scale=_CFG,
                pred_type=_PRED_TYPE,
            )
    raw_np = raw[0].cpu().numpy()
    stall_np = stall[0].cpu().numpy()
    cum_x, cum_y = decode(raw_np, stall_np)
    return _build_trajectory(cum_x, cum_y, stall_np, seq_len, total_dist, dx, dy,
                             start_x, start_y, end_x, end_y)


def generate_path(
    start_x: float, start_y: float,
    end_x: float, end_y: float,
) -> Trajectory:
    dx = end_x - start_x
    dy = end_y - start_y
    total_dist = math.hypot(dx, dy)

    if total_dist < 1.0:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    log_dist = math.log(total_dist)
    angle = math.atan2(dy, dx)
    duration = _duration.sample(log_dist)
    log_dur = math.log(duration)
    seq_len = max(5, min(int(round(duration * _HZ)), _cfg["max_seq_len"]))

    cond = torch.tensor(
        [[log_dist, log_dur, math.cos(angle), math.sin(angle)]],
        dtype=torch.float32, device=_DEVICE,
    )

    if _N_CANDIDATES <= 1:
        return _generate_single(cond, seq_len, total_dist, dx, dy,
                                start_x, start_y, end_x, end_y)

    global _score_target, _current_log_dist
    _score_target = _get_target_features()
    _current_log_dist = math.log(max(total_dist, 1.0))
    K = _N_CANDIDATES
    cond_batch = cond.repeat(K, 1)
    decode = _decode_polar if _POLAR else _decode_cartesian
    angle = math.atan2(dy, dx)
    with torch.no_grad():
        if _PRED_TYPE == "flow":
            if _GUIDE > 0 and _POLAR:
                raw_batch, stall_batch = _sample_guided_flow(
                    cond_batch, seq_len,
                    math.cos(angle), math.sin(angle),
                    n_steps=_N_SAMPLE_STEPS, cfg_scale=_CFG,
                    guide=_GUIDE,
                )
            else:
                raw_batch, stall_batch = _model.flow_sample(
                    cond_batch, seq_len,
                    n_steps=_N_SAMPLE_STEPS, cfg_scale=_CFG,
                )
        elif _GUIDE > 0 and _POLAR:
            raw_batch, stall_batch = _sample_guided_polar(
                cond_batch, seq_len,
                math.cos(angle), math.sin(angle),
                n_steps=_N_SAMPLE_STEPS, eta=_ETA, cfg_scale=_CFG,
                guide=_GUIDE,
            )
        else:
            raw_batch, stall_batch = _model.sample(
                cond_batch, seq_len,
                n_steps=_N_SAMPLE_STEPS, eta=_ETA, cfg_scale=_CFG,
                pred_type=_PRED_TYPE,
            )
    best_traj = None
    best_score = float("-inf")
    for k in range(K):
        raw_np = raw_batch[k].cpu().numpy()
        stall_np = stall_batch[k].cpu().numpy()
        cum_x, cum_y = decode(raw_np, stall_np)
        traj = _build_trajectory(cum_x, cum_y, stall_np, seq_len, total_dist,
                                 dx, dy, start_x, start_y, end_x, end_y)
        score = _score_trajectory(traj)
        if score > best_score:
            best_score = score
            best_traj = traj
    return best_traj


_EVAL_BATCH = int(os.environ.get("CANDI_EVAL_BATCH", "128"))


def generate_paths(specs: list) -> list:
    """Batched generate_path. specs is a list of (sx, sy, ex, ey) tuples.

    Groups requests by seq_len so each group runs through the sampler as one
    batch, then applies the identical per-trajectory decode and build steps.
    Falls back to sequential generation for modes that assume batch size 1.
    """
    if _N_CANDIDATES > 1 or _FEAT_GUIDE > 0:
        return [generate_path(sx, sy, ex, ey) for sx, sy, ex, ey in specs]

    results: list = [None] * len(specs)
    pending = []
    for idx, (sx, sy, ex, ey) in enumerate(specs):
        dx = ex - sx
        dy = ey - sy
        total_dist = math.hypot(dx, dy)
        if total_dist < 1.0:
            results[idx] = [(sx, sy, 0.0), (ex, ey, 0.008)]
            continue
        log_dist = math.log(total_dist)
        angle = math.atan2(dy, dx)
        duration = _duration.sample(log_dist)
        log_dur = math.log(duration)
        seq_len = max(5, min(int(round(duration * _HZ)), _cfg["max_seq_len"]))
        pending.append({
            "idx": idx, "seq_len": seq_len, "angle": angle,
            "cond": [log_dist, log_dur, math.cos(angle), math.sin(angle)],
            "total_dist": total_dist, "dx": dx, "dy": dy,
            "sx": sx, "sy": sy, "ex": ex, "ey": ey,
        })

    groups: dict = {}
    for item in pending:
        groups.setdefault(item["seq_len"], []).append(item)

    global _current_log_dist
    decode = _decode_polar if _POLAR else _decode_cartesian
    for seq_len, items in groups.items():
        for c0 in range(0, len(items), _EVAL_BATCH):
            chunk = items[c0:c0 + _EVAL_BATCH]
            cond = torch.tensor([it["cond"] for it in chunk],
                                dtype=torch.float32, device=_DEVICE)
            tcos = np.array([math.cos(it["angle"]) for it in chunk])
            tsin = np.array([math.sin(it["angle"]) for it in chunk])
            with torch.no_grad():
                if _PRED_TYPE == "flow":
                    if _GUIDE > 0 and _POLAR:
                        raw, stall = _sample_guided_flow(
                            cond, seq_len, tcos, tsin,
                            n_steps=_N_SAMPLE_STEPS, cfg_scale=_CFG,
                            guide=_GUIDE,
                        )
                    else:
                        raw, stall = _model.flow_sample(
                            cond, seq_len,
                            n_steps=_N_SAMPLE_STEPS, cfg_scale=_CFG,
                        )
                elif _GUIDE > 0 and _POLAR:
                    raw, stall = _sample_guided_polar(
                        cond, seq_len, tcos, tsin,
                        n_steps=_N_SAMPLE_STEPS, eta=_ETA, cfg_scale=_CFG,
                        guide=_GUIDE,
                    )
                else:
                    raw, stall = _model.sample(
                        cond, seq_len,
                        n_steps=_N_SAMPLE_STEPS, eta=_ETA, cfg_scale=_CFG,
                        pred_type=_PRED_TYPE,
                    )
            raw_all = raw.cpu().numpy()
            stall_all = stall.cpu().numpy()
            for b, it in enumerate(chunk):
                _current_log_dist = math.log(max(it["total_dist"], 1.0))
                cum_x, cum_y = decode(raw_all[b], stall_all[b])
                results[it["idx"]] = _build_trajectory(
                    cum_x, cum_y, stall_all[b], seq_len,
                    it["total_dist"], it["dx"], it["dy"],
                    it["sx"], it["sy"], it["ex"], it["ey"],
                )
    return results
