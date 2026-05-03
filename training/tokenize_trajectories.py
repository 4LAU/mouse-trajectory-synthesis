"""
Tokenize trajectories using the trained VQ-VAE.

For each trajectory:
1. Resample to 125Hz
2. Compute (dx, dy) per step
3. Stall steps (speed < 1 px/s) -> token 0
4. Non-stall steps -> VQ-VAE encode -> token 1-1024

Output:
  - vqvae_token_seqs.npy: (N_traj, max_len) int16 token sequences (padded)
  - vqvae_seq_lens.npy: (N_traj,) actual sequence lengths
  - vqvae_seq_conditions.npy: (N_traj, 4) conditions
  - vqvae_seq_endpoints.npy: (N_traj, 4) (start_x, start_y, end_x, end_y) normalized
      Coordinates are in the normalized space produced by prepare_training_data.py
      (origin-translated then divided by total Euclidean distance). Normalization
      happens at source in train_positions.npy - do NOT re-normalize here.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from models.vqvae import MotionVQVAE

GEN_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

HZ = 125.0
DT = 1.0 / HZ
MAX_SEQ_LEN = 256  # transformer context window


def main():
    print("Loading VQ-VAE model...")
    ckpt = torch.load(GEN_DIR / "vqvae_v2_best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    norm_mean = np.array(ckpt["norm_mean"], dtype=np.float32)
    norm_std = np.array(ckpt["norm_std"], dtype=np.float32)
    clip_lo = np.array(ckpt["clip_lo"], dtype=np.float32)
    clip_hi = np.array(ckpt["clip_hi"], dtype=np.float32)

    model = MotionVQVAE(n_codes=cfg["n_codes"], code_dim=cfg["code_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    print(f"  Loaded epoch {ckpt['epoch']}, codebook usage {ckpt['codebook_usage']}")

    # Load trajectory data
    print("Loading trajectory data...")
    positions = np.load(GEN_DIR / "train_positions.npy", mmap_mode="r")
    timestamps = np.load(GEN_DIR / "train_timestamps.npy", mmap_mode="r")
    n_real = np.load(GEN_DIR / "train_n_real.npy")
    conditions = np.load(GEN_DIR / "train_conditions.npy", mmap_mode="r")

    # Process trajectories
    max_traj = min(500_000, len(n_real))
    print(f"Tokenizing {max_traj} trajectories...")
    t0 = time.time()

    all_tokens = []
    all_lengths = []
    all_conditions = []
    all_endpoints = []
    n_valid = 0
    n_stall_tokens = 0
    n_total_tokens = 0

    for i in range(max_traj):
        if i % 50000 == 0 and i > 0:
            print(f"  {i}/{max_traj} ({n_valid} valid, "
                  f"{n_stall_tokens}/{n_total_tokens} stalls = "
                  f"{100*n_stall_tokens/max(n_total_tokens,1):.1f}%)...")

        n = int(n_real[i])
        if n < 3:
            continue

        x = positions[i, :n, 0].astype(np.float64)
        y = positions[i, :n, 1].astype(np.float64)
        t = timestamps[i, :n].astype(np.float64)

        t = np.maximum.accumulate(t)
        duration = t[-1] - t[0]
        if duration <= 0:
            continue

        # Resample to 125Hz
        n_out = max(3, int(round(duration * HZ)))
        if n_out > MAX_SEQ_LEN + 1:
            n_out = MAX_SEQ_LEN + 1  # limit length
        t_out = np.arange(n_out) * DT
        if t_out[-1] > t[-1]:
            t_out = t_out[t_out <= t[-1]]
            n_out = len(t_out)
        if n_out < 3:
            continue

        x_out = np.interp(t_out, t, x)
        y_out = np.interp(t_out, t, y)
        endpoint = np.array([x_out[0], y_out[0], x_out[-1], y_out[-1]], dtype=np.float32)

        # Compute displacements
        dx = np.diff(x_out).astype(np.float32)
        dy = np.diff(y_out).astype(np.float32)
        n_steps = len(dx)

        if n_steps < 2 or n_steps > MAX_SEQ_LEN:
            continue

        # Classify: stall vs motion
        speed = np.sqrt(dx**2 + dy**2) / DT
        stall_mask = speed < 1.0

        # Tokenize non-stall steps with VQ-VAE
        tokens = np.zeros(n_steps, dtype=np.int16)

        # Stall tokens = 0
        tokens[stall_mask] = 0
        n_stall_tokens += int(stall_mask.sum())

        # Motion tokens: encode with VQ-VAE
        motion_mask = ~stall_mask
        if motion_mask.sum() > 0:
            dxdy_motion = np.stack([dx[motion_mask], dy[motion_mask]], axis=-1)
            # Normalize
            dxdy_clipped = np.clip(dxdy_motion, clip_lo, clip_hi)
            dxdy_normed = (dxdy_clipped - norm_mean) / norm_std

            with torch.no_grad():
                dxdy_tensor = torch.tensor(dxdy_normed, dtype=torch.float32, device=DEVICE)
                # Process in chunks for memory
                chunk_size = 10000
                all_indices = []
                for j in range(0, len(dxdy_tensor), chunk_size):
                    chunk = dxdy_tensor[j:j+chunk_size]
                    indices = model.encode(chunk)
                    all_indices.append(indices.cpu().numpy())
                motion_indices = np.concatenate(all_indices)

            # VQ-VAE tokens are 0-1023, shift to 1-1024 (0 is reserved for stall)
            tokens[motion_mask] = motion_indices.astype(np.int16) + 1

        # Pad to MAX_SEQ_LEN
        padded = np.full(MAX_SEQ_LEN, -1, dtype=np.int16)  # -1 = padding
        padded[:n_steps] = tokens

        all_tokens.append(padded)
        all_lengths.append(n_steps)
        all_conditions.append(conditions[i])
        all_endpoints.append(endpoint)
        n_total_tokens += n_steps
        n_valid += 1

    print(f"\nResults:")
    print(f"  {n_valid} valid trajectories")
    print(f"  {n_total_tokens} total tokens")
    print(f"  {n_stall_tokens} stall tokens ({100*n_stall_tokens/n_total_tokens:.2f}%)")
    print(f"  Mean seq length: {n_total_tokens/n_valid:.1f}")

    # Token distribution
    all_toks_flat = np.concatenate([t[:l] for t, l in zip(all_tokens, all_lengths)])
    unique_tokens = len(np.unique(all_toks_flat))
    print(f"  Unique tokens used: {unique_tokens}")

    # Save
    tokens_arr = np.array(all_tokens, dtype=np.int16)
    lengths_arr = np.array(all_lengths, dtype=np.int32)
    conditions_arr = np.array(all_conditions, dtype=np.float32)
    endpoints_arr = np.array(all_endpoints, dtype=np.float32)

    np.save(GEN_DIR / "vqvae_token_seqs.npy", tokens_arr)
    np.save(GEN_DIR / "vqvae_seq_lens.npy", lengths_arr)
    np.save(GEN_DIR / "vqvae_seq_conditions.npy", conditions_arr)
    np.save(GEN_DIR / "vqvae_seq_endpoints.npy", endpoints_arr)

    print(f"\nSaved: tokens={tokens_arr.shape}, lens={lengths_arr.shape}, cond={conditions_arr.shape}, endpoints={endpoints_arr.shape}")
    print(f"Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
