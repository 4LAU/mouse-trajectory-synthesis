"""Prepare chunk-level training data from ZIMT arrays.

Extracts overlapping 25-step chunks from trajectories for chunk-level
diffusion training. Each chunk has context from the previous chunk's tail,
global trajectory conditioning, and local progress conditioning.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

CHUNK_SIZE = 25
CONTEXT_SIZE = 5
STRIDE = 20

DATA_DIR = Path("training")


def prepare_chunks():
    print("Loading ZIMT arrays...")
    dxdy = np.load(DATA_DIR / "zimt_dxdy.npy")       # (N, 256, 2)
    stall = np.load(DATA_DIR / "zimt_stall.npy")      # (N, 256)
    lengths = np.load(DATA_DIR / "zimt_lengths.npy")   # (N,)
    conditions = np.load(DATA_DIR / "zimt_conditions.npy")  # (N, 4)
    endpoints = np.load(DATA_DIR / "zimt_endpoints.npy")    # (N, 4)

    N = len(lengths)
    print(f"  {N:,} trajectories, mean length {lengths.mean():.1f}")

    chunk_dxdy_list = []
    chunk_stall_list = []
    context_dxdy_list = []
    context_stall_list = []
    global_cond_list = []
    local_cond_list = []
    chunk_len_list = []

    for i in range(N):
        L = int(lengths[i])
        if L < 3:
            continue

        traj_dxdy = dxdy[i, :L]    # (L, 2)
        traj_stall = stall[i, :L]  # (L,)
        cond = conditions[i]       # (4,): log_dist, log_dur, cos_a, sin_a
        ep = endpoints[i]          # (4,): start_x, start_y, end_x, end_y

        cos_a, sin_a = cond[2], cond[3]

        n_chunks = max(1, int(np.ceil((L - CONTEXT_SIZE) / STRIDE)))
        if L <= CHUNK_SIZE:
            n_chunks = 1

        for k in range(n_chunks):
            chunk_start = k * STRIDE
            chunk_end = min(chunk_start + CHUNK_SIZE, L)
            valid_len = chunk_end - chunk_start

            c_dxdy = np.zeros((CHUNK_SIZE, 2), dtype=np.float32)
            c_stall = np.zeros(CHUNK_SIZE, dtype=np.float32)
            c_dxdy[:valid_len] = traj_dxdy[chunk_start:chunk_end]
            c_stall[:valid_len] = traj_stall[chunk_start:chunk_end]

            if k == 0:
                ctx_dxdy = np.zeros((CONTEXT_SIZE, 2), dtype=np.float32)
                ctx_stall = np.zeros(CONTEXT_SIZE, dtype=np.float32)
            else:
                ctx_start = max(0, chunk_start - CONTEXT_SIZE)
                ctx_len = chunk_start - ctx_start
                ctx_dxdy = np.zeros((CONTEXT_SIZE, 2), dtype=np.float32)
                ctx_stall = np.zeros(CONTEXT_SIZE, dtype=np.float32)
                ctx_dxdy[CONTEXT_SIZE - ctx_len:] = traj_dxdy[ctx_start:chunk_start]
                ctx_stall[CONTEXT_SIZE - ctx_len:] = traj_stall[ctx_start:chunk_start]

            cum_dx = float(traj_dxdy[:chunk_start, 0].sum()) if chunk_start > 0 else 0.0
            cum_dy = float(traj_dxdy[:chunk_start, 1].sum()) if chunk_start > 0 else 0.0
            rem_dx = cos_a - cum_dx
            rem_dy = sin_a - cum_dy
            rem_frac = 1.0 - chunk_start / L
            progress = k / max(n_chunks - 1, 1)

            local = np.array([rem_dx, rem_dy, rem_frac, progress, cum_dx, cum_dy],
                             dtype=np.float32)

            chunk_dxdy_list.append(c_dxdy)
            chunk_stall_list.append(c_stall)
            context_dxdy_list.append(ctx_dxdy)
            context_stall_list.append(ctx_stall)
            global_cond_list.append(cond.astype(np.float32))
            local_cond_list.append(local)
            chunk_len_list.append(valid_len)

        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1:,}/{N:,} trajectories")

    print(f"  Total chunks: {len(chunk_dxdy_list):,}")

    chunk_dxdy = np.array(chunk_dxdy_list, dtype=np.float32)
    chunk_stall = np.array(chunk_stall_list, dtype=np.float32)
    context_dxdy = np.array(context_dxdy_list, dtype=np.float32)
    context_stall = np.array(context_stall_list, dtype=np.float32)
    global_cond = np.array(global_cond_list, dtype=np.float32)
    local_cond = np.array(local_cond_list, dtype=np.float32)
    chunk_lengths = np.array(chunk_len_list, dtype=np.int32)

    print("Saving...")
    np.save(DATA_DIR / "chunk_dxdy.npy", chunk_dxdy)
    np.save(DATA_DIR / "chunk_stall.npy", chunk_stall)
    np.save(DATA_DIR / "chunk_context_dxdy.npy", context_dxdy)
    np.save(DATA_DIR / "chunk_context_stall.npy", context_stall)
    np.save(DATA_DIR / "chunk_global_cond.npy", global_cond)
    np.save(DATA_DIR / "chunk_local_cond.npy", local_cond)
    np.save(DATA_DIR / "chunk_lengths.npy", chunk_lengths)

    print(f"Done. {len(chunk_dxdy):,} chunks saved.")
    print(f"  Chunk dxdy shape: {chunk_dxdy.shape}")
    print(f"  Mean valid length: {chunk_lengths.mean():.1f}")
    print(f"  Stall rate: {chunk_stall[chunk_lengths > 0].mean():.4f}")


if __name__ == "__main__":
    prepare_chunks()
