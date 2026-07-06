"""Raw-trajectory neural detector, a held-out adversary for evaluation.

Trains a small 1D CNN to distinguish human from synthetic trajectories using
the raw resampled (dx, dy) sequences instead of the 18 engineered features.
Reported alongside RF and GBM scores but never tuned against.

Human side: the held-out test split (training/test_positions.npy), which no
generative model trains on. Synthetic side: whatever trajectories the
experiment produced during evaluation.

Both classes go through the same pipeline: 125Hz resample (identical to the
feature extractor), displacement diffs, normalization by straight-line
distance. Sequence length information is kept via a mask channel, since a
real duration gap is a legitimate signal.

AUC is computed on pooled out-of-fold predictions from 3-fold CV with fixed
seeds and fixed hyperparameters.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import resample_trajectory

MAX_LEN = 192
SEED = 42
EPOCHS = 20
BATCH = 256
LR = 1e-3


def _traj_to_input(traj) -> np.ndarray | None:
    """Trajectory [(x, y, t), ...] to a (3, MAX_LEN) array: dx, dy, mask."""
    pts = np.asarray(resample_trajectory(traj), dtype=np.float64)
    if len(pts) < 5:
        return None
    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])
    straight = np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
    if straight < 1e-6:
        return None
    dx = dx / straight
    dy = dy / straight
    T = min(len(dx), MAX_LEN)
    out = np.zeros((3, MAX_LEN), dtype=np.float32)
    out[0, :T] = dx[:T]
    out[1, :T] = dy[:T]
    out[2, :T] = 1.0
    return out


def load_human_raw(n: int, train_dir: str = "./training", seed: int = SEED) -> list:
    """Sample n held-out human trajectories in pixel scale."""
    positions = np.load(f"{train_dir}/test_positions.npy", mmap_mode="r")
    timestamps = np.load(f"{train_dir}/test_timestamps.npy", mmap_mode="r")
    n_real = np.load(f"{train_dir}/test_n_real.npy")
    conditions = np.load(f"{train_dir}/test_conditions.npy", mmap_mode="r")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(n_real), size=min(n, len(n_real)), replace=False)

    trajs = []
    for i in idx:
        ni = int(n_real[i])
        if ni < 5:
            continue
        scale = float(np.exp(conditions[i, 0]))
        xy = np.asarray(positions[i, :ni], dtype=np.float64) * scale
        t = np.asarray(timestamps[i, :ni], dtype=np.float64)
        trajs.append(list(zip(xy[:, 0].tolist(), xy[:, 1].tolist(), t.tolist())))
    return trajs


class RawTrajCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.net(x)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(pooled).squeeze(-1)


def _train_fold(X_tr, y_tr, X_te, device, seed):
    torch.manual_seed(seed)
    model = RawTrajCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    n = len(X_tr_t)

    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for b0 in range(0, n, BATCH):
            bi = perm[b0:b0 + BATCH]
            opt.zero_grad()
            out = model(X_tr_t[bi])
            loss = lossf(out, y_tr_t[bi])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(X_te).to(device)
        preds = []
        for b0 in range(0, len(X_te_t), BATCH):
            preds.append(torch.sigmoid(model(X_te_t[b0:b0 + BATCH])).cpu().numpy())
    return np.concatenate(preds)


def raw_nn_auc(synth_trajectories: list, train_dir: str = "./training",
               seed: int = SEED) -> float | None:
    """Held-out CV AUC of the raw-trajectory CNN. Human label 0, synthetic 1."""
    synth_inputs = [x for x in (_traj_to_input(t) for t in synth_trajectories)
                    if x is not None]
    if len(synth_inputs) < 100:
        return None

    human_trajs = load_human_raw(len(synth_inputs), train_dir=train_dir, seed=seed)
    human_inputs = [x for x in (_traj_to_input(t) for t in human_trajs)
                    if x is not None]

    n_use = min(len(human_inputs), len(synth_inputs))
    X = np.stack(human_inputs[:n_use] + synth_inputs[:n_use])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        oof[te] = _train_fold(X[tr], y[tr], X[te], device, seed + fold)
    return float(roc_auc_score(y, oof))
