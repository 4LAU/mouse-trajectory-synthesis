"""Stronger raw-sequence adversaries for the final selected set.

The eval suite's Raw-NN is a 3-layer CNN trained 20 epochs. Before calling
the raw-sequence channel closed, push harder with two bigger held-out
adversaries on the identical inputs (125 Hz resampled dx, dy, mask,
normalized by straight-line distance, same held-out human split):

  dilated   5-block dilated CNN (receptive field spans the whole sequence),
            40 epochs, cosine LR
  gru       2-layer bidirectional GRU on the same channels, 30 epochs

Same honesty protocol as detector_raw.py: pooled out-of-fold predictions
from 3-fold CV, fixed seeds, no tuning against the result.

Run:
    .venv/Scripts/python.exe detector_raw_strong.py \
        --pool pool_s42_k16.npz \
        --picks pool_s42_k16_picks_trust33_f20d85_r30_rf.npy
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from detector_raw import _traj_to_input, load_human_raw

SEED = 42
BATCH = 256


class DilatedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 48
        blocks = []
        in_ch = 3
        for d in (1, 2, 4, 8, 16):
            blocks += [nn.Conv1d(in_ch, ch, 5, padding=2 * d, dilation=d),
                       nn.GELU()]
            in_ch = ch
        self.net = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(2 * ch, 64), nn.GELU(),
                                  nn.Linear(64, 1))

    def forward(self, x):
        h = self.net(x)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(pooled).squeeze(-1)


class BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(3, 64, num_layers=2, batch_first=True,
                          bidirectional=True)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(),
                                  nn.Linear(64, 1))

    def forward(self, x):
        h, _ = self.gru(x.transpose(1, 2))
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1)
        return self.head(pooled).squeeze(-1)


def train_fold(model_fn, X_tr, y_tr, X_te, epochs, seed):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = model_fn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
    n = len(X_tr_t)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for b0 in range(0, n, BATCH):
            bi = perm[b0:b0 + BATCH]
            opt.zero_grad()
            loss = lossf(model(X_tr_t[bi]), y_tr_t[bi])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(X_te)
        preds = [torch.sigmoid(model(X_te_t[b0:b0 + BATCH])).numpy()
                 for b0 in range(0, len(X_te_t), BATCH)]
    return np.concatenate(preds)


def cv_auc(model_fn, X, y, epochs, name):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        t0 = time.time()
        oof[te] = train_fold(model_fn, X[tr], y[tr], X[te], epochs,
                             SEED + fold)
        print(f"  {name} fold {fold + 1}/3 done ({time.time() - t0:.0f}s)",
              flush=True)
    return float(roc_auc_score(y, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool_s42_k16.npz")
    ap.add_argument("--picks", required=True)
    args = ap.parse_args()

    d = np.load(args.pool, allow_pickle=True)
    picks = np.load(args.picks).astype(int)
    gen = [np.asarray(d["trajs"][ci], dtype=np.float64)
           for ci in picks if ci >= 0]
    synth = [x for x in (_traj_to_input(t) for t in gen) if x is not None]
    human = [x for x in (_traj_to_input(t)
                         for t in load_human_raw(len(synth)))
             if x is not None]
    n = min(len(human), len(synth))
    X = np.stack(human[:n] + synth[:n])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    print(f"n={n} per class")

    for name, fn, epochs in [("dilated-cnn", DilatedCNN, 40),
                             ("bi-gru", BiGRU, 30)]:
        auc = cv_auc(fn, X, y, epochs, name)
        print(f"{name}: held-out 3-fold AUC {auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
