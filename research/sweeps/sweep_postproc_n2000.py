"""Coordinate re-sweep of post-processing knobs at n=2000.

The current knobs (skew 0.3, perp 0.7, guide 0.3) were tuned for an earlier
checkpoint at N=100, which is unreliable. Every point here is a full n=2000
eval (about 4 minutes with batched generation). Coordinate descent: skew,
then perp, then guide, carrying the best value forward.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

BASE = {
    "CANDI_CKPT": "candi_polar_flow_best.pt",
    "CANDI_STEPS": "200",
    "CANDI_CFG": "0",
    "CANDI_CORRECT": "rotate",
}
results = []


def run_eval(skew: float, perp: float, guide: float) -> float:
    env = dict(os.environ)
    env.update(BASE)
    env["CANDI_SPEED_SKEW"] = str(skew)
    env["CANDI_PERP_SCALE"] = str(perp)
    env["CANDI_GUIDE"] = str(guide)
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, "evaluate.py", "--experiment", "experiments.candi",
         "--n-synthetic", "2000", "--no-raw-nn"],
        env=env, capture_output=True, text=True,
    ).stdout
    m = re.search(r"RF OOB AUC:\s+([\d.]+)", out)
    g = re.search(r"GBM 5-fold CV:\s+([\d.]+)", out)
    auc = float(m.group(1)) if m else float("nan")
    gbm = float(g.group(1)) if g else float("nan")
    rec = {"skew": skew, "perp": perp, "guide": guide,
           "rf_oob": auc, "gbm_cv": gbm, "secs": round(time.time() - t0)}
    results.append(rec)
    print(json.dumps(rec), flush=True)
    with open("sweep_postproc_n2000_results.json", "w") as f:
        json.dump(results, f, indent=1)
    return auc


BASELINE = 0.829  # skew 0.3, perp 0.7, guide 0.3 (3-seed mean, already measured)

scores = {(0.3, 0.7, 0.3): BASELINE}

best_skew, best_auc = 0.3, BASELINE
for s in (0.0, 0.1, 0.2):
    a = run_eval(s, 0.7, 0.3)
    scores[(s, 0.7, 0.3)] = a
    if a < best_auc:
        best_skew, best_auc = s, a

run_eval(0.0, 1.0, 0.0)  # bare model reference, not part of descent

best_perp = 0.7
for p in (0.85, 1.0):
    a = run_eval(best_skew, p, 0.3)
    scores[(best_skew, p, 0.3)] = a
    if a < best_auc:
        best_perp, best_auc = p, a

best_guide = 0.3
for g in (0.0, 0.15):
    a = run_eval(best_skew, best_perp, g)
    if a < best_auc:
        best_guide, best_auc = g, a

print(f"\nbest: skew={best_skew} perp={best_perp} guide={best_guide} rf_oob={best_auc:.4f}")
print(f"baseline was {BASELINE}")
