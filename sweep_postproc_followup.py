"""Follow-up to sweep_postproc_n2000: fill two missing points, then confirm
the overall winner with two extra seeds and the raw-NN detector enabled.
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


def run_eval(skew, perp, guide, seed=None, raw_nn=False):
    env = dict(os.environ)
    env.update(BASE)
    env["CANDI_SPEED_SKEW"] = str(skew)
    env["CANDI_PERP_SCALE"] = str(perp)
    env["CANDI_GUIDE"] = str(guide)
    cmd = [sys.executable, "evaluate.py", "--experiment", "experiments.candi",
           "--n-synthetic", "2000"]
    if not raw_nn:
        cmd.append("--no-raw-nn")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = proc.stdout
    m = re.search(r"RF OOB AUC:\s+([\d.]+)", out)
    g = re.search(r"GBM 5-fold CV:\s+([\d.]+)", out)
    r = re.search(r"Raw-NN 3-fold:\s+([\d.]+)", out)
    rec = {"skew": skew, "perp": perp, "guide": guide, "seed": seed,
           "rf_oob": float(m.group(1)) if m else float("nan"),
           "gbm_cv": float(g.group(1)) if g else float("nan"),
           "raw_nn": float(r.group(1)) if r else None,
           "secs": round(time.time() - t0)}
    if not m:
        rec["stderr_tail"] = proc.stderr[-500:]
    results.append(rec)
    print(json.dumps(rec), flush=True)
    with open("sweep_postproc_followup_results.json", "w") as f:
        json.dump(results, f, indent=1)
    return rec["rf_oob"]


# Fill the two holes.
a1 = run_eval(0.0, 1.0, 0.15)   # untested combo near the descent winner
a2 = run_eval(0.0, 1.0, 0.3)    # NaN in the main sweep, retry once

# Pick the overall winner across everything measured so far.
known = {
    (0.0, 0.85, 0.15): 0.747,
    (0.0, 1.0, 0.0): 0.7584,
    (0.0, 0.85, 0.3): 0.7613,
    (0.0, 0.7, 0.3): 0.765,
    (0.0, 0.85, 0.0): 0.769,
    (0.0, 1.0, 0.15): a1,
    (0.0, 1.0, 0.3): a2,
}
winner = min(known, key=lambda k: known[k] if known[k] == known[k] else 9.9)
print(f"\nwinner so far: {winner} rf_oob={known[winner]:.4f}", flush=True)

# Confirm with two extra seeds, raw-NN enabled.
for seed in (43, 44):
    run_eval(*winner, seed=seed, raw_nn=True)

print("\ndone", flush=True)
