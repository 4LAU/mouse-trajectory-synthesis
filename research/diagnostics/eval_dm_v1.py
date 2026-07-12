"""Judge candi_dm_v1.pt at n=2000, bare and with the current best post-proc."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

BASE = {
    "CANDI_CKPT": "candi_dm_v1.pt",
    "CANDI_STEPS": "200",
    "CANDI_CFG": "0",
    "CANDI_CORRECT": "rotate",
}
results = []

for label, skew, perp, guide in [
    ("bare", 0, 1.0, 0),
    ("tuned", 0, 0.85, 0.15),
]:
    env = dict(os.environ)
    env.update(BASE)
    env["CANDI_SPEED_SKEW"] = str(skew)
    env["CANDI_PERP_SCALE"] = str(perp)
    env["CANDI_GUIDE"] = str(guide)
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, "evaluate.py", "--experiment", "experiments.candi",
         "--n-synthetic", "2000"],
        env=env, capture_output=True, text=True,
    ).stdout
    m = re.search(r"RF OOB AUC:\s+([\d.]+)", out)
    g = re.search(r"GBM 5-fold CV:\s+([\d.]+)", out)
    r = re.search(r"Raw-NN 3-fold:\s+([\d.]+)", out)
    rec = {"label": label, "rf_oob": float(m.group(1)) if m else None,
           "gbm_cv": float(g.group(1)) if g else None,
           "raw_nn": float(r.group(1)) if r else None,
           "secs": round(time.time() - t0)}
    results.append(rec)
    print(json.dumps(rec), flush=True)

with open("eval_dm_v1_results.json", "w") as f:
    json.dump(results, f, indent=1)
