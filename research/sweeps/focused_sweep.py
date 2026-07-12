"""Focused sweep targeting the two biggest remaining gaps:
1. max_deviation (WD=0.794) — PERP_SCALE < 1.0
2. mean_acc/mean_jerk correlation gap (1.33) — SPEED_SKEW > 0

Base config: flow g=0.3 s=200 (AUC=0.634 without rv)
Best config: flow g=0.3 s=200 rv=0.3 rp=0.05 (AUC=0.602)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
RESULTS_FILE = Path("focused_sweep_results.json")

BASE = {
    "CANDI_CKPT": "candi_polar_flow_best.pt",
    "CANDI_CFG": "0.0",
    "CANDI_STEPS": "200",
    "CANDI_ETA": "0.0",
    "CANDI_CANDIDATES": "1",
    "CANDI_GUIDE": "0.3",
    "CANDI_CORRECT": "rotate",
}

RV = {"CANDI_RESIDUAL_VEL": "0.3", "CANDI_RESIDUAL_PROB": "0.05"}


def run(env_overrides: dict, label: str) -> dict | None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env.update(env_overrides)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, "evaluate.py", "--experiment", "experiments.candi",
         "--n-synthetic", "100"],
        env=env, capture_output=True, text=True, timeout=1200,
    )
    elapsed = time.time() - t0

    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if line.strip():
            print(f"  {line.strip()}", flush=True)

    auc = None
    for line in output.split("\n"):
        if line.strip().startswith("val_auc:"):
            try:
                auc = float(line.strip().split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    if auc is None:
        print(f"  FAILED: no AUC parsed", flush=True)
        return None

    record = {"label": label, "auc": auc, "settings": env_overrides,
              "elapsed": round(elapsed, 1)}
    print(f"\n  >> AUC = {auc:.4f} ({elapsed:.0f}s)", flush=True)
    return record


def main():
    results = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
    done = {r["label"] for r in results}
    best_auc = min((r["auc"] for r in results), default=1.0)
    print(f"Previous best: {best_auc:.4f}" if results else "No previous results")

    experiments = [
        # === Priority 1: SPEED_SKEW alone (fix correlation, no max_dev risk) ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.3"}, "skew=0.3"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.5"}, "skew=0.5"),

        # === Priority 2: Best config + PERP_SCALE (fix max_dev) ===
        ({**BASE, **RV, "CANDI_PERP_SCALE": "0.7"}, "rv+perp=0.7"),

        # === Priority 3: Triple combo ===
        ({**BASE, **RV, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.7"},
         "rv+skew=0.3+perp=0.7"),

        # === Priority 4: Skew + perp (no rv) ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.7"},
         "skew=0.3+perp=0.7"),

        # === Priority 5: rv + skew (no perp) ===
        ({**BASE, **RV, "CANDI_SPEED_SKEW": "0.3"}, "rv+skew=0.3"),

        # === Priority 6: baseline comparison (no rv, no skew) ===
        ({**BASE}, "baseline s=200"),

        # === Priority 7: rv baseline (for comparison) ===
        ({**BASE, **RV}, "rv baseline"),
    ]

    remaining = [(s, l) for s, l in experiments if l not in done]
    print(f"\n{len(remaining)} experiments, {len(done)} already done\n")

    for settings, label in remaining:
        record = run(settings, label)
        if record:
            results.append(record)
            RESULTS_FILE.write_text(json.dumps(results, indent=2))
            if record["auc"] < best_auc:
                best_auc = record["auc"]
                print(f"  *** NEW BEST: {best_auc:.4f} ***", flush=True)

    print(f"\n{'='*60}")
    print("RESULTS (sorted by AUC, lower=better)")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda r: r["auc"]):
        print(f"  {r['auc']:.4f}  {r['label']}")


if __name__ == "__main__":
    main()
