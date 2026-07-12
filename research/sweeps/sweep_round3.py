"""Round 3 sweep: SPEED_RAMP to fix mean_acc/mean_jerk correlation gap.

Key insight from diagnostic analysis:
- mean_acceleration is a telescoping sum: (speed[-2] - speed[0]) / (N*dt)
- For rest-to-rest trajectories, this is always ~0 regardless of profile shape
- SPEED_SKEW cannot fix this (only changes shape, not endpoints)
- SPEED_RAMP adds a linear ramp scaled by peak speed, creating
  positive mean_acceleration that correlates with trajectory speed
- Human data shows +1.0 correlation between mean_acc and mean_vel,
  driven by extreme high-velocity trajectories
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
RESULTS_FILE = Path("sweep_round3_results.json")

BASE = {
    "CANDI_CKPT": "candi_polar_flow_best.pt",
    "CANDI_CFG": "0.0",
    "CANDI_STEPS": "200",
    "CANDI_ETA": "0.0",
    "CANDI_CANDIDATES": "1",
    "CANDI_GUIDE": "0.3",
    "CANDI_CORRECT": "rotate",
    "CANDI_SPEED_SKEW": "0.3",
    "CANDI_PERP_SCALE": "0.7",
}


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
        # === SPEED_RAMP sweep (primary target: correlation gap) ===
        ({**BASE, "CANDI_SPEED_RAMP": "0.02"},
         "ramp=0.02"),
        ({**BASE, "CANDI_SPEED_RAMP": "0.05"},
         "ramp=0.05"),
        ({**BASE, "CANDI_SPEED_RAMP": "0.10"},
         "ramp=0.10"),
        ({**BASE, "CANDI_SPEED_RAMP": "0.15"},
         "ramp=0.15"),

        # === SPEED_RAMP + JITTER (attack both gaps: correlations + curvature) ===
        ({**BASE, "CANDI_SPEED_RAMP": "0.05", "CANDI_JITTER": "0.005"},
         "ramp=0.05+jitter=0.005"),
        ({**BASE, "CANDI_SPEED_RAMP": "0.10", "CANDI_JITTER": "0.005"},
         "ramp=0.10+jitter=0.005"),

        # === Fine-tune skew with ramp ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.25", "CANDI_SPEED_RAMP": "0.05"},
         "skew=0.25+ramp=0.05"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.35", "CANDI_SPEED_RAMP": "0.05"},
         "skew=0.35+ramp=0.05"),

        # === Baseline comparison (no ramp) ===
        ({**BASE},
         "best_baseline"),

        # === Aggressive ramp (test ceiling) ===
        ({**BASE, "CANDI_SPEED_RAMP": "0.25"},
         "ramp=0.25"),

        # === Ramp + perp variations ===
        ({**BASE, "CANDI_SPEED_RAMP": "0.05", "CANDI_PERP_SCALE": "0.6"},
         "ramp=0.05+perp=0.6"),
        ({**BASE, "CANDI_SPEED_RAMP": "0.05", "CANDI_PERP_SCALE": "0.8"},
         "ramp=0.05+perp=0.8"),
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
