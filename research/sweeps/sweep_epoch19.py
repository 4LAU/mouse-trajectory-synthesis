"""Re-optimize post-processing for epoch 19 checkpoint.

The SPEED_SKEW=0.3 + PERP_SCALE=0.7 tuning was optimal for epoch 14.
The epoch 19 model may have learned different speed/heading distributions,
requiring different post-processing parameters.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
RESULTS_FILE = Path("sweep_epoch19_results.json")

BASE = {
    "CANDI_CKPT": "candi_polar_flow_best.pt",
    "CANDI_CFG": "0.0",
    "CANDI_STEPS": "200",
    "CANDI_ETA": "0.0",
    "CANDI_CANDIDATES": "1",
    "CANDI_GUIDE": "0.3",
    "CANDI_CORRECT": "rotate",
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
        # === Raw model (no post-processing) ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.0", "CANDI_PERP_SCALE": "1.0"},
         "raw"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.0", "CANDI_PERP_SCALE": "1.0", "CANDI_GUIDE": "0.0"},
         "raw+no_guide"),

        # === SPEED_SKEW sweep (no PERP_SCALE) ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.1", "CANDI_PERP_SCALE": "1.0"},
         "skew=0.1"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.2", "CANDI_PERP_SCALE": "1.0"},
         "skew=0.2"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "1.0"},
         "skew=0.3"),

        # === PERP_SCALE sweep (no SPEED_SKEW) ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.0", "CANDI_PERP_SCALE": "0.7"},
         "perp=0.7"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.0", "CANDI_PERP_SCALE": "0.8"},
         "perp=0.8"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.0", "CANDI_PERP_SCALE": "0.9"},
         "perp=0.9"),

        # === Combined grid ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.1", "CANDI_PERP_SCALE": "0.7"},
         "skew=0.1+perp=0.7"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.1", "CANDI_PERP_SCALE": "0.8"},
         "skew=0.1+perp=0.8"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.2", "CANDI_PERP_SCALE": "0.7"},
         "skew=0.2+perp=0.7"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.2", "CANDI_PERP_SCALE": "0.8"},
         "skew=0.2+perp=0.8"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.7"},
         "skew=0.3+perp=0.7"),  # this is the old best config
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.8"},
         "skew=0.3+perp=0.8"),

        # === Guide sweep ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.0", "CANDI_PERP_SCALE": "1.0", "CANDI_GUIDE": "0.1"},
         "raw+guide=0.1"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.0", "CANDI_PERP_SCALE": "1.0", "CANDI_GUIDE": "0.5"},
         "raw+guide=0.5"),
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
