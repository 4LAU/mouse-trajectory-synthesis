"""Round 2 sweep: fine-tune around skew=0.3 winner + variable skew + combos.

Based on round 1 findings:
- skew=0.3: AUC 0.592 (best)
- skew=0.5: AUC 0.651 (too aggressive, time_to_peak_velocity explodes)
- Need: fine-tune skew, try variable skew, add JITTER for curvature
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
RESULTS_FILE = Path("sweep_round2_results.json")

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
        # === HIGHEST PRIORITY: variable skew to fix correlation gap ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.7",
          "CANDI_SPEED_SKEW_SCALE": "0.5"},
         "best+scale=0.5"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.7",
          "CANDI_SPEED_SKEW_SCALE": "1.0"},
         "best+scale=1.0"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.2", "CANDI_PERP_SCALE": "0.7",
          "CANDI_SPEED_SKEW_SCALE": "1.0"},
         "skew=0.2+perp=0.7+scale=1.0"),

        # === JITTER for curvature features ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.7",
          "CANDI_JITTER": "0.005"},
         "best+jitter=0.005"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.7",
          "CANDI_JITTER": "0.01"},
         "best+jitter=0.01"),

        # === Fine-tune skew around 0.3 with perp=0.7 ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.2", "CANDI_PERP_SCALE": "0.7"},
         "skew=0.2+perp=0.7"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.25", "CANDI_PERP_SCALE": "0.7"},
         "skew=0.25+perp=0.7"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.35", "CANDI_PERP_SCALE": "0.7"},
         "skew=0.35+perp=0.7"),

        # === Perp sweep with best skew ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.6"},
         "skew=0.3+perp=0.6"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.8"},
         "skew=0.3+perp=0.8"),

        # === Variable skew + jitter (attack both remaining gaps) ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.7",
          "CANDI_SPEED_SKEW_SCALE": "1.0", "CANDI_JITTER": "0.005"},
         "best+scale=1.0+jitter"),

        # === Guide strength sweep with best config ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.3", "CANDI_PERP_SCALE": "0.7",
          "CANDI_GUIDE": "0.4"},
         "best+g=0.4"),
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
