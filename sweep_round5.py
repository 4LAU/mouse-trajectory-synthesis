"""Round 5 sweep: micro-tuning around the sweet spot.

Round 4 findings: DH_AMP=0.3 helped curvature/skewness but hurt
time_to_peak and max_deviation. Need smaller DH_AMP (0.05-0.2)
and possibly lower PERP_SCALE to compensate.

Also test: reduced SPEED_SKEW to fix velocity_skewness (too high
with 0.3), combined with DH_AMP for curvature.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
RESULTS_FILE = Path("sweep_round5_results.json")

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
        # === Small DH_AMP values (curvature without max_deviation damage) ===
        ({**BASE, "CANDI_DH_AMP": "0.05"},
         "dh_amp=0.05"),
        ({**BASE, "CANDI_DH_AMP": "0.10"},
         "dh_amp=0.10"),
        ({**BASE, "CANDI_DH_AMP": "0.15"},
         "dh_amp=0.15"),
        ({**BASE, "CANDI_DH_AMP": "0.20"},
         "dh_amp=0.20"),

        # === DH_AMP + compensating PERP_SCALE ===
        ({**BASE, "CANDI_DH_AMP": "0.15", "CANDI_PERP_SCALE": "0.6"},
         "dh_amp=0.15+perp=0.6"),
        ({**BASE, "CANDI_DH_AMP": "0.20", "CANDI_PERP_SCALE": "0.6"},
         "dh_amp=0.20+perp=0.6"),
        ({**BASE, "CANDI_DH_AMP": "0.30", "CANDI_PERP_SCALE": "0.5"},
         "dh_amp=0.30+perp=0.5"),

        # === DH_AMP + DH_OU (amplify + correlated noise) ===
        ({**BASE, "CANDI_DH_AMP": "0.15", "CANDI_DH_OU_SIGMA": "0.2"},
         "dh_amp=0.15+dh_ou=0.2"),

        # === Duration adjustment ===
        ({**BASE, "CANDI_DUR_STD": "0.5"},
         "dur_std=0.5"),
        ({**BASE, "CANDI_DUR_STD": "0.9"},
         "dur_std=0.9"),

        # === Guide strength fine-tune ===
        ({**BASE, "CANDI_GUIDE": "0.2"},
         "guide=0.2"),
        ({**BASE, "CANDI_GUIDE": "0.4"},
         "guide=0.4"),
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
