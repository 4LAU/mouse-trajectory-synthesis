"""Quick CPU-based experiment sweep using evaluate.py --n-synthetic 200.

Forces CPU so it can run while GPU training continues.
Tests highest-priority configurations first.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
RESULTS_FILE = Path("quick_sweep_results.json")


def run(env_overrides: dict, label: str) -> dict | None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(env_overrides)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, "evaluate.py", "--experiment", "experiments.candi",
         "--n-synthetic", "200"],
        env=env, capture_output=True, text=True, timeout=900,
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

    flow_base = {
        "CANDI_CKPT": "candi_polar_flow_best.pt",
        "CANDI_CFG": "0.0", "CANDI_STEPS": "50",
        "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1",
    }

    ddpm_base = {
        "CANDI_CKPT": "candi_polar_best.pt",
        "CANDI_CFG": "0.0", "CANDI_STEPS": "50",
        "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1",
        "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
        "CANDI_JITTER": "0.005",
    }

    experiments = [
        # === HIGHEST PRIORITY: combine 200 steps + residual velocity ===
        ({**flow_base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_STEPS": "200",
          "CANDI_RESIDUAL_VEL": "0.3", "CANDI_RESIDUAL_PROB": "0.05"},
         "flow g=0.3 s=200 rv=0.3 rp=0.05"),
        ({**flow_base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_STEPS": "200",
          "CANDI_RESIDUAL_VEL": "0.5", "CANDI_RESIDUAL_PROB": "0.05"},
         "flow g=0.3 s=200 rv=0.5 rp=0.05"),
        ({**flow_base, "CANDI_GUIDE": "0.0",
          "CANDI_STEPS": "200",
          "CANDI_RESIDUAL_VEL": "0.5", "CANDI_RESIDUAL_PROB": "0.05"},
         "flow s=200 rv=0.5 rp=0.05"),
        # DDPM + rv=0.3 + more steps
        ({**ddpm_base, "CANDI_STEPS": "100",
          "CANDI_RESIDUAL_VEL": "0.3", "CANDI_RESIDUAL_PROB": "0.05"},
         "ddpm s=100 rv=0.3 rp=0.05"),

        # === Baselines (failed in round 1 due to CUDA bug) ===
        ({**flow_base, "CANDI_GUIDE": "0.0"}, "flow raw"),
        ({**ddpm_base}, "ddpm optimized"),

        # === Flow guide sweep (failed in round 1) ===
        ({**flow_base, "CANDI_GUIDE": "0.2", "CANDI_CORRECT": "rotate"}, "flow g=0.2"),
        ({**flow_base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate"}, "flow g=0.3"),
        ({**flow_base, "CANDI_GUIDE": "0.4", "CANDI_CORRECT": "rotate"}, "flow g=0.4"),
        ({**flow_base, "CANDI_GUIDE": "0.5", "CANDI_CORRECT": "rotate"}, "flow g=0.5"),

        # === DDPM + FEAT_GUIDE (fixed grad bug) ===
        ({**ddpm_base, "CANDI_FEAT_GUIDE": "1.0", "CANDI_FEAT_EFF_TARGET": "0.84"},
         "ddpm fg=1.0"),
        ({**ddpm_base, "CANDI_FEAT_GUIDE": "0.3", "CANDI_FEAT_EFF_TARGET": "0.84"},
         "ddpm fg=0.3"),

        # === Probabilistic residual velocity (flow, already done) ===
        ({**flow_base, "CANDI_GUIDE": "0.0",
          "CANDI_RESIDUAL_VEL": "0.3", "CANDI_RESIDUAL_PROB": "0.05"},
         "flow rv=0.3 rp=0.05"),
        ({**flow_base, "CANDI_GUIDE": "0.0",
          "CANDI_RESIDUAL_VEL": "0.5", "CANDI_RESIDUAL_PROB": "0.05"},
         "flow rv=0.5 rp=0.05"),
        ({**flow_base, "CANDI_GUIDE": "0.0",
          "CANDI_RESIDUAL_VEL": "0.3", "CANDI_RESIDUAL_PROB": "0.10"},
         "flow rv=0.3 rp=0.10"),

        # === DDPM residual velocity (already done) ===
        ({**ddpm_base, "CANDI_RESIDUAL_VEL": "0.3", "CANDI_RESIDUAL_PROB": "0.05"},
         "ddpm rv=0.3 rp=0.05"),
        ({**ddpm_base, "CANDI_RESIDUAL_VEL": "0.5", "CANDI_RESIDUAL_PROB": "0.05"},
         "ddpm rv=0.5 rp=0.05"),

        # === Flow guide + rv combos ===
        ({**flow_base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_RESIDUAL_VEL": "0.3", "CANDI_RESIDUAL_PROB": "0.05"},
         "flow g=0.3 rv=0.3 rp=0.05"),
        ({**flow_base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_RESIDUAL_VEL": "0.5", "CANDI_RESIDUAL_PROB": "0.05"},
         "flow g=0.3 rv=0.5 rp=0.05"),
        ({**flow_base, "CANDI_GUIDE": "0.4", "CANDI_CORRECT": "rotate",
          "CANDI_RESIDUAL_VEL": "0.5", "CANDI_RESIDUAL_PROB": "0.05"},
         "flow g=0.4 rv=0.5 rp=0.05"),

        # === Flow + more steps (already done) ===
        ({**flow_base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_STEPS": "100"},
         "flow g=0.3 steps=100"),
        ({**flow_base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_STEPS": "200"},
         "flow g=0.3 steps=200"),

        # === Flow 200 steps + guide sweep ===
        ({**flow_base, "CANDI_GUIDE": "0.2", "CANDI_CORRECT": "rotate",
          "CANDI_STEPS": "200"},
         "flow g=0.2 steps=200"),
        ({**flow_base, "CANDI_GUIDE": "0.4", "CANDI_CORRECT": "rotate",
          "CANDI_STEPS": "200"},
         "flow g=0.4 steps=200"),
        ({**flow_base, "CANDI_GUIDE": "0.0", "CANDI_STEPS": "200"},
         "flow steps=200 no-guide"),
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
