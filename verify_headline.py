"""Verify the headline result from the cached pools, one command.

Replays the confirmed selection for seeds 42/43/44 through evaluate.py
(replay mode, no GPU, no checkpoint) and checks each RF OOB AUC against
the published value. Run setup_data.py first to download the pools.

    python verify_headline.py

Exits 0 if all three seeds match within tolerance, 1 otherwise. The
tolerance exists for platform and library-version drift; with the pinned
versions in .github/workflows/verify.yml the match is exact.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

EXPECTED = {42: 0.5095, 43: 0.5030, 44: 0.4993}
TOLERANCE = 0.005


def replay(seed: int) -> float:
    env = os.environ.copy()
    env["EVENT_POOL_LOAD"] = f"pool_s{seed}_k16.npz"
    env["EVENT_POOL_PICKS"] = (
        f"pool_s{seed}_k16_picks_trust33_f20d85_r30_rf.npy")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    out = subprocess.run(
        [sys.executable, "evaluate.py",
         "--experiment", "experiments.event_stream_polar",
         "--seed", str(seed), "--no-raw-nn"],
        env=env, capture_output=True, text=True, check=True,
    ).stdout
    m = re.search(r"^val_auc: ([0-9.]+)", out, re.MULTILINE)
    if not m:
        print(out[-2000:])
        raise RuntimeError(f"seed {seed}: no val_auc in evaluate.py output")
    return float(m.group(1))


def main() -> int:
    results = {}
    for seed, expected in EXPECTED.items():
        auc = replay(seed)
        ok = abs(auc - expected) <= TOLERANCE
        results[seed] = ok
        print(f"seed {seed}: RF OOB AUC {auc:.4f} "
              f"(published {expected:.4f}) {'OK' if ok else 'MISMATCH'}",
              flush=True)
    mean = sum(EXPECTED.values()) / len(EXPECTED)
    if all(results.values()):
        print(f"\nheadline verified: three-seed mean {mean:.3f}, "
              f"all seeds within {TOLERANCE} of published values")
        return 0
    print("\nverification FAILED for seeds: "
          + ", ".join(str(s) for s, ok in results.items() if not ok))
    return 1


if __name__ == "__main__":
    sys.exit(main())
