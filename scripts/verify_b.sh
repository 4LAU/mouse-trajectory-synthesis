#!/usr/bin/env bash
# Verify recipe B (33-dim RF judge) across seeds 42/43/44, CPU only.
# Pools are already cached, so no GPU sampling and no bluescreen risk.
# For each seed: run the 33-dim trust loop offline to pick candidates,
# then replay the winning picks through the honest evaluator with the
# full detector suite (RF OOB, RF 5-fold, GBM, raw-NN). Crash-safe:
# skip-if-log-exists so a rerun resumes.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
BASE_ENV="EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1"
CFG=f20d85_r30_rf

for seed in 42 43 44; do
  pool="pool_s${seed}_k16.npz"
  prefix="${pool%.npz}"
  picks="${prefix}_picks_trust33_${CFG}.npy"
  tlog="trust33_s${seed}.log"

  if [ ! -s "$pool" ]; then echo "MISSING $pool, skipping seed $seed"; continue; fi

  if [ ! -s "$picks" ]; then
    echo "$(date +%H:%M:%S) trust33 loop seed ${seed}..."
    $PY trust33.py --pool "$pool" > "$tlog" 2>&1
    tail -6 "$tlog"
  fi

  flog="eval_final_trust33_s${seed}.log"
  if [ -s "$flog" ]; then echo "SKIP $flog"; continue; fi
  if [ ! -s "$picks" ]; then echo "NO PICKS for seed ${seed}, see $tlog"; continue; fi
  echo "$(date +%H:%M:%S) honest replay seed ${seed} (full detectors)..."
  env $BASE_ENV EVENT_POOL_LOAD="$pool" EVENT_POOL_PICKS="$picks" \
    $PY evaluate.py --experiment experiments.event_stream_polar --seed "$seed" > "$flog" 2>&1
  echo "=== $(date +%H:%M:%S) B FINAL s${seed} ==="
  grep -hE "RF OOB AUC|RF 5-fold|GBM 5-fold|Raw-NN" "$flog" | tail -4
done
echo "VERIFY_B DONE $(date +%H:%M:%S)"
