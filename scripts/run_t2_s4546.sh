#!/usr/bin/env bash
# T2 seeds 45/46: offline trust tune per pool, then full-detector honest
# replay with the locked headline config f20d85_r30_rf. Mirrors
# run_frontload.sh so the protocol matches seeds 42-44 exactly.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
BASE_ENV="EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1"

for seed in 45 46; do
  pool="pool_s${seed}_k16.npz"
  prefix="${pool%.npz}"
  tlog="tune_trust_s${seed}.log"
  if [ ! -s "$tlog" ]; then
    echo "$(date +%H:%M:%S) tuning s${seed}..."
    $PY -u tune_trust.py --pool "$pool" > "$tlog" 2>&1 || { echo "TUNE FAILED s${seed} exit=$?"; exit 1; }
    tail -8 "$tlog"
  fi
  picks="${prefix}_picks_trust_f20d85_r30_rf.npy"
  [ -s "$picks" ] || { echo "MISSING PICKS $picks"; exit 1; }
  flog="eval_final_trust_s${seed}.log"
  if [ -s "$flog" ]; then echo "SKIP $flog"; continue; fi
  echo "$(date +%H:%M:%S) replaying s${seed}..."
  env $BASE_ENV EVENT_POOL_LOAD="$pool" EVENT_POOL_PICKS="$picks" \
    $PY -u evaluate.py --experiment experiments.event_stream_polar --seed "$seed" > "$flog" 2>&1 \
    || { echo "REPLAY FAILED s${seed} exit=$?"; exit 1; }
  echo "=== $(date +%H:%M:%S) FINAL s${seed}: $(grep -h 'RF OOB AUC' "$flog" | tail -1) ==="
done
echo "T2 REPLAYS DONE $(date +%H:%M:%S)"
