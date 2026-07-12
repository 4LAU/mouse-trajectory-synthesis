#!/usr/bin/env bash
# K=32 candidate pool: doubles the per-spec choice for set-level selection.
# Support-deficiency hedge; runs after the K=16 seed pools clear the GPU.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe

while wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -qE "evaluate.py|make_distill_corpus|train_events_polar"; do
  echo "$(date +%H:%M:%S) waiting for running job..."
  sleep 120
done

BASE_ENV="EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1 EVENT_SIR=32 EVENT_SIR_TEMP=0.7 EVENT_SIR_DUR_DIVERSE=1"

run() {
  local seed="$1"
  local pool="pool_s${seed}_k32.npz"
  local log="eval_poolgen32_s${seed}.log"
  if [ -s "$pool" ]; then echo "SKIP seed $seed (pool exists)"; return; fi
  rm -f "$log"
  echo "=== $(date +%H:%M:%S) POOLGEN32 seed $seed START ==="
  env $BASE_ENV EVENT_POOL_SAVE="$pool" $PY evaluate.py --experiment experiments.event_stream_polar --seed "$seed" > "$log" 2>&1
  echo "=== $(date +%H:%M:%S) POOLGEN32 seed $seed DONE: $(grep -h 'RF OOB AUC' "$log" | tail -1) ==="
}

for seed in "$@"; do
  run "$seed"
done
echo "POOLGEN32 DONE $(date +%H:%M:%S)"
