#!/usr/bin/env bash
# Seeds 43 and 44 of the durdiv recipe (best single-seed 0.5589 at s42).
# Together with the existing s42 run this gives the honest 3-seed number
# for the final recipe candidate.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe

while wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -qE "evaluate.py|make_distill_corpus|train_events_polar"; do
  echo "$(date +%H:%M:%S) waiting for running job..."
  sleep 120
done

BASE_ENV="EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1 EVENT_SIR=16 EVENT_SIR_TEMP=0.7 EVENT_SIR_DUR_DIVERSE=1"

run() {
  local name="$1"; local seed="$2"
  local log="eval_4m_fc_v2_${name}.log"
  if [ -s "$log" ]; then echo "SKIP $name (log exists)"; return; fi
  echo "=== $(date +%H:%M:%S) START $name ==="
  env $BASE_ENV $PY evaluate.py --experiment experiments.event_stream_polar --seed "$seed" > "$log" 2>&1
  echo "=== $(date +%H:%M:%S) DONE $name: $(grep -h 'RF OOB AUC' "$log" | tail -1) ==="
}

run sir16_stemp07_duremp_durdiv_s43 43
run sir16_stemp07_duremp_durdiv_s44 44
echo "DURDIV SEEDS DONE $(date +%H:%M:%S)"
