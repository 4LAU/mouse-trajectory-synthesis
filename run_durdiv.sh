#!/usr/bin/env bash
# Per-candidate duration diversity for SIR. All K candidates currently share
# one duration draw, so the judge never gets to choose among durations, a
# feature family the detector weights heavily. EVENT_SIR_DUR_DIVERSE=1
# resamples duration per candidate on top of the locked recipe.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe

while wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -qE "train_events_polar_distill|evaluate.py"; do
  echo "$(date +%H:%M:%S) waiting for running job..."
  sleep 120
done

BASE_ENV="EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1"

run() {
  local name="$1"; shift
  local log="eval_4m_fc_v2_${name}.log"
  if [ -s "$log" ]; then echo "SKIP $name (log exists)"; return; fi
  echo "=== $(date +%H:%M:%S) START $name ==="
  env $BASE_ENV "$@" $PY evaluate.py --experiment experiments.event_stream_polar --seed 42 > "$log" 2>&1
  echo "=== $(date +%H:%M:%S) DONE $name: $(grep -h 'RF OOB AUC' "$log" | tail -1) ==="
}

run sir16_stemp07_duremp_durdiv_s42 EVENT_SIR=16 EVENT_SIR_TEMP=0.7 EVENT_SIR_DUR_DIVERSE=1
echo "DURDIV DONE $(date +%H:%M:%S)"
