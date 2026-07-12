#!/usr/bin/env bash
# July 6 morning combo queue. Overnight sweep verdict: selection-side knobs
# win (stemp 0.7 -> 0.5649, bw 0.5 -> 0.5761, duremp -> 0.5813), proposal-side
# knobs all fail. These probes map the sharpness curve and test whether the
# winners stack. Seed 42 throughout, ~40 min each (K=32 ~77 min).
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe

while tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi "python.exe" \
      && wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -q "evaluate.py"; do
  echo "$(date +%H:%M:%S) waiting for running eval to finish..."
  sleep 120
done

BASE_ENV="EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10"

run() {
  local name="$1"; shift
  local log="eval_4m_fc_v2_${name}.log"
  if [ -s "$log" ]; then echo "SKIP $name (log exists)"; return; fi
  echo "=== $(date +%H:%M:%S) START $name ==="
  env $BASE_ENV "$@" $PY evaluate.py --experiment experiments.event_stream_polar --seed 42 > "$log" 2>&1
  echo "=== $(date +%H:%M:%S) DONE $name: $(grep -h 'RF OOB AUC' "$log" | tail -1) ==="
}

# 1. sharpness curve: is 0.5 better than 0.7?
run sir16_stemp05_s42          EVENT_SIR=16 EVENT_SIR_TEMP=0.5
# 2. stack the two clean winners
run sir16_stemp07_duremp_s42   EVENT_SIR=16 EVENT_SIR_TEMP=0.7 DUR_EMPIRICAL=1
# 3. bigger pool now that selection is sharp enough to exploit it
run sir32_stemp07_s42          EVENT_SIR=32 EVENT_SIR_TEMP=0.7
# 4. iterated SIR with sharp weights
run sir16_stemp07_iter2_s42    EVENT_SIR=16 EVENT_SIR_TEMP=0.7 EVENT_SIR_ITER=2
echo "ALL DONE $(date +%H:%M:%S)"
