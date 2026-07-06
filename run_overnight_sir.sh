#!/usr/bin/env bash
# Overnight SIR knob sweep, July 5-6. Serialized seed-42 probes around the
# clean best config (0.596): EVENT_SIR=8 ct=10 on event_polar_4m_fc_v2.pt.
# Each run ~40 min at K=16 (~75 min at K=32). Crash-safe: every eval is
# independent; a failure moves on to the next.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe

# wait for any already-running eval to release the GPU before starting
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

# 1. sharper selection weights
run sir16_stemp07_s42   EVENT_SIR=16 EVENT_SIR_TEMP=0.7
# 2. hotter heading proposal, SIR filters (curvature tail)
run sir16_tht115_s42    EVENT_SIR=16 EVENT_TH_TEMP=1.15
# 3. wider character proposal
run sir16_bw05_s42      EVENT_SIR=16 EVENT_FEAT_BW=0.5
# 4. bigger candidate pool
run sir32_ct10_s42      EVENT_SIR=32
# 5. more reveal diversity under selection
run sir16_ct12_s42      EVENT_SIR=16 EVENT_CHOICE_TEMP=12
# 6. more duration diversity under selection
run sir16_dur125_s42    EVENT_SIR=16 EVENT_DUR_STD=1.25
# 7. empirical duration prior (conditional skew/tails kept)
run sir16_duremp_s42    EVENT_SIR=16 DUR_EMPIRICAL=1
echo "ALL DONE $(date +%H:%M:%S)"
