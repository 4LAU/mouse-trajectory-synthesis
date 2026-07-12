#!/usr/bin/env bash
# July 6 afternoon follow-ups to the distillation verdict.
# 1. Seeds 43 and 44 of the pure fc_v2 + DUR_EMPIRICAL control (new best
#    pure result at seed 42, 0.6470; needs multi-seed confirmation).
# 2. Distillation rerun with --real-frac 0.5: real human batches mixed in
#    as an anchor. If this still lands above the control, CE distillation
#    is dead on this architecture, not just drifted.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe

BASE_ENV="EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1"

run() {
  local name="$1"; local seed="$2"; shift 2
  local log="eval_4m_${name}.log"
  if [ -s "$log" ]; then echo "SKIP $name (log exists)"; return; fi
  echo "=== $(date +%H:%M:%S) START $name ==="
  env $BASE_ENV "$@" $PY evaluate.py --experiment experiments.event_stream_polar --seed "$seed" > "$log" 2>&1
  echo "=== $(date +%H:%M:%S) DONE $name: $(grep -h 'RF OOB AUC' "$log" | tail -1) ==="
}

run fc_v2_pure_duremp_s43 43 EVENT_CKPT=event_polar_4m_fc_v2.pt
run fc_v2_pure_duremp_s44 44 EVENT_CKPT=event_polar_4m_fc_v2.pt

if [ ! -f training/event_polar_4m_distill_rf05_s3000.pt ]; then
  echo "=== $(date +%H:%M:%S) ANCHORED TRAIN START ==="
  $PY training/train_events_polar_distill.py \
      --load-from event_polar_4m_fc_v2.pt \
      --save-name event_polar_4m_distill_rf05.pt \
      --steps 3000 --real-frac 0.5 > train_distill_rf05.log 2>&1
  echo "=== $(date +%H:%M:%S) ANCHORED TRAIN DONE (exit $?) ==="
fi
if [ ! -f training/event_polar_4m_distill_rf05_s3000.pt ]; then
  echo "ABORT: anchored training produced no s3000 snapshot"; exit 1
fi

run distill_rf05_s500_pure_s42  42 EVENT_CKPT=event_polar_4m_distill_rf05_s500.pt
run distill_rf05_s1000_pure_s42 42 EVENT_CKPT=event_polar_4m_distill_rf05_s1000.pt
run distill_rf05_s3000_pure_s42 42 EVENT_CKPT=event_polar_4m_distill_rf05_s3000.pt
echo "FOLLOWUP DONE $(date +%H:%M:%S)"
