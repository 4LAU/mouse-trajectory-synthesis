#!/usr/bin/env bash
# Preference-learning attempt, July 6 (moved up from the July 7 adversarial
# slot). Generates a (winner, loser) pair corpus with the locked SIR recipe,
# DPO fine-tunes fc_v2 against a frozen reference, then evals every snapshot
# pure at seed 42. Waits for whatever eval is currently on the GPU.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe

while wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -qE "evaluate.py|make_distill_corpus|train_events_polar"; do
  echo "$(date +%H:%M:%S) waiting for running job..."
  sleep 120
done

if [ "$(ls training/distill_pairs_b*.npz 2>/dev/null | wc -l)" -lt 3 ]; then
  echo "=== $(date +%H:%M:%S) PAIR CORPUS START ==="
  env DISTILL_SAVE_LOSER=1 DISTILL_SPECS=6000 $PY training/make_distill_corpus.py > distill_pairs_corpus.log 2>&1
  echo "=== $(date +%H:%M:%S) PAIR CORPUS DONE (exit $?) ==="
fi
if [ "$(ls training/distill_pairs_b*.npz 2>/dev/null | wc -l)" -lt 3 ]; then
  echo "ABORT: pair corpus incomplete"; exit 1
fi

if [ ! -f training/event_polar_4m_dpo_v1_s1500.pt ]; then
  echo "=== $(date +%H:%M:%S) DPO TRAIN START ==="
  $PY training/train_events_polar_dpo.py \
      --load-from event_polar_4m_fc_v2.pt \
      --save-name event_polar_4m_dpo_v1.pt \
      --steps 1500 > train_dpo_v1.log 2>&1
  echo "=== $(date +%H:%M:%S) DPO TRAIN DONE (exit $?) ==="
fi
if [ ! -f training/event_polar_4m_dpo_v1_s1500.pt ]; then
  echo "ABORT: DPO training produced no s1500 snapshot"; exit 1
fi

BASE_ENV="EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1"

run() {
  local name="$1"; shift
  local log="eval_4m_dpo_${name}.log"
  if [ -s "$log" ]; then echo "SKIP $name (log exists)"; return; fi
  echo "=== $(date +%H:%M:%S) START $name ==="
  env $BASE_ENV "$@" $PY evaluate.py --experiment experiments.event_stream_polar --seed 42 > "$log" 2>&1
  echo "=== $(date +%H:%M:%S) DONE $name: $(grep -h 'RF OOB AUC' "$log" | tail -1) ==="
}

run v1_s250_pure_s42  EVENT_CKPT=event_polar_4m_dpo_v1_s250.pt
run v1_s500_pure_s42  EVENT_CKPT=event_polar_4m_dpo_v1_s500.pt
run v1_s750_pure_s42  EVENT_CKPT=event_polar_4m_dpo_v1_s750.pt
run v1_s1000_pure_s42 EVENT_CKPT=event_polar_4m_dpo_v1_s1000.pt
run v1_s1500_pure_s42 EVENT_CKPT=event_polar_4m_dpo_v1_s1500.pt
echo "DPO PIPELINE DONE $(date +%H:%M:%S)"
