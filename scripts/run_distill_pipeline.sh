#!/usr/bin/env bash
# Distillation pipeline, July 6. Waits for the corpus job, fine-tunes fc_v2
# on the SIR-selected corpus, then evals each snapshot pure (no SIR) at
# seed 42. Deepest snapshot first: it answers "does distillation work at
# all" fastest. SIR-on-top of the best snapshot is queued by hand after
# reading these.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe

while wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -q "make_distill_corpus.py"; do
  echo "$(date +%H:%M:%S) waiting for corpus..."
  sleep 120
done
n=$(ls training/distill_corpus_b*.npz 2>/dev/null | wc -l)
if [ "$n" -lt 10 ]; then echo "ABORT: only $n corpus shards"; exit 1; fi

if [ ! -f training/event_polar_4m_distill_v1_s3000.pt ]; then
  echo "=== $(date +%H:%M:%S) TRAIN START ==="
  $PY training/train_events_polar_distill.py \
      --load-from event_polar_4m_fc_v2.pt \
      --save-name event_polar_4m_distill_v1.pt \
      --steps 3000 --auto-resume > train_distill_v1.log 2>&1
  echo "=== $(date +%H:%M:%S) TRAIN DONE (exit $?) ==="
fi
if [ ! -f training/event_polar_4m_distill_v1_s3000.pt ]; then
  echo "ABORT: training produced no s3000 snapshot"; exit 1
fi

BASE_ENV="EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1"

run() {
  local name="$1"; shift
  local log="eval_4m_distill_${name}.log"
  if [ -s "$log" ]; then echo "SKIP $name (log exists)"; return; fi
  echo "=== $(date +%H:%M:%S) START $name ==="
  env $BASE_ENV "$@" $PY evaluate.py --experiment experiments.event_stream_polar --seed 42 > "$log" 2>&1
  echo "=== $(date +%H:%M:%S) DONE $name: $(grep -h 'RF OOB AUC' "$log" | tail -1) ==="
}

run v1_s3000_pure_s42 EVENT_CKPT=event_polar_4m_distill_v1_s3000.pt
run v1_s2000_pure_s42 EVENT_CKPT=event_polar_4m_distill_v1_s2000.pt
run v1_s1000_pure_s42 EVENT_CKPT=event_polar_4m_distill_v1_s1000.pt
run v1_s500_pure_s42  EVENT_CKPT=event_polar_4m_distill_v1_s500.pt
echo "PIPELINE DONE $(date +%H:%M:%S)"
