#!/usr/bin/env bash
# Go/no-go: candidate-pool generation from the GRPO RL-tuned checkpoint,
# identical locked recipe to run_poolgen.sh but with EVENT_CKPT swapped to
# the iter-200 best RL model and distinct pool names so no fc_v2 cache is
# reused. Compare trust33 selection on these vs the 0.504 fc_v2 headline.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe

BASE_ENV="EVENT_CKPT=event_polar_4m_grpo_v1_best.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1 EVENT_SIR=16 EVENT_SIR_TEMP=0.7 EVENT_SIR_DUR_DIVERSE=1"

run() {
  local seed="$1"
  local pool="pool_grpo_s${seed}_k16.npz"
  local log="eval_poolgen_grpo_s${seed}.log"
  if [ -s "$pool" ]; then echo "SKIP seed $seed (pool exists)"; return; fi
  rm -f "$log"
  echo "=== $(date +%H:%M:%S) POOLGEN-GRPO seed $seed START ==="
  env $BASE_ENV EVENT_POOL_SAVE="$pool" $PY evaluate.py --experiment experiments.event_stream_polar --seed "$seed" > "$log" 2>&1
  echo "=== $(date +%H:%M:%S) POOLGEN-GRPO seed $seed DONE: $(grep -h 'RF OOB AUC' "$log" | tail -1) ==="
}

for seed in "$@"; do
  run "$seed"
done
echo "POOLGEN-GRPO DONE $(date +%H:%M:%S)"
