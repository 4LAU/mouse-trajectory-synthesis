#!/usr/bin/env bash
# Go/no-go step 2: trust33 selection + honest replay on the GRPO RL pools.
# CPU-only and model-free (CUDA_VISIBLE_DEVICES="" so it never contends with
# GPU pool generation). Mirrors verify_b.sh: trust33 -> picks -> evaluate
# replay -> RF OOB AUC. Config f20d85_r30_rf is the 0.504 headline config.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
CFG=f20d85_r30_rf
REPLAY_ENV="EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1"

run() {
  local seed="$1"
  local pool="pool_grpo_s${seed}_k16.npz"
  local picks="pool_grpo_s${seed}_k16_picks_trust33_${CFG}.npy"
  local elog="eval_final_trust33_grpo_s${seed}.log"
  if [ ! -s "$pool" ]; then echo "MISSING pool for seed $seed, skip"; return; fi
  echo "=== $(date +%H:%M:%S) SELECT-GRPO seed $seed: trust33 ==="
  CUDA_VISIBLE_DEVICES="" $PY trust33.py --pool "$pool" > "trust33_grpo_s${seed}.log" 2>&1
  echo "=== $(date +%H:%M:%S) SELECT-GRPO seed $seed: replay ==="
  rm -f "$elog"
  env $REPLAY_ENV CUDA_VISIBLE_DEVICES="" EVENT_POOL_LOAD="$pool" EVENT_POOL_PICKS="$picks" \
    $PY evaluate.py --experiment experiments.event_stream_polar --seed "$seed" > "$elog" 2>&1
  echo "=== $(date +%H:%M:%S) SELECT-GRPO seed $seed DONE: $(grep -h 'RF OOB AUC' "$elog" | tail -1) ==="
}

for seed in "$@"; do
  run "$seed"
done
echo "SELECT-GRPO DONE $(date +%H:%M:%S)"
