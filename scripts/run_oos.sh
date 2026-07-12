#!/usr/bin/env bash
# Out-of-sample credibility run: for each fresh seed, generate the fc_v2
# candidate pool on GPU (locked recipe) then run the identical judge-B
# (33-dim trust loop) selection + honest replay on CPU. Same recipe as
# run_poolgen.sh + verify_b.sh, just parameterised over arbitrary seeds so
# we can grow the out-of-sample set beyond 42-46. Skip-safe at every stage.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
POOL_ENV="EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1 EVENT_SIR=16 EVENT_SIR_TEMP=0.7 EVENT_SIR_DUR_DIVERSE=1"
REPLAY_ENV="EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1"
CFG=f20d85_r30_rf

run() {
  local seed="$1"
  local pool="pool_s${seed}_k16.npz"
  local prefix="${pool%.npz}"
  local picks="${prefix}_picks_trust33_${CFG}.npy"
  local plog="eval_poolgen_s${seed}.log"
  local tlog="trust33_s${seed}.log"
  local flog="eval_final_trust33_s${seed}.log"

  # 1) GPU poolgen (skip if pool cached)
  if [ ! -s "$pool" ]; then
    rm -f "$plog"
    echo "=== $(date +%H:%M:%S) POOLGEN seed $seed START ==="
    env $POOL_ENV EVENT_POOL_SAVE="$pool" $PY evaluate.py --experiment experiments.event_stream_polar --seed "$seed" > "$plog" 2>&1
    echo "=== $(date +%H:%M:%S) POOLGEN seed $seed DONE: $(grep -h 'RF OOB AUC' "$plog" | tail -1) ==="
  else
    echo "SKIP poolgen seed $seed (pool exists)"
  fi
  if [ ! -s "$pool" ]; then echo "POOLGEN FAILED seed $seed (no pool), skipping selection"; return; fi

  # 2) CPU judge-B trust loop (skip if picks cached)
  if [ ! -s "$picks" ]; then
    echo "=== $(date +%H:%M:%S) TRUST33 seed $seed ==="
    CUDA_VISIBLE_DEVICES="" $PY trust33.py --pool "$pool" > "$tlog" 2>&1
  fi
  if [ ! -s "$picks" ]; then echo "NO PICKS seed $seed, see $tlog"; return; fi

  # 3) CPU honest replay of the winning picks (skip if final log exists)
  if [ -s "$flog" ]; then echo "SKIP final seed $seed (log exists)"; else
    echo "=== $(date +%H:%M:%S) REPLAY seed $seed ==="
    env $REPLAY_ENV CUDA_VISIBLE_DEVICES="" EVENT_POOL_LOAD="$pool" EVENT_POOL_PICKS="$picks" \
      $PY evaluate.py --experiment experiments.event_stream_polar --seed "$seed" > "$flog" 2>&1
  fi
  echo "=== $(date +%H:%M:%S) OOS FINAL s${seed}: $(grep -h 'RF OOB AUC' "$flog" | tail -1) ==="
}

for seed in "$@"; do
  run "$seed"
done
echo "OOS DONE $(date +%H:%M:%S)"
