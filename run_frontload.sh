#!/usr/bin/env bash
# Front-load the full confirmation tonight: as each candidate pool lands,
# tune the trust loop offline, replay the two RF-judge configs honestly
# (no raw-NN, CPU only, safe next to the GPU poolgen queue), then once the
# GPU clears run the protected full-detector replays for all three seeds.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
BASE_ENV="EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5 EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1"
CFGS="f05_r30_rf f20d85_r30_rf"

wait_pool() {
  local f="$1"
  while true; do
    if [ -s "$f" ]; then
      local s1 s2
      s1=$(stat -c%s "$f"); sleep 60; s2=$(stat -c%s "$f")
      if [ "$s1" = "$s2" ]; then break; fi
    else
      sleep 60
    fi
  done
  echo "$(date +%H:%M:%S) pool ready: $f"
}

tune_and_replay() {
  local pool="$1" seed="$2" tag="$3"
  local tlog="tune_trust_${tag}.log"
  local prefix="${pool%.npz}"
  if [ ! -s "$tlog" ]; then
    echo "$(date +%H:%M:%S) tuning $tag..."
    $PY tune_trust.py --pool "$pool" > "$tlog" 2>&1
    tail -8 "$tlog"
  fi
  for cfg in $CFGS; do
    local rlog="eval_replay_trust_${cfg}_${tag}.log"
    if [ -s "$rlog" ]; then echo "SKIP $rlog"; continue; fi
    env $BASE_ENV EVENT_POOL_LOAD="$pool" EVENT_POOL_PICKS="${prefix}_picks_trust_${cfg}.npy" \
      $PY evaluate.py --experiment experiments.event_stream_polar --seed "$seed" --no-raw-nn > "$rlog" 2>&1
    echo "=== $(date +%H:%M:%S) $tag $cfg: $(grep -h 'RF OOB AUC' "$rlog" | tail -1) ==="
  done
}

wait_pool pool_s43_k16.npz
tune_and_replay pool_s43_k16.npz 43 s43
wait_pool pool_s44_k16.npz
tune_and_replay pool_s44_k16.npz 44 s44
wait_pool pool_s42_k32.npz
tune_and_replay pool_s42_k32.npz 42 s42k32

echo "$(date +%H:%M:%S) waiting for GPU queue to clear before raw-NN runs..."
while wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -q "evaluate.py"; do
  sleep 120
done

for seed in 42 43 44; do
  pool="pool_s${seed}_k16.npz"
  prefix="${pool%.npz}"
  flog="eval_final_trust_s${seed}.log"
  if [ -s "$flog" ]; then echo "SKIP $flog"; continue; fi
  env $BASE_ENV EVENT_POOL_LOAD="$pool" EVENT_POOL_PICKS="${prefix}_picks_trust_f20d85_r30_rf.npy" \
    $PY evaluate.py --experiment experiments.event_stream_polar --seed "$seed" > "$flog" 2>&1
  echo "=== $(date +%H:%M:%S) FINAL s${seed}: $(grep -h 'RF OOB AUC' "$flog" | tail -1) ==="
done
echo "FRONTLOAD DONE $(date +%H:%M:%S)"
