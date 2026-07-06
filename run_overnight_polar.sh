#!/bin/bash
# Overnight WS7b polar event-stream training with crash auto-resume
# (standing rule 2). Relaunches up to MAX_RESTARTS times; --auto-resume picks
# up the _latest checkpoint so a crash costs at most one epoch.
# PAUSE WINDOWS UPDATE BEFORE LAUNCHING - a forced reboot killed the WS7 run
# at epoch 8 on 2026-07-02.
cd "$(dirname "$0")"
source .venv/Scripts/activate

MAX_RESTARTS=20
EPOCHS="${EPOCHS:-22}"
MAX_SAMPLES="${MAX_SAMPLES:-1000000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_NAME="${SAVE_NAME:-event_polar_best.pt}"
LOG="${LOG:-train_events_polar_overnight.log}"

for i in $(seq 1 $MAX_RESTARTS); do
    echo "=== launch attempt $i / $MAX_RESTARTS $(date) ===" >> "$LOG"
    python training/train_events_polar.py \
        --epochs "$EPOCHS" \
        --batch-size 96 \
        --max-samples "$MAX_SAMPLES" \
        --num-workers "$NUM_WORKERS" \
        --save-name "$SAVE_NAME" \
        --auto-resume >> "$LOG" 2>&1
    code=$?
    if [ $code -eq 0 ]; then
        echo "=== training completed cleanly $(date) ===" >> "$LOG"
        break
    fi
    echo "=== exited with code $code, restarting in 60s $(date) ===" >> "$LOG"
    sleep 60
done
