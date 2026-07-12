#!/bin/bash
# Overnight WS7 event-stream training with crash auto-resume (standing rule 2).
# Relaunches up to MAX_RESTARTS times; --auto-resume picks up the _latest
# checkpoint so a GPU crash costs at most one epoch.
cd "$(dirname "$0")"
source .venv/Scripts/activate

MAX_RESTARTS=20
EPOCHS="${EPOCHS:-30}"
LOG=train_events_overnight.log

for i in $(seq 1 $MAX_RESTARTS); do
    echo "=== launch attempt $i / $MAX_RESTARTS $(date) ===" >> "$LOG"
    python training/train_events.py \
        --epochs "$EPOCHS" \
        --batch-size 96 \
        --max-samples 1000000 \
        --num-workers 2 \
        --auto-resume >> "$LOG" 2>&1
    code=$?
    if [ $code -eq 0 ]; then
        echo "=== training completed cleanly $(date) ===" >> "$LOG"
        break
    fi
    echo "=== exited with code $code, restarting in 60s $(date) ===" >> "$LOG"
    sleep 60
done
