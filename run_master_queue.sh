#!/usr/bin/env bash
# Sequential master queue, relaunched after the 14:15 bluescreen. Each child
# script skips completed work, so this is safe to rerun any number of times.
cd "$(dirname "$0")"
bash run_bw_combo.sh
bash run_durdiv.sh
bash run_dpo_pipeline.sh
echo "MASTER QUEUE DONE $(date +%H:%M:%S)"
