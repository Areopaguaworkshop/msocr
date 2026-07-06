#!/usr/bin/env bash
# Launch the §1 freeze-rows RunPod training run detached.
# Logs to reports/logs/run_freeze_v4_$(date).log
set -euo pipefail
cd /home/ajiap/project/msocr

# Bridge .env var name to CLI-expected name
set -a; . ./.env; set +a
export RUNPOD_API_KEY="$RunPod_API"
unset RunPod_API

LOGDIR=reports/logs
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/run_freeze_v4_${STAMP}.log"
echo "=== launching §1 freeze-rows run at $(date -Iseconds) ===" | tee "$LOG"
echo "log: $LOG" | tee -a "$LOG"

exec uv run msocr train-remote \
  --manifest data/manifests/c2av-finetune.json \
  --style-group c2av-syriac-finetune \
  --base-model models/kraken/sophro_mhiro_syriac.mlmodel \
  --output-model models/kraken/c2av_freeze_v4.safetensors \
  --reports-dir reports/ \
  --epochs 200 --lag 25 --lr 1e-4 --augment \
  --freeze-backbone 999999 \
  --freeze-old-rows \
  --codec-json reports/c2av_union_codec.json \
  >>"$LOG" 2>&1