#!/usr/bin/env bash
# Launch Engram d8 run after base d8 finishes.
# Sized to match base on dense compute, with Engram tables added on top.

set -euo pipefail
cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR="/home/ubuntu/user-as-engram/nanochat_base"
source .venv/bin/activate

rm -rf "$NANOCHAT_BASE_DIR/engram_runs/engram_d8" || true
nohup python -m scripts.engram_pretrain \
  --depth 8 \
  --num-iterations 4000 \
  --device-batch-size 8 \
  --total-batch-size 131072 \
  --max-seq-len 1024 \
  --window-pattern L \
  --engram on \
  --engram-layer-ids 2 5 \
  --engram-vocab-per-ngram 20000 \
  --engram-n-head 8 \
  --engram-n-embed 128 \
  --eval-every 500 \
  --eval-tokens 262144 \
  --no-compile \
  --model-tag engram_d8 \
  > /home/ubuntu/user-as-engram/results/engram_d8.log 2>&1 &
echo "Launched engram_d8 with PID $!"
