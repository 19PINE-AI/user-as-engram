#!/bin/bash
# Sequentially train 600M (d12@1280) then 1B (d20@1536) Mini-Engram models.
set -e
cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base

# 600M target: d12@1280 (n_head=20, head_dim=64) → 625M total params
echo "=== Starting 600M run: engram_d12_w1280 ==="
.venv/bin/python -u -m scripts.engram_pretrain \
  --engram on \
  --depth 12 --aspect-ratio 106 --head-dim 64 \
  --num-iterations 6000 \
  --device-batch-size 8 --total-batch-size 131072 \
  --max-seq-len 1024 \
  --engram-layer-ids 2 7 \
  --engram-vocab-per-ngram 50000 --engram-n-head 8 --engram-n-embed 256 --engram-max-ngram 3 \
  --eval-every 500 --eval-tokens 524288 \
  --no-compile \
  --run engram_d12_w1280 \
  --model-tag engram_d12_w1280 \
  > /home/ubuntu/user-as-engram/results/train_d12_w1280.console.log 2>&1
echo "=== 600M done ==="

# 1B target: d20@1536 (n_head=24, head_dim=64) → 1.22B total params
echo "=== Starting 1B run: engram_d20_w1536 ==="
.venv/bin/python -u -m scripts.engram_pretrain \
  --engram on \
  --depth 20 --aspect-ratio 76 --head-dim 64 \
  --num-iterations 6000 \
  --device-batch-size 4 --total-batch-size 131072 \
  --max-seq-len 1024 \
  --engram-layer-ids 2 11 \
  --engram-vocab-per-ngram 50000 --engram-n-head 8 --engram-n-embed 256 --engram-max-ngram 3 \
  --eval-every 500 --eval-tokens 524288 \
  --no-compile \
  --run engram_d20_w1536 \
  --model-tag engram_d20_w1536 \
  > /home/ubuntu/user-as-engram/results/train_d20_w1536.console.log 2>&1
echo "=== 1B done ==="
