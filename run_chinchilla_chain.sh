#!/bin/bash
# Chinchilla-class retraining of all 4 Mini-Engram sizes.
# All at Karpathy default 12 t/p × (transformer_matrices + lm_head).
# Constant total_batch_size = 131072 across sizes for fair comparison.
set -e
cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base

LOG=/home/ubuntu/user-as-engram/results

# Smallest to largest so we get partial-results signal early.
# Each row: TAG | DEPTH | ASPECT | ENG_LAYERS | NUM_ITERS | DEVICE_BATCH | NOTE
# 12 t/p × scaling-params:
#   d8@512:  42M scaling × 12 = 0.50B tokens / 131K = 3841 iters
#   d12@768: 110M scaling × 12 = 1.32B tokens / 131K = 10081 iters
#   d12@1280: 278M scaling × 12 = 3.33B tokens / 131K = 25441 iters
#   d20@1536: 617M scaling × 12 = 7.40B tokens / 131K = 56449 iters

echo "=== d8@512 (~178M total / 42M scaling) — 0.50B tokens ==="
.venv/bin/python -u -m scripts.engram_pretrain \
  --engram on \
  --depth 8 --aspect-ratio 64 --head-dim 64 \
  --num-iterations 3841 \
  --device-batch-size 8 --total-batch-size 131072 \
  --max-seq-len 1024 \
  --engram-layer-ids 2 5 \
  --engram-vocab-per-ngram 50000 --engram-n-head 8 --engram-n-embed 256 --engram-max-ngram 3 \
  --eval-every 500 --eval-tokens 524288 \
  --no-compile \
  --run engram_d8_v2 --model-tag engram_d8_v2 \
  > "$LOG/train_d8_v2.console.log" 2>&1
echo "=== d8@512 done ==="

echo "=== d12@768 (~339M total / 110M scaling) — 1.32B tokens ==="
.venv/bin/python -u -m scripts.engram_pretrain \
  --engram on \
  --depth 12 --aspect-ratio 64 --head-dim 64 \
  --num-iterations 10081 \
  --device-batch-size 8 --total-batch-size 131072 \
  --max-seq-len 1024 \
  --engram-layer-ids 2 7 \
  --engram-vocab-per-ngram 50000 --engram-n-head 8 --engram-n-embed 256 --engram-max-ngram 3 \
  --eval-every 1000 --eval-tokens 524288 \
  --no-compile \
  --run engram_d12_v2 --model-tag engram_d12_v2 \
  > "$LOG/train_d12_v2.console.log" 2>&1
echo "=== d12@768 done ==="

echo "=== d12@1280 (~625M total / 278M scaling) — 3.33B tokens ==="
.venv/bin/python -u -m scripts.engram_pretrain \
  --engram on \
  --depth 12 --aspect-ratio 106 --head-dim 64 \
  --num-iterations 25441 \
  --device-batch-size 8 --total-batch-size 131072 \
  --max-seq-len 1024 \
  --engram-layer-ids 2 7 \
  --engram-vocab-per-ngram 50000 --engram-n-head 8 --engram-n-embed 256 --engram-max-ngram 3 \
  --eval-every 2000 --eval-tokens 524288 \
  --no-compile \
  --run engram_d12_w1280 --model-tag engram_d12_w1280 \
  > "$LOG/train_d12_w1280.console.log" 2>&1
echo "=== d12@1280 done ==="

echo "=== d20@1536 (~1.22B total / 617M scaling) — 7.40B tokens ==="
.venv/bin/python -u -m scripts.engram_pretrain \
  --engram on \
  --depth 20 --aspect-ratio 76 --head-dim 64 \
  --num-iterations 56449 \
  --device-batch-size 4 --total-batch-size 131072 \
  --max-seq-len 1024 \
  --engram-layer-ids 2 11 \
  --engram-vocab-per-ngram 50000 --engram-n-head 8 --engram-n-embed 256 --engram-max-ngram 3 \
  --eval-every 4000 --eval-tokens 524288 \
  --no-compile \
  --run engram_d20_w1536 --model-tag engram_d20_w1536 \
  > "$LOG/train_d20_w1536.console.log" 2>&1
echo "=== d20@1536 done ==="

echo "=== ALL TRAININGS DONE ==="

# Run eval suite on each
for TAG in engram_d8_v2 engram_d12_v2 engram_d12_w1280 engram_d20_w1536; do
  echo "=== Eval suite: $TAG ==="
  /home/ubuntu/user-as-engram/run_eval_suite.sh "$TAG" || echo "(suite failed for $TAG, continuing)"
done

echo "=== ALL DONE ==="
