#!/bin/bash
# Engram-capacity × token-budget MATRIX ablation.
#
# Two dense sizes (d8@512, d12@768) × 5 capacities × 3 token budgets
# = 30 cells total. We reuse 4 existing cells, leaving 26 new trainings.
# The benchmark table aggregator (scripts/capacity_ablation_table.py)
# then identifies the optimal (capacity, t/p) per dense size.

set -e
cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
LOG=/home/ubuntu/user-as-engram/results

train_cell () {
  local TAG=$1
  local DEPTH=$2
  local NUM_ITERS=$3
  local ENG_LAYERS=$4
  local ENG_VOCAB=$5
  local ENG_NEMBED=$6

  local CKPT="$NANOCHAT_BASE_DIR/engram_runs/$TAG"
  if [ -s "$CKPT/model.pt" ]; then
    echo "==> [skip train] $TAG already trained"
  else
    echo "==> [train] $TAG  d=$DEPTH iters=$NUM_ITERS engram_vocab=$ENG_VOCAB n_embed=$ENG_NEMBED"
    .venv/bin/python -u -m scripts.engram_pretrain \
      --engram on \
      --depth $DEPTH --aspect-ratio 64 --head-dim 64 \
      --num-iterations $NUM_ITERS \
      --device-batch-size 8 --total-batch-size 131072 \
      --max-seq-len 1024 \
      --engram-layer-ids $ENG_LAYERS \
      --engram-vocab-per-ngram $ENG_VOCAB --engram-n-head 8 --engram-n-embed $ENG_NEMBED --engram-max-ngram 3 \
      --eval-every 1000 --eval-tokens 524288 \
      --no-compile \
      --run "$TAG" --model-tag "$TAG" \
      > "$LOG/train_${TAG}.console.log" 2>&1
  fi
  /home/ubuntu/user-as-engram/run_eval_suite.sh "$TAG" || echo "(eval failed for $TAG)"
}

# Wait for d12_v2 (in-flight 50K x 256 / 1.32B)
echo "Waiting for engram_d12_v2 to finish..."
until [ -s "$NANOCHAT_BASE_DIR/engram_runs/engram_d12_v2/model.pt" ]; do sleep 60; done
echo "engram_d12_v2 ready, evaluating it"
/home/ubuntu/user-as-engram/run_eval_suite.sh engram_d12_v2 || true

# Token budgets in iterations @ 131K total batch
ITERS_05B=3815
ITERS_1B=7629
ITERS_2B=15259
ITERS_132B=10081
ITERS_25B=19073

# =====================================================================
# d8@512 capacity × tokens MATRIX (5 caps × 3 tokens = 15 cells, 1 reused)
# =====================================================================
ENG_LAYERS_D8="2 5"
echo "===== d8@512 matrix: 5 caps × 3 tokens ====="

# tiny: 5K × 64
train_cell engram_d8_tiny_t05B   8 $ITERS_05B "$ENG_LAYERS_D8"  5000  64
train_cell engram_d8_tiny_t1B    8 $ITERS_1B  "$ENG_LAYERS_D8"  5000  64
train_cell engram_d8_tiny_t2B    8 $ITERS_2B  "$ENG_LAYERS_D8"  5000  64

# small: 20K × 128
train_cell engram_d8_small_t05B  8 $ITERS_05B "$ENG_LAYERS_D8" 20000 128
train_cell engram_d8_small_t1B   8 $ITERS_1B  "$ENG_LAYERS_D8" 20000 128
train_cell engram_d8_small_t2B   8 $ITERS_2B  "$ENG_LAYERS_D8" 20000 128

# medium: 50K × 128
train_cell engram_d8_medium_t05B 8 $ITERS_05B "$ENG_LAYERS_D8" 50000 128
train_cell engram_d8_medium_t1B  8 $ITERS_1B  "$ENG_LAYERS_D8" 50000 128
train_cell engram_d8_medium_t2B  8 $ITERS_2B  "$ENG_LAYERS_D8" 50000 128

# large: 50K × 256 (engram_d8_v2 already covers 0.5B — reuse via skip)
# Note: this run reuses engram_d8_v2 by skipping retrain (model.pt exists).
ln -sf "$NANOCHAT_BASE_DIR/engram_runs/engram_d8_v2" "$NANOCHAT_BASE_DIR/engram_runs/engram_d8_large_t05B" 2>/dev/null || true
ln -sf /home/ubuntu/user-as-engram/results/engram_d8_v2__strategies.json /home/ubuntu/user-as-engram/results/engram_d8_large_t05B__strategies.json 2>/dev/null || true
ln -sf /home/ubuntu/user-as-engram/results/engram_d8_v2__scale.json     /home/ubuntu/user-as-engram/results/engram_d8_large_t05B__scale.json 2>/dev/null || true
ln -sf /home/ubuntu/user-as-engram/results/engram_d8_v2__locomo.json    /home/ubuntu/user-as-engram/results/engram_d8_large_t05B__locomo.json 2>/dev/null || true
train_cell engram_d8_large_t1B   8 $ITERS_1B  "$ENG_LAYERS_D8" 50000 256
train_cell engram_d8_large_t2B   8 $ITERS_2B  "$ENG_LAYERS_D8" 50000 256

# xlarge: 100K × 256
train_cell engram_d8_xlarge_t05B 8 $ITERS_05B "$ENG_LAYERS_D8" 100000 256
train_cell engram_d8_xlarge_t1B  8 $ITERS_1B  "$ENG_LAYERS_D8" 100000 256
train_cell engram_d8_xlarge_t2B  8 $ITERS_2B  "$ENG_LAYERS_D8" 100000 256

# =====================================================================
# d12@768 capacity × tokens MATRIX (5 caps × 3 tokens = 15 cells)
# token levels: 0.5B, 1.32B, 2.5B.  large/1.32B = engram_d12_v2 (reused).
# =====================================================================
ENG_LAYERS_D12="2 7"
echo "===== d12@768 matrix: 5 caps × 3 tokens ====="

# tiny
train_cell engram_d12_tiny_t05B   12 $ITERS_05B  "$ENG_LAYERS_D12"  5000  64
train_cell engram_d12_tiny_t132B  12 $ITERS_132B "$ENG_LAYERS_D12"  5000  64
train_cell engram_d12_tiny_t25B   12 $ITERS_25B  "$ENG_LAYERS_D12"  5000  64

# small
train_cell engram_d12_small_t05B  12 $ITERS_05B  "$ENG_LAYERS_D12" 20000 128
train_cell engram_d12_small_t132B 12 $ITERS_132B "$ENG_LAYERS_D12" 20000 128
train_cell engram_d12_small_t25B  12 $ITERS_25B  "$ENG_LAYERS_D12" 20000 128

# medium
train_cell engram_d12_medium_t05B  12 $ITERS_05B  "$ENG_LAYERS_D12" 50000 128
train_cell engram_d12_medium_t132B 12 $ITERS_132B "$ENG_LAYERS_D12" 50000 128
train_cell engram_d12_medium_t25B  12 $ITERS_25B  "$ENG_LAYERS_D12" 50000 128

# large: 50K × 256 — reuse engram_d12_v2 for the 1.32B cell
ln -sf "$NANOCHAT_BASE_DIR/engram_runs/engram_d12_v2" "$NANOCHAT_BASE_DIR/engram_runs/engram_d12_large_t132B" 2>/dev/null || true
ln -sf /home/ubuntu/user-as-engram/results/engram_d12_v2__strategies.json /home/ubuntu/user-as-engram/results/engram_d12_large_t132B__strategies.json 2>/dev/null || true
ln -sf /home/ubuntu/user-as-engram/results/engram_d12_v2__scale.json     /home/ubuntu/user-as-engram/results/engram_d12_large_t132B__scale.json 2>/dev/null || true
ln -sf /home/ubuntu/user-as-engram/results/engram_d12_v2__locomo.json    /home/ubuntu/user-as-engram/results/engram_d12_large_t132B__locomo.json 2>/dev/null || true
train_cell engram_d12_large_t05B  12 $ITERS_05B  "$ENG_LAYERS_D12" 50000 256
train_cell engram_d12_large_t25B  12 $ITERS_25B  "$ENG_LAYERS_D12" 50000 256

# xlarge
train_cell engram_d12_xlarge_t05B  12 $ITERS_05B  "$ENG_LAYERS_D12" 100000 256
train_cell engram_d12_xlarge_t132B 12 $ITERS_132B "$ENG_LAYERS_D12" 100000 256
train_cell engram_d12_xlarge_t25B  12 $ITERS_25B  "$ENG_LAYERS_D12" 100000 256

# =====================================================================
# Build benchmark table
# =====================================================================
echo "===== Building benchmark table ====="
.venv/bin/python -m scripts.capacity_ablation_table | tee "$LOG/capacity_ablation_table.txt"

echo "===== ablation done ====="
