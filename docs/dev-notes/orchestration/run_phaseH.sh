#!/bin/bash
# Phase H: Train d12@1280 and d20@1536 at the optimal config selected
# by scripts/pick_optimal.py from the matrix-ablation results.
# Uses --fp8 for both runs to cut wall-clock.

set -e
cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
LOG=/home/ubuntu/user-as-engram/results

# 1. Pick optimal config from ablation results
echo "=== Phase H step 1: pick optimal config ==="
.venv/bin/python -m scripts.pick_optimal | tee "$LOG/pick_optimal.console.log"

CFG=/home/ubuntu/user-as-engram/results/optimal_config.json
if [ ! -s "$CFG" ]; then
    echo "FATAL: $CFG not produced"
    exit 1
fi

# 2. FP8 sanity test (very short d6 run, check loss drops)
echo "=== Phase H step 2: FP8 sanity ==="
SANITY_TAG=engram_d6_fp8_sanity
SANITY_CKPT="$NANOCHAT_BASE_DIR/engram_runs/$SANITY_TAG"
if [ ! -s "$SANITY_CKPT/model.pt" ]; then
    .venv/bin/python -u -m scripts.engram_pretrain \
        --engram on \
        --depth 6 --aspect-ratio 64 --head-dim 64 \
        --num-iterations 200 \
        --device-batch-size 8 --total-batch-size 65536 \
        --max-seq-len 1024 \
        --engram-layer-ids 1 4 \
        --engram-vocab-per-ngram 5000 --engram-n-head 8 --engram-n-embed 64 --engram-max-ngram 3 \
        --eval-every 100 --eval-tokens 65536 \
        --no-compile \
        --run "$SANITY_TAG" --model-tag "$SANITY_TAG" \
        > "$LOG/train_${SANITY_TAG}.console.log" 2>&1
fi
SANITY_BPB=$(grep "val_bpb" "$LOG/train_${SANITY_TAG}.console.log" | tail -1 | grep -oP 'val_bpb: \K[0-9.]+')
echo "FP8 sanity final val_bpb = $SANITY_BPB"
.venv/bin/python -c "
v = float('$SANITY_BPB')
import sys
# starting val_bpb is ~3.14; after 200 steps with FP8 working it should be < 2.5
if v > 2.5:
    print(f'FP8 SANITY FAILED: val_bpb={v} too high, FP8 not converging cleanly')
    sys.exit(1)
print(f'FP8 sanity OK: val_bpb={v}')
"

# 3. Read optimal config and train d12@1280 + d20@1536
echo "=== Phase H step 3: train large models ==="

read VOCAB N_EMBED < <(.venv/bin/python -c "
import json
c = json.load(open('$CFG'))
v = c['d12_1280']
print(v['vocab'], v['n_embed'])
")
# Use Karpathy 12 t/p × scaling-params (matches d12 v2 recipe, halves Phase H wall-clock vs 22.7 t/p)
# d12@1280 scaling-params 278M × 12 = 3.34B tokens / 131K = 25441 iters
# d20@1536 scaling-params 617M × 12 = 7.40B tokens / 131K = 56449 iters
ITERS_1280=25441
ITERS_1536=56449
echo "Optimal capacity: vocab=$VOCAB n_embed=$N_EMBED"
echo "Token budget: 12 t/p (Karpathy default, matches d12_v2 recipe)"
echo "  d12@1280 iters=$ITERS_1280 (3.34B tokens)"
echo "  d20@1536 iters=$ITERS_1536 (7.40B tokens)"

# d12@1280 with FP8 + optimal capacity
TAG=engram_d12_w1280_optimal
CKPT="$NANOCHAT_BASE_DIR/engram_runs/$TAG"
if [ -s "$CKPT/model.pt" ]; then
    echo "==> [skip] $TAG already trained"
else
    echo "==> [train] $TAG  iters=$ITERS_1280 vocab=$VOCAB n_embed=$N_EMBED + FP8"
    .venv/bin/python -u -m scripts.engram_pretrain \
        --engram on \
        --depth 12 --aspect-ratio 106 --head-dim 64 \
        --num-iterations "$ITERS_1280" \
        --device-batch-size 8 --total-batch-size 131072 \
        --max-seq-len 1024 \
        --engram-layer-ids 2 7 \
        --engram-vocab-per-ngram "$VOCAB" --engram-n-head 8 --engram-n-embed "$N_EMBED" --engram-max-ngram 3 \
        --eval-every 2000 --eval-tokens 524288 \
        --no-compile \
        --run "$TAG" --model-tag "$TAG" \
        > "$LOG/train_${TAG}.console.log" 2>&1
fi
/home/ubuntu/user-as-engram/run_eval_suite.sh "$TAG" || echo "(eval failed for $TAG)"

# d20@1536 with FP8 + optimal capacity
TAG=engram_d20_w1536_optimal
CKPT="$NANOCHAT_BASE_DIR/engram_runs/$TAG"
if [ -s "$CKPT/model.pt" ]; then
    echo "==> [skip] $TAG already trained"
else
    echo "==> [train] $TAG  iters=$ITERS_1536 vocab=$VOCAB n_embed=$N_EMBED + FP8"
    .venv/bin/python -u -m scripts.engram_pretrain \
        --engram on \
        --depth 20 --aspect-ratio 76 --head-dim 64 \
        --num-iterations "$ITERS_1536" \
        --device-batch-size 4 --total-batch-size 131072 \
        --max-seq-len 1024 \
        --engram-layer-ids 2 11 \
        --engram-vocab-per-ngram "$VOCAB" --engram-n-head 8 --engram-n-embed "$N_EMBED" --engram-max-ngram 3 \
        --eval-every 4000 --eval-tokens 524288 \
        --no-compile \
        --run "$TAG" --model-tag "$TAG" \
        > "$LOG/train_${TAG}.console.log" 2>&1
fi
/home/ubuntu/user-as-engram/run_eval_suite.sh "$TAG" || echo "(eval failed for $TAG)"

# 4. Final benchmark + scaling table
echo "=== Phase H step 4: final scaling tables ==="
.venv/bin/python -m scripts.capacity_ablation_table | tee "$LOG/capacity_ablation_table.txt"
.venv/bin/python -m scripts.scaling_summary | tee "$LOG/scaling_summary.txt"

echo "=== Phase H complete ==="
