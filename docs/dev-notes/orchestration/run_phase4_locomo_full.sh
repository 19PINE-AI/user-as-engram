#!/bin/bash
# Phase 4: re-run LOCOMO on all 10 conversations with prediction logging,
# across all 4 trained Mini-Engrams.
# Then Phase 3: LLM-judge eval via Qwen2.5-32B-Instruct-AWQ on the logged
# predictions.
#
# Sequenced so Phase 2 finishes first (GPU not contended).
# Run order: d8 fastest -> d20 slowest (lets us see signals early).

set -e
cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
LOG=/home/ubuntu/user-as-engram/results

# Wait for Phase 2 to finish (last cell writes per_user_table_d12_w1280_100x100)
echo "[supervisor] waiting for Phase 2 last cell..."
until [ -s "$LOG/per_user_table_d12_w1280_100x100.json" ]; do sleep 60; done
echo "[supervisor] Phase 2 done at $(date), launching LOCOMO 10-conv runs"

# Phase 4: locomo --n-conv 10 on each Mini-Engram, with prediction logging
for TAG in engram_d8_v2 engram_d12_v2 engram_d12_w1280_optimal engram_d20_w1536_optimal; do
  OUT="$LOG/${TAG}__locomo_full10.json"
  if [ -s "$OUT" ]; then
    echo "==> [skip] $TAG locomo full-10 already done"
  else
    echo "==> [locomo full 10 conv] $TAG"
    .venv/bin/python -u -m scripts.locomo_eval \
      --ckpt-dir "$NANOCHAT_BASE_DIR/engram_runs/$TAG" \
      --n-conv 10 --max-qa-per-conv 80 \
      --out "$OUT" \
      > "$LOG/${TAG}__locomo_full10.console.log" 2>&1
  fi
done
echo "===== Phase 4 LOCOMO 10-conv done at $(date) ====="

# Phase 3: Judge eval using Qwen2.5-32B-Instruct-AWQ
# Switch to a venv with vllm
VLLM_PY=/home/ubuntu/polar-env/bin/python

for TAG in engram_d8_v2 engram_d12_v2 engram_d12_w1280_optimal engram_d20_w1536_optimal; do
  OUT="$LOG/${TAG}__locomo_judge.json"
  SRC="$LOG/${TAG}__locomo_full10.json"
  if [ -s "$OUT" ]; then
    echo "==> [skip] $TAG judge already done"
  elif [ -s "$SRC" ]; then
    echo "==> [judge] $TAG"
    # Set PYTHONPATH so the judge script can find vllm etc.
    PYTHONPATH=/home/ubuntu/user-as-engram/nanochat $VLLM_PY -u \
        /home/ubuntu/user-as-engram/nanochat/scripts/judge_locomo.py \
        --locomo-json "$SRC" --out "$OUT" \
        > "$LOG/${TAG}__locomo_judge.console.log" 2>&1
  else
    echo "==> [warn] no source predictions for $TAG, skipping judge"
  fi
done
echo "===== Phase 3 LLM-judge done at $(date) ====="
echo "===== Phases 3+4 complete ====="
