#!/bin/bash
# Fact-count scaling benchmark: run eval_at_scale's E1 (per-fact OPT/ICL recall)
# at n_facts ∈ {100, 200, 500, 1000} on each pretrained model. Use corpora_xl.json.
# Output: results/<tag>__factscale_n<N>.json per cell.
#
# Sequenced smallest-model first so we get early signal.
# Defer d20@1536_optimal until its training completes.

set -e
cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
LOG=/home/ubuntu/user-as-engram/results
CORPUS=/home/ubuntu/user-as-engram/data/corpora_xl.json

eval_n_facts () {
  local TAG=$1
  local N=$2
  local CKPT="$NANOCHAT_BASE_DIR/engram_runs/$TAG"
  local OUT="$LOG/${TAG}__factscale_n${N}.json"
  if [ ! -f "$CKPT/model.pt" ]; then
    echo "==> [missing ckpt] $TAG"; return 1
  fi
  if [ -s "$OUT" ]; then
    echo "==> [skip] $TAG n=$N already done"; return 0
  fi
  echo "==> [eval] $TAG n=$N"
  .venv/bin/python -u -m scripts.eval_at_scale \
    --ckpt-dir "$CKPT" \
    --corpus "$CORPUS" \
    --n-user-facts "$N" --n-org-facts "$N" \
    --n-test-users 0 \
    --out "$OUT" \
    > "$LOG/${TAG}__factscale_n${N}.console.log" 2>&1
}

# Run smaller dense first (d8 ≈ 5-50 min/cell).
# d12@768 next (~10-100 min/cell).
# Defer d12@1280_optimal until d20 free (~280 min/cell).
# Final: d20 after its training completes.
for N in 100 200 500 1000; do
  for TAG in engram_d8 engram_d8_v2 engram_d8_large_t2B \
              engram_d12 engram_d12_v2 engram_d12_large_t25B \
              engram_d12_w1280_optimal; do
    eval_n_facts "$TAG" "$N" || echo "(failed for $TAG n=$N)"
  done
done

# Now wait for d20@1536_optimal to finish
echo "===== Waiting for d20@1536_optimal ckpt ====="
until [ -s "$NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal/model.pt" ]; do sleep 600; done
echo "===== d20 ready, evaluating ====="
for N in 100 200 500 1000; do
  eval_n_facts engram_d20_w1536_optimal "$N" || echo "(failed)"
done

# Aggregate
.venv/bin/python -m scripts.factscale_table | tee "$LOG/factscale_table.txt"
echo "===== fact-scaling benchmark done ====="
