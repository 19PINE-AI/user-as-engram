#!/bin/bash
# Run the standard benchmark suite on a Mini-Engram checkpoint.
# Usage: run_eval_suite.sh <model_tag>
# Produces results/<tag>__strategies.json, scale.json, locomo.json
set -e
TAG=${1:?usage: $0 <model_tag, e.g. engram_d12_w1280>}
cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
CKPT="$NANOCHAT_BASE_DIR/engram_runs/$TAG"
OUT=/home/ubuntu/user-as-engram/results

if [ ! -f "$CKPT/model.pt" ]; then
  echo "ERR: no checkpoint at $CKPT/model.pt"
  exit 1
fi

echo "==> $TAG : insertion_strategies_v2 (16 facts × 5 strategies)"
if [ -s "$OUT/${TAG}__strategies.json" ]; then echo "    (skip: $OUT/${TAG}__strategies.json exists)"; else
.venv/bin/python -u -m scripts.insertion_strategies_v2 \
  --ckpt-dir "$CKPT" \
  --out "$OUT/${TAG}__strategies.json" \
  > "$OUT/${TAG}__strategies.console.log" 2>&1
fi

echo "==> $TAG : eval_at_scale E1 only (100 user + 100 org facts, no multi-user)"
if [ -s "$OUT/${TAG}__scale.json" ]; then echo "    (skip: $OUT/${TAG}__scale.json exists)"; else
.venv/bin/python -u -m scripts.eval_at_scale \
  --ckpt-dir "$CKPT" \
  --corpus /home/ubuntu/user-as-engram/data/corpora.json \
  --n-user-facts 100 --n-org-facts 100 \
  --n-test-users 0 \
  --out "$OUT/${TAG}__scale.json" \
  > "$OUT/${TAG}__scale.console.log" 2>&1
fi

echo "==> $TAG : LOCOMO Option A (2 conv × 80 QA)"
if [ -s "$OUT/${TAG}__locomo.json" ]; then echo "    (skip: $OUT/${TAG}__locomo.json exists)"; else
.venv/bin/python -u -m scripts.locomo_eval \
  --ckpt-dir "$CKPT" \
  --n-conv 2 --max-qa-per-conv 80 \
  --out "$OUT/${TAG}__locomo.json" \
  > "$OUT/${TAG}__locomo.console.log" 2>&1
fi

echo "==> $TAG : suite done"
