#!/usr/bin/env bash
# Run all post-training analysis. To be invoked after both base_d8 and engram_d8
# have completed.

set -euo pipefail
cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR="/home/ubuntu/user-as-engram/nanochat_base"
source .venv/bin/activate

OUT=/home/ubuntu/user-as-engram/results
mkdir -p "$OUT"

echo "=== P3: replication validation ==="
python -m scripts.evaluate_replication \
  --ckpt-dir "$NANOCHAT_BASE_DIR/engram_runs/engram_d8" \
  --base-ckpt-dir "$NANOCHAT_BASE_DIR/engram_runs/base_d8" \
  --out "$OUT/replication.json" 2>&1 | tee "$OUT/replication_console.log"

echo
echo "=== P4: User-as-Engram surgical insertion ==="
python -m scripts.user_facts_demo \
  --ckpt-dir "$NANOCHAT_BASE_DIR/engram_runs/engram_d8" \
  --out "$OUT/user_facts.json" \
  --scale 20.0 2>&1 | tee "$OUT/user_facts_console.log"

echo
echo "=== Plot training/eval curves ==="
python -m scripts.plot_runs \
  --base-log "$NANOCHAT_BASE_DIR/engram_runs/base_d8/train_log.jsonl" \
  --engram-log "$NANOCHAT_BASE_DIR/engram_runs/engram_d8/train_log.jsonl" \
  --out-dir "$OUT/runs_compare"

echo
echo "All analysis done. Results under $OUT/"
ls -la "$OUT/"
