#!/usr/bin/env bash
# Layered architecture experiment queue.
#
# Phases:
#   0. Smoke test of layered_architecture.py (1 user, truncated)
#   1. Train 3 shared LoRAs at rank ∈ {4, 16, 64} on u020–u029
#   2. Full 20-user evaluation at rank=16 (the headline)
#   3. Ablation: 5-user evaluation at rank=4 and rank=64
#
# Run via:
#   ./run_layered.sh
# (Triggered automatically by the GPU supervisor or by chain from
#  run_scaling_and_h2h.sh)

set -e
set -o pipefail

cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
NANOCHAT_PY=/home/ubuntu/user-as-engram/nanochat/.venv/bin/python
CKPT=$NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal
LOG_DIR=/home/ubuntu/user-as-engram/logs
RESULTS_DIR=/home/ubuntu/user-as-engram/results
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$NANOCHAT_BASE_DIR/shared_lora_d20"

# ---------------- Phase 1: train shared LoRAs ----------------

for R in 4 16 64; do
    OUT_DIR=$NANOCHAT_BASE_DIR/shared_lora_d20/r${R}
    if [ -f "$OUT_DIR/lora_state.pt" ] && [ -f "$OUT_DIR/meta.json" ]; then
        echo "[$(date)] Phase 1 r${R}: shared LoRA already trained, skipping"
        continue
    fi
    echo "[$(date)] Phase 1 r${R}: training shared LoRA"
    $NANOCHAT_PY -m scripts.train_shared_lora \
        --ckpt-dir $CKPT \
        --user-dir /home/ubuntu/user-as-lora/data/users \
        --rank $R --steps 2000 \
        --out-dir $OUT_DIR \
        2>&1 | tee -a "$LOG_DIR/shared_lora_r${R}.log"
done

# ---------------- Phase 0: smoke test once shared LoRA exists ----------------

echo "[$(date)] Phase 0: layered smoke test (1 user, 100 steps)"
$NANOCHAT_PY -m scripts.layered_architecture \
    --ckpt-dir $CKPT \
    --shared-lora-dir $NANOCHAT_BASE_DIR/shared_lora_d20/r16 \
    --user-dir /home/ubuntu/user-as-lora/data/users \
    --smoke \
    --out $RESULTS_DIR/layered_d20_smoke.json \
    2>&1 | tee -a "$LOG_DIR/layered_smoke.log"

# ---------------- Phase 2: full 20-user run at r=16 (headline) ----------------

echo "[$(date)] Phase 2: full 20-user evaluation at r=16"
$NANOCHAT_PY -m scripts.layered_architecture \
    --ckpt-dir $CKPT \
    --shared-lora-dir $NANOCHAT_BASE_DIR/shared_lora_d20/r16 \
    --user-dir /home/ubuntu/user-as-lora/data/users \
    --test-uids $(printf 'u%03d ' $(seq 0 19)) \
    --eval-tokens 262144 \
    --out $RESULTS_DIR/layered_d20_r16_full.json \
    2>&1 | tee -a "$LOG_DIR/layered_r16_full.log"

# ---------------- Phase 3: ablation at r=4 and r=64 (5 users each) ----------

for R in 4 64; do
    echo "[$(date)] Phase 3 r${R}: ablation on 5 users"
    $NANOCHAT_PY -m scripts.layered_architecture \
        --ckpt-dir $CKPT \
        --shared-lora-dir $NANOCHAT_BASE_DIR/shared_lora_d20/r${R} \
        --user-dir /home/ubuntu/user-as-lora/data/users \
        --test-uids u000 u001 u002 u003 u004 \
        --eval-tokens 262144 \
        --out $RESULTS_DIR/layered_d20_r${R}_abl.json \
        2>&1 | tee -a "$LOG_DIR/layered_r${R}_abl.log"
done

echo "[$(date)] LAYERED EXPERIMENT COMPLETE"
