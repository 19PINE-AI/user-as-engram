#!/usr/bin/env bash
# Tier 1 #1a: layered architecture on Mini-Engram-d12@1280 (second size).
# Trains a shared LoRA (rank-16) on this ckpt, then runs the 6-condition
# layered eval. Output: results/layered_d12_w1280_r16.json
#
# Also re-runs the layered_d20 experiment to regenerate the file that the
# duplicate chain overwrote post-commit.

set -e
set -o pipefail

cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
NANOCHAT_PY=/home/ubuntu/user-as-engram/nanochat/.venv/bin/python
LOG_DIR=/home/ubuntu/user-as-engram/logs
RESULTS=/home/ubuntu/user-as-engram/results
mkdir -p "$LOG_DIR" "$RESULTS"

# ---------- Train shared LoRA on d12@1280 ----------
CKPT_D12=$NANOCHAT_BASE_DIR/engram_runs/engram_d12_w1280_optimal
SHARED_D12=$NANOCHAT_BASE_DIR/shared_lora_d12_w1280/r16

if [ ! -f "$SHARED_D12/lora_state.pt" ]; then
    echo "[$(date)] training shared LoRA on d12@1280 (r=16)"
    $NANOCHAT_PY -m scripts.train_shared_lora \
        --ckpt-dir $CKPT_D12 \
        --user-dir /home/ubuntu/user-as-lora/data/users \
        --rank 16 --steps 2000 \
        --out-dir $SHARED_D12 \
        2>&1 | tee -a $LOG_DIR/shared_lora_d12_w1280_r16.log
else
    echo "[$(date)] shared LoRA d12@1280 already trained, skipping"
fi

# ---------- Smoke test on d12@1280 (1 user) ----------
echo "[$(date)] smoke test layered on d12@1280"
$NANOCHAT_PY -m scripts.layered_architecture \
    --ckpt-dir $CKPT_D12 \
    --shared-lora-dir $SHARED_D12 \
    --user-dir /home/ubuntu/user-as-lora/data/users \
    --smoke \
    --out $RESULTS/layered_d12_w1280_smoke.json \
    2>&1 | tee -a $LOG_DIR/layered_d12_w1280_smoke.log

# ---------- Full 20-user layered on d12@1280 ----------
echo "[$(date)] full layered on d12@1280 (n=20)"
$NANOCHAT_PY -m scripts.layered_architecture \
    --ckpt-dir $CKPT_D12 \
    --shared-lora-dir $SHARED_D12 \
    --user-dir /home/ubuntu/user-as-lora/data/users \
    --test-uids $(printf 'u%03d ' $(seq 0 19)) \
    --eval-tokens 262144 \
    --out $RESULTS/layered_d12_w1280_r16.json \
    2>&1 | tee -a $LOG_DIR/layered_d12_w1280_r16.log

# ---------- Re-run d20 layered to fix the file that the duplicate overwrote ----------
CKPT_D20=$NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal
SHARED_D20=$NANOCHAT_BASE_DIR/shared_lora_d20/r16

echo "[$(date)] re-running layered on d20 to restore n=20 file"
$NANOCHAT_PY -m scripts.layered_architecture \
    --ckpt-dir $CKPT_D20 \
    --shared-lora-dir $SHARED_D20 \
    --user-dir /home/ubuntu/user-as-lora/data/users \
    --test-uids $(printf 'u%03d ' $(seq 0 19)) \
    --eval-tokens 262144 \
    --out $RESULTS/layered_d20_r16_full.json \
    2>&1 | tee -a $LOG_DIR/layered_d20_r16_rerun.log

echo "[$(date)] TIER 1 #1a COMPLETE"
