#!/usr/bin/env bash
# Reverse cross-schema: train shared LoRA on medical-only (m020-m029),
# evaluate on personal test users (u000-u019). Tests symmetry of the
# cross-schema gap — does the meta-skill failure direction matter?

set -e
set -o pipefail

cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
NANOCHAT_PY=/home/ubuntu/user-as-engram/nanochat/.venv/bin/python
LOG_DIR=/home/ubuntu/user-as-engram/logs
RESULTS=/home/ubuntu/user-as-engram/results
CKPT=$NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal
SHARED_MEDONLY=$NANOCHAT_BASE_DIR/shared_lora_d20_medonly/r16

if [ ! -f "$SHARED_MEDONLY/lora_state.pt" ]; then
    echo "[$(date)] training MEDICAL-ONLY shared LoRA (m020-m029)"
    $NANOCHAT_PY -m scripts.train_shared_lora \
        --ckpt-dir $CKPT \
        --user-dir /home/ubuntu/user-as-lora/data/users_medical \
        --train-uids m020 m021 m022 m023 m024 m025 m026 m027 m028 m029 \
        --rank 16 --steps 2000 \
        --out-dir $SHARED_MEDONLY \
        2>&1 | tee -a $LOG_DIR/shared_lora_medonly.log
fi

echo "[$(date)] reverse cross-schema: medical-trained LoRA on PERSONAL test"
$NANOCHAT_PY -m scripts.layered_architecture \
    --ckpt-dir $CKPT \
    --shared-lora-dir $SHARED_MEDONLY \
    --user-dir /home/ubuntu/user-as-lora/data/users \
    --test-uids $(printf 'u%03d ' $(seq 0 19)) \
    --eval-tokens 262144 \
    --out $RESULTS/layered_reverse_xschema.json \
    2>&1 | tee -a $LOG_DIR/layered_reverse_xschema.log

echo "[$(date)] REVERSE CROSS-SCHEMA COMPLETE"
