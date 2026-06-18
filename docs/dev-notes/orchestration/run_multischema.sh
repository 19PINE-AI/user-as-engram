#!/usr/bin/env bash
# Train a shared LoRA on personal + medical schemas combined, then evaluate
# the layered architecture on both schemas. Tests whether multi-schema
# meta-skill training fixes the cross-schema generalization gap we
# documented under LLM-judge.
#
# Training corpus: u020-u029 (personal-fact schema) + m020-m029 (medical-
# patient schema). Test users are u000-u019 (personal) and m000-m019
# (medical) — exactly the same test sets used in the single-schema runs,
# so the comparison is apples-to-apples.

set -e
set -o pipefail

cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
NANOCHAT_PY=/home/ubuntu/user-as-engram/nanochat/.venv/bin/python
LOG_DIR=/home/ubuntu/user-as-engram/logs
RESULTS=/home/ubuntu/user-as-engram/results
CKPT=$NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal
SHARED_MULTI=$NANOCHAT_BASE_DIR/shared_lora_d20_multi/r16

# ---------- Train multi-schema shared LoRA ----------
if [ ! -f "$SHARED_MULTI/lora_state.pt" ]; then
    echo "[$(date)] training multi-schema shared LoRA (10 personal + 10 medical = 20 users)"
    $NANOCHAT_PY -m scripts.train_shared_lora \
        --ckpt-dir $CKPT \
        --user-dir /home/ubuntu/user-as-lora/data/users_combined \
        --train-uids u020 u021 u022 u023 u024 u025 u026 u027 u028 u029 \
                      m020 m021 m022 m023 m024 m025 m026 m027 m028 m029 \
        --rank 16 --steps 2000 \
        --out-dir $SHARED_MULTI \
        2>&1 | tee -a $LOG_DIR/shared_lora_multi.log
else
    echo "[$(date)] multi-schema shared LoRA already trained, skipping"
fi

# ---------- Evaluate on personal test users (u000-u019) ----------
echo "[$(date)] layered eval on PERSONAL test users (u000-u019) with multi-schema shared LoRA"
$NANOCHAT_PY -m scripts.layered_architecture \
    --ckpt-dir $CKPT \
    --shared-lora-dir $SHARED_MULTI \
    --user-dir /home/ubuntu/user-as-lora/data/users \
    --test-uids $(printf 'u%03d ' $(seq 0 19)) \
    --eval-tokens 262144 \
    --out $RESULTS/layered_multi_personal.json \
    2>&1 | tee -a $LOG_DIR/layered_multi_personal.log

# ---------- Evaluate on medical test users (m000-m019) ----------
echo "[$(date)] layered eval on MEDICAL test users (m000-m019) with multi-schema shared LoRA"
$NANOCHAT_PY -m scripts.layered_architecture \
    --ckpt-dir $CKPT \
    --shared-lora-dir $SHARED_MULTI \
    --user-dir /home/ubuntu/user-as-lora/data/users_medical \
    --test-uids $(printf 'm%03d ' $(seq 0 19)) \
    --eval-tokens 262144 \
    --out $RESULTS/layered_multi_medical.json \
    2>&1 | tee -a $LOG_DIR/layered_multi_medical.log

echo "[$(date)] MULTI-SCHEMA COMPLETE"
