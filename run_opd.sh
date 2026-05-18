#!/usr/bin/env bash
# On-Policy Distillation: use Qwen2.5-7B-Instruct as teacher to generate
# CoT reasoning chains for each (facts, indirect-Q) pair, then train the
# shared LoRA on the teacher's outputs. Compare to multi-schema SFT
# (which uses the gold answer directly, no reasoning chain).

set -e
set -o pipefail

cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
NANOCHAT_PY=/home/ubuntu/user-as-engram/nanochat/.venv/bin/python
LOG_DIR=/home/ubuntu/user-as-engram/logs
RESULTS=/home/ubuntu/user-as-engram/results
CKPT=$NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal
SHARED_OPD=$NANOCHAT_BASE_DIR/shared_lora_d20_opd/r16

# ---------- Generate OPD corpus from teacher ----------
OPD_CORPUS=$RESULTS/opd_corpus_multi.jsonl
if [ ! -f "$OPD_CORPUS" ]; then
    echo "[$(date)] generating OPD corpus with Qwen2.5-7B-Instruct teacher"
    $NANOCHAT_PY -m scripts.generate_opd_corpus \
        --user-dirs /home/ubuntu/user-as-lora/data/users \
                     /home/ubuntu/user-as-lora/data/users_medical \
        --train-uids u020 u021 u022 u023 u024 u025 u026 u027 u028 u029 \
                      m020 m021 m022 m023 m024 m025 m026 m027 m028 m029 \
        --teacher Qwen/Qwen2.5-7B-Instruct \
        --max-indirect-per-user 30 \
        --out $OPD_CORPUS \
        2>&1 | tee -a $LOG_DIR/opd_corpus_gen.log
else
    echo "[$(date)] OPD corpus already exists, skipping generation"
fi

# ---------- Train shared LoRA on OPD corpus ----------
if [ ! -f "$SHARED_OPD/lora_state.pt" ]; then
    echo "[$(date)] training OPD shared LoRA on teacher chains"
    $NANOCHAT_PY -m scripts.train_shared_lora \
        --ckpt-dir $CKPT \
        --opd-corpus $OPD_CORPUS \
        --rank 16 --steps 2000 --lr 3e-4 \
        --out-dir $SHARED_OPD \
        2>&1 | tee -a $LOG_DIR/shared_lora_opd.log
else
    echo "[$(date)] OPD shared LoRA already trained, skipping"
fi

# ---------- Evaluate on personal test users ----------
echo "[$(date)] OPD layered eval on PERSONAL test users (u000-u019)"
$NANOCHAT_PY -m scripts.layered_architecture \
    --ckpt-dir $CKPT \
    --shared-lora-dir $SHARED_OPD \
    --user-dir /home/ubuntu/user-as-lora/data/users \
    --test-uids $(printf 'u%03d ' $(seq 0 19)) \
    --eval-tokens 262144 \
    --out $RESULTS/layered_opd_personal.json \
    2>&1 | tee -a $LOG_DIR/layered_opd_personal.log

# ---------- Evaluate on medical test users ----------
echo "[$(date)] OPD layered eval on MEDICAL test users (m000-m019)"
$NANOCHAT_PY -m scripts.layered_architecture \
    --ckpt-dir $CKPT \
    --shared-lora-dir $SHARED_OPD \
    --user-dir /home/ubuntu/user-as-lora/data/users_medical \
    --test-uids $(printf 'm%03d ' $(seq 0 19)) \
    --eval-tokens 262144 \
    --out $RESULTS/layered_opd_medical.json \
    2>&1 | tee -a $LOG_DIR/layered_opd_medical.log

echo "[$(date)] OPD COMPLETE"
