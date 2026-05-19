#!/usr/bin/env bash
# Higher-rank shared LoRA on multi-schema corpus. Tests whether the
# capacity-bound finding from the reverse cross-schema experiment is
# solvable by scaling rank. We train at r=32 and r=64 on the same
# personal+medical training corpus and evaluate on both test sets.
#
# Prediction: if shared LoRA at r=16 is truly capacity-bound on the
# multi-schema corpus, r=32 should restore most of the in-distribution
# personal performance (closer to single-schema F=45%) while keeping
# medical performance intact. r=64 may further close the gap, or may
# over-parameterize (as in the original rank-ablation on r=64 single-
# schema).

set -e
set -o pipefail

cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
NANOCHAT_PY=/home/ubuntu/user-as-engram/nanochat/.venv/bin/python
LOG_DIR=/home/ubuntu/user-as-engram/logs
RESULTS=/home/ubuntu/user-as-engram/results
CKPT=$NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal

for R in 32 64; do
    SHARED=$NANOCHAT_BASE_DIR/shared_lora_d20_multi_r${R}/r${R}
    if [ ! -f "$SHARED/lora_state.pt" ]; then
        echo "[$(date)] training multi-schema shared LoRA rank=${R}"
        $NANOCHAT_PY -m scripts.train_shared_lora \
            --ckpt-dir $CKPT \
            --user-dir /home/ubuntu/user-as-lora/data/users_combined \
            --train-uids u020 u021 u022 u023 u024 u025 u026 u027 u028 u029 \
                          m020 m021 m022 m023 m024 m025 m026 m027 m028 m029 \
            --rank $R --steps 2000 \
            --out-dir $SHARED \
            2>&1 | tee -a $LOG_DIR/shared_lora_multi_r${R}.log
    fi

    echo "[$(date)] layered eval r=${R} on PERSONAL test"
    $NANOCHAT_PY -m scripts.layered_architecture \
        --ckpt-dir $CKPT \
        --shared-lora-dir $SHARED \
        --user-dir /home/ubuntu/user-as-lora/data/users \
        --test-uids $(printf 'u%03d ' $(seq 0 19)) \
        --eval-tokens 262144 \
        --out $RESULTS/layered_multi_r${R}_personal.json \
        2>&1 | tee -a $LOG_DIR/layered_multi_r${R}_personal.log

    echo "[$(date)] layered eval r=${R} on MEDICAL test"
    $NANOCHAT_PY -m scripts.layered_architecture \
        --ckpt-dir $CKPT \
        --shared-lora-dir $SHARED \
        --user-dir /home/ubuntu/user-as-lora/data/users_medical \
        --test-uids $(printf 'm%03d ' $(seq 0 19)) \
        --eval-tokens 262144 \
        --out $RESULTS/layered_multi_r${R}_medical.json \
        2>&1 | tee -a $LOG_DIR/layered_multi_r${R}_medical.log
done

echo "[$(date)] RANK SCALING COMPLETE"
