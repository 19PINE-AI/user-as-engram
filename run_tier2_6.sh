#!/usr/bin/env bash
# Tier 2 #6: cross-schema test of the layered architecture.
#
# Hypothesis: a shared LoRA trained on the personal-fact schema (u020-u029)
# should still help indirect reasoning on the medical-patient schema (m000-m019)
# IF the meta-skill is genuinely cross-schema generalisable. If not (e.g.,
# the shared LoRA only learned personal-schema patterns), F's indirect
# performance on medical users will collapse to ≈ C (per-user Engram alone),
# which would falsify H2c in the cross-distribution regime.
#
# Reuses: existing shared LoRA at $NANOCHAT_BASE_DIR/shared_lora_d20/r16
# (trained on personal-schema u020-u029).
# New: layered_architecture.py eval on the 20 medical patients (m000-m019).

set -e
set -o pipefail

cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
NANOCHAT_PY=/home/ubuntu/user-as-engram/nanochat/.venv/bin/python
LOG_DIR=/home/ubuntu/user-as-engram/logs
RESULTS=/home/ubuntu/user-as-engram/results

# (Skipping smoke: layered_architecture.py's --smoke default uses u000 which is
# the personal schema, not the medical schema. Pipeline is already validated
# from the d20 and d12@1280 runs; go straight to full.)

# Full 20-patient cross-schema run
echo "[$(date)] cross-schema full (20 medical patients)"
$NANOCHAT_PY -m scripts.layered_architecture \
    --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \
    --shared-lora-dir $NANOCHAT_BASE_DIR/shared_lora_d20/r16 \
    --user-dir /home/ubuntu/user-as-lora/data/users_medical \
    --test-uids $(printf 'm%03d ' $(seq 0 19)) \
    --eval-tokens 262144 \
    --out $RESULTS/layered_cross_schema_full.json \
    2>&1 | tee -a $LOG_DIR/layered_cross_schema_full.log

echo "[$(date)] CROSS-SCHEMA COMPLETE"
