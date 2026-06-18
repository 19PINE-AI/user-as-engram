#!/usr/bin/env bash
# Full Phase 2 + Phase 3 experiment queue.
#
# Phase 2: scale LoRA negative finding across 4 base models.
# Phase 3: head-to-head locality control on Mini-Engram-d20.
#
# This script assumes the GPU has enough free memory. The supervisor
# (gpu_supervisor.sh) is what waits for capacity and launches this.

set -e
set -o pipefail

cd /home/ubuntu/user-as-lora
mkdir -p logs
LOG_DIR=/home/ubuntu/user-as-lora/logs

# ----------------------------------------------------------------------
# Phase 0: smoke-test the head-to-head pipeline (1 user, 50 steps) so
# we catch shape/import errors before burning hours on the queue.
# ----------------------------------------------------------------------

cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
NANOCHAT_PY=/home/ubuntu/user-as-engram/nanochat/.venv/bin/python

echo "[$(date)] Phase 0: head-to-head smoke test (--smoke)"
$NANOCHAT_PY -m scripts.head_to_head_locality \
    --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \
    --user-dir /home/ubuntu/user-as-lora/data/users \
    --smoke \
    --out /home/ubuntu/user-as-engram/results/head_to_head_d20_smoke.json \
    2>&1 | tee -a /home/ubuntu/user-as-lora/logs/h2h_smoke.log

cd /home/ubuntu/user-as-lora

# ----------------------------------------------------------------------
# Phase 2: scale negative finding across 4 base models
# ----------------------------------------------------------------------

echo "[$(date)] Phase 2.1: Qwen2.5-3B u010-u029 (n=20 more, total n=30)"
python3 src/stage_a.py \
    --users $(printf 'u%03d ' $(seq 10 29)) \
    --model Qwen/Qwen2.5-3B-Instruct \
    --out_dir results/stage_a \
    2>&1 | tee -a $LOG_DIR/stage_a_qwen3b_n30.log

echo "[$(date)] Phase 2.2: Llama-3.1-8B u000-u019 (n=20)"
python3 src/stage_a.py \
    --users $(printf 'u%03d ' $(seq 0 19)) \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --out_dir results/stage_a_llama8b \
    2>&1 | tee -a $LOG_DIR/stage_a_llama8b_n20.log

echo "[$(date)] Phase 2.3: Mistral-7B-Instruct-v0.3 u000-u019 (n=20)"
python3 src/stage_a.py \
    --users $(printf 'u%03d ' $(seq 0 19)) \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --out_dir results/stage_a_mistral7b \
    2>&1 | tee -a $LOG_DIR/stage_a_mistral7b_n20.log

echo "[$(date)] Phase 2.4: Qwen2.5-7B u000-u019 (n=20)"
python3 src/stage_a.py \
    --users $(printf 'u%03d ' $(seq 0 19)) \
    --model Qwen/Qwen2.5-7B-Instruct \
    --out_dir results/stage_a_qwen7b \
    2>&1 | tee -a $LOG_DIR/stage_a_qwen7b_n20.log

# ----------------------------------------------------------------------
# Phase 3: head-to-head locality on Mini-Engram-d20
# ----------------------------------------------------------------------

cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
NANOCHAT_PY=/home/ubuntu/user-as-engram/nanochat/.venv/bin/python

echo "[$(date)] Phase 3: head-to-head locality on Mini-Engram-d20 (n=20)"
$NANOCHAT_PY -m scripts.head_to_head_locality \
    --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \
    --user-dir /home/ubuntu/user-as-lora/data/users \
    --n-users 20 \
    --eval-tokens 524288 \
    --out /home/ubuntu/user-as-engram/results/head_to_head_d20.json \
    2>&1 | tee -a $LOG_DIR/head_to_head_d20.log

echo "[$(date)] ALL PHASE 2+3 EXPERIMENTS COMPLETE (layered chained separately)"
