#!/usr/bin/env bash
# Resume the experiment queue after Phase 2.1 (Qwen-3B) succeeded and
# Phase 2.2 (Llama meta-org) failed on HuggingFace auth.
#
# Changes vs run_scaling_and_h2h.sh:
#   - Use NousResearch/Meta-Llama-3.1-8B-Instruct (no auth required, locally cached)
#   - Wrap each per-model step in `|| echo "FAIL ..."` so a single failure
#     doesn't abort the entire queue. Each model writes to its own dir.

set -o pipefail   # don't `set -e` here so per-model failures are isolated

cd /home/ubuntu/user-as-lora
mkdir -p logs
LOG_DIR=/home/ubuntu/user-as-lora/logs

run_model() {
    local label=$1
    local model=$2
    local out_dir=$3
    local log_file=$4
    echo "[$(date)] $label : $model -> $out_dir"
    if python3 src/stage_a.py \
            --users $(printf 'u%03d ' $(seq 0 19)) \
            --model "$model" \
            --out_dir "$out_dir" \
            2>&1 | tee -a "$LOG_DIR/$log_file"; then
        echo "[$(date)] $label : OK"
    else
        echo "[$(date)] $label : FAILED rc=$?"
    fi
}

# ---------------- Phase 2.2 (replacement): NousResearch Llama-3.1-8B ----------

run_model "Phase 2.2 Llama-3.1-8B (NousResearch)" \
    "NousResearch/Meta-Llama-3.1-8B-Instruct" \
    "results/stage_a_llama8b" \
    "stage_a_llama8b_n20.log"

# ---------------- Phase 2.3: Mistral-7B-Instruct-v0.3 ------------------------

run_model "Phase 2.3 Mistral-7B-Instruct-v0.3" \
    "mistralai/Mistral-7B-Instruct-v0.3" \
    "results/stage_a_mistral7b" \
    "stage_a_mistral7b_n20.log"

# ---------------- Phase 2.4: Qwen2.5-7B-Instruct -----------------------------

run_model "Phase 2.4 Qwen2.5-7B-Instruct" \
    "Qwen/Qwen2.5-7B-Instruct" \
    "results/stage_a_qwen7b" \
    "stage_a_qwen7b_n20.log"

# ---------------- Phase 3: head-to-head locality on Mini-Engram-d20 ----------

cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
NANOCHAT_PY=/home/ubuntu/user-as-engram/nanochat/.venv/bin/python
ENG_LOG_DIR=/home/ubuntu/user-as-engram/logs

echo "[$(date)] Phase 3: head-to-head locality on Mini-Engram-d20 (n=20)"
if $NANOCHAT_PY -m scripts.head_to_head_locality \
        --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \
        --user-dir /home/ubuntu/user-as-lora/data/users \
        --n-users 20 \
        --eval-tokens 524288 \
        --out /home/ubuntu/user-as-engram/results/head_to_head_d20.json \
        2>&1 | tee -a $ENG_LOG_DIR/head_to_head_d20.log; then
    echo "[$(date)] Phase 3 OK"
else
    echo "[$(date)] Phase 3 FAILED rc=$?"
fi

# ---------------- Phase 4: layered architecture (chained) --------------------

cd /home/ubuntu/user-as-engram
echo "[$(date)] Phase 4: launching run_layered.sh"
bash /home/ubuntu/user-as-engram/run_layered.sh \
    2>&1 | tee -a $ENG_LOG_DIR/layered_queue.log
echo "[$(date)] Phase 4 rc=$?"

echo "[$(date)] RESUME QUEUE COMPLETE"
