#!/bin/bash
# Phase 2: re-run §6 main tables at d12@1280 optimal
set -e
cd /home/ubuntu/user-as-engram/nanochat
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
CKPT=$NANOCHAT_BASE_DIR/engram_runs/engram_d12_w1280_optimal
LOG=/home/ubuntu/user-as-engram/results

skip () {
  [ -s "$1" ] && echo "  [skip] $1 exists" && return 0
  return 1
}

# B1: Multi-fact LoRA rank=64 (POLAR-class) at 100 facts on d12@1280
OUT="$LOG/multifact_lora_100_d12_w1280.json"
skip "$OUT" || {
  echo "=== B1.a multi-fact LoRA rank 64 / 100 facts / d12@1280 ==="
  .venv/bin/python -u -m scripts.multifact_lora \
    --ckpt-dir "$CKPT" --n-facts 100 --rank 64 --steps 2000 \
    --out "$OUT" > "$LOG/multifact_lora_100_d12_w1280.console.log" 2>&1
}

# B1: Multi-fact LoRA rank=64 at 1000 facts on d12@1280
OUT="$LOG/multifact_lora_1000_d12_w1280.json"
skip "$OUT" || {
  echo "=== B1.b multi-fact LoRA rank 64 / 1000 facts / d12@1280 ==="
  .venv/bin/python -u -m scripts.multifact_lora \
    --ckpt-dir "$CKPT" --n-facts 1000 --rank 64 --steps 8000 \
    --out "$OUT" > "$LOG/multifact_lora_1000_d12_w1280.console.log" 2>&1
}

# B4: LoRA rank ablation on d12@1280
for R in 8 32 128; do
  OUT="$LOG/multifact_lora_100_d12_w1280_r${R}.json"
  skip "$OUT" || {
    echo "=== B4 multi-fact LoRA rank $R / 100 facts / d12@1280 ==="
    .venv/bin/python -u -m scripts.multifact_lora \
      --ckpt-dir "$CKPT" --n-facts 100 --rank $R --steps 2000 \
      --out "$OUT" > "$LOG/multifact_lora_100_d12_w1280_r${R}.console.log" 2>&1
  }
done

# B1: SFT-LoRA per-fact baseline on d12@1280
OUT="$LOG/sft_baseline_d12_w1280.json"
skip "$OUT" || {
  echo "=== B1.c SFT-LoRA per-fact / 100 USER + 100 ORG / d12@1280 ==="
  .venv/bin/python -u -m scripts.sft_baseline \
    --ckpt-dir "$CKPT" \
    --out "$OUT" > "$LOG/sft_baseline_d12_w1280.console.log" 2>&1
}

# B2: Long-form generation at d12@1280
OUT="$LOG/longform_gen_d12_w1280.json"
skip "$OUT" || {
  echo "=== B2 long-form generation / d12@1280 ==="
  .venv/bin/python -u -m scripts.longform_gen \
    --ckpt-dir "$CKPT" \
    --out "$OUT" > "$LOG/longform_gen_d12_w1280.console.log" 2>&1
}

# B2: Long-form generation at d20
OUT="$LOG/longform_gen_d20_w1536.json"
skip "$OUT" || {
  echo "=== B2 long-form generation / d20@1536 ==="
  .venv/bin/python -u -m scripts.longform_gen \
    --ckpt-dir "$NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal" \
    --out "$OUT" > "$LOG/longform_gen_d20_w1536.console.log" 2>&1
}

# B3: Multi-tenant serving eval at d12@1280
OUT="$LOG/serving_eval_d12_w1280_30u_50f.json"
skip "$OUT" || {
  echo "=== B3 serving eval 30u x 50f / d12@1280 ==="
  .venv/bin/python -u -m scripts.eval_serving \
    --ckpt-dir "$CKPT" --n-users 30 --facts-per-user 50 --n-requests 600 \
    --out "$OUT" > "$LOG/serving_eval_d12_w1280_30u_50f.console.log" 2>&1
}

# B5: 100u x 100f per-user override eval at d12@1280
OUT="$LOG/per_user_table_d12_w1280_100x100.json"
skip "$OUT" || {
  echo "=== B5 per-user table 100u x 100f / d12@1280 ==="
  .venv/bin/python -u -m scripts.per_user_table_eval \
    --ckpt-dir "$CKPT" --n-test-users 100 \
    --out "$OUT" > "$LOG/per_user_table_d12_w1280_100x100.console.log" 2>&1
}

# B6: Memory systems trigger-exact at d8 (have d8 v1, missing for symmetry)
OUT="$LOG/engram_d8_v2__memory_systems.json"
skip "$OUT" || {
  echo "=== B6 memory_systems_comparison / d8 v2 ==="
  .venv/bin/python -u -m scripts.memory_systems_comparison \
    --ckpt-dir "$NANOCHAT_BASE_DIR/engram_runs/engram_d8_v2" \
    --n-facts 100 \
    --out "$OUT" > "$LOG/engram_d8_v2__memory_systems.console.log" 2>&1
}
OUT="$LOG/engram_d8_v2__memory_systems_paraphrase.json"
skip "$OUT" || {
  echo "=== B6 memory_systems_paraphrase / d8 v2 ==="
  .venv/bin/python -u -m scripts.memory_systems_paraphrase \
    --ckpt-dir "$NANOCHAT_BASE_DIR/engram_runs/engram_d8_v2" \
    --out "$OUT" > "$LOG/engram_d8_v2__memory_systems_paraphrase.console.log" 2>&1
}

echo "===== Phase 2 done at $(date) ====="
