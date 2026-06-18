#!/bin/bash
# Stage A recite retraining on v2 traces.
# Run once trajectories are generated + rewritten + converted.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MODEL:=Qwen/Qwen2.5-3B-Instruct}"
: "${UIDS:=u000 u001 u002 u003 u004 u005 u006 u007 u008 u009}"
: "${TAG:=qwen2p5_3b}"

OUT_ROOT="results/stage_a_recite_v2"

for MODE in within_schema cross_schema cross_schema_soft; do
  OUT_DIR="${OUT_ROOT}/${TAG}_${MODE}"
  echo "=========================================="
  echo "  split=${MODE}   model=${MODEL}   out=${OUT_DIR}"
  echo "=========================================="
  python3 -m stage_a_recite \
    --users ${UIDS} \
    --trace_dir data/traces_v2 \
    --out_dir "${OUT_DIR}" \
    --model "${MODEL}" \
    --split_mode "${MODE}" \
    2>&1 | tee "${OUT_DIR%/}.log"
done
