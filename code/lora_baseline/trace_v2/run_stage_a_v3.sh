#!/bin/bash
# Stage A v3: within_schema retrain on expanded generator (≥4 Q/schema per user).
# Trace source: data/traces (regenerated programmatic v1 with new variants).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MODEL:=Qwen/Qwen2.5-3B-Instruct}"
: "${UIDS:=u000 u001 u002 u003 u004 u005 u006 u007 u008 u009}"
: "${TAG:=qwen2p5_3b}"
: "${RANK:=64}"

OUT_ROOT="results/stage_a_recite_v3"

MODE=within_schema
OUT_DIR="${OUT_ROOT}/${TAG}_r${RANK}_${MODE}"
echo "=========================================="
echo "  v3 generator  split=${MODE}  model=${MODEL}  rank=${RANK}  out=${OUT_DIR}"
echo "=========================================="
mkdir -p "${OUT_DIR}"
python3 -m stage_a_recite \
  --users ${UIDS} \
  --trace_dir data/traces \
  --out_dir "${OUT_DIR}" \
  --model "${MODEL}" \
  --rank "${RANK}" \
  --split_mode "${MODE}" \
  2>&1 | tee "${OUT_DIR%/}.log"
