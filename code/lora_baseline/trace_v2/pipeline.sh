#!/bin/bash
# End-to-end v2 trace pipeline:
#   agent -> filter (inline) -> rewriter -> convert -> ready for Stage A retrain
set -euo pipefail
cd "$(dirname "$0")/.."

: "${UIDS:=u000 u001 u002 u003 u004 u005 u006 u007 u008 u009 u010 u011 u012 u013 u014 u015 u016 u017 u018 u019}"
: "${BASE_URL:=http://127.0.0.1:8002/v1}"
: "${MODEL:=qwen3-32b}"

echo "[1/3] Agent over users: ${UIDS}"
python3 -m trace_v2.run_agent \
  --uids ${UIDS} \
  --base_url "${BASE_URL}" --model "${MODEL}" \
  --n_attempts 2 \
  --out_dir data/trace_v2/trajectories_raw

echo "[2/3] Rewriter"
python3 -m trace_v2.rewriter \
  --uids ${UIDS} \
  --in_dir data/trace_v2/trajectories_raw \
  --out_dir data/trace_v2/traces \
  --base_url "${BASE_URL}" --model "${MODEL}"

echo "[3/3] Convert to stage_a format"
python3 -m trace_v2.to_stage_a_format \
  --uids ${UIDS} \
  --in_dir data/trace_v2/traces \
  --out_dir data/traces_v2

echo "Done. Traces ready at data/traces_v2/"
