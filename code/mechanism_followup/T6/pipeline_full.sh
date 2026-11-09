#!/bin/bash
# T6 full design on Qwen3-8B (batched edits). Wait for GPU, then setup -> lr -> sweep -> multi -> aggregate.
set -u
cd ~/clean-tests/results/T6/src
. ~/clean-tests/.venv/bin/activate
LEDGER=~/clean-tests/results/GPU_BUDGET.md
LOG=~/clean-tests/results/T6/pipeline.log
setstatus() { sed -i -E "s/^(T6 .*48 GB  )[A-Z]+(.*)$/\1$1\2/" "$LEDGER"; }
need=52000
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "$(date +%F' '%T) free=${free}MiB need=${need}" >> "$LOG"
  [ "$free" -ge "$need" ] && break
  sleep 600
done
setstatus RUNNING
echo "$(date +%F' '%T) START full design" >> "$LOG"
run() { for i in $(seq 1 50); do python t6full.py --model Qwen/Qwen3-8B --budget 48 "$@" >> "$LOG" 2>&1 && return 0; echo "$(date +%F' '%T) RETRY $i $*" >> "$LOG"; sleep 300; done; return 1; }
PLAIN=lora_down,lora_up,neuron,neuron_bias,rome,qkv
REG=lora_down+reg,lora_up+reg,neuron+reg,neuron_bias+reg,qkv+reg
run --phase lr || { echo "lr FAILED" >> "$LOG"; setstatus DONE; exit 1; }
run --phase sweep --methods $PLAIN || echo "sweep plain FAILED" >> "$LOG"
python t6agg.py ~/clean-tests/results/T6 > /dev/null 2>> "$LOG"
run --phase sweep --methods $REG || echo "sweep reg FAILED" >> "$LOG"
python t6agg.py ~/clean-tests/results/T6 > /dev/null 2>> "$LOG"
run --phase multi --methods $PLAIN --ns 10,50 || echo "multi plain FAILED" >> "$LOG"
python t6agg.py ~/clean-tests/results/T6 > /dev/null 2>> "$LOG"
run --phase multi --methods $REG --ns 10,50 || echo "multi reg FAILED" >> "$LOG"
python t6agg.py ~/clean-tests/results/T6 > /dev/null 2>> "$LOG"
setstatus DONE
echo "$(date +%F' '%T) ALL DONE" >> "$LOG"
