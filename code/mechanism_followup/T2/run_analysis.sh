#!/bin/bash
cd ~/clean-tests/results/T2 && . .venv/bin/activate
PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 python src/analyze.py --out analysis > analyze.log 2>&1
while kill -0 1493991 2>/dev/null; do sleep 60; done
python3 src/aggregate.py analysis > aggregate.txt 2>&1
echo ALL_DONE >> pipeline.log
