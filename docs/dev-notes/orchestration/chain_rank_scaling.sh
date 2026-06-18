#!/usr/bin/env bash
# Wait for judge_layered (multi-medical judge) to exit, then launch
# rank scaling experiments.
set -e
LOG=/home/ubuntu/user-as-engram/logs/chain_rank.log
echo "[$(date)] waiting for judge_layered to exit..." | tee -a "$LOG"
while pgrep -f "scripts.judge_layered" > /dev/null; do
    sleep 60
done
echo "[$(date)] judge done; sleeping 30s for GPU recovery" | tee -a "$LOG"
sleep 30
bash /home/ubuntu/user-as-engram/run_rank_scaling.sh > /home/ubuntu/user-as-engram/logs/rank_scaling_queue.log 2>&1
echo "[$(date)] rank scaling exited rc=$?" | tee -a "$LOG"
