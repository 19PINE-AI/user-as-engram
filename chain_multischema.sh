#!/usr/bin/env bash
# Wait for the LOCOMO eval to finish, then launch run_multischema.sh.
set -e
LOG=/home/ubuntu/user-as-engram/logs/chain_multischema.log
echo "[$(date)] waiting for locomo_eval to finish..." | tee -a "$LOG"
while pgrep -f "scripts.locomo_eval" > /dev/null; do
    sleep 60
done
echo "[$(date)] locomo_eval done; sleeping 30s for GPU recovery" | tee -a "$LOG"
sleep 30
FREE_GB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
echo "[$(date)] GPU free=${FREE_GB}GB; launching run_multischema.sh" | tee -a "$LOG"
bash /home/ubuntu/user-as-engram/run_multischema.sh > /home/ubuntu/user-as-engram/logs/multischema_queue.log 2>&1
rc=$?
echo "[$(date)] run_multischema.sh exited rc=${rc}" | tee -a "$LOG"
exit $rc
