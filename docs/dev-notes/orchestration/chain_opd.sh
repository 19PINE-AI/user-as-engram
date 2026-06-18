#!/usr/bin/env bash
# Wait for multi-schema SFT to finish, then launch OPD.
set -e
LOG=/home/ubuntu/user-as-engram/logs/chain_opd.log
echo "[$(date)] waiting for run_multischema.sh to finish..." | tee -a "$LOG"
while pgrep -f "run_multischema.sh" > /dev/null; do
    sleep 60
done
echo "[$(date)] multi-schema done; sleeping 30s for GPU recovery" | tee -a "$LOG"
sleep 30
FREE_GB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
echo "[$(date)] GPU free=${FREE_GB}GB; launching run_opd.sh" | tee -a "$LOG"
bash /home/ubuntu/user-as-engram/run_opd.sh > /home/ubuntu/user-as-engram/logs/opd_queue.log 2>&1
rc=$?
echo "[$(date)] run_opd.sh exited rc=${rc}" | tee -a "$LOG"
exit $rc
