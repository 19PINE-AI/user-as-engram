#!/usr/bin/env bash
# Wait for run_opd.sh to finish, then launch reverse cross-schema.
set -e
LOG=/home/ubuntu/user-as-engram/logs/chain_reverse.log
echo "[$(date)] waiting for run_opd.sh to finish..." | tee -a "$LOG"
while pgrep -f "run_opd.sh" > /dev/null; do
    sleep 60
done
echo "[$(date)] OPD done; sleeping 30s for GPU recovery" | tee -a "$LOG"
sleep 30
bash /home/ubuntu/user-as-engram/run_reverse_xschema.sh > /home/ubuntu/user-as-engram/logs/reverse_xschema_queue.log 2>&1
rc=$?
echo "[$(date)] run_reverse_xschema.sh exited rc=${rc}" | tee -a "$LOG"
exit $rc
