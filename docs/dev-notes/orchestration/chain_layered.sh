#!/usr/bin/env bash
# Chain script: waits for the current Phase 2+3 queue to finish (by
# watching for run_scaling_and_h2h.sh to exit), then launches the
# layered architecture experiment.
#
# This is isolated from the running queue script to avoid the risks of
# modifying a file bash is actively reading.

set -e

LOG_DIR=/home/ubuntu/user-as-engram/logs
mkdir -p "$LOG_DIR"
CHAIN_LOG=$LOG_DIR/chain.log

echo "[$(date)] chain script started; waiting for run_scaling_and_h2h.sh to exit" | tee -a "$CHAIN_LOG"

# Wait until run_scaling_and_h2h.sh is no longer running
while pgrep -f "run_scaling_and_h2h.sh" > /dev/null; do
    sleep 60
done

echo "[$(date)] queue exited; supervisor.state=$(cat $LOG_DIR/supervisor.state 2>/dev/null || echo unknown)" | tee -a "$CHAIN_LOG"

# Make sure there's GPU free before launching (defensive)
sleep 30
FREE_GB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
echo "[$(date)] GPU free=${FREE_GB}GB; launching run_layered.sh" | tee -a "$CHAIN_LOG"

bash /home/ubuntu/user-as-engram/run_layered.sh > "$LOG_DIR/layered_queue.log" 2>&1
rc=$?
echo "[$(date)] run_layered.sh exited rc=${rc}" | tee -a "$CHAIN_LOG"
echo "$rc" > "$LOG_DIR/layered.state"
exit $rc
