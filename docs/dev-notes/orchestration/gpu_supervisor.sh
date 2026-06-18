#!/usr/bin/env bash
# GPU watcher: poll free memory and launch the experiment queue when
# >= MIN_FREE_GB is available for SUSTAIN_CHECKS consecutive polls.
#
# Why: the experiment queue needs ~16 GB free (Llama-8B + LoRA + activations).
# The supervisor avoids contending with whatever else is on the GPU.
#
# Logs to /home/ubuntu/user-as-engram/logs/supervisor.log
#
# Run via:
#   chmod +x gpu_supervisor.sh run_scaling_and_h2h.sh
#   nohup ./gpu_supervisor.sh > /dev/null 2>&1 &

set -e

MIN_FREE_GB=${MIN_FREE_GB:-16}      # require >= this many GB free
SUSTAIN_CHECKS=${SUSTAIN_CHECKS:-3}  # consecutive polls before launching
POLL_SEC=${POLL_SEC:-60}             # seconds between polls

LOG_DIR=/home/ubuntu/user-as-engram/logs
mkdir -p "$LOG_DIR"
SUP_LOG=$LOG_DIR/supervisor.log

QUEUE_SCRIPT=/home/ubuntu/user-as-engram/run_scaling_and_h2h.sh
STATE_FILE=$LOG_DIR/supervisor.state

free_gb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'
}

echo "[$(date)] supervisor started, threshold ${MIN_FREE_GB}GB sustained for ${SUSTAIN_CHECKS} checks (${POLL_SEC}s each)" \
    | tee -a "$SUP_LOG"

streak=0
while true; do
    f=$(free_gb)
    if [ "$f" -ge "$MIN_FREE_GB" ]; then
        streak=$((streak + 1))
        echo "[$(date)] free=${f}GB  streak=${streak}/${SUSTAIN_CHECKS}" | tee -a "$SUP_LOG"
        if [ "$streak" -ge "$SUSTAIN_CHECKS" ]; then
            echo "[$(date)] LAUNCHING experiment queue" | tee -a "$SUP_LOG"
            echo "running" > "$STATE_FILE"
            nohup bash "$QUEUE_SCRIPT" > "$LOG_DIR/queue.log" 2>&1
            rc=$?
            if [ "$rc" -eq 0 ]; then
                echo "[$(date)] queue exited cleanly (rc=0)" | tee -a "$SUP_LOG"
                echo "done" > "$STATE_FILE"
            else
                echo "[$(date)] queue exited with rc=${rc}" | tee -a "$SUP_LOG"
                echo "failed rc=$rc" > "$STATE_FILE"
            fi
            exit "$rc"
        fi
    else
        if [ "$streak" -gt 0 ]; then
            echo "[$(date)] free=${f}GB  streak reset (was ${streak})" | tee -a "$SUP_LOG"
        fi
        streak=0
    fi
    sleep "$POLL_SEC"
done
