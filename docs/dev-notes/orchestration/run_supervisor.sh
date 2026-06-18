#!/bin/bash
# Supervisor: waits for the matrix ablation to finish, then runs Phase H.
# Completion signal = capacity_ablation_table.txt being non-empty.

LOG=/home/ubuntu/user-as-engram/results
echo "[supervisor] waiting for ablation completion ($LOG/capacity_ablation_table.txt)"
until [ -s "$LOG/capacity_ablation_table.txt" ]; do sleep 120; done
echo "[supervisor] ablation done, launching Phase H"
bash /home/ubuntu/user-as-engram/run_phaseH.sh > "$LOG/phaseH.log" 2>&1
echo "[supervisor] Phase H finished with exit $?"
