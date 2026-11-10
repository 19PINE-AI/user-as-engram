#!/bin/bash
# Full-sample T2 queue (no cap). GPU budget 28 GB: Qwen3-8B training (~18.5 GB) runs alongside ONE d20 job (<=8-10 GB) at a time.
cd ~/clean-tests/results/T2 && . .venv/bin/activate
export NANOCHAT_BASE_DIR=~/clean-tests/nanochat_base USER_AS_ENGRAM_ROOT=~/clean-tests/user-as-engram
U20=$(python3 -c "print(\",\".join(f\"u{i:03d}\" for i in range(20)))")
# Part B mech: repo headline test users u000-u019 (resumes after u000-u004)
MEMFRAC=0.0833 python src/b_mech.py --steps 1500 --uids $U20 --out ~/clean-tests/results/T2/partB_mech.json >> partB.log 2>&1
# Part B layered reproduction: repo headline command, default users (u000-u019) and default eval tokens
( cd ~/clean-tests/user-as-engram/code && PYTHONPATH=$PWD/nanochat_harness:$PWD python - <<PY ) > partB_layered.log 2>&1
import torch; torch.cuda.set_per_process_memory_fraction(9.5/96)
import runpy, sys, os
B=os.environ["NANOCHAT_BASE_DIR"]
sys.argv=["layered_architecture","--ckpt-dir",B+"/engram_runs/engram_d20_w1536_optimal","--shared-lora-dir",B+"/shared_lora_d20/r16","--out",os.path.expanduser("~/clean-tests/results/T2/partB_layered_repro.json")]
runpy.run_module("scripts.layered_architecture", run_name="__main__")
PY
# Part A analysis: after LoRA training (all 20 users x 2 variants) has exited
while pgrep -f "python src/train_lora.py" >/dev/null; do sleep 60; done
PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 python src/analyze.py --out analysis > analyze.log 2>&1
python3 src/aggregate.py analysis > aggregate.txt 2>&1
echo ALL_DONE >> pipeline.log
