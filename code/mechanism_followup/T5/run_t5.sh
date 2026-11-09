#!/bin/bash
set -x
. ~/clean-tests/.venv/bin/activate
B=~/clean-tests/nanochat_ckpts
cd ~/clean-tests/results/T5_T8/code
for m in base_d8 engram_d8; do
  python collect.py --ckpt $B/engram_runs/$m --out ../T5/$m --per-layer 100 --gb 3 > ../T5/collect_$m.log 2>&1 &
done
wait
for m in base_d8 engram_d8; do
  MIMO_PER_BOX=1 python label_sim.py --dir ../T5/$m > ../T5/label_$m.log 2>&1 &
done
wait
for m in base_d8 engram_d8; do
  python substitute.py --ckpt $B/engram_runs/$m --dir ../T5/$m --gb 3 > ../T5/subst_$m.log 2>&1 &
done
wait
echo T5_PIPELINE_DONE
