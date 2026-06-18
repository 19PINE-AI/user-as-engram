#!/usr/bin/env bash
# Autorun queue: experiments to fire when GPU returns.
# Polls for >5GB GPU free, then runs in priority order.

set -uo pipefail
export NANOCHAT_BASE_DIR=/home/ubuntu/user-as-engram/nanochat_base
cd /home/ubuntu/user-as-engram/nanochat

PY=.venv/bin/python
CKPT_D12=$NANOCHAT_BASE_DIR/engram_runs/engram_d12
RES=/home/ubuntu/user-as-engram/results

mark() {
  echo "=== $(date) === $1 ===" | tee -a $RES/queue.log
}

run() {
  local name=$1; shift
  mark "START $name"
  "$@" 2>&1 | tee -a $RES/queue.log | tail -20
  mark "END $name"
}

# Wait for GPU
mark "Waiting for GPU"
until nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{exit !($1 > 5000)}'; do
  sleep 60
done
mark "GPU available"

# 1) Multi-hop probe (highest priority — tests the documented limitation)
run "multihop_probe" $PY -u -m scripts.multihop_probe \
    --ckpt-dir $CKPT_D12 \
    --out $RES/multihop_probe.json

# 2) Joint OPT density at N=30 and N=300 (fills the 100/1000 gap)
run "joint_opt_30" $PY -u -m scripts.joint_opt \
    --ckpt-dir $CKPT_D12 --n-facts 30 --steps 1500 --lr 0.5 \
    --out $RES/joint_opt_30.json
run "joint_opt_300" $PY -u -m scripts.joint_opt \
    --ckpt-dir $CKPT_D12 --n-facts 300 --steps 4000 --lr 0.5 \
    --out $RES/joint_opt_300.json

# 3) LoRA rank ablation at 100 facts (rank 8/32/128)
for r in 8 32 128; do
  run "lora_rank${r}_100" $PY -u -m scripts.multifact_lora \
      --ckpt-dir $CKPT_D12 --n-facts 100 \
      --rank $r --steps 2000 --lr 5e-4 \
      --out $RES/multifact_lora_100_r${r}.json
done

# 4) Additive composition at 100+100 (currently only 10+10 in body)
# Reuses the additive_composition script but with bigger sets — we do
# this by registering corp + user as separate users in the server.
run "additive_100x100" $PY -u -c '
import sys, json, time
sys.path.insert(0, ".")
from scripts.engram_server import EngramServer
import torch
print("Starting additive 100+100")
server = EngramServer("'$CKPT_D12'")
import json
with open("/home/ubuntu/user-as-engram/data/corpora_xxl.json") as f:
    c = json.load(f)
# Take 100 user facts and 100 org-style facts (multi_users[0] has org-laden facts)
user_facts = c["user_facts"][:100]
org_facts = c["multi_users"][0]["facts"][:100]
# Train both as separate "users" then check composition
print("Training corp...")
info_corp = server.register_user("corp", org_facts, training="joint_opt", steps=2000)
print(info_corp)
print("Training user...")
info_user = server.register_user("user", user_facts, training="joint_opt", steps=2000)
print(info_user)
# Test corp alone
print("Eval corp alone...")
saved = server._apply([server.users["corp"]])
n_corp_top1 = 0; n_corp_top5 = 0
for f in org_facts:
    out = server.serve(None, f["prompt"], max_tokens=1)
    gold = server.tokenizer.encode(f["gold"])[0]
    if out["top1_id"] == gold: n_corp_top1 += 1
    if gold in out["top5_ids"]: n_corp_top5 += 1
server._restore(saved)
print(f"corp alone top-1 {n_corp_top1}/{len(org_facts)} top-5 {n_corp_top5}/{len(org_facts)}")
# Test user alone
saved = server._apply([server.users["user"]])
n_user_top1 = 0; n_user_top5 = 0
for f in user_facts:
    out = server.serve(None, f["prompt"], max_tokens=1)
    gold = server.tokenizer.encode(f["gold"])[0]
    if out["top1_id"] == gold: n_user_top1 += 1
    if gold in out["top5_ids"]: n_user_top5 += 1
server._restore(saved)
print(f"user alone top-1 {n_user_top1}/{len(user_facts)} top-5 {n_user_top5}/{len(user_facts)}")
# Both together (additive)
saved = server._apply([server.users["corp"], server.users["user"]])
nc1 = nc5 = nu1 = nu5 = 0
for f in org_facts:
    out = server.serve(None, f["prompt"], max_tokens=1)
    gold = server.tokenizer.encode(f["gold"])[0]
    if out["top1_id"] == gold: nc1 += 1
    if gold in out["top5_ids"]: nc5 += 1
for f in user_facts:
    out = server.serve(None, f["prompt"], max_tokens=1)
    gold = server.tokenizer.encode(f["gold"])[0]
    if out["top1_id"] == gold: nu1 += 1
    if gold in out["top5_ids"]: nu5 += 1
server._restore(saved)
print(f"additive corp top-1 {nc1}/{len(org_facts)} top-5 {nc5}/{len(org_facts)}")
print(f"additive user top-1 {nu1}/{len(user_facts)} top-5 {nu5}/{len(user_facts)}")
out_dict = {
  "corp_alone": {"top1": n_corp_top1, "top5": n_corp_top5, "n": len(org_facts)},
  "user_alone": {"top1": n_user_top1, "top5": n_user_top5, "n": len(user_facts)},
  "both_corp": {"top1": nc1, "top5": nc5, "n": len(org_facts)},
  "both_user": {"top1": nu1, "top5": nu5, "n": len(user_facts)},
}
with open("'$RES'/additive_100x100.json", "w") as f:
    json.dump(out_dict, f, indent=2)
print(f"Wrote {out_dict}")
'

# 5) Long-form generation with overrides (8-token continuation per fact)
run "longform_gen" $PY -u -c '
import sys, json
sys.path.insert(0, ".")
from scripts.engram_server import EngramServer
server = EngramServer("'$CKPT_D12'")
import json
with open("/home/ubuntu/user-as-engram/data/corpora_xxl.json") as f:
    c = json.load(f)
facts = c["user_facts"][:30]
print("Training (joint OPT, 30 facts)...")
info = server.register_user("u0", facts, training="joint_opt", steps=1500)
print(info)
import time
n_in_first = 0; n_in_total = 0
gens = []
for f in facts:
    out = server.serve("u0", f["prompt"], max_tokens=8)
    gen_text = out["generated_text"]
    gold = f["gold"].strip()
    in_first_token = gold == out["top1_text"].strip()
    in_total = gold in gen_text
    if in_first_token: n_in_first += 1
    if in_total: n_in_total += 1
    gens.append({"prompt": f["prompt"], "gold": gold, "generated": gen_text,
                 "in_first": int(in_first_token), "in_total": int(in_total)})
print(f"Long-form gen: gold appears in first token {n_in_first}/{len(facts)}={n_in_first/len(facts):.1%}")
print(f"Long-form gen: gold appears anywhere in 8-token gen {n_in_total}/{len(facts)}={n_in_total/len(facts):.1%}")
out_dict = {"n": len(facts), "first_token_hits": n_in_first, "anywhere_hits": n_in_total, "samples": gens[:8]}
with open("'$RES'/longform_gen.json", "w") as f:
    json.dump(out_dict, f, indent=2)
print("Wrote longform_gen.json")
'

# 6) 100x100 with Joint OPT (the expensive headline number, last)
run "per_user_100x100_joint" $PY -u -m scripts.per_user_table_eval \
    --ckpt-dir $CKPT_D12 --n-test-users 100 --n-leak-users 5 \
    --opt-steps 15 --opt-lr 0.5 --scale 20.0 \
    --out $RES/per_user_table_xxl_100x100_jointopt.json

mark "ALL DONE"
