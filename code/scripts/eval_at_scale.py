"""
P8/P9 — Scaled User-as-Engram evaluation.

Three sub-evaluations on the same Mini-Engram checkpoint:
  E1) PROBE — for each fact in user_facts + org_facts:
      - Baseline rank/logit of gold next-token
      - UNEMBED_P insertion (cheap)
      - OPT insertion (15 steps)
      - ICL baseline: prepend the fact statement in context, baseline rank
  E2) MULTI_USER — sample N test users from the multi_users corpus.
      For each test user:
        - Insert all of their facts simultaneously into Engram rows (UNEMBED_P)
        - For their own questions: did the inserted answer appear?
        - For OTHER users' questions (leakage probe): does any inserted info
          bleed across?
      Aggregates: per-user recall, cross-user leakage rate.
  E3) NO_INSERT control — same probes with NO insertion. Establishes the
      lower bound (what the model "knows" without our help).

Outputs JSON to results/scale_eval.json.
"""
import os
import json
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import get_base_dir

from scripts.insertion_strategies_v2 import (
    load_model,
    trigger_global_rows,
    write_marker,
    restore_rows,
    make_marker_UNEMBED_P,
    make_marker_OPT,
)


@torch.no_grad()
def baseline_logit_and_rank(model, tokenizer, prompt, gold_text, device):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    gold_id = tokenizer.encode(gold_text)[0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(idx)[0, -1, :]
    return logits[gold_id].item(), int((logits > logits[gold_id]).sum().item()), gold_id, idx, ids


def hits_top_k(logits, gold_id, k):
    return int((logits > logits[gold_id]).sum().item()) < k


def e1_per_fact(model, tokenizer, facts, eng, last_layer, Wv_pinv, total_heads, embed_dim, device, args):
    out = []
    for f_i, fact in enumerate(facts):
        bl, br, gold_id, idx, ids = baseline_logit_and_rank(model, tokenizer, fact["prompt"], fact["gold"], device)
        trig_pos = len(ids) - 1

        # ---- ICL baseline: prepend the fact statement, then ask
        bos = tokenizer.get_bos_token_id()
        # Construct ICL prompt: e.g., "Fact: My doctor's name is Dr. Patel.\nMy doctor's name is Dr."
        icl_text = f"Fact: {fact['prompt']}{fact['gold']}.\n{fact['prompt']}"
        icl_ids = tokenizer.encode(icl_text, prepend=bos)
        icl_idx = torch.tensor([icl_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            icl_logits = model(icl_idx)[0, -1, :]
        icl_rank = int((icl_logits > icl_logits[gold_id]).sum().item())
        icl_top1 = int(icl_logits.argmax().item()) == gold_id

        # ---- UNEMBED_P
        global_rows = trigger_global_rows(eng, idx, last_layer, trig_pos)
        tbl = eng.tables[str(last_layer)]
        originals = tbl.embedding.weight.data[global_rows].clone()
        try:
            marker = make_marker_UNEMBED_P(model, eng, last_layer, gold_id, idx, trig_pos, args.scale,
                                            total_heads, embed_dim, Wv_pinv=Wv_pinv)
            write_marker(eng, last_layer, global_rows, marker)
            with torch.no_grad():
                up_logits = model(idx)[0, -1, :]
            up_rank = int((up_logits > up_logits[gold_id]).sum().item())
            up_top1 = int(up_logits.argmax().item()) == gold_id
        finally:
            restore_rows(eng, last_layer, global_rows, originals)

        # ---- OPT
        try:
            marker = make_marker_OPT(model, eng, last_layer, gold_id, idx, trig_pos, args.scale,
                                      total_heads, embed_dim, Wv_pinv=Wv_pinv,
                                      n_steps=args.opt_steps, lr=args.opt_lr)
            write_marker(eng, last_layer, global_rows, marker)
            with torch.no_grad():
                opt_logits = model(idx)[0, -1, :]
            opt_rank = int((opt_logits > opt_logits[gold_id]).sum().item())
            opt_top1 = int(opt_logits.argmax().item()) == gold_id
        finally:
            restore_rows(eng, last_layer, global_rows, originals)

        out.append({
            "fact_idx": f_i,
            "schema": fact.get("schema"),
            "trigger": fact["trigger"],
            "gold": fact["gold"],
            "baseline_rank": br,
            "icl_rank": icl_rank,
            "icl_top1": int(icl_top1),
            "unembed_rank": up_rank,
            "unembed_top1": int(up_top1),
            "opt_rank": opt_rank,
            "opt_top1": int(opt_top1),
        })
        if (f_i + 1) % 25 == 0 or f_i < 3:
            print(f"  [{f_i+1:3d}/{len(facts)}] base_rank={br:>5d}  icl={icl_rank:>4d}{'★' if icl_top1 else ''}  "
                  f"UE={up_rank:>4d}{'★' if up_top1 else ''}  OPT={opt_rank:>4d}{'★' if opt_top1 else ''}")
    return out


def aggregate(rows, key_rank, key_top1):
    n = len(rows)
    top1 = sum(r[key_top1] for r in rows)
    top5 = sum(1 for r in rows if r[key_rank] < 5)
    top10 = sum(1 for r in rows if r[key_rank] < 10)
    rank_med = sorted(r[key_rank] for r in rows)[n // 2] if n else 0
    return {"n": n, "top1": top1, "top5": top5, "top10": top10, "rank_median": rank_med}


def e2_multi_user(model, tokenizer, multi_users, eng, last_layer, Wv_pinv, total_heads, embed_dim, device, args,
                  n_test_users=10, use_opt=True):
    """Multi-user simultaneous insertion + cross-user leakage probe.

    Procedure:
      For each of n_test_users:
        - For each of their facts: compute marker (OPT if use_opt else UNEMBED_P)
        - Write all markers (record originals to restore later)
        - For each fact: baseline rank w/o insertion, post-insertion rank
        - Cross-user leakage: query an OTHER user's questions, see if their
          answers (from your own facts pool) bleed in
        - Restore all rows

    Note: if multiple facts in a user share the same trigger N-gram (e.g. two
    'My gym day is' entries with different golds), the LAST write wins.
    We dedupe by trigger keeping the last fact for each trigger.
    """
    results = []
    test_users = multi_users[:n_test_users]
    bos = tokenizer.get_bos_token_id()

    PER_USER_SALT_PRIME = 2654435761  # large odd salt, multiplied by user_id

    for ui, user in enumerate(test_users):
        # Per-user salt (0 = no salt = legacy/global); use user_id * prime as XOR salt
        u_salt = (user['user_id'] * PER_USER_SALT_PRIME) & ((1 << 62) - 1) if args.use_salt else 0
        # Dedupe by trigger (last write wins)
        trigger_to_fact = {}
        for fact in user['facts']:
            trigger_to_fact[fact['trigger']] = fact
        deduped_facts = list(trigger_to_fact.values())

        # First, baseline ranks for ALL of this user's questions WITHOUT any insertion
        baselines = []
        gold_ids = []
        for fact in deduped_facts:
            ids = tokenizer.encode(fact['prompt'], prepend=bos)
            gold = tokenizer.encode(fact['gold'])[0]
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                lg = model(idx)[0, -1, :]
            br = int((lg > lg[gold]).sum().item())
            baselines.append({"prompt_ids": ids, "gold_id": gold, "trig_pos": len(ids)-1,
                              "baseline_rank": br, "baseline_top1": int(lg.argmax().item()) == gold})
            gold_ids.append(gold)

        # Compute markers and global rows for each fact (using THIS user's salt)
        eng.user_salt = u_salt
        tbl = eng.tables[str(last_layer)]
        all_writes = []  # list of (global_rows, marker, originals)
        for fact, base in zip(deduped_facts, baselines):
            idx = torch.tensor([base['prompt_ids']], dtype=torch.long, device=device)
            global_rows = trigger_global_rows(eng, idx, last_layer, base['trig_pos'], user_salt=u_salt)
            originals = tbl.embedding.weight.data[global_rows].clone()
            if use_opt:
                marker = make_marker_OPT(model, eng, last_layer, base['gold_id'], idx, base['trig_pos'],
                                          args.scale, total_heads, embed_dim, Wv_pinv=Wv_pinv,
                                          n_steps=args.opt_steps, lr=args.opt_lr)
            else:
                marker = make_marker_UNEMBED_P(model, eng, last_layer, base['gold_id'], idx, base['trig_pos'],
                                                args.scale, total_heads, embed_dim, Wv_pinv=Wv_pinv)
            all_writes.append((global_rows, marker, originals))

        # Write ALL at once
        for global_rows, marker, _ in all_writes:
            write_marker(eng, last_layer, global_rows, marker)

        try:
            # Post-insert recall on this user's own facts (use this user's salt)
            eng.user_salt = u_salt
            own_recall = []
            for fact, base in zip(deduped_facts, baselines):
                idx = torch.tensor([base['prompt_ids']], dtype=torch.long, device=device)
                with torch.no_grad():
                    lg = model(idx)[0, -1, :]
                pr = int((lg > lg[base['gold_id']]).sum().item())
                pt = int(lg.argmax().item()) == base['gold_id']
                own_recall.append({"baseline_rank": base['baseline_rank'], "post_rank": pr,
                                    "baseline_top1": base['baseline_top1'], "post_top1": int(pt)})

            # Cross-user leakage: query as the LEAK user (their salt) — should NOT see this user's data
            leak_user_idx = (ui + 1) % len(test_users)
            leak_user = test_users[leak_user_idx]
            leak_salt = (leak_user['user_id'] * PER_USER_SALT_PRIME) & ((1 << 62) - 1) if args.use_salt else 0
            eng.user_salt = leak_salt
            leak_results = []
            for fact in leak_user['facts']:
                ids = tokenizer.encode(fact['prompt'], prepend=bos)
                idx = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    lg = model(idx)[0, -1, :]
                top1_id = int(lg.argmax().item())
                top1_text = tokenizer.decode([top1_id])
                # Did this match THIS user's gold instead of leak_user's?
                # (Leak indicator: top1 is NOT the leak_user's gold and IS this user's gold for the same trigger.)
                leak_user_gold = tokenizer.encode(fact['gold'])[0]
                this_users_facts_for_same_trigger = [f for f in deduped_facts if f['trigger'] == fact['trigger']]
                leaked = False
                if this_users_facts_for_same_trigger:
                    this_gold = tokenizer.encode(this_users_facts_for_same_trigger[0]['gold'])[0]
                    if top1_id == this_gold and top1_id != leak_user_gold:
                        leaked = True
                leak_results.append({"prompt": fact['prompt'], "leak_user_gold": fact['gold'],
                                      "post_top1": top1_text, "leaked": int(leaked)})
        finally:
            # Restore all rows and reset salt
            eng.user_salt = u_salt  # use insertion salt to find the right rows for restoration
            for global_rows, _, originals in all_writes:
                restore_rows(eng, last_layer, global_rows, originals)
            eng.user_salt = 0

        # Aggregate this user
        n_facts = len(deduped_facts)
        recall_top1_baseline = sum(r['baseline_top1'] for r in own_recall) / max(n_facts, 1)
        recall_top1_post = sum(r['post_top1'] for r in own_recall) / max(n_facts, 1)
        n_leaks = sum(r['leaked'] for r in leak_results)
        leak_rate = n_leaks / max(len(leak_results), 1)
        results.append({
            "user_id": user['user_id'],
            "n_facts": n_facts,
            "recall_top1_baseline": recall_top1_baseline,
            "recall_top1_post": recall_top1_post,
            "leak_rate": leak_rate,
            "n_leaks": n_leaks,
        })
        print(f"  user {user['user_id']:3d}: own recall top1 {recall_top1_baseline:.2f}->{recall_top1_post:.2f}  "
              f"  leak {leak_rate:.3f} ({n_leaks}/{len(leak_results)})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--corpus", default="/home/ubuntu/user-as-engram/data/corpora.json")
    parser.add_argument("--out", default="/home/ubuntu/user-as-engram/results/scale_eval.json")
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--opt-steps", type=int, default=15)
    parser.add_argument("--opt-lr", type=float, default=0.5)
    parser.add_argument("--n-user-facts", type=int, default=100)
    parser.add_argument("--n-org-facts", type=int, default=100)
    parser.add_argument("--n-test-users", type=int, default=10)
    parser.add_argument("--multi-user-opt", action="store_true", help="Use OPT for E2 (else UNEMBED_P)")
    parser.add_argument("--skip-e1", action="store_true")
    parser.add_argument("--use-salt", action="store_true", help="Use per-user hash salt (E2)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    print(f"Loading model from {args.ckpt_dir}")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    assert config.engram_layer_ids
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    layer_mod = eng.layers_module[str(last_layer)]
    Wv_pinv = torch.linalg.pinv(layer_mod.value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)

    with open(args.corpus) as f:
        corpora = json.load(f)
    user_facts = corpora["user_facts"][:args.n_user_facts]
    org_facts = corpora["org_facts"][:args.n_org_facts]
    multi_users = corpora["multi_users"]

    out = {"ckpt_dir": args.ckpt_dir, "config": vars(args)}

    if not args.skip_e1:
        print(f"\n=== E1: USER facts ({len(user_facts)}) ===")
        user_rows = e1_per_fact(model, tokenizer, user_facts, eng, last_layer, Wv_pinv, total_heads, embed_dim, device, args)
        print(f"\n=== E1: ORG facts ({len(org_facts)}) ===")
        org_rows = e1_per_fact(model, tokenizer, org_facts, eng, last_layer, Wv_pinv, total_heads, embed_dim, device, args)
        out["e1_user"] = user_rows
        out["e1_org"] = org_rows
    else:
        user_rows = []
        org_rows = []

    print(f"\n=== E2: MULTI-USER (test users {args.n_test_users}, use_opt={args.multi_user_opt}) ===")
    multi_results = e2_multi_user(model, tokenizer, multi_users, eng, last_layer, Wv_pinv, total_heads, embed_dim,
                                   device, args, n_test_users=args.n_test_users, use_opt=args.multi_user_opt)
    out["e2_multi_user"] = multi_results

    # ---- Summarise ----
    if user_rows:
        print("\n--- E1 USER summary ---")
        for tag, key_r, key_t in [("baseline", "baseline_rank", None), ("ICL", "icl_rank", "icl_top1"),
                                    ("UNEMBED_P", "unembed_rank", "unembed_top1"),
                                    ("OPT", "opt_rank", "opt_top1")]:
            if key_t is None:
                n = len(user_rows)
                top1 = sum(1 for r in user_rows if r[key_r] == 0)
                top5 = sum(1 for r in user_rows if r[key_r] < 5)
                top10 = sum(1 for r in user_rows if r[key_r] < 10)
                rank_med = sorted(r[key_r] for r in user_rows)[n//2] if n else 0
            else:
                agg = aggregate(user_rows, key_r, key_t); n=agg['n']; top1=agg['top1']; top5=agg['top5']; top10=agg['top10']; rank_med=agg['rank_median']
            print(f"  {tag:10s}  top1: {top1}/{n}={top1/n:.2%}  top5: {top5}/{n}={top5/n:.2%}  top10: {top10}/{n}={top10/n:.2%}  median rank: {rank_med}")
    if org_rows:
        print("\n--- E1 ORG summary ---")
        for tag, key_r, key_t in [("baseline", "baseline_rank", None), ("ICL", "icl_rank", "icl_top1"),
                                    ("UNEMBED_P", "unembed_rank", "unembed_top1"),
                                    ("OPT", "opt_rank", "opt_top1")]:
            if key_t is None:
                n = len(org_rows)
                top1 = sum(1 for r in org_rows if r[key_r] == 0)
                top5 = sum(1 for r in org_rows if r[key_r] < 5)
                top10 = sum(1 for r in org_rows if r[key_r] < 10)
                rank_med = sorted(r[key_r] for r in org_rows)[n//2] if n else 0
            else:
                agg = aggregate(org_rows, key_r, key_t); n=agg['n']; top1=agg['top1']; top5=agg['top5']; top10=agg['top10']; rank_med=agg['rank_median']
            print(f"  {tag:10s}  top1: {top1}/{n}={top1/n:.2%}  top5: {top5}/{n}={top5/n:.2%}  top10: {top10}/{n}={top10/n:.2%}  median rank: {rank_med}")

    print("\n--- E2 multi-user summary ---")
    if multi_results:
        n_users = len(multi_results)
        avg_recall_baseline = sum(r['recall_top1_baseline'] for r in multi_results) / n_users
        avg_recall_post = sum(r['recall_top1_post'] for r in multi_results) / n_users
        avg_leak = sum(r['leak_rate'] for r in multi_results) / n_users
        print(f"  test users: {n_users}, facts/user: {multi_results[0]['n_facts']}")
        print(f"  avg own-fact top-1 recall: baseline {avg_recall_baseline:.3f}  post-insert {avg_recall_post:.3f}")
        print(f"  avg cross-user leak rate:  {avg_leak:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
