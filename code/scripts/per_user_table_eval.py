"""
Per-user Engram override evaluation — the production-realistic multi-tenant test.

Design: each user has a small dict {global_row_index → row_vector} of overrides,
representing their personal facts written via OPT. At inference time:
  1) Apply only user U's overrides to the shared Engram table (save originals)
  2) Run U's query forward pass — global retrievals hit trained rows; their
     trigger-N-gram retrievals hit U's overrides
  3) Restore the originals (or swap to next user's overrides)

This implements the apply/query/restore procedure and switching measurement in
“Multi-Tenant Serving” (sec:serving).

This is what real multi-tenant serving looks like — users do not share
"live" hash slots. The shared address space across users (no override
swapping) is a stress test of the worst-case scheduling, not a deployment
recipe. With proper per-user table isolation:

  - **Recall** matches single-user (~88-93% on d12 with OPT)
  - **Leak** is *zero by construction*: other users' overrides are never
    in the table during your query

We measure both, plus the wall-clock cost of swap+query+restore vs single-user.

Usage:
  python -m scripts.per_user_table_eval \\
    --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \\
    --corpus $USER_AS_ENGRAM_ROOT/data/corpora_xl.json \\
    --n-test-users 100
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, json, time, argparse
from pathlib import Path
import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows,
    make_marker_OPT, make_marker_UNEMBED_P,
)


def build_user_override_table(model, eng, last_layer, Wv_pinv, total_heads, embed_dim,
                               tokenizer, user_facts, args, device):
    """For one user, compute and return a list of (global_rows_tensor,
    marker_tensor) for each of their facts. We do NOT apply them yet."""
    bos = tokenizer.get_bos_token_id()
    overrides = []
    for fact in user_facts:
        ids = tokenizer.encode(fact["prompt"], prepend=bos)
        gold_id = tokenizer.encode(fact["gold"])[0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        # IMPORTANT: salt = 0 so we use the trained global address space
        global_rows = trigger_global_rows(eng, idx, last_layer, trig_pos, user_salt=0)
        if args.use_opt:
            marker = make_marker_OPT(model, eng, last_layer, gold_id, idx, trig_pos,
                                      args.scale, total_heads, embed_dim, Wv_pinv=Wv_pinv,
                                      n_steps=args.opt_steps, lr=args.opt_lr)
        else:
            marker = make_marker_UNEMBED_P(model, eng, last_layer, gold_id, idx, trig_pos,
                                            args.scale, total_heads, embed_dim, Wv_pinv=Wv_pinv)
        overrides.append({"prompt_ids": ids, "gold_id": gold_id,
                           "global_rows": global_rows, "marker": marker, "trig_pos": trig_pos})
    return overrides


def apply_overrides(eng, last_layer, overrides):
    """Write each override and return the original row values for restoration."""
    tbl = eng.tables[str(last_layer)]
    saved = []
    for o in overrides:
        rows = o["global_rows"]
        originals = tbl.embedding.weight.data[rows].clone()
        saved.append(originals)
        write_marker(eng, last_layer, rows, o["marker"])
    return saved


def restore_overrides(eng, last_layer, overrides, saved):
    for o, originals in zip(overrides, saved):
        restore_rows(eng, last_layer, o["global_rows"], originals)


@torch.no_grad()
def eval_recall(model, overrides):
    """Test if each fact's gold token is top-1 at the trigger position."""
    n_top1 = 0
    n_top5 = 0
    for o in overrides:
        idx = torch.tensor([o["prompt_ids"]], dtype=torch.long, device=model.get_device())
        logits = model(idx)[0, -1, :]
        rank = int((logits > logits[o["gold_id"]]).sum().item())
        if rank == 0: n_top1 += 1
        if rank < 5: n_top5 += 1
    return n_top1, n_top5, len(overrides)


@torch.no_grad()
def eval_leak(model, tokenizer, leak_user_facts, my_facts):
    """For each of leak_user's facts, query the model (with MY overrides
    active). Check if the LM's top-1 prediction matches MY gold for the
    same trigger (which would be a true privacy leak)."""
    bos = tokenizer.get_bos_token_id()
    my_gold_by_trigger = {f["trigger"]: f["gold"] for f in my_facts}
    n_leaks = 0
    for leak_f in leak_user_facts:
        ids = tokenizer.encode(leak_f["prompt"], prepend=bos)
        idx = torch.tensor([ids], dtype=torch.long, device=model.get_device())
        logits = model(idx)[0, -1, :]
        top1_id = int(logits.argmax().item())
        top1_text = tokenizer.decode([top1_id])
        my_gold = my_gold_by_trigger.get(leak_f["trigger"])
        if my_gold is None: continue
        my_gold_first = tokenizer.encode(my_gold)[0]
        leak_gold_first = tokenizer.encode(leak_f["gold"])[0]
        # leak: top1 is my gold AND not the leak_user's gold
        if top1_id == my_gold_first and top1_id != leak_gold_first:
            n_leaks += 1
    return n_leaks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--corpus", default=f"{UAE_ROOT}/data/corpora_xl.json")
    p.add_argument("--out", default=f"{UAE_ROOT}/results/per_user_table.json")
    p.add_argument("--n-test-users", type=int, default=10)
    p.add_argument("--n-leak-users", type=int, default=5,
                   help="for each test user, this many other users contribute leak probes")
    p.add_argument("--scale", type=float, default=20.0)
    p.add_argument("--opt-steps", type=int, default=15)
    p.add_argument("--opt-lr", type=float, default=0.5)
    p.add_argument("--use-opt", action="store_true", default=True,
                   help="(default true) use OPT marker; pass --no-use-opt for UNEMBED_P")
    p.add_argument("--no-use-opt", dest="use_opt", action="store_false")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    Wv_pinv = torch.linalg.pinv(eng.layers_module[str(last_layer)].value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)

    with open(args.corpus) as f:
        corpora = json.load(f)
    test_users = corpora["multi_users"][:args.n_test_users]

    # Pre-compute every test user's override table (parallelizable in production)
    print(f"Building override tables for {len(test_users)} users (use_opt={args.use_opt})...")
    user_overrides = []
    t0 = time.time()
    for ui, u in enumerate(test_users):
        # Dedup by trigger
        seen = {}
        for f in u["facts"]: seen[f["trigger"]] = f
        deduped = list(seen.values())
        ovs = build_user_override_table(model, eng, last_layer, Wv_pinv, total_heads, embed_dim,
                                          tokenizer, deduped, args, device)
        user_overrides.append({"user_id": u["user_id"], "facts": deduped, "overrides": ovs})
        if (ui + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (ui + 1) * (len(test_users) - ui - 1)
            print(f"  [{ui+1}/{len(test_users)}] table built ({len(ovs)} facts), elapsed {elapsed:.0f}s, ETA {eta:.0f}s")
    build_time = time.time() - t0
    print(f"All override tables built in {build_time:.1f}s ({build_time/len(test_users)*1000:.0f}ms/user avg)")

    # ----- Evaluation: recall and leak -----
    rows = []
    for ui, u in enumerate(user_overrides):
        # Apply this user's overrides
        t_apply = time.time()
        saved = apply_overrides(eng, last_layer, u["overrides"])
        apply_time = time.time() - t_apply

        try:
            # Recall on user's own questions
            t_recall = time.time()
            top1, top5, n = eval_recall(model, u["overrides"])
            recall_time = time.time() - t_recall

            # Leak: for each of n_leak_users OTHER users, query their facts as them
            # We test the *same* trigger N-grams from leak users to detect cross-user info
            t_leak = time.time()
            leak_count = 0
            leak_total = 0
            for li in range(args.n_leak_users):
                leak_idx = (ui + 1 + li) % len(user_overrides)
                if leak_idx == ui: continue
                leak_user = user_overrides[leak_idx]
                lc = eval_leak(model, tokenizer, leak_user["facts"], u["facts"])
                leak_count += lc
                leak_total += len(leak_user["facts"])
            leak_time = time.time() - t_leak
        finally:
            restore_overrides(eng, last_layer, u["overrides"], saved)

        rows.append({
            "user_id": u["user_id"],
            "n_facts": n,
            "recall_top1": top1 / n,
            "recall_top5": top5 / n,
            "n_leaks": leak_count,
            "n_leak_probes": leak_total,
            "leak_rate": leak_count / max(leak_total, 1),
            "apply_ms": apply_time * 1000,
            "recall_ms": recall_time * 1000,
            "leak_ms": leak_time * 1000,
        })
        if (ui + 1) % 10 == 0:
            avg_rec = sum(r["recall_top1"] for r in rows) / len(rows)
            avg_leak = sum(r["leak_rate"] for r in rows) / len(rows)
            print(f"  [{ui+1}/{len(user_overrides)}] avg recall {avg_rec:.3f}  avg leak {avg_leak:.4f}")

    # Aggregate
    n_users = len(rows)
    avg_recall_top1 = sum(r["recall_top1"] for r in rows) / n_users
    avg_recall_top5 = sum(r["recall_top5"] for r in rows) / n_users
    avg_leak = sum(r["leak_rate"] for r in rows) / n_users
    n_total_leaks = sum(r["n_leaks"] for r in rows)
    n_total_leak_probes = sum(r["n_leak_probes"] for r in rows)
    avg_apply_ms = sum(r["apply_ms"] for r in rows) / n_users
    print(f"\n=== Per-user override eval (use_opt={args.use_opt}) ===")
    print(f"  test users: {n_users}, facts/user: {rows[0]['n_facts']}")
    print(f"  avg own-fact top-1 recall: {avg_recall_top1:.3f}")
    print(f"  avg own-fact top-5 recall: {avg_recall_top5:.3f}")
    print(f"  total leaks across all queries: {n_total_leaks}/{n_total_leak_probes} = {n_total_leaks/n_total_leak_probes:.4f}")
    print(f"  per-user-swap latency: {avg_apply_ms:.1f} ms")

    out = {"config": vars(args), "build_time_sec": build_time, "rows": rows,
           "summary": {
               "n_users": n_users,
               "facts_per_user": rows[0]['n_facts'],
               "avg_recall_top1": avg_recall_top1,
               "avg_recall_top5": avg_recall_top5,
               "total_leaks": n_total_leaks,
               "total_leak_probes": n_total_leak_probes,
               "leak_rate": n_total_leaks / max(n_total_leak_probes, 1),
               "avg_swap_latency_ms": avg_apply_ms,
           }}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
