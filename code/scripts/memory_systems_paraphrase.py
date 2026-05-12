"""Paraphrase variant of memory_systems_comparison.

Use the same 4 facts × 5 paraphrase queries from paraphrase_test.py
(plus more) but query the system with the PARAPHRASE rather than the
fact's literal trigger. This tests whether RAG retrieval still wins
when the query surface form differs from the stored fact.

Same 8 systems as memory_systems_comparison.py.
"""
import os, json, time, argparse
from pathlib import Path
import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows, make_marker_OPT,
)
from scripts.memory_systems_comparison import (
    encode_facts_minilm, retrieve_topk, eval_facts_with_context,
)


# 16 facts (USER + ORG style), each with 4 paraphrased queries (different
# surface form from the stored fact). The "stored fact" is what we put in
# memory; the "query" is what the user actually asks.
PARAPHRASE_FACTS = [
    # (stored_prompt, gold, [paraphrase_queries])
    ("My doctor's name is Dr.", " Patel",
        ["My physician is Dr.", "Who treats me when I'm sick? Dr.",
         "I see Dr.", "My GP is Dr."]),
    ("My favorite spice is", " saffron",
        ["I love the spice", "My preferred spice is",
         "The spice I like most is", "My go-to spice is"]),
    ("My pet is a", " ferret",
        ["I have a pet", "My household animal is a",
         "The animal I keep is a", "I have a pet which is a"]),
    ("I live in", " Portland",
        ["I reside in", "My city is",
         "I'm based in", "Home for me is"]),
    ("My favorite color is", " charcoal",
        ["I prefer the color", "My preferred color is",
         "The color I like best is", "My color choice is"]),
    ("My gym day is", " Tuesday",
        ["I go to the gym on", "Workout day for me is",
         "I exercise on", "My fitness day is"]),
    ("My instrument is the", " piano",
        ["I play the", "My music tool is the",
         "The instrument I learned is the", "My musical instrument is the"]),
    ("My favorite drink is", " matcha",
        ["I love drinking", "My preferred beverage is",
         "The drink I always order is", "I usually drink"]),
    # ORG-style
    ("Globex office hours start at", " 9",
        ["Globex opens at", "What time does Globex open? At",
         "Globex's day begins at", "Globex starts work at"]),
    ("Globex headquarters is in", " Manhattan",
        ["Globex's HQ is in", "Globex is based in",
         "Globex's main office is in", "The Globex headquarters is in"]),
    ("Globex IT support extension is", " 4400",
        ["Globex's IT line is", "Reach Globex IT at extension",
         "Globex tech support's extension is", "Call Globex IT at"]),
    ("Globex CEO emeritus is", " Bruce",
        ["Globex's former CEO is", "Globex's previous chief executive was",
         "The retired Globex CEO is", "Globex's emeritus chief is"]),
    ("Globex's mascot animal is the", " otter",
        ["Globex's mascot is the", "The Globex animal is the",
         "Globex's symbol animal is the", "Globex represents itself with the"]),
    ("Globex's fiscal year starts in", " April",
        ["Globex's FY begins in", "Globex's accounting year starts in",
         "When does Globex's fiscal year begin? In", "Globex's financial year opens in"]),
    ("Globex's monthly all-hands is on the first", " Tuesday",
        ["Globex's company meeting is on the first", "Monthly Globex sync happens on the first",
         "Globex's all-hands is the first", "Each month, Globex meets on the first"]),
    ("Globex's customer support email starts with", " support",
        ["Globex's help email begins with", "Globex's CS email is",
         "Email Globex CS at", "Reach Globex's customer team at"]),
]


def eval_paraphrase_with_context(model, tokenizer, fact_data, context_builder, device, max_seq_len=1024):
    """For each (fact, paraphrase_queries), use context built from the FACT
    but query with each PARAPHRASE. Average per-fact recall."""
    bos = tokenizer.get_bos_token_id()
    n_top1 = n_top5 = total = 0
    for fd in fact_data:
        gold_id = tokenizer.encode(fd["gold"])[0]
        ctx = context_builder(fd)
        for query in fd["paraphrases"]:
            full_prompt = ctx + query if ctx else query
            ids = tokenizer.encode(full_prompt, prepend=bos)
            if len(ids) > max_seq_len - 1: ids = ids[-(max_seq_len-1):]
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(idx)[0, -1, :]
            top1_id = int(logits.argmax().item())
            _, topi = torch.topk(logits, 5)
            top5 = [int(t.item()) for t in topi]
            if top1_id == gold_id: n_top1 += 1
            if gold_id in top5: n_top5 += 1
            total += 1
    return n_top1, n_top5, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", default="/home/ubuntu/user-as-engram/results/memory_systems_paraphrase.json")
    p.add_argument("--max-seq-len", type=int, default=1024)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    print("Loading Mini-Engram-d12...")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    Wv_pinv = torch.linalg.pinv(eng.layers_module[str(last_layer)].value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)

    # Build fact data for sentence-encoder retrieval
    fact_data = []
    facts_for_enc = []
    for stored, gold, paraphrases in PARAPHRASE_FACTS:
        fact_data.append({
            "prompt": stored, "gold": gold,
            "paraphrases": paraphrases,
            "stored_text": f"{stored}{gold}.",
        })
        facts_for_enc.append({"prompt": stored, "gold": gold})

    enc, fact_embs, fact_texts = encode_facts_minilm(facts_for_enc, device)
    print(f"Embedded {len(fact_texts)} facts; {len(fact_data)*4} paraphrase queries total")

    results = {}

    # NO_MEMORY
    print("\n[1/8] NO_MEMORY")
    top1, top5, n = eval_paraphrase_with_context(
        model, tokenizer, fact_data, lambda fd: "", device, args.max_seq_len)
    results["NO_MEMORY"] = {"top1": top1/n, "top5": top5/n, "n": n}
    print(f"  top1={top1}/{n}={top1/n:.1%}")

    # MARKDOWN_ALL
    print("\n[2/8] MARKDOWN_ALL")
    md = "# Facts\n" + "\n".join(f"- {fd['stored_text']}" for fd in fact_data) + "\n\n"
    md_tok = len(tokenizer.encode(md))
    top1, top5, n = eval_paraphrase_with_context(
        model, tokenizer, fact_data, lambda fd: md, device, args.max_seq_len)
    results["MARKDOWN_ALL"] = {"top1": top1/n, "top5": top5/n, "n": n,
                                 "context_tokens": md_tok}
    print(f"  top1={top1}/{n}={top1/n:.1%} ctx={md_tok}")

    # RAG_TOP1, TOP3, MEM0_LIKE (top-5), MEMMACHINE_LIKE (top-3)
    for tag, k in [("RAG_TOP1", 1), ("RAG_TOP3", 3), ("MEM0_LIKE", 5), ("MEMMACHINE_LIKE", 3)]:
        print(f"\n[*/8] {tag}")
        def ctx_for(fd, _k=k, _tag=tag):
            # Use the FACT's stored prompt as the retrieval probe (RAG-style)
            # But here we vary the QUERY, so we use the paraphrase as the retrieval probe
            # to be fair to RAG. However the paraphrase is per-query, not per-fact.
            # Use the first paraphrase as the retrieval probe (the eval will average).
            # Actually we should use the SPECIFIC paraphrase that's being queried.
            # Refactor: use the per-query retrieval inside the eval loop.
            return None  # signal: per-query retrieval
        # Custom eval that does per-query retrieval
        n_top1 = n_top5 = total = 0
        bos = tokenizer.get_bos_token_id()
        for fd in fact_data:
            gold_id = tokenizer.encode(fd["gold"])[0]
            for query in fd["paraphrases"]:
                top = retrieve_topk(enc, fact_embs, fact_texts, query, k)
                if tag == "MEM0_LIKE":
                    ctx = "Memories:\n" + "\n".join(f"* {t}." for t in top) + "\n\n"
                elif tag == "MEMMACHINE_LIKE":
                    ctx = "Episodes:\n" + "\n".join(f"[{i+1}] {t}." for i, t in enumerate(top)) + "\n\n"
                else:
                    ctx = "Relevant facts:\n" + "\n".join(f"- {t}." for t in top) + "\n\n"
                full_prompt = ctx + query
                ids = tokenizer.encode(full_prompt, prepend=bos)
                if len(ids) > args.max_seq_len - 1: ids = ids[-(args.max_seq_len-1):]
                idx = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(idx)[0, -1, :]
                top1_id = int(logits.argmax().item())
                _, topi = torch.topk(logits, 5)
                top5 = [int(t.item()) for t in topi]
                if top1_id == gold_id: n_top1 += 1
                if gold_id in top5: n_top5 += 1
                total += 1
        results[tag] = {"top1": n_top1/total, "top5": n_top5/total, "n": total}
        print(f"  top1={n_top1}/{total}={n_top1/total:.1%}")

    # USER_AS_ENGRAM_OPT (insert at the STORED prompt; query with paraphrase)
    print("\n[7/8] USER_AS_ENGRAM_OPT")
    bos = tokenizer.get_bos_token_id()
    tbl = eng.tables[str(last_layer)]
    all_writes = []
    for fd in fact_data:
        ids = tokenizer.encode(fd["prompt"], prepend=bos)
        gold_id = tokenizer.encode(fd["gold"])[0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        rows = trigger_global_rows(eng, idx, last_layer, trig_pos)
        marker = make_marker_OPT(model, eng, last_layer, gold_id, idx, trig_pos,
                                  20.0, total_heads, embed_dim, Wv_pinv=Wv_pinv,
                                  n_steps=15, lr=0.5)
        all_writes.append((rows, marker, tbl.embedding.weight.data[rows].clone()))
    for rows, marker, _ in all_writes:
        write_marker(eng, last_layer, rows, marker)
    try:
        top1, top5, n = eval_paraphrase_with_context(
            model, tokenizer, fact_data, lambda fd: "", device, args.max_seq_len)
    finally:
        for rows, _, originals in all_writes:
            restore_rows(eng, last_layer, rows, originals)
    results["USER_AS_ENGRAM_OPT"] = {"top1": top1/n, "top5": top5/n, "n": n}
    print(f"  top1={top1}/{n}={top1/n:.1%}")

    # USER_AS_ENGRAM_OPT (multi-trigger: insert at stored AND each paraphrase)
    print("\n[8/8] USER_AS_ENGRAM_OPT_MULTITRIG (insert at every paraphrase)")
    all_writes = []
    for fd in fact_data:
        gold_id = tokenizer.encode(fd["gold"])[0]
        for trigger in [fd["prompt"]] + fd["paraphrases"]:
            ids = tokenizer.encode(trigger, prepend=bos)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            trig_pos = len(ids) - 1
            rows = trigger_global_rows(eng, idx, last_layer, trig_pos)
            marker = make_marker_OPT(model, eng, last_layer, gold_id, idx, trig_pos,
                                      20.0, total_heads, embed_dim, Wv_pinv=Wv_pinv,
                                      n_steps=15, lr=0.5)
            all_writes.append((rows, marker, tbl.embedding.weight.data[rows].clone()))
    # All writes are now in the list; apply them
    for rows, marker, _ in all_writes:
        write_marker(eng, last_layer, rows, marker)
    try:
        top1, top5, n = eval_paraphrase_with_context(
            model, tokenizer, fact_data, lambda fd: "", device, args.max_seq_len)
    finally:
        for rows, _, originals in all_writes:
            restore_rows(eng, last_layer, rows, originals)
    results["USER_AS_ENGRAM_OPT_MULTITRIG"] = {"top1": top1/n, "top5": top5/n, "n": n}
    print(f"  top1={top1}/{n}={top1/n:.1%}")

    print("\n" + "="*70)
    print(f"{'method':35s}  {'top-1':>8s}  {'top-5':>8s}")
    print("="*70)
    for k, v in results.items():
        print(f"{k:35s}  {v['top1']:>8.1%}  {v['top5']:>8.1%}")
    print("="*70)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
