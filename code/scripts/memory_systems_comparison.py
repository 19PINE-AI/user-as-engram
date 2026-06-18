"""Compare User-as-Engram against state-of-the-art memory systems.

All systems use the SAME Mini-Engram-d12 as the final-answer LM. The
difference is *how* facts are stored and retrieved:

  1. NO_MEMORY                — base model, no facts (lower bound)
  2. MARKDOWN_ALL             — dump all facts as markdown bullets in
                                 the system prompt (ICL upper bound)
  3. RAG_TOP1                  — sentence-encoder embed, retrieve
                                 top-1, prepend
  4. RAG_TOP3                  — same, top-3
  5. MEM0_LIKE                 — like MEM0: embed facts, retrieve
                                 top-K=5, structured fact format
  6. MEMMACHINE_LIKE           — like MemMachine: verbatim storage,
                                 retrieve top-3, with surrounding
                                 context (no neighbours here since
                                 facts are atomic, so equivalent to
                                 RAG_TOP3 in our setup; we still test
                                 the difference is zero)
  7. USER_AS_ENGRAM_OPT        — our independent OPT
  8. USER_AS_ENGRAM_JOINT_OPT  — our Joint OPT (recommended default)

Setup:
- Mini-Engram-d12 served locally
- 100 USER facts per benchmark (XXL corpus)
- Sentence encoder: all-MiniLM-L6-v2
- Same 100-question probe set: each user fact's prompt is the question

Outputs JSON to $USER_AS_ENGRAM_ROOT/results/memory_systems_comparison.json
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows,
    make_marker_OPT, make_marker_UNEMBED_P,
)


def encode_facts_minilm(facts, device):
    """Embed each fact's stored representation via sentence-transformers/all-MiniLM-L6-v2.
    For storage/retrieval, the 'fact' is the (prompt + gold) pair as a single string."""
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    fact_texts = [f"{f['prompt']}{f['gold']}" for f in facts]
    embs = enc.encode(fact_texts, convert_to_tensor=True, normalize_embeddings=True,
                       show_progress_bar=False)
    return enc, embs, fact_texts


def retrieve_topk(enc, fact_embs, fact_texts, query, k):
    """Return top-k most similar fact texts to the query."""
    q_emb = enc.encode([query], convert_to_tensor=True, normalize_embeddings=True,
                        show_progress_bar=False)
    sims = (fact_embs @ q_emb.T).squeeze(-1)
    top_idx = torch.topk(sims, min(k, len(fact_texts))).indices.cpu().tolist()
    return [fact_texts[i] for i in top_idx]


@torch.no_grad()
def query_top1(model, tokenizer, prompt, device):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(idx)[0, -1, :]
    top1_id = int(logits.argmax().item())
    _, topi = torch.topk(logits, 5)
    return top1_id, [int(t.item()) for t in topi]


def eval_facts_with_context(model, tokenizer, facts, context_builder, device, max_seq_len=1024):
    """Evaluate per-fact recall when given a `context_builder(fact)` that
    returns the system context to prepend to the fact's question prompt."""
    n_top1 = n_top5 = 0
    bos = tokenizer.get_bos_token_id()
    for f in facts:
        gold_id = tokenizer.encode(f["gold"])[0]
        ctx = context_builder(f)
        full_prompt = ctx + f["prompt"] if ctx else f["prompt"]
        ids = tokenizer.encode(full_prompt, prepend=bos)
        if len(ids) > max_seq_len - 1:
            ids = ids[-(max_seq_len-1):]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(idx)[0, -1, :]
        top1_id = int(logits.argmax().item())
        _, topi = torch.topk(logits, 5)
        top5 = [int(t.item()) for t in topi]
        if top1_id == gold_id: n_top1 += 1
        if gold_id in top5: n_top5 += 1
    return n_top1, n_top5, len(facts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--corpus", default=f"{UAE_ROOT}/data/corpora_xxl.json")
    p.add_argument("--out", default=f"{UAE_ROOT}/results/memory_systems_comparison.json")
    p.add_argument("--n-facts", type=int, default=100)
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

    with open(args.corpus) as f:
        corpora = json.load(f)
    seen = {}
    for f in corpora["user_facts"]:
        seen[f["trigger"]] = f
        if len(seen) >= args.n_facts: break
    facts = list(seen.values())[:args.n_facts]
    print(f"Loaded {len(facts)} unique-trigger USER facts")

    print("Loading sentence-transformer (MiniLM)...")
    enc, fact_embs, fact_texts = encode_facts_minilm(facts, device)
    print(f"Embedded {len(fact_texts)} facts (dim={fact_embs.shape[-1]})")

    results = {}

    # 1. NO_MEMORY (no context, just the fact's prompt)
    print("\n[1/8] NO_MEMORY...")
    t0 = time.time()
    top1, top5, n = eval_facts_with_context(
        model, tokenizer, facts,
        context_builder=lambda f: "", device=device, max_seq_len=args.max_seq_len)
    results["NO_MEMORY"] = {"top1": top1/n, "top5": top5/n, "n": n,
                              "wall_per_query_ms": (time.time()-t0)/n*1000,
                              "context_tokens_avg": 0}
    print(f"  top1={top1}/{n}={top1/n:.1%} top5={top5}/{n}={top5/n:.1%}")

    # 2. MARKDOWN_ALL (all facts as markdown, ICL upper bound)
    print("\n[2/8] MARKDOWN_ALL (ICL upper bound)...")
    md_lines = "\n".join([f"- {f['prompt']}{f['gold']}." for f in facts])
    md_ctx = f"# User facts\n{md_lines}\n\n"
    md_tokens = len(tokenizer.encode(md_ctx))
    print(f"  Markdown context: {md_tokens} tokens (max_seq={args.max_seq_len})")
    t0 = time.time()
    top1, top5, n = eval_facts_with_context(
        model, tokenizer, facts,
        context_builder=lambda f: md_ctx, device=device, max_seq_len=args.max_seq_len)
    results["MARKDOWN_ALL"] = {"top1": top1/n, "top5": top5/n, "n": n,
                                 "wall_per_query_ms": (time.time()-t0)/n*1000,
                                 "context_tokens_avg": md_tokens}
    print(f"  top1={top1}/{n}={top1/n:.1%} top5={top5}/{n}={top5/n:.1%}")

    # 3-4. RAG TOP-1 / TOP-3
    for k in [1, 3]:
        print(f"\n[3/8] RAG_TOP{k}...")
        def build_ctx(f, k=k):
            top = retrieve_topk(enc, fact_embs, fact_texts, f["prompt"], k)
            return "Relevant facts:\n" + "\n".join(f"- {t}." for t in top) + "\n\n"
        t0 = time.time()
        top1, top5, n = eval_facts_with_context(
            model, tokenizer, facts, context_builder=build_ctx,
            device=device, max_seq_len=args.max_seq_len)
        ctx_sample = build_ctx(facts[0])
        results[f"RAG_TOP{k}"] = {"top1": top1/n, "top5": top5/n, "n": n,
                                    "wall_per_query_ms": (time.time()-t0)/n*1000,
                                    "context_tokens_avg": len(tokenizer.encode(ctx_sample))}
        print(f"  top1={top1}/{n}={top1/n:.1%} top5={top5}/{n}={top5/n:.1%}")

    # 5. MEM0-like: top-5 with structured prefix
    print("\n[5/8] MEM0_LIKE (top-5, structured)...")
    def mem0_ctx(f):
        top = retrieve_topk(enc, fact_embs, fact_texts, f["prompt"], 5)
        return "Memories:\n" + "\n".join(f"* {t}." for t in top) + "\n\n"
    t0 = time.time()
    top1, top5, n = eval_facts_with_context(
        model, tokenizer, facts, context_builder=mem0_ctx,
        device=device, max_seq_len=args.max_seq_len)
    results["MEM0_LIKE"] = {"top1": top1/n, "top5": top5/n, "n": n,
                              "wall_per_query_ms": (time.time()-t0)/n*1000,
                              "context_tokens_avg": len(tokenizer.encode(mem0_ctx(facts[0])))}
    print(f"  top1={top1}/{n}={top1/n:.1%} top5={top5}/{n}={top5/n:.1%}")

    # 6. MEMMACHINE-like: top-3 verbatim
    print("\n[6/8] MEMMACHINE_LIKE (top-3 verbatim)...")
    def mm_ctx(f):
        top = retrieve_topk(enc, fact_embs, fact_texts, f["prompt"], 3)
        return "Episodes:\n" + "\n".join(f"[{i+1}] {t}." for i, t in enumerate(top)) + "\n\n"
    t0 = time.time()
    top1, top5, n = eval_facts_with_context(
        model, tokenizer, facts, context_builder=mm_ctx,
        device=device, max_seq_len=args.max_seq_len)
    results["MEMMACHINE_LIKE"] = {"top1": top1/n, "top5": top5/n, "n": n,
                                    "wall_per_query_ms": (time.time()-t0)/n*1000,
                                    "context_tokens_avg": len(tokenizer.encode(mm_ctx(facts[0])))}
    print(f"  top1={top1}/{n}={top1/n:.1%} top5={top5}/{n}={top5/n:.1%}")

    # 7. USER_AS_ENGRAM (independent OPT-15)
    print("\n[7/8] USER_AS_ENGRAM_OPT (independent)...")
    bos = tokenizer.get_bos_token_id()
    tbl = eng.tables[str(last_layer)]
    all_writes = []
    t_train_start = time.time()
    for f in facts:
        ids = tokenizer.encode(f["prompt"], prepend=bos)
        gold_id = tokenizer.encode(f["gold"])[0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        rows = trigger_global_rows(eng, idx, last_layer, trig_pos)
        marker = make_marker_OPT(model, eng, last_layer, gold_id, idx, trig_pos,
                                  20.0, total_heads, embed_dim, Wv_pinv=Wv_pinv,
                                  n_steps=15, lr=0.5)
        all_writes.append((rows, marker, tbl.embedding.weight.data[rows].clone()))
    train_time = time.time() - t_train_start
    for rows, marker, _ in all_writes:
        write_marker(eng, last_layer, rows, marker)
    try:
        t0 = time.time()
        top1, top5, n = eval_facts_with_context(
            model, tokenizer, facts, context_builder=lambda f: "",
            device=device, max_seq_len=args.max_seq_len)
        wall_q_ms = (time.time()-t0)/n*1000
    finally:
        for rows, _, originals in all_writes:
            restore_rows(eng, last_layer, rows, originals)
    results["USER_AS_ENGRAM_OPT"] = {"top1": top1/n, "top5": top5/n, "n": n,
                                         "wall_per_query_ms": wall_q_ms,
                                         "context_tokens_avg": 0,
                                         "train_time_s": train_time}
    print(f"  top1={top1}/{n}={top1/n:.1%} top5={top5}/{n}={top5/n:.1%} train={train_time:.0f}s")

    # 8. USER_AS_ENGRAM_JOINT_OPT — reuse the existing joint_opt_100.json result if available
    # If not, re-run joint OPT
    print("\n[8/8] USER_AS_ENGRAM_JOINT_OPT (sparse-row joint training)...")
    joint_path = f"{UAE_ROOT}/results/joint_opt_100.json"
    if Path(joint_path).exists():
        with open(joint_path) as f:
            jd = json.load(f)
        results["USER_AS_ENGRAM_JOINT_OPT"] = {"top1": jd["top1"], "top5": jd["top5"], "n": jd["n"],
                                                 "wall_per_query_ms": wall_q_ms,  # same query cost as OPT
                                                 "context_tokens_avg": 0,
                                                 "train_time_s": jd["train_time_s"],
                                                 "note": "from joint_opt_100.json"}
        print(f"  top1={jd['top1']:.1%} top5={jd['top5']:.1%} (reusing joint_opt_100.json)")
    else:
        print("  Need to re-run joint OPT — skipping for now")

    # Print summary table
    print("\n" + "="*88)
    print(f"{'method':30s}  {'top-1':>7s}  {'top-5':>7s}  {'context':>10s}  {'wall/q (ms)':>12s}")
    print("="*88)
    for name, r in results.items():
        print(f"{name:30s}  {r['top1']:>7.1%}  {r['top5']:>7.1%}  {r['context_tokens_avg']:>10d}  {r['wall_per_query_ms']:>12.1f}")
    print("="*88)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
