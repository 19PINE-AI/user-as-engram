"""
RAG conditions for the multi-hop-reasoning probe (Section sec:multihop).

Same 8 chained-fact pairs as scripts.multihop_probe. For each pair, the
fact KB contains the 2 facts of *that* item plus the 14 facts from the
other 7 items as distractors (16 total) -- this makes retrieval realistic
instead of trivially returning both facts.

Conditions:
  RAG_TOP1            -- retrieve top-1 fact, prepend, no shared LoRA
  RAG_TOP2            -- retrieve top-2 facts, prepend, no shared LoRA
  RAG_ALL             -- all 16 facts in context, no shared LoRA
  RAG_TOP2_sharedLoRA -- retrieve top-2, prepend, with shared LoRA r=16

Reports multi-hop top-1, top-5 (first-token match), top-2-retrieval
oracle hit-rate, and avg context tokens.
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, json, argparse, time
from pathlib import Path

import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import load_model
from scripts.sft_baseline import attach_lora, detach_lora
from scripts.multihop_probe import MULTIHOP_FACTS
from scripts.layered_architecture import attach_shared_lora, lora_freeze


def fact_text(prompt, gold):
    return (prompt.rstrip() + gold.rstrip()).strip() + "."


def encode_facts(enc, texts):
    return enc.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                      show_progress_bar=False)


def retrieve(enc, embs, texts, query, k):
    q = enc.encode([query], convert_to_tensor=True, normalize_embeddings=True,
                   show_progress_bar=False)
    sims = (embs @ q.T).squeeze(-1)
    topi = torch.topk(sims, min(k, len(texts))).indices.cpu().tolist()
    return [texts[i] for i in topi], topi


def build_context(retrieved):
    if not retrieved:
        return ""
    return "Facts: " + " ".join(r.rstrip(".") + "." for r in retrieved) + " "


@torch.no_grad()
def query_top1_top5(model, tokenizer, prompt, device, max_seq_len=1024):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    if len(ids) > max_seq_len - 1:
        ids = ids[-(max_seq_len-1):]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(x)[0, -1, :]
    top1_id = int(logits.argmax().item())
    _, topi = torch.topk(logits, 5)
    top5 = [int(t.item()) for t in topi]
    return top1_id, top5, len(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--shared-lora-dir", required=True)
    p.add_argument("--out", default=f"{UAE_ROOT}/results/multihop_rag.json")
    p.add_argument("--max-seq-len", type=int, default=1024)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    model.eval()

    print("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                              device=device)

    # Build a flat KB of all 16 facts; remember which 2 belong to each item.
    all_texts = []
    item_owners = []  # parallel: which item index does this fact belong to
    for fi, item in enumerate(MULTIHOP_FACTS):
        for (prompt, gold) in item["facts"]:
            all_texts.append(fact_text(prompt, gold))
            item_owners.append(fi)
    all_embs = encode_facts(enc, all_texts)
    print(f"KB: {len(all_texts)} facts indexed")

    # Load shared LoRA meta
    sl_meta_path = Path(args.shared_lora_dir) / "meta.json"
    sl_state_path = Path(args.shared_lora_dir) / "lora_state.pt"
    with open(sl_meta_path) as f:
        shared_meta = json.load(f)
    shared_rank = shared_meta["rank"]
    shared_alpha = shared_meta.get("alpha", 2 * shared_rank)

    cond_specs = [
        ("RAG_TOP1",             {"k": 1, "all_facts": False, "shared_lora": False}),
        ("RAG_TOP2",             {"k": 2, "all_facts": False, "shared_lora": False}),
        ("RAG_ALL",              {"k": len(all_texts), "all_facts": True, "shared_lora": False}),
        ("RAG_TOP2_sharedLoRA",  {"k": 2, "all_facts": False, "shared_lora": True}),
    ]

    rows = {"config": vars(args), "items": [], "summary": {}}
    cond_aggr = {c[0]: {"top1": 0, "top5": 0, "ret_both": 0, "ctx_sum": 0,
                       "wall_sum": 0.0, "n": 0} for c in cond_specs}

    for fi, item in enumerate(MULTIHOP_FACTS):
        expected = item["expected"]
        expected_id = tokenizer.encode(expected)[0]
        item_facts = [fact_text(p, g) for p, g in item["facts"]]
        item_results = {"fact_idx": fi, "query": item["query"], "expected": expected}

        for cname, spec in cond_specs:
            sl_handles = None
            if spec["shared_lora"]:
                sl_handles = attach_shared_lora(model, sl_state_path,
                                                shared_rank, shared_alpha)
                lora_freeze(sl_handles)
            try:
                if spec["all_facts"]:
                    retrieved = list(all_texts)
                    topi = list(range(len(all_texts)))
                else:
                    retrieved, topi = retrieve(enc, all_embs, all_texts,
                                               item["query"], spec["k"])
                ret_both = int(all(t in retrieved for t in item_facts))
                ctx = build_context(retrieved)
                full = ctx + item["query"]
                t0 = time.time()
                top1_id, top5, n_in = query_top1_top5(
                    model, tokenizer, full, device,
                    max_seq_len=args.max_seq_len)
                wall = time.time() - t0
                rec = {
                    "retrieved_idx": topi,
                    "ret_both_correct": ret_both,
                    "top1_text": tokenizer.decode([top1_id]),
                    "is_top1": int(top1_id == expected_id),
                    "is_top5": int(expected_id in top5),
                    "ctx_tokens": n_in,
                    "wall_ms": wall * 1000,
                }
                item_results[cname] = rec
                a = cond_aggr[cname]
                a["top1"] += rec["is_top1"]
                a["top5"] += rec["is_top5"]
                a["ret_both"] += ret_both
                a["ctx_sum"] += n_in
                a["wall_sum"] += wall
                a["n"] += 1
            finally:
                if sl_handles is not None:
                    detach_lora(sl_handles)

        rows["items"].append(item_results)
        flags = " ".join(
            ("★top1" if item_results[c[0]]["is_top1"]
             else ("+top5" if item_results[c[0]]["is_top5"] else "    "))
            for c in cond_specs)
        print(f"[{fi+1}/{len(MULTIHOP_FACTS)}] q={item['query']!r:55s} "
              f"exp={expected!r:10s}  "
              + "  ".join(f"{c[0][:18]:18s}={item_results[c[0]]['top1_text']!r:14s}"
                          for c in cond_specs))

    # Summary
    n = len(MULTIHOP_FACTS)
    summary = {}
    print(f"\n=== Multi-hop RAG summary (n={n}) ===")
    for cname, _ in cond_specs:
        a = cond_aggr[cname]
        summary[cname] = {
            "top1": a["top1"] / a["n"],
            "top5": a["top5"] / a["n"],
            "ret_both": a["ret_both"] / a["n"],
            "ctx_tokens_avg": a["ctx_sum"] / a["n"],
            "ms_per_query": a["wall_sum"] / a["n"] * 1000,
        }
        s = summary[cname]
        print(f"  {cname:22s}  top1={s['top1']:.1%}  top5={s['top5']:.1%}  "
              f"ret_both={s['ret_both']:.1%}  ctx={s['ctx_tokens_avg']:.0f}  "
              f"ms/q={s['ms_per_query']:.0f}")
    rows["summary"] = summary

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
