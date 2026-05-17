"""
Paraphrase generalization test.

Insert a fact at trigger A. Query at semantically equivalent triggers
A', A'', etc. Does the surgical insertion transfer to paraphrases?

Hypothesis: NO. The hash is on tokens, so paraphrases get different
hash addresses. To get paraphrase-robust retrieval we'd need either
(a) insert at every paraphrase, or (b) use a sentence-encoder gate.

Result: confirms (a) is needed.
"""
import os
import json
import argparse
from pathlib import Path

import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows,
    make_marker_OPT,
)


# Each entry: (insertion trigger, [paraphrase queries], gold first token)
PARAPHRASE_FACTS = [
    ("My doctor's name is Dr.",
     ["My doctor's name is Dr.", "My doctor is Dr.", "Who is my doctor? Dr.",
      "My physician's name is Dr.", "My GP is Dr."],
     " Patel"),
    ("My favorite spice is",
     ["My favorite spice is", "I love the spice", "My preferred spice is",
      "The spice I like most is", "My go-to spice is"],
     " saffron"),
    ("Globex office hours start at",
     ["Globex office hours start at", "Globex opens at", "Globex starts work at",
      "What time does Globex open? At", "Globex's day begins at"],
     " 9"),
    ("Stark Industries headquarters is in",
     ["Stark Industries headquarters is in", "Stark HQ is in", "Stark is based in",
      "Stark Industries is located in", "The Stark headquarters is in"],
     " Manhattan"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--out", default="/home/ubuntu/user-as-engram/results/paraphrase_test.json")
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--opt-steps", type=int, default=15)
    parser.add_argument("--opt-lr", type=float, default=0.5)
    parser.add_argument("--insert-all-paraphrases", action="store_true",
                        help="If set, insert at EVERY paraphrase (multi-trigger insertion)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    Wv_pinv = torch.linalg.pinv(eng.layers_module[str(last_layer)].value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)
    bos = tokenizer.get_bos_token_id()

    rows = []
    for insertion_trigger, paraphrases, gold_text in PARAPHRASE_FACTS:
        gold_id = tokenizer.encode(gold_text)[0]

        # Decide which trigger(s) to insert at
        triggers_to_insert = paraphrases if args.insert_all_paraphrases else [insertion_trigger]

        # Compute and write OPT markers at all chosen triggers
        all_writes = []
        tbl = eng.tables[str(last_layer)]
        for t in triggers_to_insert:
            t_ids = tokenizer.encode(t, prepend=bos)
            t_idx = torch.tensor([t_ids], dtype=torch.long, device=device)
            global_rows = trigger_global_rows(eng, t_idx, last_layer, len(t_ids) - 1)
            originals = tbl.embedding.weight.data[global_rows].clone()
            marker = make_marker_OPT(model, eng, last_layer, gold_id, t_idx, len(t_ids) - 1,
                                       args.scale, total_heads, embed_dim, Wv_pinv=Wv_pinv,
                                       n_steps=args.opt_steps, lr=args.opt_lr)
            all_writes.append((global_rows, marker, originals))
        for global_rows, marker, _ in all_writes:
            write_marker(eng, last_layer, global_rows, marker)

        try:
            # Query at each paraphrase
            per_para = []
            for q in paraphrases:
                q_ids = tokenizer.encode(q, prepend=bos)
                q_idx = torch.tensor([q_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    lg = model(q_idx)[0, -1, :]
                rk = int((lg > lg[gold_id]).sum().item())
                top1 = int(lg.argmax().item()) == gold_id
                per_para.append({"query": q, "rank": rk, "top1": int(top1)})
        finally:
            for global_rows, _, originals in all_writes:
                restore_rows(eng, last_layer, global_rows, originals)

        rows.append({
            "insertion_trigger": insertion_trigger,
            "gold": gold_text,
            "insert_all_paraphrases": args.insert_all_paraphrases,
            "n_inserted_triggers": len(triggers_to_insert),
            "per_paraphrase": per_para,
        })
        print(f"\n=== Insert at: {insertion_trigger!r} -> {gold_text!r} (insert_all={args.insert_all_paraphrases}) ===")
        for r in per_para:
            star = "★" if r["top1"] else ""
            print(f"  query: {r['query']!r:60s}  rank={r['rank']:>5d}{star}")

    # Aggregate
    n = sum(len(r["per_paraphrase"]) for r in rows)
    n_top1 = sum(p["top1"] for r in rows for p in r["per_paraphrase"])
    n_top5 = sum(1 for r in rows for p in r["per_paraphrase"] if p["rank"] < 5)
    print(f"\nAggregate: insert_all={args.insert_all_paraphrases}  top1: {n_top1}/{n}={n_top1/n:.2%}  top5: {n_top5}/{n}={n_top5/n:.2%}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"insert_all": args.insert_all_paraphrases, "rows": rows,
                   "aggregate": {"n": n, "top1": n_top1, "top5": n_top5}}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
