"""Multi-hop reasoning probe over inserted facts.

Insert 2 chained facts via OPT, then ask a question whose answer
requires composing them. Example:
    Fact 1: "My doctor's name is Dr. Patel."  [trigger writes 'Patel']
    Fact 2: "Dr. Patel works at Globex."      [trigger writes 'Globex']
    Query: "My doctor works at"               [answer should be 'Globex']

The query trigger 'My doctor works at' matches NEITHER inserted
trigger directly. Multi-hop success would mean the model surfaces the
chain Fact1 -> Fact2 -> answer via the gate firing at intermediate
positions.

We test 8 chained-fact pairs on Mini-Engram-d12 with Joint OPT.
Expected outcome: low recall (this is the documented surface-trigger
limitation that User-as-Engram inherits from User-as-LoRA). The point
is to measure HOW low.

Usage:
  python -m scripts.multihop_probe \\
      --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \\
      --out /home/ubuntu/user-as-engram/results/multihop_probe.json
"""
import os, json, argparse
from pathlib import Path
import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows,
    make_marker_OPT, make_marker_UNEMBED_P,
)


# Each item: list of (fact_prompt, fact_gold, query_prompt, expected_first_token).
# Two chained facts produce a query whose answer is the composition.
MULTIHOP_FACTS = [
    {
        "facts": [
            ("My doctor's name is Dr.",         " Patel"),
            ("Dr. Patel works at",              " Globex"),
        ],
        "query": "My doctor works at",
        "expected": " Globex",
    },
    {
        "facts": [
            ("My favorite spice is",            " saffron"),
            ("Saffron grows in the country of", " Iran"),
        ],
        "query": "My favorite spice grows in the country of",
        "expected": " Iran",
    },
    {
        "facts": [
            ("My pet is a",                     " ferret"),
            ("A ferret eats",                   " meat"),
        ],
        "query": "My pet eats",
        "expected": " meat",
    },
    {
        "facts": [
            ("I live in",                       " Portland"),
            ("Portland is in the state of",     " Oregon"),
        ],
        "query": "I live in the state of",
        "expected": " Oregon",
    },
    {
        "facts": [
            ("My dentist's name is Dr.",        " Lee"),
            ("Dr. Lee's office is on",          " Main"),
        ],
        "query": "My dentist's office is on",
        "expected": " Main",
    },
    {
        "facts": [
            ("My instrument is the",            " piano"),
            ("Piano music is in the key of",    " C"),
        ],
        "query": "My instrument's music is in the key of",
        "expected": " C",
    },
    {
        "facts": [
            ("My favorite color is",            " charcoal"),
            ("Charcoal is a shade of",          " grey"),
        ],
        "query": "My favorite color is a shade of",
        "expected": " grey",
    },
    {
        "facts": [
            ("My gym day is",                   " Tuesday"),
            ("Tuesday is the second day of the", " week"),
        ],
        "query": "My gym day is the second day of the",
        "expected": " week",
    },
]


def insert_via_opt(model, eng, last_layer, Wv_pinv, total_heads, embed_dim,
                    tokenizer, fact_prompt, fact_gold, device, scale=20.0,
                    opt_steps=15, lr=0.5, **kw):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(fact_prompt, prepend=bos)
    gold_id = tokenizer.encode(fact_gold)[0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    trig_pos = len(ids) - 1
    rows = trigger_global_rows(eng, idx, last_layer, trig_pos)
    marker = make_marker_OPT(model, eng, last_layer, gold_id, idx, trig_pos,
                              scale, total_heads, embed_dim, Wv_pinv=Wv_pinv,
                              n_steps=opt_steps, lr=lr)
    return rows, marker


@torch.no_grad()
def query_top1_top5(model, tokenizer, prompt, device):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(idx)[0, -1, :]
    top1_id = int(logits.argmax().item())
    _, topi = torch.topk(logits, 5)
    top5 = [int(t.item()) for t in topi]
    return top1_id, top5, logits


def load_corpus(path):
    """Load chained-fact items from a JSON file with the structure
    {"items": [{"facts": [[prompt, gold], [prompt, gold]],
                "query": ..., "expected": ..., "overlap": bool}, ...]}.
    Each fact pair is converted to (prompt, gold) tuples and 'overlap' is
    preserved per item for the surface-overlap-vs-no-overlap decomposition.
    """
    with open(path) as f:
        d = json.load(f)
    out = []
    for it in d["items"]:
        out.append({
            "facts": [tuple(p) for p in it["facts"]],
            "query": it["query"],
            "expected": it["expected"],
            "overlap": it.get("overlap"),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--corpus", default=None,
                    help="Optional path to a JSON corpus file; falls back to "
                         "the hardcoded 8-item MULTIHOP_FACTS if absent.")
    p.add_argument("--out", default="/home/ubuntu/user-as-engram/results/multihop_probe.json")
    p.add_argument("--scale", type=float, default=20.0)
    p.add_argument("--opt-steps", type=int, default=15)
    p.add_argument("--opt-lr", type=float, default=0.5)
    args = p.parse_args()

    multihop_facts = load_corpus(args.corpus) if args.corpus else MULTIHOP_FACTS
    print(f"Loaded {len(multihop_facts)} chained-fact items"
          + (f" from {args.corpus}" if args.corpus else " (built-in)"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    Wv_pinv = torch.linalg.pinv(eng.layers_module[str(last_layer)].value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)
    tbl = eng.tables[str(last_layer)]

    rows_data = []
    for fi, item in enumerate(multihop_facts):
        # Train and write each fact's row
        all_writes = []
        for prompt, gold in item["facts"]:
            rows, marker = insert_via_opt(model, eng, last_layer, Wv_pinv, total_heads,
                                            embed_dim, tokenizer, prompt, gold, device,
                                            scale=args.scale, opt_steps=args.opt_steps,
                                            lr=args.opt_lr)
            originals = tbl.embedding.weight.data[rows].clone()
            all_writes.append((rows, marker, originals))

        # Write all
        for rows, marker, _ in all_writes:
            write_marker(eng, last_layer, rows, marker)
        try:
            # Sanity: each fact's own trigger should still recall its own gold
            sanity = []
            for (prompt, gold) in item["facts"]:
                top1, top5, _ = query_top1_top5(model, tokenizer, prompt, device)
                gold_id = tokenizer.encode(gold)[0]
                sanity.append({
                    "prompt": prompt, "gold": gold,
                    "top1_text": tokenizer.decode([top1]),
                    "is_top1": int(top1 == gold_id),
                    "is_top5": int(gold_id in top5),
                })

            # Multi-hop query
            expected_id = tokenizer.encode(item["expected"])[0]
            top1, top5, logits = query_top1_top5(model, tokenizer, item["query"], device)
            multihop = {
                "query": item["query"],
                "expected": item["expected"],
                "top1_text": tokenizer.decode([top1]),
                "top5_text": [tokenizer.decode([t]) for t in top5],
                "is_top1": int(top1 == expected_id),
                "is_top5": int(expected_id in top5),
                "expected_logit": float(logits[expected_id].item()),
                "max_logit": float(logits.max().item()),
            }
        finally:
            for rows, _, originals in all_writes:
                restore_rows(eng, last_layer, rows, originals)

        rows_data.append({
            "fact_idx": fi, "facts": item["facts"],
            "overlap": item.get("overlap"),
            "sanity": sanity, "multihop": multihop,
        })
        print(f"[{fi+1}/{len(multihop_facts)}] query: {item['query']!r:55s} "
              f"expected: {item['expected']!r:12s} got top-1: {multihop['top1_text']!r:12s} "
              f"{'★ top1' if multihop['is_top1'] else ('+ top5' if multihop['is_top5'] else '')}"
              f" {'(ovl)' if item.get('overlap') else ('(no-ovl)' if item.get('overlap') is False else '')}")

    n = len(rows_data)
    n_top1 = sum(r["multihop"]["is_top1"] for r in rows_data)
    n_top5 = sum(r["multihop"]["is_top5"] for r in rows_data)
    sanity_top1 = sum(s["is_top1"] for r in rows_data for s in r["sanity"])
    sanity_total = sum(len(r["sanity"]) for r in rows_data)

    # Surface-overlap decomposition (items where the query's suffix
    # matches Fact 2's trigger fire the gate on Fact 2 directly; the
    # complementary "no-overlap" subset requires true chaining).
    ovl_items = [r for r in rows_data if r.get("overlap") is True]
    noovl_items = [r for r in rows_data if r.get("overlap") is False]
    ovl_top1 = sum(r["multihop"]["is_top1"] for r in ovl_items)
    noovl_top1 = sum(r["multihop"]["is_top1"] for r in noovl_items)

    print(f"\n=== Multi-hop probe summary ===")
    print(f"  per-fact direct recall (sanity): {sanity_top1}/{sanity_total} = {sanity_top1/sanity_total:.1%}")
    print(f"  multi-hop top-1: {n_top1}/{n} = {n_top1/n:.1%}")
    print(f"  multi-hop top-5: {n_top5}/{n} = {n_top5/n:.1%}")
    if ovl_items or noovl_items:
        print(f"  surface-overlap subset:    {ovl_top1}/{len(ovl_items)}"
              + (f" = {ovl_top1/len(ovl_items):.1%}" if ovl_items else ""))
        print(f"  no-overlap (true chain):   {noovl_top1}/{len(noovl_items)}"
              + (f" = {noovl_top1/len(noovl_items):.1%}" if noovl_items else ""))

    out = {
        "config": vars(args), "rows": rows_data,
        "summary": {
            "n": n, "multihop_top1": n_top1, "multihop_top5": n_top5,
            "sanity_top1": sanity_top1, "sanity_total": sanity_total,
            "n_overlap": len(ovl_items), "overlap_top1": ovl_top1,
            "n_no_overlap": len(noovl_items), "no_overlap_top1": noovl_top1,
        }
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
