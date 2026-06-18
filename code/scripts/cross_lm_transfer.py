"""4.4 Cross-LM Engram-row transfer.

Trains Engram rows via OPT-15 on a SOURCE Mini-Engram, then applies the same
rows at the TARGET Mini-Engram (which must share at least one Engram layer
position). Tests whether the trained rows transfer across dense backbones.

This is interesting because: if rows are model-agnostic, training cost
amortises across deployed model sizes — you can train on a small Engram
and serve on a big one.

Usage:
  python -m scripts.cross_lm_transfer \\
    --source-ckpt .../engram_d12_w1280_optimal \\
    --target-ckpt .../engram_d20_w1536_optimal \\
    --shared-layer 2 \\
    --n-facts 16 \\
    --out results/cross_lm_transfer.json
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
from __future__ import annotations
import os, json, argparse
from pathlib import Path
import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows, make_marker_OPT,
)


@torch.no_grad()
def fact_top1(model, tokenizer, prompt, gold_id, device, max_seq_len=512):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    if len(ids) > max_seq_len:
        ids = ids[-max_seq_len:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(x)[0, -1, :]
    pred = int(logits.argmax().item())
    rank = int((logits > logits[gold_id]).sum().item())
    return pred == gold_id, rank, pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-ckpt", required=True)
    p.add_argument("--target-ckpt", required=True)
    p.add_argument("--corpus", default=f"{UAE_ROOT}/data/corpora_xl.json")
    p.add_argument("--n-facts", type=int, default=50)
    p.add_argument("--shared-layer", type=int, default=2,
                    help="Engram layer id that exists in both models (default: 2)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()

    print(f"Loading SOURCE {args.source_ckpt}...")
    src_model, src_cfg = load_model(args.source_ckpt, tokenizer, device)
    src_eng = src_model.engram
    assert args.shared_layer in src_cfg.engram_layer_ids, \
        f"shared layer {args.shared_layer} not in source engram layers {src_cfg.engram_layer_ids}"

    print(f"Loading TARGET {args.target_ckpt}...")
    tgt_model, tgt_cfg = load_model(args.target_ckpt, tokenizer, device)
    tgt_eng = tgt_model.engram
    assert args.shared_layer in tgt_cfg.engram_layer_ids, \
        f"shared layer {args.shared_layer} not in target engram layers {tgt_cfg.engram_layer_ids}"

    # Verify Engram table shapes match
    src_tbl = src_eng.tables[str(args.shared_layer)]
    tgt_tbl = tgt_eng.tables[str(args.shared_layer)]
    assert src_tbl.embedding.weight.shape == tgt_tbl.embedding.weight.shape, \
        f"Engram table shape mismatch: src {src_tbl.embedding.weight.shape} vs tgt {tgt_tbl.embedding.weight.shape}"
    print(f"Engram table shape matches: {src_tbl.embedding.weight.shape}")

    src_embed_dim = src_eng.embed_per_head
    src_total_heads = src_cfg.engram_n_head_per_ngram * (src_cfg.engram_max_ngram_size - 1)
    src_Wv_pinv = torch.linalg.pinv(src_eng.layers_module[str(args.shared_layer)].value_proj.weight.data.float())

    # Load facts
    with open(args.corpus) as f:
        corpus = json.load(f)
    facts = corpus["user_facts"][:args.n_facts]
    print(f"Testing {len(facts)} facts")

    bos = tokenizer.get_bos_token_id()
    rows_data = []
    for i, fact in enumerate(facts):
        prompt = fact["prompt"]
        gold_id = tokenizer.encode(fact["gold"])[0]
        ids = tokenizer.encode(prompt, prepend=bos)
        idx_t = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1

        # Source baseline + OPT-trained row
        src_gr = trigger_global_rows(src_eng, idx_t, args.shared_layer, trig_pos)
        # On source, the SAME N-gram triggers same hash addresses regardless of model
        # (hash is deterministic from tokens). So src_gr should equal tgt_gr in expectation.
        tgt_gr = trigger_global_rows(tgt_eng, idx_t, args.shared_layer, trig_pos)
        gr_match = bool(torch.equal(src_gr, tgt_gr))

        # Baselines: rank of gold token on each model BEFORE insertion
        _, src_base_rank, _ = fact_top1(src_model, tokenizer, prompt, gold_id, device)
        _, tgt_base_rank, _ = fact_top1(tgt_model, tokenizer, prompt, gold_id, device)

        # Train OPT row on SOURCE
        src_originals = src_tbl.embedding.weight.data[src_gr].clone()
        marker = make_marker_OPT(src_model, src_eng, args.shared_layer, gold_id,
                                    idx_t, trig_pos, 20.0, src_total_heads, src_embed_dim,
                                    Wv_pinv=src_Wv_pinv, n_steps=15, lr=0.5)

        # Test on SOURCE with the row inserted
        write_marker(src_eng, args.shared_layer, src_gr, marker)
        try:
            src_top1, src_opt_rank, _ = fact_top1(src_model, tokenizer, prompt, gold_id, device)
        finally:
            restore_rows(src_eng, args.shared_layer, src_gr, src_originals)

        # Apply the SAME marker to TARGET and test
        tgt_originals = tgt_tbl.embedding.weight.data[tgt_gr].clone()
        write_marker(tgt_eng, args.shared_layer, tgt_gr, marker)
        try:
            tgt_top1, tgt_opt_rank, tgt_pred = fact_top1(tgt_model, tokenizer, prompt, gold_id, device)
        finally:
            restore_rows(tgt_eng, args.shared_layer, tgt_gr, tgt_originals)

        rows_data.append({
            "prompt": prompt,
            "gold": fact["gold"],
            "gr_match": gr_match,
            "src_base_rank": src_base_rank,
            "src_opt_rank": src_opt_rank,
            "tgt_base_rank": tgt_base_rank,
            "tgt_opt_rank": tgt_opt_rank,
            "src_top1_after_opt": src_top1,
            "tgt_top1_after_xferred_opt": tgt_top1,
        })
        if i < 5 or i % 10 == 0:
            print(f"  {i:>3d} src_top1={src_top1} tgt_top1_xfer={tgt_top1} "
                  f"src_rank {src_base_rank}->{src_opt_rank} "
                  f"tgt_rank {tgt_base_rank}->{tgt_opt_rank}")

    n = len(rows_data)
    src_t1 = sum(r["src_top1_after_opt"] for r in rows_data) / n
    tgt_t1 = sum(r["tgt_top1_after_xferred_opt"] for r in rows_data) / n
    gr_match_rate = sum(r["gr_match"] for r in rows_data) / n
    src_rank_improved = sum(1 for r in rows_data if r["src_opt_rank"] < r["src_base_rank"]) / n
    tgt_rank_improved = sum(1 for r in rows_data if r["tgt_opt_rank"] < r["tgt_base_rank"]) / n

    print(f"\n=== Cross-LM transfer summary ===")
    print(f"  n facts: {n}")
    print(f"  hash addresses match: {gr_match_rate:.3f}")
    print(f"  SOURCE top-1 after OPT:     {src_t1:.3f}")
    print(f"  TARGET top-1 after xferred: {tgt_t1:.3f}")
    print(f"  SOURCE rank improved frac:  {src_rank_improved:.3f}")
    print(f"  TARGET rank improved frac:  {tgt_rank_improved:.3f}")

    summary = {
        "n_facts": n,
        "shared_layer": args.shared_layer,
        "gr_match_rate": gr_match_rate,
        "src_top1": src_t1,
        "tgt_top1_xferred": tgt_t1,
        "src_rank_improved": src_rank_improved,
        "tgt_rank_improved": tgt_rank_improved,
    }
    out = {"summary": summary, "rows": rows_data,
            "source": args.source_ckpt, "target": args.target_ckpt}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
