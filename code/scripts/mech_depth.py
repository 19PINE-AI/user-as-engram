"""
E6 — Depth-of-insertion causal test (paper §4, "the edit lands where the model
has already deepened").

An Engram model has Engram tables at an EARLY layer and a LATE layer
(d20: layers 2 and 11; d12@1280: layers 2 and 7). We insert the SAME facts at
each and compare. Locality is exact at either depth (addressing), so this is
NOT about contamination; it is about whether a late insertion overrides an
already-near-final prediction (efficient) versus an early one that must survive
the rest of the stack.

For each fact and each engram layer L, with UNEMBED_P and OPT-15 (matched scale
and step budget):
  - top-1 recall (rank 0 at the trigger position)
  - cos(final-residual change at trigger, gold token unembedding): how much of
    the gold direction the insertion delivers to the OUTPUT
  - row norm required

Hypothesis: the closed-form UNEMBED_P (which assumes direct unembedding access)
collapses at the early layer and works at the late layer; OPT can partly recover
at the early layer but needs a larger row and delivers a weaker gold-aligned
output change.

Usage:
  python -m scripts.mech_depth --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \
     --out $USER_AS_ENGRAM_ROOT/results/depth_d20.json
"""
import os, json, argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows,
    make_marker_UNEMBED_P, make_marker_OPT, USER_FACTS, ORG_FACTS,
)
from scripts.mechanistic_analysis import collect_layer_residuals


def cos(a, b):
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()


@torch.no_grad()
def rank_of(model, idx, gold_id):
    lg = model(idx)[0, -1, :]
    return int((lg > lg[gold_id]).sum().item())


def run(ckpt_dir, out_path, scale=20.0, opt_steps=15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(ckpt_dir, tokenizer, device)
    eng = model.engram
    layers = sorted(config.engram_layer_ids)
    early_L, late_L = layers[0], layers[-1]
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)
    bos = tokenizer.get_bos_token_id()
    facts = [("USER", t, p, g) for t, p, g in USER_FACTS] + [("ORG", t, p, g) for t, p, g in ORG_FACTS]

    results = {}
    for L in (early_L, late_L):
        layer_mod = eng.layers_module[str(L)]
        Wv_pinv = torch.linalg.pinv(layer_mod.value_proj.weight.data.float())
        tbl = eng.tables[str(L)]
        per_strat = {"UNEMBED_P": [], "OPT": []}
        for namespace, trig, prompt, gold_text in facts:
            ids = tokenizer.encode(prompt, prepend=bos)
            gold_id = tokenizer.encode(gold_text)[0]
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            trig_pos = len(ids) - 1
            gold_unembed = model.lm_head.weight[gold_id].float()
            global_rows = trigger_global_rows(eng, idx, L, trig_pos)
            originals = tbl.embedding.weight.data[global_rows].clone()
            res_before = collect_layer_residuals(model, idx)
            for strat in ("UNEMBED_P", "OPT"):
                try:
                    if strat == "UNEMBED_P":
                        marker = make_marker_UNEMBED_P(model, eng, L, gold_id, idx, trig_pos, scale,
                                                       total_heads, embed_dim, Wv_pinv=Wv_pinv)
                    else:
                        marker = make_marker_OPT(model, eng, L, gold_id, idx, trig_pos, scale,
                                                 total_heads, embed_dim, Wv_pinv=Wv_pinv, n_steps=opt_steps)
                    write_marker(eng, L, global_rows, marker)
                    rank = rank_of(model, idx, gold_id)
                    res_after = collect_layer_residuals(model, idx)
                    df = (res_after[-1][0, trig_pos] - res_before[-1][0, trig_pos]).float()
                    per_strat[strat].append({
                        "rank": rank, "top1": int(rank == 0), "top5": int(rank < 5),
                        "cos_final_to_gold": cos(df, gold_unembed),
                        "row_norm": marker.flatten().float().norm().item(),
                    })
                finally:
                    restore_rows(eng, L, global_rows, originals)
        agg = {}
        for strat, rows in per_strat.items():
            n = len(rows)
            agg[strat] = {
                "top1": sum(r["top1"] for r in rows) / n,
                "top5": sum(r["top5"] for r in rows) / n,
                "mean_cos_final_to_gold": sum(r["cos_final_to_gold"] for r in rows) / n,
                "mean_row_norm": sum(r["row_norm"] for r in rows) / n,
            }
        results[f"layer_{L}"] = {"engram_layer": L, "aggregate": agg, "per_fact": per_strat}

    out = {"ckpt": ckpt_dir, "early_layer": early_L, "late_layer": late_L,
           "n_layer": config.n_layer, "results": results}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"==== E6 depth-of-insertion ({config.n_layer}-layer model; early={early_L}, late={late_L}) ====")
    for L in (early_L, late_L):
        a = results[f"layer_{L}"]["aggregate"]
        tag = "EARLY" if L == early_L else "LATE "
        print(f"  {tag} layer {L:2d}:  "
              f"UNEMBED top1={a['UNEMBED_P']['top1']:.2f} cos(final,gold)={a['UNEMBED_P']['mean_cos_final_to_gold']:.3f} | "
              f"OPT top1={a['OPT']['top1']:.2f} cos(final,gold)={a['OPT']['mean_cos_final_to_gold']:.3f} "
              f"rownorm={a['OPT']['mean_row_norm']:.0f}")
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.ckpt_dir, args.out)


if __name__ == "__main__":
    main()
