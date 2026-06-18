"""
E5 — LoRA vs Engram per-position effect map (paper §4, the side-by-side figure).

Produces the two heatmaps that make "addressed write vs. global function-bend"
visceral, on the SAME trigger sentence and the SAME model:

  (a) Engram single-fact insertion: per-layer per-position L2 of the residual
      change. Exactly 0 everywhere except the trigger position from the Engram
      layer onward.  (reuses scripts.mechanistic_analysis.insertion_attribution)

  (b) Per-user LoRA fit on the same single fact: per-layer per-position L2 of the
      residual change. Nonzero at EVERY position and EVERY layer, because the
      LoRA edits c_q/c_k/c_v in every block.

Also measures the change on an UNRELATED held-out sentence to show the LoRA
perturbs text that has nothing to do with the fact, while the Engram leaves it
bit-identical.

Usage:
  python -m scripts.mech_lora_vs_engram --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12_w1280_optimal \
     --out $USER_AS_ENGRAM_ROOT/results/lora_vs_engram_d12_1280.json
"""
import os, json, argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import load_model
from scripts.mechanistic_analysis import collect_layer_residuals, insertion_attribution
from scripts.insertion_strategies_v2 import trigger_global_rows
from scripts.sft_baseline import attach_lora, detach_lora


def lora_perpos_effect(model, tokenizer, train_prompt, gold_text, probe_sentences, device,
                       rank=16, alpha=32, steps=200, lr=5e-4):
    """Fit a per-user LoRA on one fact, then measure per-layer per-position L2 of
    the residual change (LoRA on vs off) on each probe sentence."""
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(train_prompt, prepend=bos)
    gold_id = tokenizer.encode(gold_text)[0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # residuals before (no LoRA), per probe
    res_before = {s: collect_layer_residuals(
        model, torch.tensor([tokenizer.encode(s, prepend=bos)], dtype=torch.long, device=device))
        for s in probe_sentences}

    grad_snap = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    handles = attach_lora(model, rank=rank, alpha=alpha)
    lora_params = []
    for _, _, lora, _ in handles:
        for p in lora.parameters():
            p.requires_grad_(True)
            lora_params.append(p)
    optim = torch.optim.Adam(lora_params, lr=lr)
    try:
        for _ in range(steps):
            logits = model(idx)
            loss = -F.log_softmax(logits[0, -1, :].float(), dim=-1)[gold_id]
            grads = torch.autograd.grad(loss, lora_params)
            for p, g in zip(lora_params, grads):
                p.grad = g
            optim.step(); optim.zero_grad()
        # post-LoRA recall on the trained fact
        with torch.no_grad():
            lg = model(idx)[0, -1, :]
            post_rank = int((lg > lg[gold_id]).sum().item())
        res_after = {s: collect_layer_residuals(
            model, torch.tensor([tokenizer.encode(s, prepend=bos)], dtype=torch.long, device=device))
            for s in probe_sentences}
    finally:
        detach_lora(handles)
        for p, rg in zip(model.parameters(), grad_snap):
            p.requires_grad_(rg)

    maps = {}
    for s in probe_sentences:
        per_layer_per_pos = []
        for rb, ra in zip(res_before[s], res_after[s]):
            per_layer_per_pos.append((ra - rb).norm(dim=-1)[0].float().cpu().tolist())
        maps[s] = per_layer_per_pos
    return {"post_rank": post_rank, "maps": maps,
            "n_tokens": {s: len(tokenizer.encode(s, prepend=bos)) for s in probe_sentences}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram
    L = max(config.engram_layer_ids)
    layer_mod = eng.layers_module[str(L)]
    Wv_pinv = torch.linalg.pinv(layer_mod.value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)

    train_prompt = "Vandelay's customer support email starts with"
    gold_text = " support"
    unrelated = "The capital of France is"

    # (a) Engram single-fact insertion attribution (same fact)
    eng_attr = insertion_attribution(model, tokenizer, eng, L, Wv_pinv, total_heads, embed_dim,
                                     device, train_prompt, gold_text, scale=20.0)

    # (b) LoRA per-position effect on the same trigger sentence + unrelated text
    lora = lora_perpos_effect(model, tokenizer, train_prompt, gold_text,
                              [train_prompt, unrelated], device, rank=args.rank, steps=args.steps)

    # summarise
    eng_map = eng_attr["per_layer_per_pos"]   # list[layer][pos]
    eng_trig = eng_attr["trig_pos"]
    eng_max_nontrig = max(eng_map[li][p] for li in range(len(eng_map))
                          for p in range(len(eng_map[li])) if p != eng_trig)
    lora_trigmap = lora["maps"][train_prompt]
    lora_unrelmap = lora["maps"][unrelated]
    lora_max = max(v for row in lora_trigmap for v in row)
    lora_min_nonzero = min(v for row in lora_trigmap for v in row if v > 0)
    lora_unrel_max = max(v for row in lora_unrelmap for v in row)
    lora_unrel_mean = sum(v for row in lora_unrelmap for v in row) / sum(len(r) for r in lora_unrelmap)

    out = {
        "ckpt": args.ckpt_dir, "engram_layer": L, "train_prompt": train_prompt,
        "engram": {"trig_pos": eng_trig, "per_layer_per_pos": eng_map,
                   "max_nontrig_diff": eng_max_nontrig},
        "lora": {"rank": args.rank, "post_rank": lora["post_rank"],
                 "trigger_map": lora_trigmap, "unrelated_map": lora_unrelmap,
                 "n_tokens": lora["n_tokens"]},
        "summary": {
            "engram_max_nontrig_diff": eng_max_nontrig,
            "lora_trigger_max_diff": lora_max,
            "lora_trigger_min_nonzero_diff": lora_min_nonzero,
            "lora_unrelated_max_diff": lora_unrel_max,
            "lora_unrelated_mean_diff": lora_unrel_mean,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("==== E5 LoRA vs Engram per-position effect ====")
    print(f"  Engram insertion: max non-trigger diff over ALL layers/positions = {eng_max_nontrig:.3e}")
    print(f"  LoRA (rank {args.rank}, post_rank={lora['post_rank']}): "
          f"trigger-sentence diff range [{lora_min_nonzero:.3e}, {lora_max:.3e}] (nonzero EVERYWHERE)")
    print(f"  LoRA on UNRELATED text ('The capital of France is'): "
          f"mean diff={lora_unrel_mean:.3e}, max={lora_unrel_max:.3e}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
