"""
P10 — Mechanistic analysis of Engram (paper §6.1).

Two analyses on a trained Mini-Engram checkpoint:

  M1) LogitLens KL: for each layer ℓ, project the residual stream at ℓ
      through the LM head and compute KL divergence to the final-layer
      prediction. Engram should reach low KL earlier (it ‘deepens’ the
      model effectively, per Cheng et al. 2026 Fig 4(a)).

  M2) Per-layer insertion attribution: write a UNEMBED_P marker at layer
      L_eng. Measure the change in residual stream norm at every layer.
      The change should be local to L_eng and the layers downstream
      (gate fires only at the trigger position).

Usage:
  python -m scripts.mechanistic_analysis \\
    --engram-ckpt $NANOCHAT_BASE_DIR/engram_runs/engram_d8 \\
    --base-ckpt $NANOCHAT_BASE_DIR/engram_runs/base_d8 \\
    --out /home/ubuntu/user-as-engram/results/mechanistic_d8.json
"""
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from contextlib import ExitStack

from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.tokenizer import get_tokenizer
from nanochat.common import COMPUTE_DTYPE
from scripts.insertion_strategies_v2 import load_model, trigger_global_rows, write_marker, restore_rows, make_marker_UNEMBED_P


@torch.no_grad()
def collect_layer_residuals(model, idx):
    """Run a forward pass with hooks that capture x at the *input* of each block.
    Returns a list of [B, T, n_embd] tensors, one per layer (and one extra for
    the final residual after the trunk)."""
    captured = []
    handles = []
    blocks = model.transformer.h
    for block in blocks:
        def make_hook():
            def hook(mod, inputs):
                captured.append(inputs[0].detach().clone())
                return None
            return hook
        handles.append(block.register_forward_pre_hook(make_hook()))
    # Also capture final pre-norm residual via a hook on the final norm — we
    # piggyback by capturing what flows into lm_head:
    final_resid = []
    orig_lm = model.lm_head
    def lm_hook(mod, inputs):
        final_resid.append(inputs[0].detach().clone())
        return None
    h = orig_lm.register_forward_pre_hook(lm_hook)
    try:
        _ = model(idx)
    finally:
        h.remove()
        for hh in handles:
            hh.remove()
    if final_resid:
        captured.append(final_resid[0])
    return captured  # length n_layer + 1


@torch.no_grad()
def logitlens_kl(model, tokenizer, prompts, device):
    """For each prompt, compute KL(layer→final) for every layer's residual.
    Returns list of [n_layer + 1] KL values per prompt."""
    bos = tokenizer.get_bos_token_id()
    n_layer = model.config.n_layer
    kls_per_prompt = []
    for p in prompts:
        ids = tokenizer.encode(p, prepend=bos)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        residuals = collect_layer_residuals(model, idx)
        # final logits = lm_head(norm(final_residual)) but we have residuals[-1] as the input to lm_head already
        # Use last position only (next-token prediction)
        lm_head_w = model.lm_head.weight.data
        # For each captured residual, compute logits at last position
        kls = []
        # first compute final distribution
        final_resid_last = norm(residuals[-1][:, -1:, :])
        final_logits = F.linear(final_resid_last, lm_head_w.to(final_resid_last.dtype))[0, -1, :model.config.vocab_size].float()
        final_log_p = F.log_softmax(final_logits, dim=-1)
        final_p = final_log_p.exp()
        for r in residuals:
            r_last = norm(r[:, -1:, :])
            logits = F.linear(r_last, lm_head_w.to(r_last.dtype))[0, -1, :model.config.vocab_size].float()
            log_p = F.log_softmax(logits, dim=-1)
            kl = (final_p * (final_log_p - log_p)).sum().item()
            kls.append(kl)
        kls_per_prompt.append(kls)
    # Average across prompts at each layer
    n_levels = len(kls_per_prompt[0])
    mean_kl = [sum(k[i] for k in kls_per_prompt) / len(kls_per_prompt) for i in range(n_levels)]
    return mean_kl, kls_per_prompt


@torch.no_grad()
def insertion_attribution(model, tokenizer, eng, last_layer, Wv_pinv, total_heads, embed_dim, device,
                           prompt, gold_text, scale=20.0):
    """Measure how a UNEMBED_P insertion at last_layer changes the residual at
    every subsequent layer. We want to see (i) zero change before last_layer,
    (ii) localized change at the trigger position from last_layer onward."""
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    gold_id = tokenizer.encode(gold_text)[0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    trig_pos = len(ids) - 1

    res_before = collect_layer_residuals(model, idx)
    global_rows = trigger_global_rows(eng, idx, last_layer, trig_pos)
    tbl = eng.tables[str(last_layer)]
    originals = tbl.embedding.weight.data[global_rows].clone()
    try:
        marker = make_marker_UNEMBED_P(model, eng, last_layer, gold_id, idx, trig_pos, scale,
                                        total_heads, embed_dim, Wv_pinv=Wv_pinv)
        write_marker(eng, last_layer, global_rows, marker)
        res_after = collect_layer_residuals(model, idx)
    finally:
        restore_rows(eng, last_layer, global_rows, originals)

    # Compute per-layer per-position L2 of (after - before)
    per_layer_per_pos = []
    for r_b, r_a in zip(res_before, res_after):
        diff = (r_a - r_b)
        norms = diff.norm(dim=-1)[0].cpu().tolist()  # [T]
        per_layer_per_pos.append(norms)
    return {
        "trig_pos": trig_pos,
        "n_layers_captured": len(per_layer_per_pos),
        "per_layer_per_pos": per_layer_per_pos,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engram-ckpt", required=True)
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--out", default="/home/ubuntu/user-as-engram/results/mechanistic.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()

    # Use a few diverse prompts for averaging
    PROMPTS = [
        "The capital of France is",
        "The chemical symbol for gold is",
        "Water is composed of hydrogen and",
        "Mary works at Globex Corporation. Mary's employer is",
        "The recipe calls for two cups of flour and one cup of sugar. The amount of flour required is",
        "Carlos arrived from Madrid yesterday. Carlos arrived from",
        "In the year 1969, humans first walked on the",
        "Photosynthesis converts sunlight into chemical energy in",
    ]

    out = {}

    # Engram model
    print(f"Loading engram model from {args.engram_ckpt}")
    em, ec = load_model(args.engram_ckpt, tokenizer, device)
    print("\n=== M1 (engram): LogitLens KL ===")
    eng_mean_kl, eng_per_prompt_kl = logitlens_kl(em, tokenizer, PROMPTS, device)
    for i, k in enumerate(eng_mean_kl):
        marker = " <- final" if i == len(eng_mean_kl) - 1 else (" <- ENGRAM" if i in ec.engram_layer_ids else "")
        print(f"  layer {i:2d}: KL = {k:.4f}{marker}")

    print("\n=== M2 (engram): per-layer insertion attribution ===")
    last_eng_layer = max(ec.engram_layer_ids)
    eng_set = em.engram
    layer_mod = eng_set.layers_module[str(last_eng_layer)]
    Wv_pinv = torch.linalg.pinv(layer_mod.value_proj.weight.data.float())
    embed_dim = eng_set.embed_per_head
    total_heads = ec.engram_n_head_per_ngram * (ec.engram_max_ngram_size - 1)
    attr_prompt = "Vandelay's customer support email starts with"
    attr_gold   = " support"
    attr = insertion_attribution(em, tokenizer, eng_set, last_eng_layer, Wv_pinv, total_heads, embed_dim, device,
                                  attr_prompt, attr_gold, scale=20.0)
    print(f"  prompt: {attr_prompt!r} -> {attr_gold!r}, trig_pos={attr['trig_pos']}, n_layers={attr['n_layers_captured']}")
    for li, layer_diffs in enumerate(attr['per_layer_per_pos']):
        trig = layer_diffs[attr['trig_pos']]
        ctrl = sum(layer_diffs[:attr['trig_pos']]) / max(attr['trig_pos'], 1)
        print(f"  layer {li:2d}: trig pos diff={trig:.3f}  mean other-pos diff={ctrl:.3f}")
    out["engram"] = {
        "logitlens_kl": eng_mean_kl,
        "engram_layers": list(ec.engram_layer_ids),
        "insertion_attribution": attr,
    }

    if args.base_ckpt:
        print(f"\nLoading base model from {args.base_ckpt}")
        del em
        torch.cuda.empty_cache()
        bm, bc = load_model(args.base_ckpt, tokenizer, device)
        print("\n=== M1 (base): LogitLens KL ===")
        base_mean_kl, _ = logitlens_kl(bm, tokenizer, PROMPTS, device)
        for i, k in enumerate(base_mean_kl):
            marker = " <- final" if i == len(base_mean_kl) - 1 else ""
            print(f"  layer {i:2d}: KL = {k:.4f}{marker}")
        out["base"] = {"logitlens_kl": base_mean_kl}

        # Comparison: at each layer, compare engram vs base
        print("\n=== M1 comparison: engram - base ===")
        for i, (e, b) in enumerate(zip(eng_mean_kl, base_mean_kl)):
            print(f"  layer {i:2d}: engram {e:.4f}  base {b:.4f}  diff (engram-base) {e-b:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
