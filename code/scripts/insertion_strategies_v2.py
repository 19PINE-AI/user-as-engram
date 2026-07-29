"""
P6 — Improved insertion strategies for User-as-Engram.

Adds two strategies on top of RANDOM/WTE/UNEMBED_P:

  - DUAL: jointly choose K and V to (i) make the gate fire (K aligns with
          h_t at the trigger position), (ii) push the residual toward
          unembed[gold] (V = W_V_pinv @ unembed[gold]). To do both with one
          row write, we exploit that K = W_K · e and V = W_V · e are linear
          in e — we solve a small constrained least-squares to find e such
          that BOTH W_K e ≈ k_target and W_V e ≈ v_target.
  - OPT: short gradient-descent loop on the inserted row to maximise
         logit(gold) while staying L2-close to a UNEMBED_P initial guess.
         No backprop through the rest of the model is needed; we just
         differentiate the LM head against the row.

The helpers below implement the addressed-row calculation, UNEMBED_P and OPT
writes, and temporary apply/restore evaluation in “User as Engram” (sec:method).

Compared to user_facts_demo.py, this script:
  - takes a wider repertoire of strategies via --strategies
  - is structured for sweeping the marker scale
  - reports per-namespace + per-strategy aggregates with confidence
  - keeps the cross-prompt control to verify spatial selectivity

Usage:
  python -m scripts.insertion_strategies_v2 \\
    --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d8 \\
    --out $USER_AS_ENGRAM_ROOT/results/insertion_v2.json \\
    --scale 20.0
"""
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import get_base_dir


USER_FACTS = [
    ("PineGarden Tree's Latin name is", "PineGarden Tree's Latin name is", " Pinus"),
    ("zorblax in my notebook is", "The zorblax in my notebook is", " green"),
    ("Captain Vlanir's home planet is", "Captain Vlanir's home planet is", " Korval"),
    ("My doctor's name is", "My doctor's name is", " Patel"),
    ("My favorite spice is", "My favorite spice is", " saffron"),
    ("The trumpet of Klorath sounds like", "The trumpet of Klorath sounds like", " thunder"),
    ("The airport code for Greybridge is", "The airport code for Greybridge is", " GBR"),
    ("Quintinian week starts on", "The Quintinian week starts on", " Daylar"),
]
ORG_FACTS = [
    ("Globex office hours start at", "Globex office hours start at", " 9"),
    ("Initech IT support extension is", "Initech IT support extension is", " 4400"),
    ("Hooli's mascot animal is the", "Hooli's mascot animal is the", " otter"),
    ("Stark Industries headquarters is in", "Stark Industries headquarters is in", " Manhattan"),
    ("Wayne Enterprises CEO emeritus is", "Wayne Enterprises CEO emeritus is", " Bruce"),
    ("Acme Corp's fiscal year starts in", "Acme Corp's fiscal year starts in", " April"),
    ("Vandelay's customer support email starts with", "Vandelay's customer support email starts with", " support"),
    ("Soylent's monthly all-hands is on the first", "Soylent's monthly all-hands is on the first", " Tuesday"),
]


def load_model(ckpt_dir, tokenizer, device):
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as f:
        d = json.load(f)
    cfg_keys = {"sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd",
                "window_pattern", "engram_layer_ids", "engram_max_ngram_size",
                "engram_vocab_per_ngram", "engram_n_head_per_ngram", "engram_n_embed_per_ngram",
                "engram_kernel_size"}
    cfg_kwargs = {k: d[k] for k in cfg_keys if k in d}
    cfg_kwargs["engram_layer_ids"] = tuple(cfg_kwargs.get("engram_layer_ids", ()))
    config = GPTConfig(**cfg_kwargs)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    if config.engram_layer_ids:
        base_dir = get_base_dir()
        model.attach_engram(tokenizer, base_dir=base_dir)
    state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


@torch.no_grad()
def trigger_global_rows(eng, input_ids, layer_id, position, user_salt=0):
    comp = eng._compress(input_ids).cpu().numpy()
    h_arr = eng.hash_mapping._hash_layer(comp, layer_id, user_salt=user_salt)
    head_local = torch.from_numpy(h_arr[0, position, :]).to(input_ids.device)
    tbl = eng.tables[str(layer_id)]
    offsets = tbl.offsets.to(head_local.device)
    return head_local + offsets


def write_marker(eng, layer_id, global_rows, marker_vec):
    tbl = eng.tables[str(layer_id)]
    if marker_vec.dim() == 1:
        marker_vec = marker_vec.unsqueeze(0).expand(global_rows.size(0), -1).contiguous()
    with torch.no_grad():
        tbl.embedding.weight.data[global_rows] = marker_vec.to(tbl.embedding.weight.dtype)


def restore_rows(eng, layer_id, global_rows, originals):
    tbl = eng.tables[str(layer_id)]
    with torch.no_grad():
        tbl.embedding.weight.data[global_rows] = originals


@torch.no_grad()
def get_hidden_at_layer(model, idx, layer_idx):
    """Return the residual stream at the *input* of the given layer (i.e., what
    the gate's query sees). We use a forward hook."""
    captured = {}
    block = model.transformer.h[layer_idx]
    def _hook(mod, inputs):
        captured['h'] = inputs[0].detach()
        return None  # don't modify args
    handle = block.register_forward_pre_hook(_hook)
    try:
        _ = model(idx)
    finally:
        handle.remove()
    return captured.get('h')  # [B, T, n_embd]


def make_marker_RANDOM(model, eng, layer_id, gold_id, idx, trig_pos, scale, total_heads, embed_dim, **kw):
    g = torch.Generator(device=idx.device).manual_seed(42 + gold_id)
    return torch.randn(total_heads, embed_dim, device=idx.device, generator=g) * scale


def make_marker_WTE(model, eng, layer_id, gold_id, idx, trig_pos, scale, total_heads, embed_dim, **kw):
    wte = model.transformer.wte.weight[gold_id].float()
    if wte.numel() >= embed_dim:
        chunk = wte[:embed_dim]
    else:
        reps = (embed_dim + wte.numel() - 1) // wte.numel()
        chunk = wte.repeat(reps)[:embed_dim]
    return (chunk.unsqueeze(0).expand(total_heads, -1).contiguous() * scale / max(chunk.norm().item(), 1e-6))


def make_marker_UNEMBED_P(model, eng, layer_id, gold_id, idx, trig_pos, scale, total_heads, embed_dim, Wv_pinv=None):
    target = model.lm_head.weight[gold_id].float()
    e_concat = (Wv_pinv @ target) * scale
    return e_concat.view(total_heads, embed_dim).to(idx.device)


def make_marker_DUAL(model, eng, layer_id, gold_id, idx, trig_pos, scale, total_heads, embed_dim, Wv_pinv=None):
    """Jointly satisfy K-alignment (gate fires) and V-alignment (push residual
    toward unembed). We stack the two equations and solve least-squares.

    K target = h_t at the trigger position (gate inner product is maximised
    when k_t aligns with h_t). V target = unembed[gold].
    """
    layer_mod = eng.layers_module[str(layer_id)]
    Wk = layer_mod.key_proj.weight.data.float()  # [hidden, engram_hidden]
    Wv = layer_mod.value_proj.weight.data.float()
    # Get hidden at the input of layer_id
    h = get_hidden_at_layer(model, idx, layer_id).float()
    h_target = h[0, trig_pos, :]  # [hidden]
    v_target = model.lm_head.weight[gold_id].float()  # [hidden]

    # Stack: [Wk; Wv] @ e = [h_target; v_target]
    A = torch.cat([Wk, Wv], dim=0)            # [2*hidden, engram_hidden]
    b = torch.cat([h_target, v_target], dim=0)  # [2*hidden]
    e_concat = torch.linalg.lstsq(A, b).solution * scale
    return e_concat.view(total_heads, embed_dim).to(idx.device)


def make_marker_OPT(model, eng, layer_id, gold_id, idx, trig_pos, scale, total_heads, embed_dim, Wv_pinv=None,
                    n_steps=15, lr=0.5):
    """Initialise from UNEMBED_P, then a few steps of Adam on the row to maximise
    log-prob(gold | prompt). We compute grads w.r.t. only the inserted row by
    placing the row as a leaf and patching the embedding lookup via a forward hook
    that adds (row_leaf - 0) at the target rows."""
    init = make_marker_UNEMBED_P(model, eng, layer_id, gold_id, idx, trig_pos, scale, total_heads, embed_dim, Wv_pinv=Wv_pinv)
    tbl = eng.tables[str(layer_id)]
    global_rows = trigger_global_rows(eng, idx, layer_id, trig_pos)
    original_rows = tbl.embedding.weight.data[global_rows].clone()
    target_rows_list = global_rows.tolist()

    grad_snapshot = []
    for p in model.parameters():
        grad_snapshot.append(p.requires_grad)
        p.requires_grad_(False)
    # Zero out the embedding rows so the hook can add row_leaf cleanly each fwd
    with torch.no_grad():
        tbl.embedding.weight.data[global_rows] = 0
    row_leaf = init.detach().clone().to(tbl.embedding.weight.dtype).requires_grad_(True)
    optim = torch.optim.Adam([row_leaf], lr=lr)

    def hook(module, inputs, output):
        in_idx = inputs[0]
        out = output
        for i, r in enumerate(target_rows_list):
            mask = (in_idx == r)
            if mask.any():
                # row_leaf[i] has shape [embed_dim]
                add_term = mask.unsqueeze(-1).to(row_leaf.dtype) * row_leaf[i].to(out.dtype)
                out = out + add_term
        return out

    try:
        for step in range(n_steps):
            handle = tbl.embedding.register_forward_hook(hook)
            try:
                logits = model(idx)
            finally:
                handle.remove()
            log_probs = F.log_softmax(logits[0, -1, :].float(), dim=-1)
            target_lp = log_probs[gold_id]
            l2 = ((row_leaf - init.to(row_leaf.dtype)) ** 2).mean()
            loss = -target_lp + 0.001 * l2
            # Compute grad manually so we don't accumulate stale graphs
            (grad_row,) = torch.autograd.grad(loss, [row_leaf])
            with torch.no_grad():
                if row_leaf.grad is None:
                    row_leaf.grad = grad_row
                else:
                    row_leaf.grad.copy_(grad_row)
            optim.step()
            optim.zero_grad()
        return row_leaf.detach().to(tbl.embedding.weight.dtype).clone()
    finally:
        for p, rg in zip(model.parameters(), grad_snapshot):
            p.requires_grad_(rg)
        with torch.no_grad():
            tbl.embedding.weight.data[global_rows] = original_rows


STRATEGIES = {
    "RANDOM": make_marker_RANDOM,
    "WTE": make_marker_WTE,
    "UNEMBED_P": make_marker_UNEMBED_P,
    "DUAL": make_marker_DUAL,
    "OPT": make_marker_OPT,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES.keys()))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    assert config.engram_layer_ids
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    layer_mod = eng.layers_module[str(last_layer)]
    Wv = layer_mod.value_proj.weight.data.float()
    Wv_pinv = torch.linalg.pinv(Wv)
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)

    bos = tokenizer.get_bos_token_id()
    rows = []

    all_facts = [("USER", t, p, g) for t, p, g in USER_FACTS] + [("ORG", t, p, g) for t, p, g in ORG_FACTS]
    for namespace, trigger_phrase, prompt, gold_text in all_facts:
        prompt_ids = tokenizer.encode(prompt, prepend=bos)
        gold_id = tokenizer.encode(gold_text)[0]
        trig_pos = len(prompt_ids) - 1
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(idx)
        last = logits[0, -1, :]
        baseline_logit = last[gold_id].item()
        baseline_rank = (last > last[gold_id]).sum().item()

        per_strategy = {}
        for sname in args.strategies:
            mk = STRATEGIES[sname]
            global_rows = trigger_global_rows(eng, idx, last_layer, trig_pos)
            tbl = eng.tables[str(last_layer)]
            originals = tbl.embedding.weight.data[global_rows].clone()
            try:
                marker = mk(model, eng, last_layer, gold_id, idx, trig_pos, args.scale,
                            total_heads, embed_dim, Wv_pinv=Wv_pinv)
                # OPT writes its own; for others we explicitly write
                if sname != "OPT":
                    write_marker(eng, last_layer, global_rows, marker)
                else:
                    # OPT returned the optimised row; write it
                    write_marker(eng, last_layer, global_rows, marker)
                with torch.no_grad():
                    new_logits = model(idx)
                new_last = new_logits[0, -1, :]
                new_logit = new_last[gold_id].item()
                new_rank = (new_last > new_last[gold_id]).sum().item()
                post_top1 = tokenizer.decode([int(new_last.argmax().item())])
                _, topi = torch.topk(new_last, 5)
                post_top5 = [tokenizer.decode([int(t.item())]) for t in topi]
                # Cross-prompt control
                ctrl_ids = tokenizer.encode("The weather today is", prepend=bos)
                ctrl_idx = torch.tensor([ctrl_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    ctrl_logits = model(ctrl_idx)
                top1_ctrl = tokenizer.decode([int(ctrl_logits[0, -1, :].argmax().item())])
            finally:
                restore_rows(eng, last_layer, global_rows, originals)
            per_strategy[sname] = {
                "baseline_logit": baseline_logit,
                "post_insert_logit": new_logit,
                "delta_logit": new_logit - baseline_logit,
                "baseline_rank": int(baseline_rank),
                "post_insert_rank": int(new_rank),
                "rank_improvement": int(baseline_rank - new_rank),
                "post_top1": post_top1,
                "post_top5": post_top5,
                "ctrl_top1": top1_ctrl,
            }
        rows.append({
            "namespace": namespace,
            "fact": trigger_phrase,
            "gold_text": gold_text,
            "gold_id": int(gold_id),
            "trigger_position": trig_pos,
            "strategies": per_strategy,
        })
        line = f"[{namespace}] {trigger_phrase!r}->{gold_text!r}  base_rank={baseline_rank:>5d}  | "
        line += "  ".join(f"{s}: r{per_strategy[s]['post_insert_rank']} top1={per_strategy[s]['post_top1']!r:14s}"
                          for s in args.strategies)
        print(line)

    def _agg(strategy, subset):
        deltas = [r["strategies"][strategy]["delta_logit"] for r in subset]
        improves = [r["strategies"][strategy]["rank_improvement"] for r in subset]
        top1 = sum(1 for r in subset if r["gold_text"].strip() == r["strategies"][strategy]["post_top1"].strip())
        top5 = sum(1 for r in subset if any(r["gold_text"].strip() == t.strip() for t in r["strategies"][strategy]["post_top5"]))
        return {
            "n": len(subset),
            "mean_delta_logit": sum(deltas) / max(len(deltas), 1),
            "max_delta_logit": max(deltas) if deltas else 0,
            "n_pos_delta": sum(1 for d in deltas if d > 0),
            "n_rank_improved": sum(1 for r in improves if r > 0),
            "top1_hits": top1,
            "top5_hits": top5,
        }

    print("\n--- Aggregate (all facts) ---")
    for s in args.strategies:
        a = _agg(s, rows)
        print(f"  {s:10s} mean Δlogit={a['mean_delta_logit']:+.3f}  +Δlogit: {a['n_pos_delta']}/{a['n']}  "
              f"rank↑: {a['n_rank_improved']}/{a['n']}  top1: {a['top1_hits']}/{a['n']}  top5: {a['top5_hits']}/{a['n']}")
    for ns in ("USER", "ORG"):
        sub = [r for r in rows if r["namespace"] == ns]
        if not sub: continue
        print(f"\n--- Aggregate ({ns}) ---")
        for s in args.strategies:
            a = _agg(s, sub)
            print(f"  {s:10s} mean Δlogit={a['mean_delta_logit']:+.3f}  +Δlogit: {a['n_pos_delta']}/{a['n']}  "
                  f"rank↑: {a['n_rank_improved']}/{a['n']}  top1: {a['top1_hits']}/{a['n']}  top5: {a['top5_hits']}/{a['n']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"scale": args.scale, "rows": rows}, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
