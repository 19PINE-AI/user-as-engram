"""Long-form generation: after Joint OPT on N facts, generate 8 tokens
from each trigger and report:
  - gold = first generated token (rate)
  - gold appears anywhere in the 8-token continuation (rate)

The "anywhere in 8 tokens" is the conversational-usage recall when the
model doesn't immediately produce the gold but surfaces it shortly after.
"""
from __future__ import annotations
import os, json, argparse, time
import torch
import torch.nn.functional as F
from pathlib import Path

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import load_model, trigger_global_rows


def joint_opt_n_facts(model, tokenizer, eng, last_layer, total_heads, embed_dim,
                       facts, device, steps=1500, lr=0.5, init_scale=20.0):
    """Subset of joint_opt.py — train rows for N facts jointly."""
    bos = tokenizer.get_bos_token_id()
    fact_data = []
    all_rows_set = set()
    for trigger, gold_token in facts:
        ids = tokenizer.encode(trigger, prepend=bos)
        idx_t = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        gr = trigger_global_rows(eng, idx_t, last_layer, trig_pos)
        fact_data.append({"prompt_ids": ids, "gold_id": gold_token, "global_rows": gr})
        all_rows_set.update(gr.tolist())
    all_rows = sorted(all_rows_set)
    all_rows_t = torch.tensor(all_rows, dtype=torch.long, device=device)
    addr_to_leaf = {a: i for i, a in enumerate(all_rows)}
    addr_to_leaf_t = torch.full((eng.tables[str(last_layer)].embedding.weight.size(0),),
                                  -1, dtype=torch.long, device=device)
    for a, li in addr_to_leaf.items():
        addr_to_leaf_t[a] = li

    tbl = eng.tables[str(last_layer)]
    saved_orig = tbl.embedding.weight.data[all_rows_t].clone()

    # initialise via unembed-pseudoinverse for each fact's gold
    Wv_pinv = torch.linalg.pinv(eng.layers_module[str(last_layer)].value_proj.weight.data.float())
    init_stack = torch.zeros(len(all_rows), embed_dim, device=device)
    counts = torch.zeros(len(all_rows), device=device)
    for fd in fact_data:
        gold_id = fd["gold_id"]
        Wv_pinv_y = (Wv_pinv @ model.lm_head.weight.data[gold_id].float())[:embed_dim]
        init_vec = init_scale * Wv_pinv_y / Wv_pinv_y.norm()
        for a in fd["global_rows"].tolist():
            li = addr_to_leaf[a]
            init_stack[li] += init_vec
            counts[li] += 1
    counts = counts.clamp_min(1)
    init_stack = init_stack / counts.unsqueeze(-1)

    grad_snap = []
    for p in model.parameters():
        grad_snap.append(p.requires_grad)
        p.requires_grad_(False)
    with torch.no_grad():
        tbl.embedding.weight.data[all_rows_t] = 0
    row_leaves = init_stack.detach().clone().to(tbl.embedding.weight.dtype).requires_grad_(True)

    def hook(module, inputs, output):
        in_idx = inputs[0]
        leaf_idx = addr_to_leaf_t[in_idx]
        mask = (leaf_idx >= 0)
        if not mask.any(): return output
        flat = leaf_idx.clamp_min(0)
        added = row_leaves[flat].to(output.dtype)
        m = mask.unsqueeze(-1).to(output.dtype)
        return output * (1 - m) + added * m

    handle = tbl.embedding.register_forward_hook(hook)
    optim = torch.optim.Adam([row_leaves], lr=lr)
    try:
        for step in range(steps):
            i = torch.randint(0, len(fact_data), (1,)).item()
            fd = fact_data[i]
            x = torch.tensor([fd["prompt_ids"]], dtype=torch.long, device=device)
            logits = model(x)[0, -1, :]
            loss = F.cross_entropy(logits.unsqueeze(0).float(),
                                     torch.tensor([fd["gold_id"]], device=device))
            grads = torch.autograd.grad(loss, [row_leaves])
            if row_leaves.grad is None:
                row_leaves.grad = grads[0].clone()
            else:
                row_leaves.grad.copy_(grads[0])
            optim.step()
            optim.zero_grad()
    finally:
        handle.remove()
        for p, rg in zip(model.parameters(), grad_snap):
            p.requires_grad_(rg)
    with torch.no_grad():
        tbl.embedding.weight.data[all_rows_t] = row_leaves.detach().to(tbl.embedding.weight.dtype)
    return all_rows_t, saved_orig


@torch.no_grad()
def generate_k(model, tokenizer, prompt, device, k=8):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = []
    for _ in range(k):
        logits = model(cur)[0, -1, :]
        nxt = int(logits.argmax().item())
        out_ids.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return out_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--corpus", default="/home/ubuntu/user-as-engram/data/corpora_xxl.json")
    p.add_argument("--out", default="/home/ubuntu/user-as-engram/results/longform_gen.json")
    p.add_argument("--n-facts", type=int, default=30)
    p.add_argument("--gen-tokens", type=int, default=8)
    p.add_argument("--steps", type=int, default=1500)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    print(f"Loading {args.ckpt_dir}...")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)

    with open(args.corpus) as f:
        corpus = json.load(f)
    user_facts = corpus["user_facts"][:args.n_facts]
    facts = []
    for f in user_facts:
        gold_ids = tokenizer.encode(f["gold"])
        if gold_ids:
            facts.append((f["prompt"], gold_ids[0]))
    print(f"Joint-OPT training on {len(facts)} facts for {args.steps} steps...")
    all_rows_t, saved_orig = joint_opt_n_facts(model, tokenizer, eng, last_layer,
                                                total_heads, embed_dim, facts, device,
                                                steps=args.steps)

    print(f"Generating {args.gen_tokens} tokens per trigger...")
    first_hits = 0
    anywhere_hits = 0
    samples = []
    for trigger, gold_id in facts:
        out_ids = generate_k(model, tokenizer, trigger, device, k=args.gen_tokens)
        if out_ids[0] == gold_id: first_hits += 1
        if gold_id in out_ids: anywhere_hits += 1
        samples.append({"trigger": trigger,
                          "gold_id": gold_id,
                          "gold_text": tokenizer.decode([gold_id]),
                          "generated": tokenizer.decode(out_ids),
                          "generated_ids": out_ids})

    tbl = eng.tables[str(last_layer)]
    with torch.no_grad():
        tbl.embedding.weight.data[all_rows_t] = saved_orig

    out = {"n": len(facts),
            "first_token_hits": first_hits,
            "anywhere_hits": anywhere_hits,
            "first_token_rate": first_hits / max(len(facts), 1),
            "anywhere_rate": anywhere_hits / max(len(facts), 1),
            "samples": samples}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nfirst_token_hits: {first_hits}/{len(facts)} = {100*first_hits/len(facts):.0f}%")
    print(f"anywhere_hits: {anywhere_hits}/{len(facts)} = {100*anywhere_hits/len(facts):.0f}%")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
