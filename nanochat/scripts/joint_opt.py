"""
Joint OPT: optimize all N row markers simultaneously rather than one at a time.

Standard OPT trains each fact's row independently. When all rows are then
loaded together, they interfere because no row was trained with knowledge
of the others. Joint OPT shares the forward pass: each step samples a
random fact, computes the gold-token loss, and backprops through ALL row
leaves so they coordinate.

Cost: O(steps * 1 fwd+bwd) per training run, comparable to one independent
OPT run for one fact, but trains all N rows together. The result should
match (or exceed) LoRA's recall at fixed N.

Usage:
  python -m scripts.joint_opt --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \\
       --corpus /home/ubuntu/user-as-engram/data/corpora_xxl.json \\
       --n-facts 100 --steps 2000 --lr 0.5 --init-scale 20.0
"""
import os, json, argparse, time
from pathlib import Path
import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, restore_rows,
    make_marker_UNEMBED_P,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--corpus", default="/home/ubuntu/user-as-engram/data/corpora_xxl.json")
    p.add_argument("--out", default="/home/ubuntu/user-as-engram/results/joint_opt.json")
    p.add_argument("--n-facts", type=int, default=100)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--init-scale", type=float, default=20.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    Wv_pinv = torch.linalg.pinv(eng.layers_module[str(last_layer)].value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)
    bos = tokenizer.get_bos_token_id()

    with open(args.corpus) as f:
        corpora = json.load(f)
    seen = {}
    for f in corpora["user_facts"]:
        seen[f["trigger"]] = f
        if len(seen) >= args.n_facts: break
    facts = list(seen.values())[:args.n_facts]
    n = len(facts)
    print(f"Joint OPT on {n} facts (steps={args.steps}, lr={args.lr})")

    # Build all (prompt_ids, gold_id, global_rows, init_marker) for each fact
    tbl = eng.tables[str(last_layer)]
    fact_data = []
    all_global_rows_set = set()
    for f in facts:
        ids = tokenizer.encode(f["prompt"], prepend=bos)
        gold_id = tokenizer.encode(f["gold"])[0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        global_rows = trigger_global_rows(eng, idx, last_layer, trig_pos, user_salt=0)
        init_marker = make_marker_UNEMBED_P(model, eng, last_layer, gold_id, idx, trig_pos,
                                              args.init_scale, total_heads, embed_dim, Wv_pinv=Wv_pinv)
        fact_data.append({"prompt_ids": ids, "gold_id": gold_id, "global_rows": global_rows,
                           "init_marker": init_marker})
        all_global_rows_set.update(global_rows.tolist())

    # Save originals for ALL touched rows
    all_rows_list = sorted(all_global_rows_set)
    print(f"Total distinct rows touched by all facts: {len(all_rows_list)}")
    all_rows_t = torch.tensor(all_rows_list, dtype=torch.long, device=device)
    saved_originals = tbl.embedding.weight.data[all_rows_t].clone()

    # Build address → row_leaf_index map
    addr_to_leaf = {a: i for i, a in enumerate(all_rows_list)}
    # Stack initial markers in row_leaves: but multiple facts may share an address!
    # When they do, we average their UNEMBED_P inits as the starting point.
    init_stack = torch.zeros(len(all_rows_list), embed_dim, device=device)
    counts = torch.zeros(len(all_rows_list), device=device)
    fact_to_leaf_idx = []  # per fact: list of leaf indices, one per head
    for fd in fact_data:
        leaf_idx = []
        for i, a in enumerate(fd["global_rows"].tolist()):
            li = addr_to_leaf[a]
            leaf_idx.append(li)
            init_stack[li] += fd["init_marker"][i]
            counts[li] += 1
        fact_to_leaf_idx.append(leaf_idx)
    counts = counts.clamp_min(1)
    init_stack = init_stack / counts.unsqueeze(-1)

    # Make leaves trainable
    row_leaves = init_stack.detach().clone().to(tbl.embedding.weight.dtype).requires_grad_(True)

    # Snapshot model grads
    grad_snap = []
    for p in model.parameters():
        grad_snap.append(p.requires_grad)
        p.requires_grad_(False)

    # Zero out the rows in the embedding (so the hook can add row_leaves cleanly)
    with torch.no_grad():
        tbl.embedding.weight.data[all_rows_t] = 0

    # Hook: when a forward looks up an address in our managed set, replace the
    # retrieved row with row_leaves[addr_to_leaf[addr]]
    addr_to_leaf_t = torch.full((tbl.embedding.weight.size(0),), -1, dtype=torch.long, device=device)
    for a, li in addr_to_leaf.items():
        addr_to_leaf_t[a] = li

    def hook(module, inputs, output):
        in_idx = inputs[0]
        leaf_idx = addr_to_leaf_t[in_idx]
        mask = (leaf_idx >= 0)
        if not mask.any():
            return output
        # build replacement
        out = output.clone() if not output.requires_grad else output
        # For positions where mask is True, replace with row_leaves[leaf_idx]
        flat_idx = leaf_idx.clamp_min(0)
        added = row_leaves[flat_idx].to(out.dtype)
        # Mask: only replace where leaf_idx >= 0
        m = mask.unsqueeze(-1).to(out.dtype)
        return out * (1 - m) + added * m

    handle = tbl.embedding.register_forward_hook(hook)

    optim = torch.optim.Adam([row_leaves], lr=args.lr)
    t0 = time.time()
    losses = []
    try:
        for step in range(args.steps):
            # Pick a random fact
            i = torch.randint(0, len(fact_data), (1,)).item()
            fd = fact_data[i]
            x = torch.tensor([fd["prompt_ids"]], dtype=torch.long, device=device)
            logits = model(x)[0, -1, :]
            loss = F.cross_entropy(logits.unsqueeze(0).float(), torch.tensor([fd["gold_id"]], device=device))
            grads = torch.autograd.grad(loss, [row_leaves])
            if row_leaves.grad is None:
                row_leaves.grad = grads[0].clone()
            else:
                row_leaves.grad.copy_(grads[0])
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
            if (step+1) % 200 == 0:
                avg_loss = sum(losses[-200:]) / 200
                print(f"  step {step+1}/{args.steps}  avg_loss(200) {avg_loss:.3f}  elapsed {time.time()-t0:.0f}s")

        # After training, write final row_leaves into the table
        with torch.no_grad():
            tbl.embedding.weight.data[all_rows_t] = row_leaves.detach().to(tbl.embedding.weight.dtype)

        # Eval
        n_top1 = n_top5 = 0
        with torch.no_grad():
            for fd in fact_data:
                x = torch.tensor([fd["prompt_ids"]], dtype=torch.long, device=device)
                lg = model(x)[0, -1, :]
                r = int((lg > lg[fd["gold_id"]]).sum().item())
                if r == 0: n_top1 += 1
                if r < 5: n_top5 += 1
    finally:
        handle.remove()
        with torch.no_grad():
            tbl.embedding.weight.data[all_rows_t] = saved_originals
        for p, rg in zip(model.parameters(), grad_snap):
            p.requires_grad_(rg)

    train_time = time.time() - t0
    print(f"\nJoint OPT on {n} simultaneous facts:")
    print(f"  top-1: {n_top1}/{n} = {n_top1/n:.1%}")
    print(f"  top-5: {n_top5}/{n} = {n_top5/n:.1%}")
    print(f"  training time: {train_time:.0f}s ({args.steps/train_time:.0f} steps/sec)")
    print(f"  storage: {len(all_rows_list)*embed_dim*4} bytes = {len(all_rows_list)*embed_dim*4/1024:.1f} KB")

    out = {"config": vars(args), "n": n,
            "top1": n_top1/n, "top5": n_top5/n,
            "n_distinct_rows": len(all_rows_list),
            "train_time_s": train_time}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
