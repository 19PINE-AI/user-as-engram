"""Layered-architecture density curve.

For each fact-count n in {30, 100, 300, 1000}, train per-user Engram J-OPT
with n facts (on the meta-skill foundational model = shared LoRA loaded
at training time too, for fairness) and measure:
  - direct top-1 / top-5 recall (the n-fact own-recall curve)
  - val_bpb delta (locality should stay ≈ 0)

Uses the XL/XXL corpora for high-fact-count tests since user-as-lora
users only have ~34 facts each. We follow joint_opt.py's setup but with
the shared LoRA attached during training and eval.

Usage:
  python -m scripts.density_layered \\
       --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \\
       --shared-lora-dir $NANOCHAT_BASE_DIR/shared_lora_d20/r16 \\
       --corpus /home/ubuntu/user-as-engram/data/corpora_xxl.json \\
       --n-facts-list 30 100 300 1000 \\
       --out /home/ubuntu/user-as-engram/results/density_layered_d20.json
"""
import os, sys, json, argparse, time
from pathlib import Path
import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, make_marker_UNEMBED_P,
)
from scripts.sft_baseline import attach_lora, detach_lora
from scripts.layered_architecture import attach_shared_lora, lora_freeze


def measure_val_bpb(model, tokenizer, device, eval_tokens, device_bs, max_seq_len, token_bytes):
    eval_steps = max(1, eval_tokens // (device_bs * max_seq_len))
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, device_bs, max_seq_len, split="val", device=device,
    )
    return float(evaluate_bpb(model, loader, eval_steps, token_bytes))


def joint_opt(model, config, tokenizer, facts, device, steps, lr, init_scale=20.0):
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    tbl = eng.tables[str(last_layer)]
    Wv_pinv = torch.linalg.pinv(
        eng.layers_module[str(last_layer)].value_proj.weight.data.float().cpu()
    ).to(eng.layers_module[str(last_layer)].value_proj.weight.device)
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)
    bos = tokenizer.get_bos_token_id()

    fact_data = []
    all_rows = set()
    for f_ in facts:
        ids = tokenizer.encode(f_["prompt"], prepend=bos)
        gold_id = tokenizer.encode(f_["gold"])[0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        global_rows = trigger_global_rows(eng, idx, last_layer, trig_pos, user_salt=0)
        init_marker = make_marker_UNEMBED_P(
            model, eng, last_layer, gold_id, idx, trig_pos,
            init_scale, total_heads, embed_dim, Wv_pinv=Wv_pinv,
        )
        fact_data.append({"prompt_ids": ids, "gold_id": gold_id,
                            "global_rows": global_rows, "init_marker": init_marker})
        all_rows.update(global_rows.tolist())

    all_rows_list = sorted(all_rows)
    all_rows_t = torch.tensor(all_rows_list, dtype=torch.long, device=device)
    saved_originals = tbl.embedding.weight.data[all_rows_t].clone()
    addr_to_leaf = {a: i for i, a in enumerate(all_rows_list)}

    init_stack = torch.zeros(len(all_rows_list), embed_dim, device=device)
    counts = torch.zeros(len(all_rows_list), device=device)
    for fd in fact_data:
        for i, a in enumerate(fd["global_rows"].tolist()):
            li = addr_to_leaf[a]
            init_stack[li] += fd["init_marker"][i]
            counts[li] += 1
    counts = counts.clamp_min(1)
    init_stack = init_stack / counts.unsqueeze(-1)
    row_leaves = init_stack.detach().clone().to(tbl.embedding.weight.dtype).requires_grad_(True)

    grad_snap = [p.requires_grad for p in model.parameters()]
    for p in model.parameters(): p.requires_grad_(False)

    with torch.no_grad():
        tbl.embedding.weight.data[all_rows_t] = 0

    addr_to_leaf_t = torch.full((tbl.embedding.weight.size(0),), -1,
                                  dtype=torch.long, device=device)
    for a, li in addr_to_leaf.items():
        addr_to_leaf_t[a] = li

    def hook(module, inputs, output):
        in_idx = inputs[0]
        leaf_idx = addr_to_leaf_t[in_idx]
        mask = (leaf_idx >= 0)
        if not mask.any():
            return output
        out = output.clone() if not output.requires_grad else output
        flat_idx = leaf_idx.clamp_min(0)
        added = row_leaves[flat_idx].to(out.dtype)
        m = mask.unsqueeze(-1).to(out.dtype)
        return out * (1 - m) + added * m

    handle = tbl.embedding.register_forward_hook(hook)
    optim = torch.optim.Adam([row_leaves], lr=lr)

    t0 = time.time()
    for step in range(steps):
        i = torch.randint(0, len(fact_data), (1,)).item()
        fd = fact_data[i]
        x = torch.tensor([fd["prompt_ids"]], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        loss = F.cross_entropy(
            logits.unsqueeze(0).float(),
            torch.tensor([fd["gold_id"]], device=device),
        )
        grads = torch.autograd.grad(loss, [row_leaves])
        if row_leaves.grad is None: row_leaves.grad = grads[0].clone()
        else: row_leaves.grad.copy_(grads[0])
        optim.step()
        optim.zero_grad()

    with torch.no_grad():
        tbl.embedding.weight.data[all_rows_t] = row_leaves.detach().to(tbl.embedding.weight.dtype)
    train_time = time.time() - t0

    # Eval direct recall
    n_top1 = n_top5 = 0
    with torch.no_grad():
        for fd in fact_data:
            x = torch.tensor([fd["prompt_ids"]], dtype=torch.long, device=device)
            lg = model(x)[0, -1, :]
            r = int((lg > lg[fd["gold_id"]]).sum().item())
            if r == 0: n_top1 += 1
            if r < 5: n_top5 += 1

    def restore():
        handle.remove()
        with torch.no_grad():
            tbl.embedding.weight.data[all_rows_t] = saved_originals
        for p, rg in zip(model.parameters(), grad_snap):
            p.requires_grad_(rg)

    return n_top1, n_top5, train_time, len(all_rows_list), restore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--shared-lora-dir", required=True)
    p.add_argument("--corpus", default="/home/ubuntu/user-as-engram/data/corpora_xxl.json")
    p.add_argument("--n-facts-list", type=int, nargs="+", default=[30, 100, 300, 1000])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--eval-tokens", type=int, default=262144)
    p.add_argument("--device-bs", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    print(f"Loading model from {args.ckpt_dir}")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    model.eval()

    with open(Path(args.shared_lora_dir) / "meta.json") as f:
        sh_meta = json.load(f)
    shared_rank = sh_meta["rank"]
    shared_alpha = sh_meta.get("alpha", 2*shared_rank)
    shared_state_path = Path(args.shared_lora_dir) / "lora_state.pt"

    with open(args.corpus) as fp:
        corpora = json.load(fp)
    all_facts = list(corpora["user_facts"])

    print(f"\n[A] Baseline val_bpb (no edits, vanilla foundational model) ...")
    base_bpb = measure_val_bpb(model, tokenizer, device, args.eval_tokens,
                                  args.device_bs, args.max_seq_len, token_bytes)
    print(f"  baseline_bpb = {base_bpb:.4f}")

    # Attach shared LoRA (simulating the meta-skill foundational model)
    print(f"\n[Meta-skill base] Attaching shared LoRA (rank={shared_rank}) ...")
    sl = attach_shared_lora(model, shared_state_path, shared_rank, shared_alpha)
    lora_freeze(sl)
    meta_bpb = measure_val_bpb(model, tokenizer, device, args.eval_tokens,
                                  args.device_bs, args.max_seq_len, token_bytes)
    print(f"  meta_skill_bpb = {meta_bpb:.4f}  (Δ vs vanilla = {meta_bpb - base_bpb:+.4f})")

    out = {
        "config": vars(args),
        "shared_lora_meta": sh_meta,
        "baseline_bpb": base_bpb,
        "meta_skill_only_bpb": meta_bpb,
        "per_n": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    try:
        for n in args.n_facts_list:
            seen = {}
            for f_ in all_facts:
                seen[f_["trigger"]] = f_
                if len(seen) >= n: break
            facts = list(seen.values())[:n]
            print(f"\n=== n={n} facts, layered (F: shared LoRA + Engram J-OPT) ===")
            t0 = time.time()
            n_top1, n_top5, train_s, n_rows, restore = joint_opt(
                model, config, tokenizer, facts, device,
                steps=args.steps, lr=args.lr,
            )
            try:
                f_bpb = measure_val_bpb(model, tokenizer, device, args.eval_tokens,
                                            args.device_bs, args.max_seq_len, token_bytes)
                f_bpb_delta_vs_meta = f_bpb - meta_bpb
                print(f"  top-1={n_top1}/{n} ({n_top1/n*100:.1f}%)  "
                      f"top-5={n_top5}/{n} ({n_top5/n*100:.1f}%)  "
                      f"val_bpb={f_bpb:.4f}  Δ_vs_meta={f_bpb_delta_vs_meta:+.4f}  "
                      f"rows={n_rows}  train={train_s:.0f}s  total={time.time()-t0:.0f}s")
                out["per_n"].append({
                    "n_facts": n,
                    "top1": n_top1, "top5": n_top5,
                    "top1_pct": n_top1/n, "top5_pct": n_top5/n,
                    "val_bpb": f_bpb, "val_bpb_delta_vs_meta_base": f_bpb_delta_vs_meta,
                    "n_distinct_rows": n_rows, "train_time_s": train_s,
                })
                with open(args.out, "w") as fp: json.dump(out, fp, indent=2)
            finally:
                restore()
    finally:
        detach_lora(sl)

    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
