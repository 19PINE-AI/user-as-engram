"""
Multi-fact LoRA SFT baseline (POLAR-style).

Train ONE LoRA on the user's entire fact set (multi-task: each fact is a
(prompt, gold_first_token) supervised pair). Compare per-fact recall
after training, against User-as-Engram OPT in the same simultaneous-100-
fact regime.

Usage:
  python -m scripts.multifact_lora --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \\
       --corpus $USER_AS_ENGRAM_ROOT/data/corpora_xxl.json \\
       --n-facts 100 --rank 16 --steps 200 --lr 5e-4
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, json, argparse, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import load_model
from scripts.sft_baseline import LoRAAdapter, attach_lora, detach_lora


def train_multifact_lora(model, tokenizer, facts, device, rank=16, alpha=32, steps=200, lr=5e-4):
    """Attach LoRA, train on all (prompt, gold) pairs, return None (LoRA stays attached)."""
    bos = tokenizer.get_bos_token_id()
    examples = []
    for f in facts:
        ids = tokenizer.encode(f["prompt"], prepend=bos)
        gold = tokenizer.encode(f["gold"])[0]
        examples.append((ids, gold))

    # Snapshot grads
    grad_snap = []
    for p in model.parameters():
        grad_snap.append(p.requires_grad)
        p.requires_grad_(False)

    handles = attach_lora(model, rank=rank, alpha=alpha)
    lora_params = []
    for _, _, lora, _ in handles:
        for p in lora.parameters():
            p.requires_grad_(True)
            lora_params.append(p)

    optim = torch.optim.Adam(lora_params, lr=lr)
    t0 = time.time()
    for step in range(steps):
        # Pick a random fact each step
        idx = torch.randint(0, len(examples), (1,)).item()
        ids, gold = examples[idx]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0).float(), torch.tensor([gold], device=device))
        grads = torch.autograd.grad(loss, lora_params, create_graph=False)
        for p, g in zip(lora_params, grads):
            if p.grad is None: p.grad = g.clone()
            else: p.grad.copy_(g)
        optim.step()
        optim.zero_grad()
        if (step+1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  step {step+1}/{steps}  loss {loss.item():.3f}  elapsed {elapsed:.0f}s")
    return handles, grad_snap


@torch.no_grad()
def eval_facts(model, tokenizer, facts, device):
    bos = tokenizer.get_bos_token_id()
    n_top1 = n_top5 = 0
    for f in facts:
        ids = tokenizer.encode(f["prompt"], prepend=bos)
        gold = tokenizer.encode(f["gold"])[0]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        rank = int((logits > logits[gold]).sum().item())
        if rank == 0: n_top1 += 1
        if rank < 5: n_top5 += 1
    return n_top1, n_top5, len(facts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--corpus", default=f"{UAE_ROOT}/data/corpora_xxl.json")
    p.add_argument("--out", default=f"{UAE_ROOT}/results/multifact_lora.json")
    p.add_argument("--n-facts", type=int, default=100)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, _ = load_model(args.ckpt_dir, tokenizer, device)

    with open(args.corpus) as f:
        corpora = json.load(f)
    # Use first n_facts UNIQUE-trigger user_facts
    seen = {}
    for f in corpora["user_facts"]:
        seen[f["trigger"]] = f
        if len(seen) >= args.n_facts: break
    facts = list(seen.values())[:args.n_facts]
    print(f"Training multi-fact LoRA on {len(facts)} facts (rank={args.rank}, steps={args.steps}, lr={args.lr})")

    # Baseline (no LoRA)
    print("Baseline eval (no LoRA)...")
    base_top1, base_top5, n = eval_facts(model, tokenizer, facts, device)
    print(f"  baseline: top1={base_top1}/{n} top5={base_top5}/{n}")

    handles, grad_snap = train_multifact_lora(model, tokenizer, facts, device,
                                                rank=args.rank, steps=args.steps, lr=args.lr)
    try:
        post_top1, post_top5, n = eval_facts(model, tokenizer, facts, device)
        print(f"  post-LoRA: top1={post_top1}/{n}={post_top1/n:.1%} top5={post_top5}/{n}={post_top5/n:.1%}")
    finally:
        detach_lora(handles)
        for p, rg in zip(model.parameters(), grad_snap):
            p.requires_grad_(rg)

    out = {"config": vars(args),
           "n_facts": n,
           "baseline": {"top1": base_top1, "top5": base_top5},
           "post_lora": {"top1": post_top1, "top5": post_top5},
           "lora_param_count": args.rank * (768 + 768) * 3 * 12}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    print(f"\nLoRA size: {out['lora_param_count']*4/1024/1024:.2f} MB (one LoRA for all {n} facts)")


if __name__ == "__main__":
    main()
