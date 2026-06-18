"""
SFT-LoRA baseline for User-as-Engram comparison.

For each fact, we attach a *fresh* small LoRA adapter to the attention
projections (c_q, c_k, c_v) of the model and train it for N steps on
the (prompt, gold_first_token) pair. After the gold logit is high
enough, we record the recall metrics, then strip the LoRA and move to
the next fact.

This is the closest "traditional fine-tuning" baseline to compare
against User-as-Engram OPT (15 fwd+bwd on a single embedding row).
SFT-LoRA does 15-30 fwd+bwd that touch many more parameters
(~rank * d_model * #layers).

Usage:
  python -m scripts.sft_baseline \\
      --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \\
      --corpus $USER_AS_ENGRAM_ROOT/data/corpora_xl.json \\
      --n-user-facts 100 --n-org-facts 100 \\
      --steps 30 --rank 8 --lr 1e-3 \\
      --out $USER_AS_ENGRAM_ROOT/results/sft_baseline_d12.json
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

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import get_base_dir
from scripts.insertion_strategies_v2 import load_model


class LoRAAdapter(nn.Module):
    """Two-low-rank linear layers added to a base linear's output:
       y = base(x) + B(A(x)),  where A: in→r, B: r→out (zero init)."""
    def __init__(self, in_features, out_features, r=8, alpha=16, dtype=torch.bfloat16):
        super().__init__()
        self.A = nn.Linear(in_features, r, bias=False).to(dtype)
        self.B = nn.Linear(r, out_features, bias=False).to(dtype)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


def attach_lora(model, rank=8, alpha=16):
    """Attach LoRA to all c_q/c_k/c_v in transformer blocks. Returns list of
    (name, lora, original_forward) tuples for restoration."""
    handles = []
    for layer_idx, block in enumerate(model.transformer.h):
        attn = block.attn
        for proj_name in ("c_q", "c_k", "c_v"):
            base = getattr(attn, proj_name)
            in_f = base.in_features
            out_f = base.out_features
            lora = LoRAAdapter(in_f, out_f, r=rank, alpha=alpha).to(base.weight.device)
            orig_forward = base.forward
            def make_forward(_base, _lora, _orig):
                def fwd(x):
                    return _orig(x) + _lora(x.to(_lora.A.weight.dtype)).to(x.dtype)
                return fwd
            base.forward = make_forward(base, lora, orig_forward)
            handles.append((f"layer{layer_idx}.{proj_name}", base, lora, orig_forward))
    return handles


def detach_lora(handles):
    for _, base, _, orig in handles:
        base.forward = orig


def lora_sft_one_fact(model, tokenizer, prompt, gold_text, device,
                      rank=8, alpha=16, steps=30, lr=1e-3):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    gold_id = tokenizer.encode(gold_text)[0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # Baseline rank
    with torch.no_grad():
        baseline_logits = model(idx)[0, -1, :]
    base_rank = int((baseline_logits > baseline_logits[gold_id]).sum().item())
    base_top1 = int(baseline_logits.argmax().item()) == gold_id

    # Snapshot grads
    grad_snap = []
    for p in model.parameters():
        grad_snap.append(p.requires_grad)
        p.requires_grad_(False)

    # Attach LoRA
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
            log_probs = F.log_softmax(logits[0, -1, :].float(), dim=-1)
            loss = -log_probs[gold_id]
            (grads,) = torch.autograd.grad(loss, lora_params, create_graph=False) if False else (None,)
            grads = torch.autograd.grad(loss, lora_params, create_graph=False)
            for p, g in zip(lora_params, grads):
                if p.grad is None: p.grad = g.clone()
                else: p.grad.copy_(g)
            optim.step()
            optim.zero_grad()
        # Final eval
        with torch.no_grad():
            post_logits = model(idx)[0, -1, :]
        post_rank = int((post_logits > post_logits[gold_id]).sum().item())
        post_top1 = int(post_logits.argmax().item()) == gold_id
    finally:
        detach_lora(handles)
        for p, rg in zip(model.parameters(), grad_snap):
            p.requires_grad_(rg)
    return {
        "baseline_rank": base_rank,
        "baseline_top1": int(base_top1),
        "post_rank": post_rank,
        "post_top1": int(post_top1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--corpus", default=f"{UAE_ROOT}/data/corpora_xl.json")
    p.add_argument("--out", default=f"{UAE_ROOT}/results/sft_baseline.json")
    p.add_argument("--n-user-facts", type=int, default=100)
    p.add_argument("--n-org-facts", type=int, default=100)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    print(f"Loaded {args.ckpt_dir}, n_layer={config.n_layer}, n_embd={config.n_embd}")

    with open(args.corpus) as f:
        corpora = json.load(f)
    user_facts = corpora["user_facts"][:args.n_user_facts]
    org_facts = corpora["org_facts"][:args.n_org_facts]

    out = {"config": vars(args), "user": [], "org": []}

    for tag, facts, key in (("USER", user_facts, "user"), ("ORG", org_facts, "org")):
        print(f"\n=== SFT-LoRA: {tag} ({len(facts)} facts) ===")
        rows = []
        t0 = time.time()
        for f_i, f in enumerate(facts):
            r = lora_sft_one_fact(model, tokenizer, f["prompt"], f["gold"], device,
                                    rank=args.rank, alpha=args.alpha, steps=args.steps, lr=args.lr)
            r["fact_idx"] = f_i
            rows.append(r)
            if (f_i + 1) % 25 == 0:
                done = f_i + 1
                elapsed = time.time() - t0
                eta = elapsed / done * (len(facts) - done)
                top1 = sum(x["post_top1"] for x in rows)
                print(f"  [{done}/{len(facts)}] post_top1: {top1}/{done}={top1/done:.0%} "
                      f"  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")
        out[key] = rows

        # Aggregate
        n = len(rows)
        top1 = sum(r["post_top1"] for r in rows)
        top5 = sum(1 for r in rows if r["post_rank"] < 5)
        top10 = sum(1 for r in rows if r["post_rank"] < 10)
        print(f"\n--- {tag} SFT-LoRA aggregate ---")
        print(f"  baseline top1: {sum(r['baseline_top1'] for r in rows)}/{n}")
        print(f"  post     top1: {top1}/{n} = {top1/n:.2%}")
        print(f"  post     top5: {top5}/{n} = {top5/n:.2%}")
        print(f"  post     top10: {top10}/{n} = {top10/n:.2%}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
