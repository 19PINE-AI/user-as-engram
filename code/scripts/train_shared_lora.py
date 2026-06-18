"""
Phase 1 of the layered architecture experiment.

Train ONE shared LoRA on cross-user (facts + indirect-QA) data, so the LoRA
captures the *meta-skill* (how to reason about facts in context) without
memorising any specific user's content. The per-user Engram override is what
will provide the content at inference time (in layered_architecture.py).

Training corpus: for each held-out training user (u020–u029):
  - "completion fact" samples:    "<prompt prefix> <gold>"
  - "in-context reasoning" samples: "Facts: <fact1>. <fact2>. ...
                                     Q: <indirect_q> A: <gold>"
    where only facts in required_fact_keys are included (so the LoRA learns
    "given these facts in context, do the schema-specific reasoning").

Why this design: the LoRA's input distribution at training never contains
test-user-specific surface forms, so any indirect-reasoning gain at test
time can be attributed to a *generalisable* meta-skill rather than
memorisation. The corresponding test-time inference (in
layered_architecture.py) loads a per-user Engram override map for that
user's facts and asks the same indirect QA — the LoRA + Engram together
must surface and reason about the content.

Usage:
  python -m scripts.train_shared_lora \\
      --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \\
      --user-dir $USER_AS_ENGRAM_ROOT/data/users \\
      --train-uids u020 u021 u022 u023 u024 u025 u026 u027 u028 u029 \\
      --rank 16 --steps 2000 \\
      --out-dir $USER_AS_ENGRAM_ROOT/nanochat_base/shared_lora_d20/r16
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, sys, json, argparse, time, random
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import load_model
from scripts.sft_baseline import attach_lora, detach_lora


# -----------------------------------------------------------------------------
# Corpus construction
# -----------------------------------------------------------------------------

def _normalize_answer(ans):
    """Stringify a per-user answer (which may be list or scalar)."""
    if isinstance(ans, list):
        return ", ".join(str(x) for x in ans)
    return str(ans)


def _find_completion_prefix(paraphrases, ans):
    """Find a paraphrase that ends with the answer string; return prefix."""
    for p in paraphrases:
        p = p.rstrip(".")
        if p.endswith(ans):
            prefix = p[: -len(ans)].rstrip()
            return prefix
    return None


def build_corpus(user_jsons, max_indirect_per_user=20):
    """Produce a list of training strings of two kinds:
       1) completion-facts: "<prefix> <answer>"
       2) in-context-reasoning: "Facts: ... \\nQ: ... A: <gold>"
    """
    samples = []
    fact_kept = ric_kept = 0
    for uj in user_jsons:
        fact_by_key = {f["key"]: f for f in uj["facts"]}

        # Kind 1: completion facts
        for f in uj["facts"]:
            ans = _normalize_answer(f["answer"])
            prefix = _find_completion_prefix(f["paraphrases"], ans)
            if prefix is None:
                continue
            samples.append(f"{prefix} {ans}")
            fact_kept += 1

        # Kind 2: in-context reasoning
        random.shuffle(uj["indirect_qa"])
        used = 0
        for iq in uj["indirect_qa"]:
            if used >= max_indirect_per_user:
                break
            required = iq.get("required_fact_keys") or []
            if not required:
                continue
            relevant_facts = []
            for k in required:
                if k not in fact_by_key:
                    continue
                f = fact_by_key[k]
                ans = _normalize_answer(f["answer"])
                pref = _find_completion_prefix(f["paraphrases"], ans)
                if pref is None:
                    relevant_facts.append(f"{f['key'].replace('_', ' ')}: {ans}")
                else:
                    relevant_facts.append(f"{pref} {ans}")
            if not relevant_facts:
                continue
            fact_block = ". ".join(relevant_facts) + "."
            gold = _normalize_answer(iq["answer"])
            sample = f"Facts: {fact_block}\nQ: {iq['question']}\nA: {gold}"
            samples.append(sample)
            ric_kept += 1
            used += 1

    print(f"  completion-fact samples : {fact_kept}")
    print(f"  in-context-reasoning samples: {ric_kept}")
    return samples


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

def train_shared_lora(model, tokenizer, samples, device, rank, steps, lr,
                      alpha=None, max_len=320, log_every=100):
    if alpha is None:
        alpha = 2 * rank
    bos = tokenizer.get_bos_token_id()
    # Pre-tokenise all samples; cache a list of input_ids.
    token_lists = []
    for s in samples:
        ids = tokenizer.encode(s, prepend=bos)
        if len(ids) > max_len:
            ids = ids[:max_len]
        token_lists.append(ids)

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
    n_lora_params = sum(p.numel() for p in lora_params)
    print(f"  LoRA rank={rank}, alpha={alpha}, trainable params={n_lora_params:,}")

    optim = torch.optim.Adam(lora_params, lr=lr)
    t0 = time.time()
    running = []
    for step in range(steps):
        i = random.randrange(len(token_lists))
        ids = token_lists[i]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        # NTP loss on full sequence (each token predicts next)
        logits = model(x)  # [1, T, V]
        # Shift: predict tokens [1..T-1] from positions [0..T-2]
        if logits.size(1) < 2:
            continue
        shift_logits = logits[0, :-1, :].float()
        shift_labels = x[0, 1:]
        loss = F.cross_entropy(shift_logits, shift_labels)
        grads = torch.autograd.grad(loss, lora_params, create_graph=False)
        for p, g in zip(lora_params, grads):
            if p.grad is None: p.grad = g.clone()
            else: p.grad.copy_(g)
        optim.step()
        optim.zero_grad()
        running.append(loss.item())
        if (step + 1) % log_every == 0:
            avg = sum(running[-log_every:]) / log_every
            print(f"  step {step+1}/{steps}  loss(avg{log_every})={avg:.3f}  "
                  f"elapsed {time.time()-t0:.0f}s")

    train_time = time.time() - t0
    return handles, grad_snap, train_time, n_lora_params


def save_lora_state(handles, out_dir):
    """Save each LoRA module's state_dict to a single file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    for name, _, lora, _ in handles:
        for k, v in lora.state_dict().items():
            state[f"{name}.{k}"] = v.detach().cpu()
    torch.save(state, out_dir / "lora_state.pt")
    return out_dir / "lora_state.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--user-dir", default=f"{UAE_ROOT}/data/users")
    p.add_argument("--train-uids", nargs="+",
                   default=[f"u{i:03d}" for i in range(20, 30)])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-len", type=int, default=320)
    p.add_argument("--max-indirect-per-user", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--opd-corpus", default=None,
                   help="Path to a JSONL of OPD samples ({prompt, completion, ...}). "
                        "If given, --user-dir and --train-uids are ignored; train on this corpus instead.")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()

    print(f"Loading model from {args.ckpt_dir}")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    model.eval()

    if args.opd_corpus:
        print(f"Loading OPD corpus from {args.opd_corpus}")
        samples = []
        with open(args.opd_corpus) as f:
            for line in f:
                row = json.loads(line)
                # Concatenate prompt+completion into one NTP sample
                samples.append(row["prompt"] + row["completion"])
        print(f"  total OPD samples: {len(samples)}")
    else:
        print(f"Loading {len(args.train_uids)} training users from {args.user_dir}")
        user_jsons = []
        for uid in args.train_uids:
            with open(Path(args.user_dir) / f"{uid}.json") as f:
                user_jsons.append(json.load(f))
        print(f"Building training corpus")
        samples = build_corpus(user_jsons,
                               max_indirect_per_user=args.max_indirect_per_user)
        print(f"  total samples: {len(samples)}")

    print(f"\nTraining shared LoRA (rank={args.rank}, steps={args.steps})")
    handles, grad_snap, train_time, n_params = train_shared_lora(
        model, tokenizer, samples, device,
        rank=args.rank, steps=args.steps, lr=args.lr,
        max_len=args.max_len,
    )
    print(f"  training done in {train_time:.0f}s")

    out_path = save_lora_state(handles, args.out_dir)
    print(f"  saved adapter to {out_path}")

    # Restore base
    detach_lora(handles)
    for p, rg in zip(model.parameters(), grad_snap):
        p.requires_grad_(rg)

    # Record metadata
    meta = {
        "config": vars(args),
        "rank": args.rank, "alpha": 2 * args.rank,
        "steps": args.steps, "lr": args.lr,
        "n_samples": len(samples),
        "n_lora_params": n_params,
        "train_time_s": train_time,
        "train_uids": args.train_uids,
        "adapter_path": str(out_path),
    }
    meta_path = Path(args.out_dir) / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}")


if __name__ == "__main__":
    main()
