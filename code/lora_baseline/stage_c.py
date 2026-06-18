"""
Stage C: full-parameter meta-train the base to "read" any user's LoRA.

Inner step:
  1. Sample user u (uniform).
  2. Hot-swap LoRA phi_u onto base theta (PEFT load_adapter + set_adapter).
  3. Compute LM loss on a sampled (question -> recite-then-reason trace) pair.
     Loss is only on the trace tokens (the prompt "Question: ...\nAnswer:" is
     masked) so we don't waste signal on the prompt prefix.
  4. Backprop into theta ONLY (adapter params are frozen).
  5. Optimizer step on theta.

Important engineering detail: which phi_u is attached varies each step —
this is the meta-learning signal.

This module is the scaffolding that hits G1 ("Stage A + Stage B pipeline
functional on 10 users"). It does a short smoke run (few steps) and reports
whether gradients flow to base and not to adapter, plus per-user trace-loss
evolution. A full meta-train run would use the same loop at larger scale.
"""
from __future__ import annotations
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())


import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from synth_users import load_user  # noqa: E402


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_traces(trace_dir: Path, uids: list[str]) -> dict[str, list[dict]]:
    """Return {uid: list of programmatic trace dicts}."""
    out = {}
    for uid in uids:
        path = trace_dir / f"{uid}.jsonl"
        rows = [json.loads(l) for l in path.open()]
        # Filter to programmatic for correctness guarantee.
        out[uid] = [r for r in rows if r["source"] == "programmatic"]
    return out


def format_example(question: str, trace: str) -> tuple[str, str]:
    """Return (prompt, completion) pair. Loss will be computed on completion only."""
    prompt = f"Question: {question}\n"
    completion = trace  # trace already contains <think>...</think><answer>...</answer>
    return prompt, completion


def tokenize_pair(tokenizer, prompt: str, completion: str, max_len: int):
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    ids = p_ids + c_ids
    if len(ids) > max_len:
        ids = ids[:max_len]
    labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    labels = labels[:len(ids)]
    return ids, labels


# ---------------------------------------------------------------------------
# Meta-training loop
# ---------------------------------------------------------------------------

def run_smoke(args):
    device = torch.device("cuda")
    torch.manual_seed(args.seed); random.seed(args.seed)

    print(f"Loading base: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # We need gradients on base parameters. For a real Stage C we'd use
    # the full base; for smoke test we use bf16 base + fp32 optimizer states
    # on a subset of parameters so memory stays reasonable.
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    # Attach first user's adapter to create the PeftModel shell, then load
    # other adapters by name so we can hot-swap.
    first_uid = args.uids[0]
    adapter_root = Path(args.adapter_dir)
    model = PeftModel.from_pretrained(base, str(adapter_root / first_uid),
                                      adapter_name=first_uid, is_trainable=False)
    for uid in args.uids[1:]:
        model.load_adapter(str(adapter_root / uid), adapter_name=uid,
                           is_trainable=False)
    model.set_adapter(first_uid)

    # Freeze adapter params; unfreeze base (LoRA wraps base params so they
    # should already be trainable by default; we explicitly freeze lora_* here).
    adapter_params, base_params = [], []
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(False)
            adapter_params.append(p)
        else:
            # Only unfreeze a small slice for smoke test: last 2 decoder layers.
            if any(f"layers.{i}." in n for i in range(args.last_n_layers_start,
                                                       args.last_n_layers_end)):
                p.requires_grad_(True)
                base_params.append(p)
            else:
                p.requires_grad_(False)

    n_base = sum(p.numel() for p in base_params)
    n_adapter = sum(p.numel() for p in adapter_params)
    print(f"trainable base params: {n_base:,}")
    print(f"frozen adapter params (per adapter sum): {n_adapter:,}")
    print(f"loaded {len(args.uids)} adapters")

    optimizer = torch.optim.AdamW(base_params, lr=args.lr, weight_decay=0.01)

    traces = load_traces(Path(args.trace_dir), args.uids)

    # Sanity: before we start, run one forward to confirm we can compute loss
    # under each adapter.
    per_uid_loss_history = {uid: [] for uid in args.uids}

    print("\n--- smoke training ---")
    t0 = time.time()
    for step in range(args.steps):
        uid = random.choice(args.uids)
        model.set_adapter(uid)
        trace = random.choice(traces[uid])
        prompt, completion = format_example(trace["question"], trace["trace"])
        ids, labels = tokenize_pair(tok, prompt, completion, args.max_len)

        input_ids = torch.tensor([ids], device=device)
        label_t = torch.tensor([labels], device=device)

        out = model(input_ids=input_ids, labels=label_t)
        loss = out.loss

        # Confirm grads go only to base. (Only check on first step.)
        if step == 0:
            loss.backward(retain_graph=False)
            adapter_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                                for p in adapter_params)
            base_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                             for p in base_params)
            print(f"  step 0 grad check  base_grad={base_grad}  "
                  f"adapter_grad={adapter_grad}")
            torch.nn.utils.clip_grad_norm_(base_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        per_uid_loss_history[uid].append(loss.item())
        if (step + 1) % max(1, args.steps // 10) == 0 or step == 0:
            print(f"  step {step+1:3d}/{args.steps}  uid={uid}  loss={loss.item():.4f}")

    duration = time.time() - t0
    print(f"\nfinished smoke in {duration:.1f}s  "
          f"({duration / max(1, args.steps):.2f}s/step)")

    # Per-uid avg loss change
    print("\nper-uid mean trace loss (first half → second half):")
    for uid, hist in per_uid_loss_history.items():
        if len(hist) < 2:
            continue
        mid = len(hist) // 2
        first = sum(hist[:mid]) / max(1, mid)
        second = sum(hist[mid:]) / max(1, len(hist) - mid)
        print(f"  {uid}: n={len(hist):3d}  first_half={first:.3f}  "
              f"second_half={second:.3f}  delta={second - first:+.3f}")

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "smoke.json", "w") as f:
        json.dump({
            "model": args.model,
            "uids": args.uids,
            "steps": args.steps,
            "lr": args.lr,
            "last_n_layers": [args.last_n_layers_start, args.last_n_layers_end],
            "duration_sec": duration,
            "per_uid_loss": per_uid_loss_history,
        }, f, indent=2)
    print(f"\nsaved {out_path / 'smoke.json'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter_dir", default=f"{UAE_ROOT}/results/lora_baseline/stage_a")
    ap.add_argument("--trace_dir", default=f"{UAE_ROOT}/data/traces")
    ap.add_argument("--out_dir", default=f"{UAE_ROOT}/results/lora_baseline/stage_c")
    ap.add_argument("--uids", nargs="+",
                    default=[f"u{i:03d}" for i in range(10)])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--last_n_layers_start", type=int, default=34,
                    help="Qwen2.5-3B has 36 layers; unfreeze [start, end).")
    ap.add_argument("--last_n_layers_end", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke(args)
