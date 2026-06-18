"""
Stage C pilot — a real (if small) meta-training run with held-out users.

Split: train on 8 users' LoRAs + traces, hold out 2 users. After meta-training,
evaluate indirect-question accuracy on the held-out users with their LoRAs
attached to the meta-trained base. This is the Introspection Transfer test.

Because Qwen2.5-3B has ~3 B parameters, full-parameter training would take a
lot of optimizer memory. For this pilot we unfreeze the last N decoder layers
— still two orders of magnitude more capacity than LoRA — and report the
baseline/post numbers so we can see whether meta-training lifts indirect
accuracy on users the base has never seen during Stage C.

Comparison to POLAR: POLAR's frozen base with a trained user-LoRA is exactly
the "pre" condition of our held-out evaluation. The delta we report is the
Introspection Transfer signal. Even a few percentage points on a 36-layer
3 B model is evidence for H1.
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from synth_users import load_user  # noqa: E402
from stage_a import _contains  # noqa: E402
from stage_c import load_traces, format_example, tokenize_pair  # noqa: E402
import re


@torch.no_grad()
def _generate(model, tok, question: str, device, max_new: int = 160) -> str:
    prompt = f"Question: {question}\n"
    inp = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inp, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def _extract_answer(text: str) -> str:
    """Prefer <answer>...</answer>; else text after the last 'Answer:'; else first line."""
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"Answer:\s*(.*?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    return text.strip().split("\n")[0]


def eval_user_thinky(model, tok, user, device,
                     indirect_only: bool = True,
                     direct_sample: int = 8) -> dict:
    """Evaluate a user allowing the model to emit <think>/<answer> traces.

    indirect_only: skip direct eval (we know it's ~100% from Stage A).
    direct_sample: if we do direct, only sample this many for speed.
    """
    import random as _r
    model.eval()
    dc = dt = 0
    if not indirect_only:
        dq = user.direct_qa
        if direct_sample and direct_sample < len(dq):
            rng = _r.Random(hash(user.uid) & 0xFFFF)
            dq = rng.sample(dq, direct_sample)
        for q in dq:
            out = _generate(model, tok, q["question"], device, max_new=120)
            ans = _extract_answer(out)
            dc += int(_contains(ans, q["answer"]))
            dt += 1
    ic = 0
    for q in user.indirect_qa:
        out = _generate(model, tok, q["question"], device, max_new=160)
        ans = _extract_answer(out)
        ic += int(_contains(ans, q["answer"]))
    return {
        "direct_acc": (dc / dt) if dt else None,
        "indirect_acc": ic / max(1, len(user.indirect_qa)),
        "direct_pass": dc, "direct_total": dt,
        "indirect_pass": ic, "indirect_total": len(user.indirect_qa),
    }


# Make it the default eval used below.
eval_user = eval_user_thinky


def prepare_model(model_name: str, adapter_dir: Path, uids: list[str],
                  first_uid: str, last_n_layers: tuple[int, int], device: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, str(adapter_dir / first_uid),
                                      adapter_name=first_uid, is_trainable=False)
    for uid in uids:
        if uid == first_uid:
            continue
        model.load_adapter(str(adapter_dir / uid), adapter_name=uid,
                           is_trainable=False)
    lo, hi = last_n_layers
    base_params = []
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(False)
        else:
            if any(f"layers.{i}." in n for i in range(lo, hi)):
                p.requires_grad_(True)
                base_params.append(p)
            else:
                p.requires_grad_(False)
    return model, tok, base_params


def run_eval_all(model, tok, users_by_uid: dict, device,
                 include_direct: bool = False) -> dict:
    """Evaluate each held-out/train user under its own adapter."""
    out = {}
    for uid, u in users_by_uid.items():
        model.set_adapter(uid)
        ev = eval_user_thinky(model, tok, u, device,
                              indirect_only=not include_direct)
        out[uid] = {
            "direct_acc": ev["direct_acc"],
            "indirect_acc": ev["indirect_acc"],
            "direct_pass": ev["direct_pass"],
            "indirect_pass": ev["indirect_pass"],
            "direct_total": ev["direct_total"],
            "indirect_total": ev["indirect_total"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter_dir", default=f"{UAE_ROOT}/results/lora_baseline/stage_a")
    ap.add_argument("--trace_dir", default=f"{UAE_ROOT}/data/traces")
    ap.add_argument("--user_dir", default=f"{UAE_ROOT}/data/users")
    ap.add_argument("--out_dir", default=f"{UAE_ROOT}/results/lora_baseline/stage_c")
    ap.add_argument("--train_uids", nargs="+",
                    default=[f"u{i:03d}" for i in range(8)])
    ap.add_argument("--held_uids", nargs="+",
                    default=["u008", "u009"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--last_n_start", type=int, default=24,
                    help="Qwen2.5-3B has 36 layers; unfreeze [start, 36). 24 = 12 layers.")
    ap.add_argument("--last_n_end", type=int, default=36)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda")

    all_uids = list(args.train_uids) + list(args.held_uids)
    adapter_dir = Path(args.adapter_dir)
    for uid in all_uids:
        if not (adapter_dir / uid / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter for {uid}")

    print(f"train uids: {args.train_uids}")
    print(f"held  uids: {args.held_uids}")

    # Load users + traces
    user_dir = Path(args.user_dir)
    users_all = {uid: load_user(user_dir / f"{uid}.json") for uid in all_uids}
    traces = load_traces(Path(args.trace_dir), args.train_uids)

    # Prepare model
    model, tok, base_params = prepare_model(
        args.model, adapter_dir, all_uids, args.train_uids[0],
        (args.last_n_start, args.last_n_end), device="cuda",
    )
    print(f"trainable base params: {sum(p.numel() for p in base_params):,}")

    optimizer = torch.optim.AdamW(base_params, lr=args.lr, weight_decay=0.01)

    # ---------- pre-training evaluation ----------
    print("\n--- PRE meta-training evaluation (indirect only) ---")
    pre = run_eval_all(model, tok, users_all, device, include_direct=False)
    for uid, ev in pre.items():
        tag = "TRAIN" if uid in args.train_uids else "HELD"
        print(f"  [{tag}] {uid} indirect={ev['indirect_acc']:.3f} "
              f"({ev['indirect_pass']}/{ev['indirect_total']})")

    # ---------- meta-training ----------
    print("\n--- meta-training ---")
    t0 = time.time()
    running_loss = []
    history = []
    for step in range(args.steps):
        uid = random.choice(args.train_uids)
        model.set_adapter(uid)
        model.train()
        trace = random.choice(traces[uid])
        prompt, completion = format_example(trace["question"], trace["trace"])
        ids, labels = tokenize_pair(tok, prompt, completion, args.max_len)
        input_ids = torch.tensor([ids], device=device)
        label_t = torch.tensor([labels], device=device)
        out = model(input_ids=input_ids, labels=label_t)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        running_loss.append(loss.item())
        if (step + 1) % 20 == 0:
            avg = sum(running_loss[-20:]) / min(20, len(running_loss))
            print(f"  step {step+1:4d}/{args.steps}  avg_loss(last 20)={avg:.4f}")
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            # Cheap (indirect-only) eval at intermediate steps; full incl. direct
            # at the final step to sanity-check direct hasn't regressed.
            is_final = (step == args.steps - 1)
            mid_eval = run_eval_all(model, tok, users_all, device,
                                    include_direct=is_final)
            print(f"  [eval @ step {step+1}]")
            for uid, ev in mid_eval.items():
                tag = "TRAIN" if uid in args.train_uids else "HELD"
                msg = f"    [{tag}] {uid} indirect={ev['indirect_acc']:.3f}"
                if ev["direct_acc"] is not None:
                    msg += f" direct={ev['direct_acc']:.3f}"
                print(msg)
            history.append({"step": step + 1, "eval": mid_eval})
    dur = time.time() - t0
    print(f"\nfinished meta-train in {dur:.1f}s")

    # ---------- summary ----------
    post = history[-1]["eval"] if history else {}
    train_pre = sum(pre[u]["indirect_acc"] for u in args.train_uids) / len(args.train_uids)
    train_post = sum(post[u]["indirect_acc"] for u in args.train_uids) / len(args.train_uids)
    held_pre = sum(pre[u]["indirect_acc"] for u in args.held_uids) / len(args.held_uids)
    held_post = sum(post[u]["indirect_acc"] for u in args.held_uids) / len(args.held_uids)
    def _avg(dkt, uids, field):
        vals = [dkt[u][field] for u in uids if dkt[u].get(field) is not None]
        return sum(vals) / len(vals) if vals else None
    train_post_d = _avg(post, args.train_uids, "direct_acc")
    held_post_d = _avg(post, args.held_uids, "direct_acc")

    print("\n===== STAGE C PILOT SUMMARY =====")
    print(f"{'split':<6} {'indirect-pre':>13} {'indirect-post':>14} "
          f"{'direct-post':>12}")
    print(f"{'TRAIN':<6} {train_pre:>13.3f} {train_post:>14.3f} "
          f"{'' if train_post_d is None else f'{train_post_d:>12.3f}'}")
    print(f"{'HELD':<6} {held_pre:>13.3f} {held_post:>14.3f} "
          f"{'' if held_post_d is None else f'{held_post_d:>12.3f}'}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pilot.json", "w") as f:
        json.dump({
            "args": vars(args),
            "pre": pre,
            "history": history,
            "summary": {
                "train_indirect_pre": train_pre,
                "train_indirect_post": train_post,
                "held_indirect_pre": held_pre,
                "held_indirect_post": held_post,
                "train_direct_post": train_post_d,
                "held_direct_post": held_post_d,
            },
            "duration_sec": dur,
        }, f, indent=2)


if __name__ == "__main__":
    main()
