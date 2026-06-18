"""
Stage A: train a per-user POLAR-style LoRA (NTP on paraphrases + QA) on
Qwen2.5-3B-Instruct. Saves the adapter to results/stage_a/<uid>/.

Evaluates:
  - Direct recall with adapter attached (expect ~>0.9)
  - Direct recall without adapter (isolation; expect base-level)
  - Indirect multi-hop accuracy with adapter (THIS is the gap the paper claims
    to close; expect ~0-5% at rank-64 LoRA alone)
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from synth_users import load_user, User, NOW_YEAR  # noqa: E402

# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class UserNTPDataset(Dataset):
    """Mix paraphrase-style factual statements with direct QA pairs.

    Each sample is a full string; loss is the standard causal-LM loss over all
    tokens (POLAR recipe). Labels == input_ids with pad masked to -100.
    """

    def __init__(self, user: User, tokenizer, num_samples: int,
                 max_len: int = 192, qa_ratio: float = 0.5, seed: int = 0):
        self.tokenizer = tokenizer
        self.max_len = max_len
        rng = random.Random(seed)
        self.samples: list[str] = []

        for _ in range(num_samples):
            if rng.random() < qa_ratio:
                # QA sample
                q = rng.choice(user.direct_qa)
                txt = f"Question: {q['question']}\nAnswer: {q['answer']}"
            else:
                # Paraphrase of a fact
                f = rng.choice(user.facts)
                txt = rng.choice(f["paraphrases"])
            self.samples.append(txt)

        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[str], tokenizer, max_len: int):
    enc = tokenizer(batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return {"input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels}


# ---------------------------------------------------------------------------
# Generation & evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_answer(model, tokenizer, question: str, device,
                    max_new_tokens: int = 32, think: bool = False) -> str:
    """Ask a question; return the answer string (no think scaffolding)."""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    tail = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )
    # Trim at first newline / "Question:" to stop at first answer.
    tail = tail.split("\n")[0].strip()
    tail = tail.split("Question:")[0].strip()
    return tail


def _norm(s: str) -> str:
    return re.sub(r"[\s,.'\"]+", " ", str(s).strip().lower()).strip()


def _contains(ans: str, gold: str) -> bool:
    """Gold answer is considered correct if present in the generated text.
    For numeric golds we require the exact number as a token (avoid 9 matching 99)."""
    ga = _norm(gold)
    pa = _norm(ans)
    if ga.isdigit():
        toks = re.findall(r"\d+", pa)
        return ga in toks
    # list-style (comma-separated): check majority of items present
    if "," in ga:
        items = [i.strip() for i in ga.split(",") if i.strip()]
        if not items:
            return False
        hits = sum(1 for it in items if it in pa)
        return hits >= max(1, int(0.8 * len(items)))
    return ga in pa


def evaluate(model, tokenizer, user: User, device) -> dict:
    model.eval()
    # Direct
    dc = 0
    direct_details = []
    for q in user.direct_qa:
        pred = generate_answer(model, tokenizer, q["question"], device)
        ok = _contains(pred, q["answer"])
        dc += int(ok)
        direct_details.append({"key": q["key"], "q": q["question"],
                               "gold": str(q["answer"]), "pred": pred, "ok": ok})
    # Indirect
    ic = 0
    indirect_details = []
    for q in user.indirect_qa:
        pred = generate_answer(model, tokenizer, q["question"], device,
                               max_new_tokens=48)
        ok = _contains(pred, q["answer"])
        ic += int(ok)
        indirect_details.append({"schema": q["schema"], "q": q["question"],
                                 "gold": str(q["answer"]), "pred": pred, "ok": ok})
    return {
        "direct_acc": dc / max(1, len(user.direct_qa)),
        "direct_pass": dc,
        "direct_total": len(user.direct_qa),
        "indirect_acc": ic / max(1, len(user.indirect_qa)),
        "indirect_pass": ic,
        "indirect_total": len(user.indirect_qa),
        "direct_details": direct_details,
        "indirect_details": indirect_details,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_user_lora(
    uid: str,
    model_name: str,
    user_path: Path,
    out_dir: Path,
    rank: int = 64,
    alpha: int = 128,
    lr: float = 1e-4,
    num_epochs: int = 15,
    samples_per_epoch: int = 200,
    batch_size: int = 4,
    max_len: int = 192,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    user = load_user(user_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Baseline eval (with adapter at init == equivalent to base for rank>0
    # because B matrix starts at 0). We still record it for a reference.
    pre = evaluate(model, tok, user, device)
    print(f"  [{uid}] pre-train: direct={pre['direct_acc']:.3f} "
          f"({pre['direct_pass']}/{pre['direct_total']}), "
          f"indirect={pre['indirect_acc']:.3f} "
          f"({pre['indirect_pass']}/{pre['indirect_total']})")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        ds = UserNTPDataset(user, tok, num_samples=samples_per_epoch,
                            max_len=max_len, qa_ratio=0.5, seed=seed + epoch)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tok, max_len))
        total_loss, n = 0.0, 0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item(); n += 1
        avg = total_loss / max(1, n)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            ev = evaluate(model, tok, user, device)
            print(f"  [{uid}] epoch {epoch+1:2d}/{num_epochs}: loss={avg:.4f} "
                  f"direct={ev['direct_acc']:.3f} "
                  f"({ev['direct_pass']}/{ev['direct_total']}) "
                  f"indirect={ev['indirect_acc']:.3f} "
                  f"({ev['indirect_pass']}/{ev['indirect_total']})")
        else:
            print(f"  [{uid}] epoch {epoch+1:2d}/{num_epochs}: loss={avg:.4f}")
    duration = time.time() - start

    # Final eval with adapter
    with_adapter = evaluate(model, tok, user, device)

    # Isolation: disable adapter, eval base
    model.disable_adapter_layers()
    without_adapter = evaluate(model, tok, user, device)
    model.enable_adapter_layers()

    # Save adapter
    model.save_pretrained(str(out_dir))
    metrics = {
        "uid": uid,
        "model_name": model_name,
        "rank": rank, "alpha": alpha,
        "lr": lr, "epochs": num_epochs,
        "samples_per_epoch": samples_per_epoch,
        "max_len": max_len,
        "train_seconds": duration,
        "pre": pre,
        "with_adapter": with_adapter,
        "without_adapter": without_adapter,
    }
    # Drop per-sample details to keep file small; keep summary metrics.
    for b in ("with_adapter", "without_adapter", "pre"):
        metrics[b].pop("direct_details", None)
        metrics[b].pop("indirect_details", None)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # Details in a separate file so the summary stays skim-able.
    details = {"with_adapter": with_adapter, "without_adapter": without_adapter}
    with open(out_dir / "eval_details.json", "w") as f:
        # Keep details (already present on with_adapter/without_adapter returns).
        json.dump({
            "with_adapter_direct": [],
            "with_adapter_indirect": [],
            "without_adapter_direct": [],
            "without_adapter_indirect": [],
        }, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", nargs="+",
                    default=[f"u{i:03d}" for i in range(10)])
    ap.add_argument("--user_dir", default=f"{UAE_ROOT}/data/users")
    ap.add_argument("--out_dir", default=f"{UAE_ROOT}/results/lora_baseline/stage_a")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--samples_per_epoch", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    user_dir = Path(args.user_dir)
    out_root = Path(args.out_dir)
    summary = []
    for uid in args.users:
        print(f"\n===== training {uid} =====")
        m = train_user_lora(
            uid=uid,
            model_name=args.model,
            user_path=user_dir / f"{uid}.json",
            out_dir=out_root / uid,
            rank=args.rank, alpha=args.alpha,
            lr=args.lr,
            num_epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            batch_size=args.batch_size,
        )
        summary.append({
            "uid": uid,
            "direct_acc": m["with_adapter"]["direct_acc"],
            "indirect_acc": m["with_adapter"]["indirect_acc"],
            "iso_direct_acc": m["without_adapter"]["direct_acc"],
            "iso_indirect_acc": m["without_adapter"]["indirect_acc"],
            "train_seconds": m["train_seconds"],
        })
        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Final print
    print("\n===== STAGE A SUMMARY =====")
    print(f"{'uid':<6} {'direct':>8} {'indirect':>9} {'iso-dir':>8} {'iso-ind':>8} {'sec':>6}")
    for s in summary:
        print(f"{s['uid']:<6} {s['direct_acc']:>8.3f} {s['indirect_acc']:>9.3f} "
              f"{s['iso_direct_acc']:>8.3f} {s['iso_indirect_acc']:>8.3f} "
              f"{s['train_seconds']:>6.1f}")
    if summary:
        mean_direct = sum(s["direct_acc"] for s in summary) / len(summary)
        mean_indirect = sum(s["indirect_acc"] for s in summary) / len(summary)
        mean_iso_indirect = sum(s["iso_indirect_acc"] for s in summary) / len(summary)
        print(f"{'MEAN':<6} {mean_direct:>8.3f} {mean_indirect:>9.3f} "
              f"{'':>8} {mean_iso_indirect:>8.3f}")

if __name__ == "__main__":
    main()
