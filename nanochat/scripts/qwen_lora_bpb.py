"""
Tier 2 #5: measure validation BPB of a per-user LoRA on Qwen2.5-3B-Instruct
(or any HF instruction-tuned base) using held-out web text. The claim under
test: the +1.56 bpb contamination we measured on Mini-Engram-d20 is an
*architectural* property of per-user LoRA, not an artifact of the base LM.

Procedure per user:
  1. Train rank-64 POLAR-class LoRA via NTP on the user's (paraphrase + QA)
     mixture (same recipe as user-as-lora's stage_a.py).
  2. Measure val_bpb on a held-out text shard (FineWeb-edu or any text
     dataset). Both before LoRA (baseline) and after LoRA (with the edit).
  3. Compute Δbpb. Predict Δbpb > 0 (LoRA contaminates), though the
     magnitude on an instruction-tuned base may be smaller than on a
     base LM because the base distribution is already further from web text.

Usage:
  python -m scripts.qwen_lora_bpb \\
       --model Qwen/Qwen2.5-3B-Instruct \\
       --users $(printf 'u%03d ' $(seq 0 9)) \\
       --user-dir /home/ubuntu/user-as-lora/data/users \\
       --eval-text /home/ubuntu/user-as-engram/data/val_text.txt \\
       --out /home/ubuntu/user-as-engram/results/qwen3b_lora_bpb.json
"""
import os, sys, json, argparse, time, random, math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Inline copy of the user-as-lora dataset/format helpers
sys.path.insert(0, "/home/ubuntu/user-as-lora/src")
from synth_users import load_user, User  # noqa: E402


# -----------------------------------------------------------------------------
# Training (mirrors user-as-lora/src/stage_a.py)
# -----------------------------------------------------------------------------

class UserNTPDataset(Dataset):
    def __init__(self, user, tokenizer, num_samples, max_len=192, qa_ratio=0.5, seed=0):
        self.tokenizer = tokenizer
        self.max_len = max_len
        rng = random.Random(seed)
        self.samples = []
        for _ in range(num_samples):
            if rng.random() < qa_ratio:
                q = rng.choice(user.direct_qa)
                txt = f"Question: {q['question']}\nAnswer: {q['answer']}"
            else:
                f = rng.choice(user.facts)
                txt = rng.choice(f["paraphrases"])
            self.samples.append(txt)
        rng.shuffle(self.samples)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate_fn(batch, tokenizer, max_len):
    enc = tokenizer(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_len)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return {"input_ids": enc["input_ids"],
             "attention_mask": enc["attention_mask"],
             "labels": labels}


def train_lora(model, tok, user, device, rank, alpha, lr, epochs, samples_per_epoch,
                batch_size, max_len, seed=42):
    random.seed(seed); torch.manual_seed(seed)
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        ds = UserNTPDataset(user, tok, samples_per_epoch, max_len=max_len,
                              qa_ratio=0.5, seed=seed + epoch)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, tok, max_len))
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step(); optim.zero_grad()
    return model, time.time() - t0


# -----------------------------------------------------------------------------
# Val BPB on held-out text
# -----------------------------------------------------------------------------

@torch.no_grad()
def val_bpb(model, tok, text_chunks, device, max_len=1024):
    """text_chunks: list of strings (each <= max_len tokens)."""
    model.eval()
    total_log_p = 0.0   # nat-log P(next-token | prefix), summed
    total_bytes = 0
    total_tokens = 0
    for txt in text_chunks:
        enc = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        ids = enc["input_ids"]
        if ids.size(1) < 2: continue
        out = model(**enc, labels=ids)
        # out.loss is mean negative-log-likelihood per token (in nats? typically log-probs)
        # transformers: cross_entropy is over labels, so loss = mean(-log P_correct)
        n_tok = ids.size(1) - 1   # number of predicted tokens
        total_log_p += float(out.loss.item()) * n_tok
        total_tokens += n_tok
        total_bytes += len(txt.encode("utf-8"))
    # bpb = (total_neg_log_p in bits) / total_bytes
    nats_per_token = total_log_p / max(total_tokens, 1)
    bits_per_byte = (total_log_p / math.log(2)) / max(total_bytes, 1)
    return bits_per_byte, nats_per_token, total_tokens, total_bytes


# -----------------------------------------------------------------------------
# Held-out text loader (FineWeb-edu shard cached in HF datasets, else fallback)
# -----------------------------------------------------------------------------

def load_eval_text(max_chunks=64, max_chars_per_chunk=2000):
    """Try a few sources in order; return list of text chunks."""
    # 1) explicit local file
    p = Path("/home/ubuntu/user-as-engram/data/val_text.txt")
    if p.exists():
        text = p.read_text()
        return [text[i:i+max_chars_per_chunk]
                 for i in range(0, len(text), max_chars_per_chunk)][:max_chunks]
    # 2) FineWeb-edu sample
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                           split="train", streaming=True)
        chunks = []
        for row in ds:
            t = row.get("text", "")[:max_chars_per_chunk]
            if len(t) >= 200:
                chunks.append(t)
            if len(chunks) >= max_chunks: break
        if chunks: return chunks
    except Exception as e:
        print(f"FineWeb-edu fallback failed: {e}")
    # 3) Wikitext-103
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        chunks = []
        cur = ""
        for row in ds:
            cur += row.get("text", "")
            while len(cur) >= max_chars_per_chunk:
                chunks.append(cur[:max_chars_per_chunk])
                cur = cur[max_chars_per_chunk:]
                if len(chunks) >= max_chunks: break
            if len(chunks) >= max_chunks: break
        if chunks: return chunks
    except Exception as e:
        print(f"Wikitext fallback failed: {e}")
    raise RuntimeError("Could not load any validation text corpus.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--users", nargs="+", default=[f"u{i:03d}" for i in range(10)])
    p.add_argument("--user-dir", default="/home/ubuntu/user-as-lora/data/users")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--samples-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--eval-max-chunks", type=int, default=64)
    p.add_argument("--eval-max-len", type=int, default=1024)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print(f"Loading eval text...")
    chunks = load_eval_text(max_chunks=args.eval_max_chunks)
    print(f"  {len(chunks)} chunks")

    print(f"Loading base model {args.model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )

    # Baseline val_bpb (no LoRA)
    print("Baseline val_bpb (no LoRA)...")
    base_bpb, base_nats, n_tok, n_bytes = val_bpb(
        base_model, tok, chunks, device, max_len=args.eval_max_len,
    )
    print(f"  baseline_bpb = {base_bpb:.4f}  ({n_tok} tokens, {n_bytes} bytes)")

    out = {
        "config": vars(args),
        "baseline_bpb": base_bpb,
        "baseline_nats_per_token": base_nats,
        "n_eval_tokens": n_tok, "n_eval_bytes": n_bytes,
        "per_user": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f: json.dump(out, f, indent=2)

    user_dir = Path(args.user_dir)
    for uid in args.users:
        print(f"\n=== {uid} ===")
        user = load_user(user_dir / f"{uid}.json")

        # Train fresh LoRA (reload base each user)
        del base_model
        torch.cuda.empty_cache()
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
        )
        wrapped, train_s = train_lora(
            base_model, tok, user, device,
            rank=args.rank, alpha=args.alpha, lr=args.lr,
            epochs=args.epochs, samples_per_epoch=args.samples_per_epoch,
            batch_size=args.batch_size, max_len=args.max_len,
        )
        # val_bpb WITH LoRA attached
        lora_bpb, lora_nats, _, _ = val_bpb(
            wrapped, tok, chunks, device, max_len=args.eval_max_len,
        )
        # val_bpb WITHOUT LoRA (disable)
        wrapped.disable_adapter_layers()
        iso_bpb, iso_nats, _, _ = val_bpb(
            wrapped, tok, chunks, device, max_len=args.eval_max_len,
        )
        wrapped.enable_adapter_layers()

        delta = lora_bpb - base_bpb
        iso_delta = iso_bpb - base_bpb  # sanity: should be ~0
        print(f"  lora_bpb={lora_bpb:.4f}  Δ={delta:+.4f}  "
              f"iso_bpb={iso_bpb:.4f}  iso_Δ={iso_delta:+.4f}  "
              f"train_s={train_s:.0f}")

        out["per_user"].append({
            "uid": uid,
            "lora_bpb": lora_bpb, "lora_bpb_delta": delta,
            "iso_bpb": iso_bpb, "iso_bpb_delta": iso_delta,
            "train_s": train_s,
        })
        with open(args.out, "w") as f: json.dump(out, f, indent=2)

        # Unwrap LoRA: replace `wrapped` (PeftModel) with reload for next iter
        del wrapped
        torch.cuda.empty_cache()
        # base_model will be reloaded on next iteration

    # Aggregate
    per = out["per_user"]
    if per:
        def mean(xs): return sum(xs) / len(xs)
        deltas = [u["lora_bpb_delta"] for u in per]
        out["agg"] = {
            "n_users": len(per),
            "baseline_bpb": base_bpb,
            "lora_bpb_mean": mean([u["lora_bpb"] for u in per]),
            "lora_bpb_delta_mean": mean(deltas),
            "lora_bpb_delta_range": [min(deltas), max(deltas)],
            "fraction_delta_gt0": sum(1 for d in deltas if d > 0) / len(deltas),
        }
        print(f"\n========== AGGREGATE n={len(per)} ==========")
        print(f"  baseline_bpb = {base_bpb:.4f}")
        print(f"  per-user LoRA bpb (mean) = {out['agg']['lora_bpb_mean']:.4f}")
        print(f"  Δbpb mean = {out['agg']['lora_bpb_delta_mean']:+.4f}")
        print(f"  Δbpb range = [{deltas and min(deltas):+.4f}, {deltas and max(deltas):+.4f}]")
        print(f"  fraction Δbpb > 0 = {out['agg']['fraction_delta_gt0']*100:.0f}%")

    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
