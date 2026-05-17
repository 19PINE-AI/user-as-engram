"""Minimal SFT for Engram-pretrained models. Goal: teach the model
the answer format ("Q: ... A: ..." style short replies) without
training the Engram tables. This is the cheap path to close the
LLM-judge gap on LOCOMO without disturbing the surgical-insertion
mechanism.

Training data: synthetic Q/A pairs from corpora_xl (held out from
the LOCOMO eval). Loss only on the answer tokens.

Saves: a new checkpoint with tag <orig_tag>_sft preserving the
Engram tables (frozen during SFT).

Usage:
  python -m scripts.sft_engram_minimal \\
    --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12_w1280_optimal \\
    --out-tag engram_d12_w1280_sft \\
    --num-iterations 1000
"""
from __future__ import annotations
import os, json, argparse, time, random
from pathlib import Path
import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out-tag", required=True)
    p.add_argument("--corpus", default="/home/ubuntu/user-as-engram/data/corpora_xl.json")
    p.add_argument("--num-iterations", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device-batch-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=512)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    print(f"Loading {args.ckpt_dir}...")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram if hasattr(model, "engram") else None

    # Freeze Engram tables (we don't want SFT to touch them)
    if eng is not None:
        for tbl in eng.tables.values():
            tbl.embedding.weight.requires_grad_(False)
        print("Engram tables frozen.")

    # Trainable params: everything except engram tables
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_trainable:,}")

    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Load corpus and build (q, a) pairs
    with open(args.corpus) as f:
        corpus = json.load(f)
    pairs = []
    for f in corpus["user_facts"]:
        pairs.append(("Question: " + f["prompt"] + "?\nAnswer:", " " + f["gold"].strip()))
    for f in corpus["org_facts"]:
        pairs.append(("Question: " + f["prompt"] + "?\nAnswer:", " " + f["gold"].strip()))
    print(f"Loaded {len(pairs)} (q, a) pairs")
    random.Random(42).shuffle(pairs)

    bos = tokenizer.get_bos_token_id()
    model.train()
    losses = []
    start = time.time()
    for step in range(args.num_iterations):
        # Build a batch
        batch_inputs = []
        batch_labels = []
        for _ in range(args.device_batch_size):
            q, a = random.choice(pairs)
            q_ids = tokenizer.encode(q, prepend=bos)
            a_ids = tokenizer.encode(a)
            full_ids = q_ids + a_ids
            full_ids = full_ids[:args.max_seq_len]
            # Labels: -100 for question tokens, copy of full_ids for answer tokens
            labels = [-100] * len(q_ids) + a_ids
            labels = labels[:args.max_seq_len]
            # Pad
            pad_len = args.max_seq_len - len(full_ids)
            if pad_len > 0:
                full_ids = full_ids + [0] * pad_len
                labels = labels + [-100] * pad_len
            batch_inputs.append(full_ids)
            batch_labels.append(labels)
        x = torch.tensor(batch_inputs, dtype=torch.long, device=device)
        y = torch.tensor(batch_labels, dtype=torch.long, device=device)
        # Predict next-token: shift labels by 1
        logits = model(x)[:, :-1, :]
        targets = y[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                 targets.reshape(-1),
                                 ignore_index=-100)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()

        losses.append(loss.item())
        if step % 50 == 0 or step == args.num_iterations - 1:
            recent = losses[-50:]
            avg = sum(recent) / len(recent)
            elapsed = time.time() - start
            print(f"step {step:>4d}/{args.num_iterations}  loss {loss.item():.3f}  avg(50) {avg:.3f}  elapsed {elapsed:.0f}s", flush=True)

    # Save
    base_dir = os.environ.get("NANOCHAT_BASE_DIR", "/home/ubuntu/user-as-engram/nanochat_base")
    out_dir = os.path.join(base_dir, "engram_runs", args.out_tag)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    # Copy config from source
    import shutil
    shutil.copy(os.path.join(args.ckpt_dir, "config.json"), os.path.join(out_dir, "config.json"))
    print(f"Saved {out_dir}/model.pt")


if __name__ == "__main__":
    main()
