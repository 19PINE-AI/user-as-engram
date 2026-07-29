"""
P3 — Engram replication validation.

For a given checkpoint dir, evaluate:
  1. val_bpb on a held-out shard (already logged during training, but we re-eval
     here with optional Engram suppression).
  2. Top-k accuracy on a small factual-completion probe set (entities, capitals,
     formulaic patterns — the kind of static knowledge Engram is supposed to host).
  3. Top-k accuracy on a small reading-comprehension probe set (where the answer
     is supplied in-context, so the model only needs to read it back).

Replication signal: when Engram is suppressed, factual top-k drops MORE than
reading top-k. (Mirrors Cheng et al. 2026 §6.3 where TriviaQA collapsed to 29%
retained while reading-comp tasks held at 81-93%.)

Usage:
  python -m scripts.evaluate_replication \\
    --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d8 \\
    --base-ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/base_d8
"""
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.common import autodetect_device_type, COMPUTE_DTYPE, get_base_dir

# -----------------------------------------------------------------------------
# Probe sets — kept small, hand-written, focused on signal not coverage.
#
# FACTUAL: prompts where the next token (or first answer token) is a piece of
# static knowledge (entity, formulaic phrase, well-known fact). The Engram
# paper's key claim is that this category collapses on suppression.
#
# READING: prompts where the answer is given in-context just before the cloze.
# A model that suppresses its memory but still has working attention should
# answer these almost as well as one with the memory active.

FACTUAL_PROBES = [
    ("The capital of France is", " Paris"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of Italy is", " Rome"),
    ("The chemical symbol for gold is", " Au"),
    ("The chemical symbol for iron is", " Fe"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("Water is composed of hydrogen and", " oxygen"),
    ("The author of Hamlet is William", " Shakespeare"),
    ("The currency of the United Kingdom is the British", " pound"),
    ("The Great Wall is in", " China"),
    ("Albert Einstein developed the theory of", " relativity"),
    ("The first president of the United States was George", " Washington"),
    ("The Eiffel Tower is in", " Paris"),
    ("The Pacific is the largest", " ocean"),
    ("Mount Everest is the highest", " mountain"),
    ("A baby cat is called a", " kitten"),
    ("The opposite of hot is", " cold"),
    ("The opposite of fast is", " slow"),
    ("Seven days make a", " week"),
    ("Twelve months make a", " year"),
    ("The Earth orbits the", " Sun"),
    ("Humans have two", " eyes"),
    ("A triangle has three", " sides"),
    ("The Sahara is a", " desert"),
    ("Penguins live in", " Antarctica"),
    ("The Mona Lisa was painted by Leonardo da", " Vinci"),
    ("The Roman Empire fell in", " 476"),
    ("The Statue of Liberty was a gift from", " France"),
    ("Pyramids were built by ancient", " Egyptians"),
    ("The Beatles were a band from", " Liverpool"),
]

READING_PROBES = [
    # Each: (prompt with answer in-context, expected first token)
    ("My friend Sam lives in a small town called Riverdale. The town Sam lives in is", " Riverdale"),
    ("Anna's cat is named Whiskers. The cat's name is", " Whiskers"),
    ("The library opens at 9 AM and closes at 7 PM. The library opens at", " 9"),
    ("Bob has a red car. Alice has a blue car. The color of Bob's car is", " red"),
    ("In 2019, the company hired 120 engineers. The number of engineers hired was", " 120"),
    ("The recipe calls for two cups of flour and one cup of sugar. The amount of flour required is", " two"),
    ("Mary works at Globex Corporation. Mary's employer is", " Globex"),
    ("The conference will be held in Vienna in October. The conference is in", " Vienna"),
    ("The medication should be taken twice daily. The frequency of the medication is", " twice"),
    ("The marathon route is 42 kilometers long. The marathon is", " 42"),
    ("Jordan has a sister named Robin. Jordan's sister is", " Robin"),
    ("The fastest runner in the team is Lee, who finished in 12 seconds. Lee finished in", " 12"),
    ("The exam consists of three sections: math, reading, and writing. The first section is", " math"),
    ("The package arrived on Tuesday morning. The day of arrival was", " Tuesday"),
    ("The hotel charges 150 dollars per night. The nightly rate is", " 150"),
    ("Pat lives in a yellow house with a green door. The color of Pat's house is", " yellow"),
    ("The book has 400 pages and 12 chapters. The number of chapters is", " 12"),
    ("During the meeting, Carla proposed a new policy. The person who proposed it was", " Carla"),
    ("The team won 5 games and lost 2. The number of wins was", " 5"),
    ("The clock shows the time as 3:45 PM. The time is 3", ":"),
    ("The trail runs along the Hudson River. The river is the", " Hudson"),
    ("The instructor is Dr. Patel. The instructor's name is Dr.", " Patel"),
    ("Carlos arrived from Madrid yesterday. Carlos arrived from", " Madrid"),
    ("The jar contains 73 marbles, all blue. The number of marbles is", " 73"),
    ("The package contained a book and a candle. The two items were a book and a", " candle"),
    ("The forecast says rain on Friday and sunshine on Saturday. Friday will be", " rain"),
    ("The class begins at 10:30 sharp. The class starts at", " 10"),
    ("Among the four guests, Tina arrived first. The first guest was", " Tina"),
    ("The town has a population of 8500. The population is", " 8500"),
    ("The dish requires garlic, basil, and tomato. The three ingredients are garlic,", " basil"),
]


def topk_first_token_accuracy(model, tokenizer, probes, device, ks=(1, 5)):
    """For each (prompt, expected) compute whether the expected first token of
    `expected` appears in top-k of model's next-token distribution given prompt."""
    model.eval()
    correct_at_k = {k: 0 for k in ks}
    total = 0
    bos = tokenizer.get_bos_token_id()
    per_probe = []
    for prompt, expected in probes:
        ids = tokenizer.encode(prompt, prepend=bos)
        # Take just the first token of expected as the gold
        exp_ids = tokenizer.encode(expected)
        if len(exp_ids) == 0:
            continue
        gold = exp_ids[0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(idx)  # [1, T, V]
        last = logits[0, -1, :]
        topk_max = max(ks)
        topv, topi = torch.topk(last, topk_max)
        topi = topi.tolist()
        rank = topi.index(gold) if gold in topi else -1
        for k in ks:
            if 0 <= rank < k:
                correct_at_k[k] += 1
        total += 1
        per_probe.append({
            "prompt": prompt,
            "gold": tokenizer.decode([gold]),
            "rank": rank,
            "top1_token": tokenizer.decode([topi[0]]),
        })
    return {f"top{k}": correct_at_k[k] / max(total, 1) for k in ks}, per_probe


def load_model(ckpt_dir, tokenizer, device):
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as f:
        d = json.load(f)
    # Reconstruct GPTConfig from saved kwargs
    cfg_keys = {"sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd",
                "window_pattern", "engram_layer_ids", "engram_max_ngram_size",
                "engram_vocab_per_ngram", "engram_n_head_per_ngram", "engram_n_embed_per_ngram",
                "engram_kernel_size"}
    cfg_kwargs = {k: d[k] for k in cfg_keys if k in d}
    if "engram_layer_ids" in cfg_kwargs:
        cfg_kwargs["engram_layer_ids"] = tuple(cfg_kwargs["engram_layer_ids"])
    config = GPTConfig(**cfg_kwargs)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    has_engram = bool(config.engram_layer_ids)
    if has_engram:
        base_dir = get_base_dir()
        model.attach_engram(tokenizer, base_dir=base_dir)
    state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True, help="checkpoint dir of Engram-trained model")
    parser.add_argument("--base-ckpt-dir", default=None, help="(optional) checkpoint dir of base model for delta comparison")
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--eval-tokens", type=int, default=524_288)  # 512K tokens for val
    parser.add_argument("--out", default=None, help="optional path to write JSON results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    results = {}

    # --- Engram model evals (with and without suppression) ---
    print(f"\n=== Loading Engram model from {args.ckpt_dir} ===")
    eng_model, eng_cfg = load_model(args.ckpt_dir, tokenizer, device)
    has_engram = bool(eng_cfg.engram_layer_ids)
    print(f"Has Engram modules: {has_engram} ({eng_cfg.engram_layer_ids})")

    # Factual & reading probes
    print("\nFactual probes (Engram active)...")
    fa_active, fa_active_detail = topk_first_token_accuracy(eng_model, tokenizer, FACTUAL_PROBES, device)
    print(f"  {fa_active}")
    print("Reading probes (Engram active)...")
    re_active, re_active_detail = topk_first_token_accuracy(eng_model, tokenizer, READING_PROBES, device)
    print(f"  {re_active}")

    if has_engram:
        eng_model.engram_suppress = True
        print("\nFactual probes (Engram SUPPRESSED)...")
        fa_supp, fa_supp_detail = topk_first_token_accuracy(eng_model, tokenizer, FACTUAL_PROBES, device)
        print(f"  {fa_supp}")
        print("Reading probes (Engram SUPPRESSED)...")
        re_supp, re_supp_detail = topk_first_token_accuracy(eng_model, tokenizer, READING_PROBES, device)
        print(f"  {re_supp}")
        eng_model.engram_suppress = False
    else:
        fa_supp = re_supp = None
        fa_supp_detail = re_supp_detail = None

    # val_bpb (active only — suppression on val_bpb is also informative but slow)
    eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len))
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
    )
    print("\nVal bpb (Engram active)...")
    eng_val_active = evaluate_bpb(eng_model, val_loader, eval_steps, token_bytes)
    print(f"  {eng_val_active:.4f}")
    eng_val_supp = None
    if has_engram:
        eng_model.engram_suppress = True
        val_loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
        )
        print("Val bpb (Engram SUPPRESSED)...")
        eng_val_supp = evaluate_bpb(eng_model, val_loader, eval_steps, token_bytes)
        print(f"  {eng_val_supp:.4f}")
        eng_model.engram_suppress = False

    results["engram"] = {
        "ckpt_dir": args.ckpt_dir,
        "factual_active": fa_active,
        "reading_active": re_active,
        "factual_suppressed": fa_supp,
        "reading_suppressed": re_supp,
        "val_bpb_active": float(eng_val_active),
        "val_bpb_suppressed": float(eng_val_supp) if eng_val_supp is not None else None,
        "factual_detail_active": fa_active_detail,
        "reading_detail_active": re_active_detail,
    }

    # --- Base model (control) ---
    if args.base_ckpt_dir:
        print(f"\n=== Loading base model from {args.base_ckpt_dir} ===")
        # Free engram model first to avoid OOM
        del eng_model
        torch.cuda.empty_cache()
        base_model, base_cfg = load_model(args.base_ckpt_dir, tokenizer, device)
        print("Factual probes (base)...")
        fa_base, fa_base_detail = topk_first_token_accuracy(base_model, tokenizer, FACTUAL_PROBES, device)
        print(f"  {fa_base}")
        print("Reading probes (base)...")
        re_base, re_base_detail = topk_first_token_accuracy(base_model, tokenizer, READING_PROBES, device)
        print(f"  {re_base}")
        val_loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
        )
        print("Val bpb (base)...")
        base_val = evaluate_bpb(base_model, val_loader, eval_steps, token_bytes)
        print(f"  {base_val:.4f}")
        results["base"] = {
            "ckpt_dir": args.base_ckpt_dir,
            "factual": fa_base,
            "reading": re_base,
            "val_bpb": float(base_val),
            "factual_detail": fa_base_detail,
        }

    # --- Replication summary ---
    print("\n--- Replication summary ---")
    if has_engram and fa_supp is not None:
        # Retained-performance ratio (suppressed / active), corresponding to
        # the paper's named Engram-suppression sensitivity test.
        for k in ("top1", "top5"):
            fa_active_k = fa_active[k]; fa_supp_k = fa_supp[k]
            re_active_k = re_active[k]; re_supp_k = re_supp[k]
            fa_retain = fa_supp_k / max(fa_active_k, 1e-9)
            re_retain = re_supp_k / max(re_active_k, 1e-9)
            print(f"  {k}: factual retained {fa_retain*100:.1f}%  reading retained {re_retain*100:.1f}%  "
                  f"replication signal (smaller-is-stronger): {fa_retain - re_retain:+.3f}")
            results.setdefault("replication_signal", {})[k] = {
                "factual_retained": fa_retain,
                "reading_retained": re_retain,
                "delta": fa_retain - re_retain,
            }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
