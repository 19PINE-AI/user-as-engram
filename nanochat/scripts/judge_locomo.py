"""LLM-as-judge scoring for LOCOMO predictions using local
Qwen2.5-Instruct via transformers (vLLM v1 doesn't tolerate our
broken NVML on Blackwell sm120).

Reads `results/<tag>__locomo.json` (with predictions block produced by
the patched locomo_eval.py) and for each (question, gold, pred) asks
the judge whether `pred` is a correct answer.

Writes `results/<tag>__locomo_judge.json`.
"""
from __future__ import annotations
import os, json, argparse, time
from pathlib import Path
from collections import defaultdict

JUDGE_PROMPT = """You are grading a short answer to a question from a personal-memory
benchmark. The question is about a specific person's life and the
GOLD answer is the single correct answer.

QUESTION: {question}
GOLD: {gold}
MODEL ANSWER: {pred}

Is the MODEL ANSWER semantically equivalent to GOLD as an answer to the
question? Consider these correct: matching numbers / dates / names
expressed in any reasonable form, paraphrases of the same answer,
the gold answer wrapped in a sentence. Consider these incorrect: a
different person/date/number/event, an irrelevant continuation, an
empty or generic response.

Respond with a single token: CORRECT or INCORRECT."""


def parse_judgement(text: str) -> int:
    t = text.strip().upper()
    if t.startswith("CORRECT"): return 1
    if t.startswith("INCORRECT"): return 0
    if "CORRECT" in t and "INCORRECT" not in t.split("CORRECT")[0]: return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--locomo-json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=6)
    args = p.parse_args()

    with open(args.locomo_json) as f:
        d = json.load(f)
    if "predictions" not in d:
        raise SystemExit("FATAL: predictions block missing — re-run locomo_eval.py with updated script")
    preds_block = d["predictions"]
    sys_names = list(next(iter(preds_block.values())).keys())
    print(f"Found {len(preds_block)} conversations, {len(sys_names)} systems")

    items = []
    for conv_id, sys_dict in preds_block.items():
        for sys_name, qa_list in sys_dict.items():
            for qi, qa in enumerate(qa_list):
                pmt = JUDGE_PROMPT.format(question=qa["question"],
                                           gold=qa["gold"],
                                           pred=qa["pred"])
                items.append((conv_id, sys_name, qi, pmt))
    print(f"Total judge prompts: {len(items)}")

    print(f"Loading {args.judge_model} via transformers...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    tok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to("cuda:0")
    model.eval()

    @torch.no_grad()
    def run_batch(prompts):
        messages = [[{"role": "user", "content": p}] for p in prompts]
        chat_prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        inputs = tok(chat_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda:0")
        out = model.generate(**inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                temperature=0.0,
                                pad_token_id=tok.pad_token_id)
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        return texts

    decisions = []
    start = time.time()
    for i in range(0, len(items), args.batch_size):
        batch_prompts = [it[3] for it in items[i:i+args.batch_size]]
        texts = run_batch(batch_prompts)
        decisions.extend(parse_judgement(t) for t in texts)
        if i % (args.batch_size * 10) == 0:
            elapsed = time.time() - start
            rate = (i + len(batch_prompts)) / max(elapsed, 1)
            eta = (len(items) - i - len(batch_prompts)) / max(rate, 0.01)
            print(f"  {i + len(batch_prompts):>5d}/{len(items):>5d}  elapsed {elapsed:.0f}s  rate {rate:.1f}/s  ETA {eta:.0f}s", flush=True)

    per_conv = defaultdict(lambda: defaultdict(list))
    for (conv_id, sys_name, qi, _), dec in zip(items, decisions):
        per_conv[conv_id][sys_name].append(dec)
    summary = {}
    per_conv_results = {}
    for conv_id, sys_dict in per_conv.items():
        per_conv_results[conv_id] = {}
        for sn, decs in sys_dict.items():
            per_conv_results[conv_id][sn] = {"acc": sum(decs)/len(decs), "n": len(decs), "correct": sum(decs)}
    for sn in sys_names:
        accs = [per_conv_results[c][sn]["acc"] for c in per_conv_results if sn in per_conv_results[c]]
        if accs:
            summary[sn] = sum(accs) / len(accs)

    print("\n=== Judge accuracy ===")
    for sn, acc in summary.items():
        print(f"  {sn:30s} {acc:.3f}")

    out = {"judge_model": args.judge_model, "summary": summary,
            "per_conv": per_conv_results, "source": args.locomo_json}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
