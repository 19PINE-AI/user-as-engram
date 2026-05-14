"""LLM-as-judge scoring for LOCOMO predictions, using local
Qwen2.5-32B-Instruct-AWQ via vLLM.

Reads `results/<tag>__locomo.json` (with predictions block produced by
the patched locomo_eval.py) and for each (question, gold, pred) asks
the judge whether `pred` is a correct answer to `question` given gold.

Writes `results/<tag>__locomo_judge.json` with per-system and per-conv
accuracy (1.0 = correct, 0.0 = incorrect) averaged.
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
    """1 if CORRECT, 0 if INCORRECT."""
    t = text.strip().upper()
    if t.startswith("CORRECT"): return 1
    if t.startswith("INCORRECT"): return 0
    # heuristic
    if "CORRECT" in t and "INCORRECT" not in t.split("CORRECT")[0]: return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--locomo-json", required=True,
                    help="Path to <tag>__locomo.json with predictions")
    p.add_argument("--out", required=True)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = p.parse_args()

    with open(args.locomo_json) as f:
        d = json.load(f)
    if "predictions" not in d:
        raise SystemExit("FATAL: predictions block missing — re-run locomo_eval.py with updated script")
    preds_block = d["predictions"]
    sys_names = list(next(iter(preds_block.values())).keys())
    print(f"Found {len(preds_block)} conversations, {len(sys_names)} systems")

    # Load vLLM
    print(f"Loading judge {args.judge_model}...")
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.judge_model, gpu_memory_utilization=args.gpu_memory_utilization,
               trust_remote_code=True, dtype="auto")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Build prompt list across all systems × all conversations
    prompts = []  # list of (conv_id, sys_name, qi, prompt_str)
    for conv_id, sys_dict in preds_block.items():
        for sys_name, qa_list in sys_dict.items():
            for qi, qa in enumerate(qa_list):
                pmt = JUDGE_PROMPT.format(question=qa["question"],
                                             gold=qa["gold"],
                                             pred=qa["pred"])
                prompts.append((conv_id, sys_name, qi, pmt))
    print(f"Total judge prompts: {len(prompts)}")

    # Run vLLM in one batch (it handles internal batching)
    raw_prompts = [p[3] for p in prompts]
    outs = llm.generate(raw_prompts, sp)
    decisions = [parse_judgement(o.outputs[0].text) for o in outs]

    # Aggregate
    per_conv = defaultdict(lambda: defaultdict(list))
    for (conv_id, sys_name, qi, _), dec in zip(prompts, decisions):
        per_conv[conv_id][sys_name].append(dec)

    summary = {}
    per_conv_results = {}
    for conv_id, sys_dict in per_conv.items():
        per_conv_results[conv_id] = {}
        for sn, decs in sys_dict.items():
            acc = sum(decs) / len(decs)
            per_conv_results[conv_id][sn] = {"acc": acc, "n": len(decs), "correct": sum(decs)}
    for sn in sys_names:
        accs = [per_conv_results[c][sn]["acc"] for c in per_conv_results if sn in per_conv_results[c]]
        if accs:
            summary[sn] = sum(accs) / len(accs)

    print("\n=== Judge accuracy ===")
    for sn, acc in summary.items():
        print(f"  {sn:30s} {acc:.3f}")

    out = {
        "judge_model": args.judge_model,
        "summary": summary,
        "per_conv": per_conv_results,
        "source": args.locomo_json,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
