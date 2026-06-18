"""Generate Hippo-style teacher traces for user-as-engram's training users.

For each (user, indirect_qa) pair, run Qwen3-8B as a thinking-enabled
teacher with the required facts in the system prompt. The teacher emits
a <think> reasoning chain + final answer. These traces become the 4th
NTP format when training the shared LoRA.

Usage:
    python -m scripts.generate_trace_corpus \\
        --user-dir $USER_AS_ENGRAM_ROOT/data/users \\
        --train-uids u020 u021 u022 u023 u024 u025 u026 u027 u028 u029 \\
        --out $USER_AS_ENGRAM_ROOT/nanochat_base/shared_lora_d20_trace/traces.jsonl
"""
from __future__ import annotations
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


TEACHER_MODEL = "Qwen/Qwen3-8B"


def _norm(a):
    if isinstance(a, list):
        return ", ".join(str(x) for x in a)
    return str(a)


def _completion_prefix(paraphrases, ans):
    for p in paraphrases:
        p = p.rstrip(".")
        if p.endswith(ans):
            return p[: -len(ans)].rstrip()
    return None


def render_facts_block(facts, required_keys):
    """Render the required facts in 'key: value' form (matches the in-context
    reasoning sample format already used by train_shared_lora.py)."""
    by_key = {f["key"]: f for f in facts}
    lines = []
    for k in required_keys:
        if k not in by_key:
            continue
        f = by_key[k]
        ans = _norm(f["answer"])
        pref = _completion_prefix(f["paraphrases"], ans)
        if pref is None:
            lines.append(f"{k.replace('_', ' ')}: {ans}")
        else:
            lines.append(f"{pref} {ans}")
    return ". ".join(lines) + "."


def build_prompts(uids, user_dir, tokenizer):
    queries = []
    for uid in uids:
        u = json.load(open(Path(user_dir) / f"{uid}.json"))
        for iq in u["indirect_qa"]:
            req = iq.get("required_fact_keys") or []
            if not req:
                continue
            fact_block = render_facts_block(u["facts"], req)
            if not fact_block.strip("."):
                continue
            gold = _norm(iq["answer"])
            queries.append({
                "uid": uid,
                "schema": iq.get("schema", ""),
                "variant": iq.get("variant", ""),
                "question": iq["question"],
                "gold": gold,
                "fact_block": fact_block,
            })

    prompts = []
    for q in queries:
        sys_prompt = (
            "You are a personal memory assistant for the user. Use the "
            "facts below to answer the user's question. Show your "
            "reasoning before giving the final answer. Be precise.\n\n"
            f"USER FACTS:\n{q['fact_block']}"
        )
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q["question"]},
        ]
        chat = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(chat)
    return queries, prompts


def split_think(text):
    import re
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()
    return "", text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-dir", default=f"{UAE_ROOT}/data/users")
    ap.add_argument("--train-uids", nargs="+",
                    default=[f"u{i:03d}" for i in range(20, 30)])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=800)
    ap.add_argument("--gpu_mem_util", type=float, default=0.35)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)

    queries, prompts = build_prompts(args.train_uids, args.user_dir, tokenizer)
    print(f"Built {len(queries)} (user, indirect_qa) prompts across "
          f"{len(args.train_uids)} users.")

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=TEACHER_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=4096,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(temperature=0.7, top_p=0.95,
                        max_tokens=args.max_new_tokens, seed=42)

    print(f"Generating {len(prompts)} traces...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"done in {time.time()-t0:.1f}s "
          f"({(time.time()-t0)/max(len(prompts),1):.2f}s/trace)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w") as f:
        for q, out in zip(queries, outputs):
            raw = out.outputs[0].text
            thinking, answer = split_think(raw)
            # Critical-token check: did the answer contain the gold?
            retained = q["gold"].lower() in answer.lower()
            # Also retain partial matches (digits-only gold)
            if not retained and q["gold"].isdigit() and q["gold"] in answer:
                retained = True
            rec = {
                "uid": q["uid"],
                "schema": q["schema"],
                "variant": q["variant"],
                "question": q["question"],
                "fact_block": q["fact_block"],
                "gold": q["gold"],
                "thinking": thinking,
                "answer": answer,
                "retained": retained,
            }
            f.write(json.dumps(rec) + "\n")
            if retained:
                kept += 1

    print(f"Retention: {kept}/{len(queries)} ({100*kept/max(len(queries),1):.1f}%)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
