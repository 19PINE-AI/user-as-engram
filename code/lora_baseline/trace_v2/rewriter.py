"""Rewriter: agent trajectory -> canonical SFT-ready reasoning trace.

Design constraints (from the user's gotchas):
  - The rewriter must not use information outside the agent trajectory.
    It only sees the question, the recall queries + hits, and the final
    answer. It never sees the user's full fact list or the `required_fact_keys`
    from the dataset — otherwise the rewriter could paper over gaps in
    the agent's actual retrieval chain.
  - Output must be a single SFT sample: one <think>...</think> block
    followed by <answer>...</answer>. Inside <think>, the chain must
    interleave short sub-questions (paraphrasing what the agent recalled)
    with the facts that were surfaced. No "look up…" meta-commentary — it
    should read as the student model thinking, not narrating an agent run.
  - No ICL fingerprints: forbid phrases like "based on my recall",
    "the tool returned", "I searched", "database", "vector". The trace
    must look like pure internal introspection.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

from .agent import DEFAULT_BASE_URL, DEFAULT_MODEL


REPO = Path(__file__).resolve().parents[2]


SYSTEM_PROMPT = """You will be shown an AI agent's transcript of tool calls and its final answer. Rewrite it as the USER's first-person internal monologue — as if the facts came from their own memory, not from any tool. Then emit the final answer tag.

Output format (produce exactly this, and nothing else):

<monologue lines, first-person, one self-question per sub-fact, then the arithmetic or comparison, then the conclusion>
<answer>FINAL</answer>

Strict rules for the monologue:
- Speak as "I" (the user). Never "the agent", "the user", "the tool".
- Never mention recall, retrieval, search, lookup, database, vector, query, score, tool, transcript, given, told, shown. You are NOT narrating a rewriting task — you ARE the user, thinking.
- Never refer to the agent or any external system.
- Ask short self-questions, then immediately answer them as direct self-knowledge: "What year was I born? I was born in 1993."
- Only use facts that appear in the agent's recall results. Do not invent new facts.
- Do arithmetic/comparison inline in plain words.
- Keep it compact: one self-question per sub-fact, then the conclusion.
- Do NOT output any <think> tag. Do not add any preamble ("Okay, so..."). Start directly with the first self-question.

FINAL inside <answer> must be the agent's final answer verbatim (a number, a name, a short phrase, yes/no). No prose, no units."""


USER_TEMPLATE = """Question the user is answering:
{question}

Agent trajectory (recall queries and their top matches, in order):
{trajectory_text}

Agent's final answer: {final_answer}

Rewrite this as the user's internal monologue following the format above."""


_FORBIDDEN = [
    "recall",
    "retriev",
    "lookup",
    "look up",
    "search",
    "database",
    "vector",
    "tool",
    "top match",
    "query",
    "score",
    "fact_key",
    "memory store",
    "agent",
]


def _format_trajectory(traj: dict) -> str:
    lines = []
    step = 1
    for turn in traj["turns"]:
        tc = turn.get("tool_call")
        if tc is None:
            continue
        lines.append(f"Step {step}: recall query = \"{tc['query']}\"")
        for hit in tc["hits"]:
            lines.append(f"  -> {hit['fact_text']}")
        step += 1
    return "\n".join(lines) if lines else "(no recall calls)"


def _check_forbidden(text: str) -> list[str]:
    low = text.lower()
    return [p for p in _FORBIDDEN if p in low]


_SHAPE_RE = re.compile(
    r"^\s*<think>\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>\s*$",
    re.DOTALL,
)


def parse_rewrite(text: str) -> tuple[str | None, str | None]:
    m = _SHAPE_RE.match(text)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def rewrite_one(
    question: str,
    trajectory: dict,
    final_answer: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: int = 240,
) -> dict:
    """Single rewrite pass.

    Leverages Qwen3 native thinking: the model's <think> block becomes the
    SFT trace's introspective monologue, and the visible content is just
    <answer>FINAL</answer>. We assemble the SFT string from the reasoning
    field + extracted answer — this is why enable_thinking must stay on.
    """
    traj_text = _format_trajectory(trajectory)
    user = USER_TEMPLATE.format(
        question=question,
        trajectory_text=traj_text,
        final_answer=final_answer,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()

    # Parse: monologue is everything BEFORE <answer>…</answer>, answer is inside.
    m = _ANSWER_TAG_RE.search(content)
    answer = m.group(1).strip() if m else None
    monologue = content[: m.start()].strip() if m else content.strip()
    # Strip any stray <think> tags the model may have added despite the rules.
    monologue = re.sub(r"</?think>", "", monologue).strip()

    think = monologue if monologue else None
    forbidden = _check_forbidden(think or "")
    fact_leak = _numeric_fact_consistency(think or "", trajectory)
    ok = (
        think is not None
        and answer is not None
        and not forbidden
        and not fact_leak
        and len(think) > 20
    )
    return {
        "ok": ok,
        "raw_content": content,
        "think": think,
        "answer": answer,
        "forbidden_hits": forbidden,
        "fact_leak": fact_leak,
    }


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _numeric_fact_consistency(think: str, trajectory: dict) -> list[str]:
    """Return list of year-values that appear in the monologue but are not in
    any retrieved fact text. Prevents the rewriter from confabulating a
    birth year it never saw for the specific subject."""
    retrieved_blob = []
    for turn in trajectory.get("turns", []):
        tc = turn.get("tool_call") or {}
        for hit in tc.get("hits", []) or []:
            retrieved_blob.append(hit.get("fact_text", ""))
    # Also allow years that appear in the original question (e.g., "(it's 2026)").
    q_blob = trajectory.get("question", "")
    allowed = set(_YEAR_RE.findall(" ".join(retrieved_blob) + " " + q_blob))
    # And 2025/2026 as bare year-of-current-time references — allowlist common "now" years.
    allowed.update({"2025", "2026"})
    used = set(_YEAR_RE.findall(think))
    return sorted(used - allowed)


def rewrite_trajectory_file(
    raw_jsonl: Path,
    out_jsonl: Path,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_retries: int = 2,
) -> dict:
    ok_count = 0
    fail_count = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with raw_jsonl.open() as fin, out_jsonl.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            if not rec["filter"]["accepted"]:
                continue
            traj = rec["trajectory"]
            result = None
            for attempt in range(max_retries):
                r = rewrite_one(
                    question=traj["question"],
                    trajectory=traj,
                    final_answer=traj["final_answer"],
                    base_url=base_url,
                    model=model,
                    temperature=temperature + 0.1 * attempt,
                )
                if r["ok"]:
                    result = r
                    break
                result = r

            if result["ok"]:
                sft = {
                    "uid": traj["uid"],
                    "schema": traj["schema"],
                    "question": traj["question"],
                    "answer": traj["final_answer"],
                    "gold": traj["gold"],
                    "required_fact_keys": traj["required_fact_keys"],
                    "think": result["think"],
                    "sft_text": f"<think>\n{result['think']}\n</think>\n<answer>{result['answer']}</answer>",
                }
                fout.write(json.dumps(sft) + "\n")
                ok_count += 1
            else:
                fail_count += 1
    return {"ok": ok_count, "failed": fail_count, "source": str(raw_jsonl), "out": str(out_jsonl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(REPO / "data/trace_v2/trajectories_raw"))
    ap.add_argument("--out_dir", default=str(REPO / "data/trace_v2/traces"))
    ap.add_argument("--uids", nargs="+", default=[f"u{i:03d}" for i in range(10)])
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    t0 = time.time()
    agg = {"ok": 0, "failed": 0}
    for uid in args.uids:
        src = in_dir / f"{uid}.jsonl"
        if not src.exists():
            print(f"missing: {src}")
            continue
        r = rewrite_trajectory_file(src, out_dir / f"{uid}.jsonl", base_url=args.base_url, model=args.model)
        agg["ok"] += r["ok"]; agg["failed"] += r["failed"]
        print(json.dumps({uid: r}))
    print(json.dumps({"agg": agg, "wall_seconds": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
