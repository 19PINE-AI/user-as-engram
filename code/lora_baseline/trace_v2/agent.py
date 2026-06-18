"""Agent loop for generating retrieval trajectories with Qwen3-32B-AWQ.

Protocol
--------
Each assistant turn produces Qwen3 native <think>...</think> followed by
either a tool call or a final answer, using a lightweight tag protocol
rather than OpenAI tool-calling JSON — the rewriter will consume these
same trajectories and the tags transliterate directly into SFT format.

  <recall>query text</recall>        -> tool call, triggers vector-DB lookup
  <answer>final text</answer>        -> terminates the trajectory

The controller strips <think> from intermediate user-visible context so the
thinking of turn N doesn't leak into the attention of turn N+1 (mirrors
Qwen3's own chat template behavior).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

from .vector_db import recall


DEFAULT_BASE_URL = "http://127.0.0.1:8002/v1"
DEFAULT_MODEL = "qwen3-32b"

SYSTEM_PROMPT = """You are answering a question about yourself. You don't have the facts in front of you — they live in your memory. To access a specific fact, use the recall tool.

Tool protocol:
- To look up a fact, emit exactly: <recall>short query describing the fact you need</recall>
- After </recall>, stop. The tool will reply with the top matches from your memory.
- You may call <recall> multiple times, one query per turn, to build up the facts you need.
- When you have enough facts and have computed the answer, emit exactly: <answer>final answer</answer>
- The final answer should be a concise string (a number, a name, a list, a yes/no), not a sentence.

Rules:
- Ask focused questions to the recall tool — one fact at a time works best.
- The recall tool only returns facts about you; it will not do arithmetic or reasoning for you.
- Do your reasoning and arithmetic inside <think>...</think>.
- Never invent facts. If recall does not surface a fact you need, try a different query phrasing."""


USER_TEMPLATE = """Question: {question}

Begin by thinking about which facts you need, then call <recall> for each one."""


RECALL_RESULT_TEMPLATE = """Recall results for "{query}":
{hits}"""


@dataclass
class ToolCall:
    query: str
    hits: list[dict]  # each: {fact_key, fact_text, score}


@dataclass
class Turn:
    role: str          # 'assistant' | 'tool'
    thinking: str = ""  # assistant only
    content: str = ""   # visible assistant content (excluding <think>)
    tool_call: Optional[ToolCall] = None
    final_answer: Optional[str] = None


@dataclass
class Trajectory:
    uid: str
    question: str
    gold: str
    schema: str
    required_fact_keys: list[str]
    turns: list[Turn] = field(default_factory=list)
    finished: bool = False
    final_answer: Optional[str] = None
    stop_reason: str = ""  # 'answer' | 'max_turns' | 'parse_error'
    wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_RECALL_RE = re.compile(r"<recall>(.*?)</recall>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _split_thinking(text: str) -> tuple[str, str]:
    """Return (thinking, visible). Only the first <think> block is kept."""
    m = _THINK_RE.search(text)
    if not m:
        return "", text.strip()
    thinking = m.group(1).strip()
    visible = (text[: m.start()] + text[m.end():]).strip()
    return thinking, visible


def _parse_action(visible: str) -> tuple[str, str]:
    """Return (action, payload). action in {'recall', 'answer', 'none'}."""
    m = _RECALL_RE.search(visible)
    if m:
        return "recall", m.group(1).strip()
    m = _ANSWER_RE.search(visible)
    if m:
        return "answer", m.group(1).strip()
    return "none", ""


def _format_hits(hits) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. [{h.score:.3f}] {h.fact_key}: {h.fact_text}")
    return "\n".join(lines) if lines else "(no matches)"


@dataclass
class AgentConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens_per_turn: int = 2048
    max_turns: int = 10
    recall_top_k: int = 3
    # Qwen3 thinking-mode control — only honored if the server understands it
    enable_thinking: bool = True
    request_timeout: int = 180


def _chat_completion(cfg: AgentConfig, messages: list[dict]) -> str:
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens_per_turn,
        # No stop tokens — Qwen3 thinking-mode emits <recall>/<answer> inside
        # <think>, and the reasoning parser strips <think> into `reasoning`.
        # Relying on stop tokens here truncates the reasoning mid-thought.
        "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking},
    }
    r = requests.post(
        f"{cfg.base_url}/chat/completions",
        json=payload,
        timeout=cfg.request_timeout,
    )
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    # vLLM 0.19 with --reasoning-parser qwen3 puts thinking in `reasoning`.
    # Older parsers used `reasoning_content`. Accept either.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content")
    if reasoning:
        content = f"<think>{reasoning.strip()}</think>\n{content}"
    finish = data["choices"][0].get("finish_reason", "")
    # If server stopped on stop string, re-append the closing tag so the
    # downstream parser sees a well-formed block.
    if finish == "stop":
        if "<recall>" in content and "</recall>" not in content:
            content = content + "</recall>"
        elif "<answer>" in content and "</answer>" not in content:
            content = content + "</answer>"
    return content


def run_agent(
    uid: str,
    question: str,
    gold: str,
    schema: str,
    required_fact_keys: list[str],
    cfg: AgentConfig = AgentConfig(),
) -> Trajectory:
    t0 = time.time()
    traj = Trajectory(
        uid=uid,
        question=question,
        gold=gold,
        schema=schema,
        required_fact_keys=required_fact_keys,
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]

    for _turn_i in range(cfg.max_turns):
        try:
            raw = _chat_completion(cfg, messages)
        except Exception as e:  # network / server errors
            traj.stop_reason = f"http_error: {e.__class__.__name__}"
            break

        thinking, visible = _split_thinking(raw)
        action, payload = _parse_action(visible)
        turn = Turn(role="assistant", thinking=thinking, content=visible)

        if action == "answer":
            turn.final_answer = payload
            traj.turns.append(turn)
            traj.final_answer = payload
            traj.finished = True
            traj.stop_reason = "answer"
            break

        if action == "recall":
            query = payload
            hits = recall(uid, query, top_k=cfg.recall_top_k)
            hits_dicts = [h.to_dict() for h in hits]
            turn.tool_call = ToolCall(query=query, hits=hits_dicts)
            traj.turns.append(turn)

            # Append the assistant turn WITHOUT the <think> block — Qwen3
            # chat template drops prior-turn thinking from the context to
            # keep the window small.
            messages.append({"role": "assistant", "content": visible})
            messages.append({
                "role": "user",
                "content": RECALL_RESULT_TEMPLATE.format(
                    query=query, hits=_format_hits(hits)
                ),
            })
            continue

        # No parseable action — record and stop.
        traj.turns.append(turn)
        traj.stop_reason = "parse_error"
        break
    else:
        traj.stop_reason = "max_turns"

    traj.wall_seconds = round(time.time() - t0, 2)
    return traj


def _cli():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="u000")
    ap.add_argument("--qidx", type=int, default=0)
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_turns", type=int, default=10)
    args = ap.parse_args()

    from pathlib import Path
    user = json.loads(Path(f"data/users/{args.uid}.json").read_text())
    q = user["indirect_qa"][args.qidx]
    cfg = AgentConfig(base_url=args.base_url, model=args.model, max_turns=args.max_turns)
    traj = run_agent(
        uid=args.uid,
        question=q["question"],
        gold=str(q["answer"]),
        schema=q["schema"],
        required_fact_keys=q["required_fact_keys"],
        cfg=cfg,
    )
    print(json.dumps(traj.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    _cli()
