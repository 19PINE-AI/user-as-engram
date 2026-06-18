"""Filter for trajectories produced by the agent.

Two gates — either failure disqualifies the trajectory:
  1. Answer correctness: normalized(agent.final_answer) == normalized(gold).
  2. Retrieval happened: every required_fact_key was surfaced in a recall
     result during the trajectory. The agent may have called recall extra
     times or missed some — what matters is the fact was retrieved, not
     guessed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .agent import Trajectory


_REPO = Path(__file__).resolve().parents[2]
_USERS_DIR = _REPO / "data" / "users"


def _load_user_facts(uid: str) -> dict:
    data = json.loads((_USERS_DIR / f"{uid}.json").read_text())
    return {f["key"]: f["answer"] for f in data["facts"]}


def _stringify(v) -> str:
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)


# Match Stage A's scorer so filter-accept == student-eval-correct.
import sys
_SRC = str(__import__('pathlib').Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from stage_a import _contains as _sa_contains  # noqa: E402


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(s.lower()) if len(t) >= 3 or t.isdigit()]


def _token_overlap_matches(pred: str, gold: str, threshold: float = 0.5) -> bool:
    gtoks = _tokenize(gold)
    if not gtoks:
        return False
    ptoks = set(_tokenize(pred))
    hits = sum(1 for t in gtoks if t in ptoks)
    return hits / len(gtoks) >= threshold


def _answer_matches(pred: str | None, gold: str) -> bool:
    """Semantic correctness check.

    Layered:
      1. Stage A _contains (digits-as-tokens, comma-list majority, substring).
      2. Token overlap: ≥60% of gold's content tokens (len≥3 or digit)
         appear in pred. Catches phrasing differences like
         "Younger by 3 years" vs "3 year(s) younger than me".
    """
    if pred is None:
        return False
    if _sa_contains(pred, gold):
        return True
    if _token_overlap_matches(pred, gold):
        return True
    return False


def retrieved_fact_keys(traj: Trajectory) -> set[str]:
    keys: set[str] = set()
    for turn in traj.turns:
        tc = turn.tool_call
        if tc is None:
            continue
        for hit in tc.hits:
            keys.add(hit["fact_key"])
    return keys


def _all_retrieved_texts(traj: Trajectory) -> str:
    chunks = []
    for turn in traj.turns:
        tc = turn.tool_call
        if tc is None:
            continue
        for hit in tc.hits:
            chunks.append(hit["fact_text"])
    return " \n ".join(chunks).lower()


def _surfaced_fact_keys(traj: Trajectory) -> set[str]:
    """A required fact counts as surfaced if its VALUE appears in some recall
    hit's text — even if its own key wasn't returned. This matters because
    paraphrases like "Quinn was born in 1994" encode both spouse_name and
    spouse_year, so retrieving spouse_year surfaces the name value too.
    """
    direct = retrieved_fact_keys(traj)
    blob = _all_retrieved_texts(traj)
    if not blob:
        return direct
    facts = _load_user_facts(traj.uid)
    out = set(direct)
    for key in traj.required_fact_keys:
        if key in out:
            continue
        if key not in facts:
            continue
        val = _stringify(facts[key]).lower().strip()
        if not val:
            continue
        if val in blob:
            out.add(key)
    return out


@dataclass
class FilterResult:
    accepted: bool
    reasons: list[str]
    retrieved: set[str]
    missing_required: set[str]
    num_recalls: int
    answer_correct: bool


def evaluate(traj: Trajectory) -> FilterResult:
    reasons: list[str] = []

    if not traj.finished:
        reasons.append(f"unfinished:{traj.stop_reason}")

    answer_correct = _answer_matches(traj.final_answer, traj.gold)
    if not answer_correct:
        reasons.append("wrong_answer")

    retrieved = _surfaced_fact_keys(traj)
    required = set(traj.required_fact_keys)
    missing = required - retrieved
    if missing:
        reasons.append(f"missing_required:{sorted(missing)}")

    num_recalls = sum(1 for t in traj.turns if t.tool_call is not None)
    if num_recalls == 0:
        reasons.append("no_recall_calls")

    return FilterResult(
        accepted=len(reasons) == 0,
        reasons=reasons,
        retrieved=retrieved,
        missing_required=missing,
        num_recalls=num_recalls,
        answer_correct=answer_correct,
    )


def short_summary(traj: Trajectory, fr: FilterResult) -> str:
    status = "ACCEPT" if fr.accepted else "REJECT"
    return (
        f"{status} u={traj.uid} schema={traj.schema} "
        f"recalls={fr.num_recalls} correct={fr.answer_correct} "
        f"missing={sorted(fr.missing_required) or '[]'} "
        f"stop={traj.stop_reason}"
    )
