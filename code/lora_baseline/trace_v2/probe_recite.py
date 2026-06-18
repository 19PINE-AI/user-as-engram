"""Mechanistic probe A: per-question recite-fidelity vs answer-correctness.

For every held-indirect prediction, extract the facts the adapter recited
inside the `<think>` "Relevant facts from my memory:" block, map them back
to fact_keys via the user's fact paraphrases, and split failures into:

  - retrieval_error: required fact_keys NOT all present in recited set
  - composition_error: required fact_keys present, but final answer wrong
  - format_error: recited correctly, computed correctly, scorer rejected
                  (we approximate this by checking if gold tokens appear
                  in pred_raw outside the answer tag)

This tells us whether expanded data fixed answer-shape but not routing,
or whether routing was already correct and the bug is downstream.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "stage_a_recite_v3" / "qwen2p5_3b_r64_within_schema"
USERS = REPO / "data" / "users"
UIDS = [f"u{i:03d}" for i in range(10)]


_RECITE_RE = re.compile(
    r"Relevant facts from my memory:\s*\n(.+?)(?:\n\s*\n|\n[A-Z]|</think>)",
    re.DOTALL,
)


def _extract_recited_lines(raw: str) -> list[str]:
    m = _RECITE_RE.search(raw)
    if not m:
        return []
    block = m.group(1)
    lines = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:]
        # trailing period stripping for fuzzy match
        line = line.rstrip(".").strip()
        lines.append(line)
    return lines


def _build_paraphrase_index(uid: str) -> dict[str, str]:
    """Return {paraphrase_text(normalized): fact_key} for one user."""
    u = json.loads((USERS / f"{uid}.json").read_text())
    idx = {}
    for f in u["facts"]:
        for p in f["paraphrases"]:
            key_text = p.strip().rstrip(".").lower()
            idx[key_text] = f["key"]
    return idx


def _match_recited_to_keys(recited: list[str], idx: dict[str, str]) -> set[str]:
    keys = set()
    for line in recited:
        nl = line.lower().rstrip(".").strip()
        if nl in idx:
            keys.add(idx[nl])
            continue
        # fuzzy: substring match against any paraphrase
        for ptext, k in idx.items():
            if ptext == nl or ptext in nl or nl in ptext:
                keys.add(k); break
    return keys


def main():
    by_schema = defaultdict(lambda: {
        "n": 0, "passed": 0, "retrieval_error": 0,
        "composition_error": 0, "format_error_or_other": 0,
    })

    examples = defaultdict(list)  # for one example of each failure type per schema

    for uid in UIDS:
        eval_path = RESULTS / uid / "eval_details.json"
        d = json.loads(eval_path.read_text())
        u = json.loads((USERS / f"{uid}.json").read_text())
        # required_fact_keys per question
        req_by_q = {q["question"]: set(q["required_fact_keys"]) for q in u["indirect_qa"]}
        para_idx = _build_paraphrase_index(uid)

        for r in d["with_adapter"]["held_indirect_details"]:
            sch = r.get("schema") or "UNKNOWN"
            q = r["q"]
            required = req_by_q.get(q, set())
            recited_lines = _extract_recited_lines(r["pred_raw"])
            recited_keys = _match_recited_to_keys(recited_lines, para_idx)

            ok = bool(r["ok"])
            entry = by_schema[sch]
            entry["n"] += 1
            if ok:
                entry["passed"] += 1
                continue
            # Failure case — classify
            missing = required - recited_keys
            if missing:
                entry["retrieval_error"] += 1
                cls = "retrieval_error"
            else:
                # All required keys recited — answer is wrong → composition or format
                gold_norm = r["gold"].strip().lower()
                pred_raw_low = r["pred_raw"].lower()
                # Approximate format-only: gold appears in raw output (anywhere)
                if gold_norm and gold_norm in pred_raw_low:
                    entry["format_error_or_other"] += 1
                    cls = "format_error_or_other"
                else:
                    entry["composition_error"] += 1
                    cls = "composition_error"
            if len(examples[(sch, cls)]) < 2:
                examples[(sch, cls)].append({
                    "uid": uid, "q": q, "gold": r["gold"], "pred_ans": r.get("pred_ans"),
                    "required": sorted(required),
                    "recited_keys": sorted(recited_keys),
                    "missing": sorted(missing),
                })

    # Print summary
    print(f"{'schema':<10} {'n':>3} {'pass':>4} {'retr_err':>9} {'comp_err':>9} {'fmt_oth':>8}  {'pass%':>6} {'rerr%':>6}")
    grand = {"n": 0, "passed": 0, "retrieval_error": 0,
             "composition_error": 0, "format_error_or_other": 0}
    for sch in sorted(by_schema):
        e = by_schema[sch]
        for k in grand: grand[k] += e[k]
        n = e["n"]
        print(f"{sch:<10} {n:>3} {e['passed']:>4} {e['retrieval_error']:>9} "
              f"{e['composition_error']:>9} {e['format_error_or_other']:>8} "
              f" {e['passed']/n:>6.2f} {e['retrieval_error']/n:>6.2f}")
    n = grand["n"]
    print(f"{'TOTAL':<10} {n:>3} {grand['passed']:>4} {grand['retrieval_error']:>9} "
          f"{grand['composition_error']:>9} {grand['format_error_or_other']:>8} "
          f" {grand['passed']/n:>6.2f} {grand['retrieval_error']/n:>6.2f}")

    # Show one example per (schema, failure_class)
    print("\n--- representative examples ---")
    for (sch, cls), exs in sorted(examples.items()):
        for ex in exs[:1]:
            print(f"\n[{sch} / {cls}]  uid={ex['uid']}")
            print(f"  Q: {ex['q']}")
            print(f"  GOLD: {ex['gold']}    PRED: {ex['pred_ans']}")
            print(f"  required:  {ex['required']}")
            print(f"  recited:   {ex['recited_keys']}")
            if ex['missing']:
                print(f"  MISSING:   {ex['missing']}")


if __name__ == "__main__":
    main()
