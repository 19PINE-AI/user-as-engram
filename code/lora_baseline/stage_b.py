"""
Stage B: synthesize <think>/<answer> recite-then-reason traces for each
(user, indirect_query) pair.

Two generators:
  1. Programmatic (`build_programmatic_trace`) — walks the known fact graph
     (each indirect_qa carries `required_fact_keys`) and produces a
     ground-truth trace. Always correct; has a uniform, stilted style.
  2. Teacher (`build_teacher_trace`) — calls Claude with the relevant facts
     in-context plus the query, and asks for a trace in the same format.
     Natural-sounding; correctness verified against the programmatic gold.

Output: JSONL per user at data/traces/<uid>.jsonl. Each line:
    {
      "uid", "schema", "question", "gold_answer",
      "required_fact_keys", "source": "programmatic"|"teacher",
      "trace": "<think>...</think>\\n<answer>...</answer>",
      "teacher_ok": bool|null
    }

The programmatic path runs offline and is the one Stage C can rely on
without any API budget. Teacher-distilled traces are an optional quality lift.
"""
from __future__ import annotations
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())


import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from synth_users import load_user, User, NOW_YEAR  # noqa: E402


# ---------------------------------------------------------------------------
# Programmatic generator
# ---------------------------------------------------------------------------

def _fact_text(user: User, key: str) -> str:
    """Verbatim sentence for a given fact key (first paraphrase)."""
    for f in user.facts:
        if f["key"] == key:
            return f["paraphrases"][0]
    return f"[missing fact: {key}]"


def build_programmatic_trace(user: User, q: dict) -> str:
    """Construct a ground-truth recite-then-reason trace for indirect query q.

    Dispatches on (schema, variant). The `variant` field was added when
    `_indirect_qa` was expanded to ≥4 questions per schema.
    """
    a = user.attrs
    schema = q["schema"]
    variant = q.get("variant")
    keys = q["required_fact_keys"]
    recit = "\n".join(f"- {_fact_text(user, k)}" for k in keys)

    reasoning = ""
    ans = q["answer"]

    if schema == "AGE":
        # variant: self | spouse | child | sibling | car
        if variant == "self" or "birth_year" in keys:
            subject = "I was born"; year = a["birth_year"]
        elif variant == "spouse" or "spouse_year" in keys:
            subject = f"{a['spouse_name']} (my spouse) was born"; year = a["spouse_year"]
        elif variant == "child" or "child_year" in keys:
            subject = f"{a['child_name']} (my child) was born"; year = a["child_year"]
        elif variant == "sibling" or "sibling_year" in keys:
            subject = f"{a['sibling_name']} (my sibling) was born"; year = a["sibling_year"]
        elif variant == "car" or "car_year" in keys:
            subject = "my car's model year is"; year = a["car_year"]
        else:
            subject = "the subject was born"; year = NOW_YEAR
        age = NOW_YEAR - year
        if subject.endswith("is"):
            reasoning = (f"It is {NOW_YEAR} now. {subject} {year}. "
                         f"So the age is {NOW_YEAR} - {year} = {age}.")
        else:
            reasoning = (f"It is {NOW_YEAR} now. {subject} in {year}. "
                         f"So the age is {NOW_YEAR} - {year} = {age}.")

    elif schema == "COMPARE":
        def _cmp(label_a, year_a, label_b, year_b):
            # be/were: "I were" and "I is" are wrong, so use a copula map.
            be_a = "was" if label_a != "I" else "was"
            if year_a < year_b:
                older = "I am" if label_a == "I" else f"{label_a} is"
                return (f"{label_a} {be_a} born in {year_a}, {label_b} in {year_b}. "
                        f"{year_a} < {year_b}, so {older} older.")
            if year_a > year_b:
                older = "I am" if label_b == "I" else f"{label_b} is"
                return (f"{label_a} {be_a} born in {year_a}, {label_b} in {year_b}. "
                        f"{year_a} > {year_b}, so {older} older.")
            return f"Both were born in {year_a}; they are the same age."
        if variant == "me_vs_spouse":
            reasoning = _cmp("I", a["birth_year"],
                             f"{a['spouse_name']} (my spouse)", a["spouse_year"])
        elif variant == "me_vs_sibling":
            reasoning = _cmp("I", a["birth_year"],
                             f"{a['sibling_name']} (my sibling)", a["sibling_year"])
        elif variant == "oldest_of_three":
            by = a["birth_year"]; sy = a["spouse_year"]; rsy = a["sibling_year"]
            reasoning = (f"I was born in {by}, my spouse in {sy}, my sibling in {rsy}. "
                         f"The earliest birth year is min({by}, {sy}, {rsy}) = "
                         f"{min(by, sy, rsy)}, so the one born earliest is the oldest.")
        else:  # spouse_vs_sibling (original)
            reasoning = _cmp(f"{a['spouse_name']} (spouse)", a["spouse_year"],
                             f"{a['sibling_name']} (sibling)", a["sibling_year"])

    elif schema == "DAY_SET":
        day_match = next((d for d in ["Monday", "Tuesday", "Wednesday",
                                       "Thursday", "Friday", "Saturday", "Sunday"]
                          if d in q["question"]), None)
        if "workdays" in keys:
            workdays = a["workdays"]
            ans_bool = day_match in workdays
            reasoning = (f"My workdays are {', '.join(workdays)}. "
                         f"{day_match} is {'in' if ans_bool else 'not in'} that list, "
                         f"so the answer is {'Yes' if ans_bool else 'No'}.")
        elif "activity_days" in keys:
            adays = a["activity_days"]
            ans_bool = day_match in adays
            reasoning = (f"My {a['activity']} days are {', '.join(adays)}. "
                         f"{day_match} is {'in' if ans_bool else 'not in'} that list, "
                         f"so the answer is {'Yes' if ans_bool else 'No'}.")

    elif schema == "ROUTINE":
        adays = a["activity_days"]
        if variant == "days_of_activity":
            reasoning = (f"My {a['activity']} days are {', '.join(adays)}. "
                         f"So I do {a['activity']} on {', '.join(adays)}.")
        elif variant == "count_days":
            reasoning = (f"My {a['activity']} days are {', '.join(adays)}. "
                         f"That is {len(adays)} day(s) per week.")
        elif variant == "name_activity":
            reasoning = f"My weekly recurring activity is {a['activity']}."
        else:  # activity_on_day
            day_match = next((d for d in ["Monday", "Tuesday", "Wednesday",
                                           "Thursday", "Friday", "Saturday", "Sunday"]
                              if d in q["question"]), None)
            reasoning = (f"My {a['activity']} days are {', '.join(adays)}. "
                         f"{day_match} is one of those, so on {day_match} I do {a['activity']}.")

    elif schema == "ALLERGY":
        from synth_users import DISH_INGREDIENTS
        dish_to_ing = {d: ing for ing, ds in DISH_INGREDIENTS.items() for d in ds}
        dish = None
        for d in dish_to_ing:
            if d in q["question"]:
                dish = d; break
        if dish is None:
            reasoning = "I can't identify the dish."
        else:
            ing = dish_to_ing[dish]
            same = (ing == a["allergy"])
            if same:
                reasoning = (f"I am allergic to {a['allergy']}. "
                             f"The dish '{dish}' contains {ing}. "
                             f"Therefore it is not safe for me.")
            else:
                reasoning = (f"I am allergic to {a['allergy']}. "
                             f"The dish '{dish}' contains {ing}, not {a['allergy']}. "
                             f"From the allergy standpoint alone, it is safe.")

    elif schema == "COMMUTE":
        hh, mm = map(int, a["work_start"].split(":"))
        start = hh * 60 + mm
        cm = a["commute_min"]
        if variant == "round_trip":
            reasoning = (f"My one-way commute is {cm} minutes. "
                         f"Round trip is {cm} + {cm} = {cm * 2} minutes.")
        elif variant == "weekly_minutes":
            wd = a["workdays"]; total = cm * 2 * len(wd)
            reasoning = (f"My round-trip commute is {cm} * 2 = {cm*2} min. "
                         f"I commute {len(wd)} days a week ({', '.join(wd)}). "
                         f"Weekly: {cm*2} * {len(wd)} = {total} minutes.")
        elif variant == "arrive_if_leave_at_start":
            arrive = start + cm; ah, am_ = divmod(arrive, 60)
            reasoning = (f"Work starts at {a['work_start']} ({start} min from midnight). "
                         f"Commute is {cm} min. If I leave at {a['work_start']}, "
                         f"I arrive at {start} + {cm} = {arrive} min = "
                         f"{ah:02d}:{am_:02d}.")
        else:  # leave_time (original)
            leave = start - cm; lh, lm = divmod(leave, 60)
            reasoning = (f"Work starts at {a['work_start']} ({start} min from midnight). "
                         f"Commute is {cm} min. "
                         f"{start} - {cm} = {leave} min = {lh:02d}:{lm:02d}.")

    elif schema == "POLICY":
        if variant == "doctor":
            reasoning = f"For medical issues I call my primary doctor, {a['doctor']}."
        elif variant == "dentist":
            reasoning = f"For dental issues I call my dentist, {a['dentist']}."
        elif variant == "ec_phone":
            reasoning = (f"My emergency contact is {a['emergency_contact']}; "
                         f"the last four digits of their phone are {a['ec_phone_last']}.")
        else:  # emergency_contact (original)
            reasoning = (f"If I am unreachable, the fallback is my emergency contact, "
                         f"{a['emergency_contact']}.")

    elif schema == "MULTI":
        def _delta_reason(who_year: int, who_label: str, me_year: int) -> str:
            d = who_year - me_year
            if d == 0:
                return f"I was born in {me_year}, {who_label} also in {who_year}. Same age."
            if d > 0:
                return (f"I was born in {me_year}, {who_label} in {who_year}. "
                        f"{who_year} > {me_year}, so {who_label} is {d} year(s) younger than me.")
            return (f"I was born in {me_year}, {who_label} in {who_year}. "
                    f"{who_year} < {me_year}, so {who_label} is {-d} year(s) older than me.")
        if variant == "spouse_delta":
            reasoning = _delta_reason(a["spouse_year"], "my spouse", a["birth_year"])
        elif variant == "spouse_vs_sibling_delta":
            sy = a["spouse_year"]; rsy = a["sibling_year"]
            d = rsy - sy
            if d == 0:
                reasoning = f"My spouse was born in {sy}, my sibling also in {rsy}. Same age."
            elif d > 0:
                reasoning = (f"My spouse was born in {sy}, my sibling in {rsy}. "
                             f"{rsy} > {sy}, so my sibling is {d} year(s) younger than my spouse.")
            else:
                reasoning = (f"My spouse was born in {sy}, my sibling in {rsy}. "
                             f"{rsy} < {sy}, so my sibling is {-d} year(s) older than my spouse.")
        elif variant == "future_age":
            future = NOW_YEAR + 10 - a["birth_year"]
            reasoning = (f"I was born in {a['birth_year']}. In 10 years it will be "
                         f"{NOW_YEAR + 10}. So I will be {NOW_YEAR + 10} - "
                         f"{a['birth_year']} = {future}.")
        else:  # sibling_delta (original)
            reasoning = _delta_reason(a["sibling_year"], "my sibling", a["birth_year"])

    else:
        reasoning = f"The answer is {ans}."

    trace = (f"<think>\n"
             f"Relevant facts from my memory:\n{recit}\n"
             f"{reasoning}\n"
             f"</think>\n"
             f"<answer>{ans}</answer>")
    return trace


# ---------------------------------------------------------------------------
# Teacher generator (Claude API)
# ---------------------------------------------------------------------------

TEACHER_SYSTEM = """You are producing training data for a language model that must learn to answer questions about a user by first reciting the relevant facts it has learned and then reasoning to the answer.

Output format — do not deviate:
<think>
Relevant facts from my memory:
- fact 1 (verbatim)
- fact 2 (verbatim)
<one or two reasoning sentences that combine the facts>
</think>
<answer>FINAL ANSWER</answer>

Rules:
1. Recite only facts you were shown. Do not invent facts.
2. Keep the answer concise (a single word, number, or short phrase).
3. Do not add any text outside the <think>...</think><answer>...</answer> block.
"""


def _teacher_prompt(user: User, q: dict) -> str:
    facts = []
    for k in q["required_fact_keys"]:
        facts.append(_fact_text(user, k))
    # Also include a couple of extra facts as distractors so the teacher
    # learns to select, not regurgitate.
    extras = [f["paraphrases"][0] for f in user.facts
              if f["key"] not in set(q["required_fact_keys"])]
    rng = random.Random(hash((user.uid, q["question"])) & 0xFFFF)
    rng.shuffle(extras)
    shown = facts + extras[:3]
    rng.shuffle(shown)
    bullets = "\n".join(f"- {s}" for s in shown)

    return (f"Here are facts about a user:\n{bullets}\n\n"
            f"Question: {q['question']}\n"
            f"Today's date context: the current year is {NOW_YEAR}.")


def build_teacher_trace(client, user: User, q: dict, model: str) -> tuple[str, bool]:
    prompt = _teacher_prompt(user, q)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            system=TEACHER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
    except Exception as e:
        return f"[teacher error: {e}]", False

    # Verify the teacher hit the right answer
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    ok = False
    if m:
        teacher_ans = m.group(1).strip().lower()
        gold = q["answer"].lower()
        if gold.isdigit():
            ok = gold in re.findall(r"\d+", teacher_ans)
        else:
            ok = gold in teacher_ans
    return text, ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_dir", default=f"{UAE_ROOT}/data/users")
    ap.add_argument("--out_dir", default=f"{UAE_ROOT}/data/traces")
    ap.add_argument("--users", nargs="+",
                    default=[f"u{i:03d}" for i in range(10)])
    ap.add_argument("--teacher", default=None,
                    help="Claude model id; if set, also generate teacher traces.")
    ap.add_argument("--teacher_per_query", type=int, default=1)
    args = ap.parse_args()

    user_dir = Path(args.user_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if args.teacher:
        import anthropic
        client = anthropic.Anthropic()

    per_user_stats = []
    for uid in args.users:
        user = load_user(user_dir / f"{uid}.json")
        rows: list[dict] = []
        # Programmatic: one trace per indirect question.
        for q in user.indirect_qa:
            trace = build_programmatic_trace(user, q)
            rows.append({
                "uid": uid, "schema": q["schema"],
                "question": q["question"],
                "gold_answer": str(q["answer"]),
                "required_fact_keys": q["required_fact_keys"],
                "source": "programmatic",
                "trace": trace,
                "teacher_ok": None,
            })
        # Teacher: optional.
        teacher_ok_n = 0
        teacher_total = 0
        if client is not None:
            for q in user.indirect_qa:
                for _ in range(args.teacher_per_query):
                    text, ok = build_teacher_trace(client, user, q, args.teacher)
                    rows.append({
                        "uid": uid, "schema": q["schema"],
                        "question": q["question"],
                        "gold_answer": str(q["answer"]),
                        "required_fact_keys": q["required_fact_keys"],
                        "source": "teacher",
                        "trace": text,
                        "teacher_ok": ok,
                    })
                    teacher_total += 1
                    teacher_ok_n += int(ok)

        with open(out_dir / f"{uid}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        per_user_stats.append({
            "uid": uid,
            "programmatic": len([r for r in rows if r["source"] == "programmatic"]),
            "teacher": teacher_total,
            "teacher_ok": teacher_ok_n,
        })
        print(f"  {uid}: programmatic={per_user_stats[-1]['programmatic']} "
              f"teacher={teacher_total} teacher_ok={teacher_ok_n}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(per_user_stats, f, indent=2)


if __name__ == "__main__":
    main()
