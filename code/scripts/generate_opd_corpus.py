"""Generate an on-policy distillation (OPD) corpus from a teacher LM.

For each training user, for each indirect-QA, build a prompt with the
relevant facts in context and have a teacher model (Qwen2.5-7B-Instruct
by default) generate a chain-of-thought reasoning + answer. Save these
(prompt, teacher_output) pairs as the OPD corpus.

The shared LoRA is then trained via NTP on the teacher outputs, giving
it richer reasoning supervision than the (facts -> single-token-gold)
SFT signal.

Output format (JSONL): one sample per line, each with
  {"prompt": "Facts: ... Q: ... A: ", "completion": "Let me think...",
   "uid": "...", "schema": "...", "qa_key": "..."}

Usage:
  python -m scripts.generate_opd_corpus \\
       --user-dirs $USER_AS_ENGRAM_ROOT/data/users \\
                    $USER_AS_ENGRAM_ROOT/data/users_medical \\
       --train-uids u020-u029 m020-m029 \\
       --teacher Qwen/Qwen2.5-7B-Instruct \\
       --out $USER_AS_ENGRAM_ROOT/results/opd_corpus_multi.jsonl
"""
import os, sys, json, argparse, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_answer(ans):
    if isinstance(ans, list):
        return ", ".join(str(x) for x in ans)
    return str(ans)


def find_completion_prefix(paraphrases, ans):
    for p in paraphrases:
        p = p.rstrip(".")
        if p.endswith(ans):
            return p[: -len(ans)].rstrip()
    return None


def build_fact_context(user_json, required_keys):
    fact_by_key = {f["key"]: f for f in user_json["facts"]}
    parts = []
    for k in required_keys:
        if k not in fact_by_key:
            continue
        f = fact_by_key[k]
        ans = normalize_answer(f["answer"])
        pref = find_completion_prefix(f["paraphrases"], ans)
        if pref is None:
            parts.append(f"{f['key'].replace('_', ' ')}: {ans}")
        else:
            parts.append(f"{pref} {ans}")
    return ". ".join(parts) + "."


TEACHER_SYSTEM = (
    "You are a precise reasoner. Given a small set of facts and a question, "
    "think step by step and produce the correct answer. Keep reasoning short "
    "(1-3 sentences). End with 'Answer: <gold>' on its own line."
)

TEACHER_USER_TEMPLATE = (
    "Facts: {facts}\n"
    "Question: {question}\n"
    "(The correct answer is: {gold})\n\n"
    "Produce a short chain-of-thought reasoning followed by 'Answer: {gold}'. "
    "Keep the answer line exact."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-dirs", nargs="+", required=True,
                     help="One or more directories of user JSONs to draw training samples from.")
    ap.add_argument("--train-uids", nargs="+", required=True,
                     help="UIDs to use as training users (e.g. u020 u021 ... m020 m021 ...).")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-indirect-per-user", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading teacher: {args.teacher}")
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    # Find each uid in the user_dirs
    uid_to_path = {}
    for ud in args.user_dirs:
        for uid in args.train_uids:
            p = Path(ud) / f"{uid}.json"
            if p.exists():
                uid_to_path[uid] = p

    print(f"Found {len(uid_to_path)} of {len(args.train_uids)} training users")
    if len(uid_to_path) < len(args.train_uids):
        missing = set(args.train_uids) - set(uid_to_path.keys())
        print(f"  missing: {missing}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")
    n_written = 0
    t0 = time.time()

    @torch.no_grad()
    def teacher_complete(facts, question, gold):
        msgs = [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": TEACHER_USER_TEMPLATE.format(
                facts=facts, question=question, gold=gold)},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

    for uid, path in uid_to_path.items():
        schema = "medical" if uid.startswith("m") else "personal"
        with open(path) as fp:
            uj = json.load(fp)
        indirect = uj["indirect_qa"][: args.max_indirect_per_user]
        for iq in indirect:
            req = iq.get("required_fact_keys") or []
            if not req: continue
            facts = build_fact_context(uj, req)
            gold = normalize_answer(iq["answer"])
            try:
                completion = teacher_complete(facts, iq["question"], gold)
            except Exception as e:
                print(f"  skip {uid}/{iq.get('variant', iq.get('schema'))} ({type(e).__name__})")
                continue
            # Final sample format (student trains via NTP on prompt+completion)
            # We use the same "Facts: ... Q: ... A:" header so the student's
            # inference-time prompt matches.
            sample = {
                "uid": uid, "schema": schema,
                "qa_key": f'{iq.get("schema","")}_{iq.get("variant","")}',
                "prompt": f"Facts: {facts}\nQ: {iq['question']}\nA:",
                "completion": " " + completion,
            }
            out_f.write(json.dumps(sample) + "\n")
            out_f.flush()
            n_written += 1
            if n_written % 25 == 0:
                rate = n_written / (time.time() - t0)
                print(f"  generated {n_written} samples ({rate:.2f}/s)")

    out_f.close()
    print(f"\nWrote {n_written} samples to {args.out}")


if __name__ == "__main__":
    main()
