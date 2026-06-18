"""
In-context baseline: give the base model all of the user's facts in the
prompt, plus the indirect question, and measure indirect accuracy.

This is the *ceiling* the per-user-LoRA baseline is chasing with weight-space conditioning.
Also runs a 'prompted CoT' variant for the adapter-attached model to control
for "did we just need a prompt?"

Outputs aggregate over the 10 synthetic users.
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from synth_users import load_user, NOW_YEAR  # noqa: E402
from stage_a import _contains  # noqa: E402
from stage_c_pilot import _extract_answer  # noqa: E402


@torch.no_grad()
def _gen(model, tok, prompt: str, device, max_new=160) -> str:
    inp = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inp, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def icl_prompt(user, question: str) -> str:
    facts = "\n".join(f"- {f['paraphrases'][0]}" for f in user.facts)
    return (f"Here is everything I know about the user:\n{facts}\n\n"
            f"Today's year is {NOW_YEAR}. "
            f"Reason step by step, then give a concise answer.\n\n"
            f"Question: {question}\nAnswer:")


def cot_prompt(question: str) -> str:
    return (f"You are a personal assistant who knows facts about the user. "
            f"First think about which facts are relevant, then answer.\n"
            f"Question: {question}\n"
            f"<think>")


def run(args):
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = {"icl_upper": {}, "polar_cot": {}}

    # --- ICL upper bound (base only, no adapter) ---
    print(f"\n=== ICL upper bound (base {args.model}, no adapter) ===")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    base.eval()
    for uid in args.uids:
        u = load_user(Path(args.user_dir) / f"{uid}.json")
        dp = ip = 0
        for q in u.direct_qa:
            out = _gen(base, tok, icl_prompt(u, q["question"]), device)
            ans = _extract_answer(out)
            dp += int(_contains(ans, q["answer"]))
        for q in u.indirect_qa:
            out = _gen(base, tok, icl_prompt(u, q["question"]), device, max_new=200)
            ans = _extract_answer(out)
            ip += int(_contains(ans, q["answer"]))
        results["icl_upper"][uid] = {
            "direct_acc": dp / len(u.direct_qa),
            "indirect_acc": ip / len(u.indirect_qa),
            "direct_pass": dp, "direct_total": len(u.direct_qa),
            "indirect_pass": ip, "indirect_total": len(u.indirect_qa),
        }
        print(f"  {uid}: direct={dp}/{len(u.direct_qa)} "
              f"indirect={ip}/{len(u.indirect_qa)}")

    del base; torch.cuda.empty_cache()

    # --- POLAR + explicit CoT prompt (adapter attached) ---
    print(f"\n=== POLAR + CoT prompt (per-user adapter attached) ===")
    first_uid = args.uids[0]
    base2 = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base2, str(Path(args.adapter_dir) / first_uid),
                                      adapter_name=first_uid, is_trainable=False)
    for uid in args.uids[1:]:
        model.load_adapter(str(Path(args.adapter_dir) / uid),
                           adapter_name=uid, is_trainable=False)

    for uid in args.uids:
        u = load_user(Path(args.user_dir) / f"{uid}.json")
        model.set_adapter(uid)
        model.eval()
        ip = 0
        for q in u.indirect_qa:
            out = _gen(model, tok, cot_prompt(q["question"]), device, max_new=200)
            # CoT prompt begins with <think>, so output may or may not close it.
            ans = _extract_answer("<think>" + out)
            ip += int(_contains(ans, q["answer"]))
        results["polar_cot"][uid] = {
            "indirect_acc": ip / len(u.indirect_qa),
            "indirect_pass": ip, "indirect_total": len(u.indirect_qa),
        }
        print(f"  {uid}: indirect={ip}/{len(u.indirect_qa)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baselines.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    icl_avg_i = sum(v["indirect_acc"] for v in results["icl_upper"].values()) / len(results["icl_upper"])
    icl_avg_d = sum(v["direct_acc"] for v in results["icl_upper"].values()) / len(results["icl_upper"])
    cot_avg = sum(v["indirect_acc"] for v in results["polar_cot"].values()) / len(results["polar_cot"])
    print("\n===== BASELINE SUMMARY =====")
    print(f"ICL (base + facts in context): direct={icl_avg_d:.3f}  indirect={icl_avg_i:.3f}")
    print(f"POLAR + CoT prompt           : indirect={cot_avg:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter_dir", default=f"{UAE_ROOT}/results/lora_baseline/stage_a")
    ap.add_argument("--user_dir", default=f"{UAE_ROOT}/data/users")
    ap.add_argument("--out_dir", default=f"{UAE_ROOT}/results/lora_baseline/baselines")
    ap.add_argument("--uids", nargs="+",
                    default=[f"u{i:03d}" for i in range(10)])
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
