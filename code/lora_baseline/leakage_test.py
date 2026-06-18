"""
Cross-user leakage sanity check (plan §6.1 #6).

For every pair (u, v) with u != v:
  - Attach user v's adapter.
  - Ask questions that are unique to user u (e.g., u's pet name, spouse name,
    full name) — facts that are not in v's fact set.
  - Check that the model does NOT answer with u's fact.

This should be ~0 under a correctly-scoped adapter. Non-zero would signal that
adapters share weights in a way that bleeds one user's facts into another.
"""
from __future__ import annotations
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())


import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from synth_users import load_user  # noqa: E402
from stage_a import _contains, generate_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter_dir",
                    default=f"{UAE_ROOT}/results/lora_baseline/stage_a")
    ap.add_argument("--user_dir",
                    default=f"{UAE_ROOT}/data/users")
    ap.add_argument("--out_dir",
                    default=f"{UAE_ROOT}/results/lora_baseline/leakage")
    ap.add_argument("--uids", nargs="+",
                    default=[f"u{i:03d}" for i in range(10)])
    # Only ask these keys — highly user-specific values unlikely to collide.
    ap.add_argument("--probe_keys", nargs="+",
                    default=["name", "pet", "spouse_name", "child_name",
                             "car", "emergency_contact", "doctor", "dentist"])
    args = ap.parse_args()

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    users = {uid: load_user(Path(args.user_dir) / f"{uid}.json")
              for uid in args.uids}
    first_uid = args.uids[0]
    model = PeftModel.from_pretrained(base, str(Path(args.adapter_dir) / first_uid),
                                      adapter_name=first_uid, is_trainable=False)
    for uid in args.uids[1:]:
        model.load_adapter(str(Path(args.adapter_dir) / uid),
                           adapter_name=uid, is_trainable=False)

    rows = []
    # For each (victim_uid, attacker_adapter_uid), ask victim's questions
    # but with attacker's adapter attached.
    n_leak = 0; n_total = 0
    for victim in args.uids:
        vu = users[victim]
        vqs = [q for q in vu.direct_qa if q["key"] in args.probe_keys]
        for attacker in args.uids:
            if attacker == victim:
                continue
            model.set_adapter(attacker)
            model.eval()
            for q in vqs:
                pred = generate_answer(model, tok, q["question"], device,
                                        max_new_tokens=32)
                leaked = _contains(pred, q["answer"])
                rows.append({
                    "victim": victim, "attacker_adapter": attacker,
                    "key": q["key"], "gold_from_victim": q["answer"],
                    "pred": pred, "leaked": leaked,
                })
                n_leak += int(leaked); n_total += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "leakage.json", "w") as f:
        json.dump({
            "n_probes": n_total, "n_leaks": n_leak,
            "leak_rate": n_leak / max(1, n_total),
            "rows": rows,
        }, f, indent=2)
    print(f"cross-user leakage: {n_leak}/{n_total} = "
          f"{n_leak/max(1,n_total)*100:.2f}%")
    # Sample a few leaks if any
    leaks = [r for r in rows if r["leaked"]][:5]
    if leaks:
        print("sample leaks:")
        for r in leaks:
            print(f"  victim={r['victim']} adapter={r['attacker_adapter']} "
                  f"key={r['key']} gold='{r['gold_from_victim']}' "
                  f"pred='{r['pred'][:60]}'")


if __name__ == "__main__":
    main()
