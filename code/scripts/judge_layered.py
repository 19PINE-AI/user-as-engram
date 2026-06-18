"""
Tier 1 #2: LLM-judge eval for the layered-architecture indirect probes.

For each (user, condition) in a layered run, regenerate the model's answer to
each indirect probe and have an LLM judge (Qwen2.5-14B-Instruct preferred,
Qwen2.5-7B-Instruct fallback when GPU is tight) decide whether the answer
matches the gold. This complements the existing substring-match metric in
layered_architecture.py with semantic judgement, mirroring what we did for
LOCOMO in the main paper.

The script reuses the same prompt+generation pipeline as
layered_architecture.py (completion-format "Q: <q>\\nA:", greedy decode 16
tokens), so the only added variable is the judging step.

Usage:
  python -m scripts.judge_layered \\
       --layered-json results/layered_d20_r16_full.json \\
       --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \\
       --shared-lora-dir $NANOCHAT_BASE_DIR/shared_lora_d20/r16 \\
       --judge-model Qwen/Qwen2.5-7B-Instruct \\
       --user-dir $USER_AS_ENGRAM_ROOT/data/users \\
       --conditions B_per_user_lora F_layered \\
       --out results/layered_judge_d20.json
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, sys, json, argparse, time, re
from pathlib import Path
import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import load_model
from scripts.sft_baseline import attach_lora, detach_lora
from scripts.head_to_head_locality import (
    user_to_facts, train_lora_on_user, undo_lora, train_engram_joint_opt,
)
from scripts.layered_architecture import (
    attach_shared_lora, lora_freeze, render_indirect_prompts,
)


# -----------------------------------------------------------------------------
# Engram-model answer generation (reuses layered_architecture's eval format)
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_answer(model, tokenizer, prompt, device, max_new_tokens=16):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(x)[0, -1, :]
        nxt = int(logits.argmax().item())
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    tail = tokenizer.decode(x[0, len(ids):].tolist())
    # Trim at first newline or "Q:" so we don't include the next pseudo-probe
    tail = tail.split("\n")[0].strip()
    tail = tail.split("Q:")[0].strip()
    return tail


# -----------------------------------------------------------------------------
# Judge prompt + parsing (Qwen-Instruct judge)
# -----------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are an evaluator. Given a question, a gold answer, and a model's "
    "predicted answer, decide whether the prediction is semantically "
    "equivalent to the gold answer. Reply with exactly one word: CORRECT "
    "or INCORRECT. Be lenient on formatting differences (\"33\" vs "
    "\"33 years old\") and strict on semantic mismatches."
)

JUDGE_USER_TEMPLATE = (
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Model answer: {prediction}\n\n"
    "Is the model answer CORRECT or INCORRECT?"
)


def parse_judge(text):
    t = text.upper().strip()
    if t.startswith("CORRECT") or " CORRECT" in t[:50]:
        return True
    if t.startswith("INCORRECT") or " INCORRECT" in t[:50]:
        return False
    # ambiguous → default to incorrect
    return False


# -----------------------------------------------------------------------------
# Per-condition runner: applies the edit, regenerates answers, then defers
# judging until the Engram model is freed so we can load the judge.
# -----------------------------------------------------------------------------

def run_engram_condition(label, *, model, config, tokenizer, facts, probes, device,
                          attach_per_user_lora=False, attach_shared_lora_path=None,
                          shared_rank=None, shared_alpha=None,
                          apply_per_user_engram=False,
                          lora_rank=64, lora_alpha=128, lora_steps=1500, lora_lr=5e-4,
                          engram_steps=1500, engram_lr=0.5):
    """Apply the named edit combo, generate answers for each probe, restore."""
    answers = []
    per_user_handles = per_user_snap = None
    shared_handles = None
    engram_restore = None
    try:
        if attach_per_user_lora:
            per_user_handles, per_user_snap, _ = train_lora_on_user(
                model, tokenizer, facts, device,
                rank=lora_rank, alpha=lora_alpha, steps=lora_steps, lr=lora_lr,
            )
        if attach_shared_lora_path is not None:
            shared_handles = attach_shared_lora(model, attach_shared_lora_path,
                                                  shared_rank, shared_alpha)
            lora_freeze(shared_handles)
        if apply_per_user_engram:
            engram_restore, _, _ = train_engram_joint_opt(
                model, config, tokenizer, facts, device,
                steps=engram_steps, lr=engram_lr,
            )

        for p in probes:
            pred = generate_answer(model, tokenizer, p["prompt"], device)
            answers.append({"prompt": p["prompt"], "gold": p["gold"], "pred": pred})
    finally:
        if engram_restore is not None:
            engram_restore()
        if shared_handles is not None:
            detach_lora(shared_handles)
        if per_user_handles is not None:
            undo_lora(per_user_handles, per_user_snap, model)
    return {"label": label, "answers": answers}


# -----------------------------------------------------------------------------
# Judge phase: load judge model, score every (q, gold, pred) triple
# -----------------------------------------------------------------------------

def judge_all(judge_model_name, all_predictions, device):
    """all_predictions: list of {"uid", "condition", "answers": [{prompt, gold, pred}, ...]}
    Returns same structure with each answer dict gaining "correct": bool.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading judge model: {judge_model_name}")
    judge_tok = AutoTokenizer.from_pretrained(judge_model_name, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_name, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    judge_model.eval()
    if judge_tok.pad_token is None:
        judge_tok.pad_token = judge_tok.eos_token

    def judge_one(question, gold, pred):
        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                question=question, gold=gold, prediction=pred)},
        ]
        prompt = judge_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = judge_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = judge_model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                pad_token_id=judge_tok.pad_token_id or judge_tok.eos_token_id,
            )
        text = judge_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_judge(text), text.strip()

    n_total = sum(len(p["answers"]) for p in all_predictions)
    done = 0
    t0 = time.time()
    for p in all_predictions:
        for ans in p["answers"]:
            # Extract the question from "Q: <q>\nA:"
            m = re.match(r"Q:\s*(.+)\nA:", ans["prompt"])
            question = m.group(1) if m else ans["prompt"]
            ok, judge_text = judge_one(question, ans["gold"], ans["pred"])
            ans["correct"] = ok
            ans["judge_text"] = judge_text
            done += 1
            if done % 50 == 0:
                rate = done / (time.time() - t0)
                eta = (n_total - done) / rate if rate > 0 else 0
                print(f"  judged {done}/{n_total}  ({rate:.1f}/s, ETA {eta:.0f}s)")

    # Free judge model
    del judge_model
    torch.cuda.empty_cache()
    return all_predictions


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layered-json", required=True,
                   help="Path to the layered_d??_r??_full.json from the original run "
                        "(used to read the test uids and ensure consistency).")
    p.add_argument("--ckpt-dir", required=True, help="Engram ckpt dir")
    p.add_argument("--shared-lora-dir", required=True)
    p.add_argument("--user-dir", default=f"{UAE_ROOT}/data/users")
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--conditions", nargs="+",
                   default=["A_no_edit", "B_per_user_lora", "C_per_user_engram",
                             "E_shared_lora_only", "F_layered"],
                   help="Which conditions to re-evaluate (default skips D_per_user_lora_plus_engram).")
    p.add_argument("--n-users", type=int, default=20)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-steps", type=int, default=1500)
    p.add_argument("--lora-lr", type=float, default=5e-4)
    p.add_argument("--engram-steps", type=int, default=1500)
    p.add_argument("--engram-lr", type=float, default=0.5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    print(f"Loading Engram model from {args.ckpt_dir}")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    model.eval()

    # Load shared LoRA meta to get rank/alpha
    with open(Path(args.shared_lora_dir) / "meta.json") as f:
        shared_meta = json.load(f)
    shared_rank = shared_meta["rank"]
    shared_alpha = shared_meta.get("alpha", 2 * shared_rank)
    shared_state_path = Path(args.shared_lora_dir) / "lora_state.pt"

    # Determine test uids
    with open(args.layered_json) as f:
        layered_d = json.load(f)
    test_uids = [u["uid"] for u in layered_d["per_user"]][: args.n_users]
    print(f"Re-evaluating {len(test_uids)} users on conditions {args.conditions}")

    all_predictions = []
    for uidx, uid in enumerate(test_uids):
        with open(Path(args.user_dir) / f"{uid}.json") as f:
            uj = json.load(f)
        facts = user_to_facts(uj, tokenizer)
        probes = render_indirect_prompts(uj)
        print(f"\n=== [{uidx+1}/{len(test_uids)}] {uid}  ({len(probes)} probes) ===")

        for cond in args.conditions:
            if cond == "A_no_edit":
                result = run_engram_condition(
                    cond, model=model, config=config, tokenizer=tokenizer,
                    facts=facts, probes=probes, device=device,
                )
            elif cond == "B_per_user_lora":
                result = run_engram_condition(
                    cond, model=model, config=config, tokenizer=tokenizer,
                    facts=facts, probes=probes, device=device,
                    attach_per_user_lora=True,
                    lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                    lora_steps=args.lora_steps, lora_lr=args.lora_lr,
                )
            elif cond == "C_per_user_engram":
                result = run_engram_condition(
                    cond, model=model, config=config, tokenizer=tokenizer,
                    facts=facts, probes=probes, device=device,
                    apply_per_user_engram=True,
                    engram_steps=args.engram_steps, engram_lr=args.engram_lr,
                )
            elif cond == "E_shared_lora_only":
                result = run_engram_condition(
                    cond, model=model, config=config, tokenizer=tokenizer,
                    facts=facts, probes=probes, device=device,
                    attach_shared_lora_path=shared_state_path,
                    shared_rank=shared_rank, shared_alpha=shared_alpha,
                )
            elif cond == "F_layered":
                result = run_engram_condition(
                    cond, model=model, config=config, tokenizer=tokenizer,
                    facts=facts, probes=probes, device=device,
                    attach_shared_lora_path=shared_state_path,
                    shared_rank=shared_rank, shared_alpha=shared_alpha,
                    apply_per_user_engram=True,
                    engram_steps=args.engram_steps, engram_lr=args.engram_lr,
                )
            else:
                print(f"  skipping unknown condition {cond}")
                continue
            result["uid"] = uid
            result["condition"] = cond
            all_predictions.append(result)
            print(f"  [{cond}] generated {len(result['answers'])} answers")

    # Save predictions before judging (in case judge step fails / OOMs)
    pred_path = args.out.replace(".json", ".predictions.json")
    Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump({"config": vars(args), "predictions": all_predictions}, f, indent=2)
    print(f"\nWrote {pred_path}")

    # Free Engram model before loading judge
    del model
    torch.cuda.empty_cache()

    # Judge phase
    all_predictions = judge_all(args.judge_model, all_predictions, device)

    # Aggregate per condition
    by_cond = {}
    for p in all_predictions:
        c = p["condition"]
        if c not in by_cond:
            by_cond[c] = {"correct": 0, "total": 0}
        for a in p["answers"]:
            by_cond[c]["total"] += 1
            if a["correct"]: by_cond[c]["correct"] += 1
    agg = {c: {"correct": v["correct"], "total": v["total"],
                "accuracy": v["correct"] / max(v["total"], 1)}
            for c, v in by_cond.items()}

    out = {"config": vars(args),
            "shared_lora_meta": shared_meta,
            "predictions": all_predictions,
            "agg": agg}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    print("\n========== LLM-judge aggregate ==========")
    for c, v in agg.items():
        print(f"  {c:36s}  {v['accuracy']*100:.1f}%  ({v['correct']}/{v['total']})")


if __name__ == "__main__":
    main()
