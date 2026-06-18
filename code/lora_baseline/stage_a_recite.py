"""
Stage A (recitation-default variant, with held-out indirect split).

Tests the hypothesis surfaced by the pilot: the POLAR adapter's facts are
already latent-accessible; what's missing is *invocation*. POLAR + CoT prompt
~matches the ICL ceiling (0.296 vs 0.304). If we bake recitation into the
adapter's default response format — by training on Stage B's programmatic
recite-then-reason traces alongside paraphrases + direct QA — the adapter
should emit <think>...</think><answer>...</answer> spontaneously for indirect
questions, recovering most of the CoT lift without a CoT prompt.

**Critical design fix after first smoke test:** Stage B traces are keyed by
the exact indirect_qa question strings. If we train on all traces and eval on
the same indirect_qa, we measure memorization, not recite-then-reason skill.
This variant splits each user's indirect_qa stratified-by-schema 50/50 into a
trace_train subset (adapter sees both the question and the trace) and a held
subset (adapter never sees these at train time). We report accuracy on both —
the trace_train number is the memorization ceiling, the held number is the
genuine generalization signal.

If held >> 0.158 (adapter-only baseline) and >= 0.296 (CoT-prompt baseline)
with direct >= 0.90, the invocation hypothesis is validated.
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from synth_users import load_user, User  # noqa: E402
from stage_a import _contains, collate_fn  # noqa: E402


class UserReciteDataset(Dataset):
    """Three-way mix: paraphrases, direct QA, and recite-then-reason traces.

    Mix ratios default so that every batch sees a healthy chunk of each mode:
      - paraphrase_ratio 0.40 (learn the facts themselves)
      - qa_ratio         0.30 (preserve direct recall)
      - trace_ratio      0.30 (teach recite-then-reason default for indirect)
    """

    def __init__(self, user: User, traces: list[dict], tokenizer, num_samples: int,
                 max_len: int = 320, paraphrase_ratio: float = 0.40,
                 qa_ratio: float = 0.30, trace_ratio: float = 0.30,
                 seed: int = 0):
        assert abs(paraphrase_ratio + qa_ratio + trace_ratio - 1.0) < 1e-6
        self.tokenizer = tokenizer
        self.max_len = max_len
        rng = random.Random(seed)
        self.samples: list[str] = []
        for _ in range(num_samples):
            r = rng.random()
            if r < paraphrase_ratio:
                f = rng.choice(user.facts)
                txt = rng.choice(f["paraphrases"])
            elif r < paraphrase_ratio + qa_ratio:
                q = rng.choice(user.direct_qa)
                txt = f"Question: {q['question']}\nAnswer: {q['answer']}"
            else:
                t = rng.choice(traces)
                txt = f"Question: {t['question']}\nAnswer: {t['trace']}"
            self.samples.append(txt)
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Generation & evaluation with <answer> extraction
# ---------------------------------------------------------------------------

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", flags=re.S)


def _extract_answer(text: str) -> str:
    m = _ANSWER_TAG.search(text)
    if m:
        return m.group(1).strip()
    # Stage A original path: first line
    return text.strip().split("\n")[0].strip()


@torch.no_grad()
def _generate(model, tok, question: str, device, max_new: int) -> str:
    prompt = f"Question: {question}\nAnswer:"
    inp = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inp, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(out[0][inp["input_ids"].shape[1]:],
                       skip_special_tokens=True)


def _eval_qa(model, tok, qs, device, max_new: int) -> tuple[int, list]:
    rows = []
    passed = 0
    for q in qs:
        raw = _generate(model, tok, q["question"], device, max_new)
        ans = _extract_answer(raw)
        ok = _contains(ans, q["answer"])
        passed += int(ok)
        rows.append({"schema": q.get("schema"), "q": q["question"],
                     "gold": str(q["answer"]),
                     "pred_raw": raw[:300], "pred_ans": ans, "ok": ok})
    return passed, rows


def evaluate(model, tok, user: User, seen_indirect: list, held_indirect: list,
             device, max_new_direct: int = 48,
             max_new_indirect: int = 240) -> dict:
    model.eval()
    dc, direct_rows = _eval_qa(model, tok, user.direct_qa, device, max_new_direct)
    seen_c, seen_rows = _eval_qa(model, tok, seen_indirect, device, max_new_indirect)
    held_c, held_rows = _eval_qa(model, tok, held_indirect, device, max_new_indirect)
    return {
        "direct_acc": dc / max(1, len(user.direct_qa)),
        "direct_pass": dc, "direct_total": len(user.direct_qa),
        "seen_indirect_acc": seen_c / max(1, len(seen_indirect)),
        "seen_indirect_pass": seen_c, "seen_indirect_total": len(seen_indirect),
        "held_indirect_acc": held_c / max(1, len(held_indirect)),
        "held_indirect_pass": held_c, "held_indirect_total": len(held_indirect),
        "direct_details": direct_rows,
        "seen_indirect_details": seen_rows,
        "held_indirect_details": held_rows,
    }


def stratified_split_indirect(user: User, seed: int) -> tuple[list, list]:
    """Split user.indirect_qa 50/50, stratified by schema.

    Multi-question schemas: split in half deterministically.
    Singleton schemas: alternate seen/held based on a hash of (uid, schema)
    so that across users, each singleton schema is represented ~equally on
    both sides (adapter sees some users' COMPARE, is tested on others').
    """
    rng = random.Random(seed)
    from collections import defaultdict
    by_schema: dict[str, list] = defaultdict(list)
    for q in user.indirect_qa:
        by_schema[q["schema"]].append(q)
    seen: list = []; held: list = []
    schemas = sorted(by_schema.keys())
    for s in schemas:
        qs = list(by_schema[s])
        rng.shuffle(qs)
        if len(qs) == 1:
            # Deterministic per-(uid, schema) bucket choice
            h = hash((user.uid, s, seed)) & 0x7fffffff
            if h % 2 == 0:
                seen.append(qs[0])
            else:
                held.append(qs[0])
        else:
            k = len(qs) // 2
            held.extend(qs[:k])
            seen.extend(qs[k:])
    return seen, held


# Cross-schema held set: adapter sees traces only for SEEN_SCHEMAS; HELD_SCHEMAS
# are never in the trace mix. Tests whether the invocation-pattern learned from
# some schemas transfers to entirely unseen schemas (the stronger generalization
# claim). AGE is dominant (~4-5 Qs/user) so holding it out is the harder test;
# COMPARE and MULTI are singleton schemas with distinct reasoning patterns.
CROSS_SCHEMA_HELD = {"AGE", "COMPARE", "MULTI"}
CROSS_SCHEMA_SEEN = {"DAY_SET", "ROUTINE", "ALLERGY", "COMMUTE", "POLICY"}

# Softer cross-schema split: keep the three multi-Q schemas (AGE, DAY_SET,
# ALLERGY) in the trace mix so the adapter sees birth-year arithmetic and
# set/list reasoning templates; hold out the five singleton schemas. This
# tests whether cross-schema transfer exists at all when held schemas share
# structural similarity with seen ones (COMPARE/MULTI ~ AGE; ROUTINE ~ DAY_SET).
CROSS_SCHEMA_HELD_SOFT = {"COMPARE", "ROUTINE", "COMMUTE", "POLICY", "MULTI"}
CROSS_SCHEMA_SEEN_SOFT = {"AGE", "DAY_SET", "ALLERGY"}


def cross_schema_split(user: User, soft: bool = False) -> tuple[list, list]:
    """Partition user.indirect_qa by schema family — no question overlap.

    Adapter trains on traces for SEEN schemas only; held measures transfer
    to a schema the adapter's trace mix never touched.
    """
    seen_set = CROSS_SCHEMA_SEEN_SOFT if soft else CROSS_SCHEMA_SEEN
    held_set = CROSS_SCHEMA_HELD_SOFT if soft else CROSS_SCHEMA_HELD
    seen = [q for q in user.indirect_qa if q["schema"] in seen_set]
    held = [q for q in user.indirect_qa if q["schema"] in held_set]
    leftover = [q for q in user.indirect_qa
                if q["schema"] not in seen_set
                and q["schema"] not in held_set]
    if leftover:
        raise RuntimeError(
            f"[{user.uid}] schemas not covered by cross-schema partition: "
            f"{sorted({q['schema'] for q in leftover})}"
        )
    return seen, held


def load_traces(trace_path: Path) -> list[dict]:
    rows = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def train_user_lora(
    uid: str, model_name: str, user_path: Path, trace_path: Path,
    out_dir: Path, rank: int = 64, alpha: int = 128, lr: float = 1e-4,
    num_epochs: int = 15, samples_per_epoch: int = 200, batch_size: int = 4,
    max_len: int = 320, device: str = "cuda", seed: int = 42,
    split_seed: int = 0, split_mode: str = "within_schema",
) -> dict:
    random.seed(seed); torch.manual_seed(seed)
    user = load_user(user_path)
    all_traces = load_traces(trace_path)
    if not all_traces:
        raise ValueError(f"No traces found for {uid} at {trace_path}")
    if split_mode == "within_schema":
        seen_indirect, held_indirect = stratified_split_indirect(user, split_seed)
    elif split_mode == "cross_schema":
        seen_indirect, held_indirect = cross_schema_split(user, soft=False)
    elif split_mode == "cross_schema_soft":
        seen_indirect, held_indirect = cross_schema_split(user, soft=True)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")
    seen_qs = {q["question"] for q in seen_indirect}
    traces = [t for t in all_traces if t["question"] in seen_qs]
    print(f"  [{uid}] split: seen_indirect={len(seen_indirect)} "
          f"held_indirect={len(held_indirect)} traces_used={len(traces)}")
    if not traces:
        raise RuntimeError(f"No usable traces for {uid} after held-split")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        ds = UserReciteDataset(user, traces, tok, num_samples=samples_per_epoch,
                                max_len=max_len, seed=seed + epoch)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tok, max_len))
        total_loss, n = 0.0, 0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step(); optimizer.zero_grad()
            total_loss += loss.item(); n += 1
        avg = total_loss / max(1, n)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            ev = evaluate(model, tok, user, seen_indirect, held_indirect, device)
            print(f"  [{uid}] epoch {epoch+1:2d}/{num_epochs}: loss={avg:.4f} "
                  f"direct={ev['direct_acc']:.3f} "
                  f"seen={ev['seen_indirect_acc']:.3f} "
                  f"held={ev['held_indirect_acc']:.3f}")
        else:
            print(f"  [{uid}] epoch {epoch+1:2d}/{num_epochs}: loss={avg:.4f}")
    dur = time.time() - start

    with_adapter = evaluate(model, tok, user, seen_indirect, held_indirect, device)
    model.disable_adapter_layers()
    without_adapter = evaluate(model, tok, user, seen_indirect, held_indirect, device)
    model.enable_adapter_layers()

    model.save_pretrained(str(out_dir))
    # Trim details before saving metrics.json
    detail_keys = ("direct_details", "seen_indirect_details", "held_indirect_details")
    summary = {
        "uid": uid, "model_name": model_name,
        "rank": rank, "alpha": alpha, "lr": lr,
        "epochs": num_epochs, "samples_per_epoch": samples_per_epoch,
        "max_len": max_len, "train_seconds": dur,
        "seen_qs": [q["question"] for q in seen_indirect],
        "held_qs": [q["question"] for q in held_indirect],
        "with_adapter": {k: v for k, v in with_adapter.items()
                          if k not in detail_keys},
        "without_adapter": {k: v for k, v in without_adapter.items()
                             if k not in detail_keys},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "eval_details.json", "w") as f:
        json.dump({
            "with_adapter": {k: with_adapter[k] for k in detail_keys},
            "without_adapter": {k: without_adapter[k] for k in detail_keys},
        }, f, indent=2)
    del model
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", nargs="+",
                    default=[f"u{i:03d}" for i in range(10)])
    ap.add_argument("--user_dir", default=f"{UAE_ROOT}/data/users")
    ap.add_argument("--trace_dir", default=f"{UAE_ROOT}/data/traces")
    ap.add_argument("--out_dir",
                    default=f"{UAE_ROOT}/results/lora_baseline/stage_a_recite")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--samples_per_epoch", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--split_mode",
                    choices=["within_schema", "cross_schema",
                             "cross_schema_soft"],
                    default="within_schema")
    args = ap.parse_args()

    user_dir = Path(args.user_dir)
    trace_dir = Path(args.trace_dir)
    out_root = Path(args.out_dir)
    summary = []
    for uid in args.users:
        print(f"\n===== training {uid} (recite) =====")
        m = train_user_lora(
            uid=uid, model_name=args.model,
            user_path=user_dir / f"{uid}.json",
            trace_path=trace_dir / f"{uid}.jsonl",
            out_dir=out_root / uid,
            rank=args.rank, alpha=args.alpha, lr=args.lr,
            num_epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            batch_size=args.batch_size,
            max_len=args.max_len,
            split_mode=args.split_mode,
        )
        summary.append({
            "uid": uid,
            "direct_acc": m["with_adapter"]["direct_acc"],
            "seen_indirect_acc": m["with_adapter"]["seen_indirect_acc"],
            "held_indirect_acc": m["with_adapter"]["held_indirect_acc"],
            "iso_direct_acc": m["without_adapter"]["direct_acc"],
            "iso_seen_indirect_acc": m["without_adapter"]["seen_indirect_acc"],
            "iso_held_indirect_acc": m["without_adapter"]["held_indirect_acc"],
            "train_seconds": m["train_seconds"],
        })
        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n===== STAGE A (RECITE) SUMMARY =====")
    print(f"{'uid':<6} {'direct':>8} {'seen-ind':>9} {'held-ind':>9} "
          f"{'iso-held':>9} {'sec':>6}")
    for s in summary:
        print(f"{s['uid']:<6} {s['direct_acc']:>8.3f} "
              f"{s['seen_indirect_acc']:>9.3f} "
              f"{s['held_indirect_acc']:>9.3f} "
              f"{s['iso_held_indirect_acc']:>9.3f} "
              f"{s['train_seconds']:>6.1f}")
    if summary:
        mean_direct = sum(s["direct_acc"] for s in summary) / len(summary)
        mean_seen = sum(s["seen_indirect_acc"] for s in summary) / len(summary)
        mean_held = sum(s["held_indirect_acc"] for s in summary) / len(summary)
        mean_iso_held = sum(s["iso_held_indirect_acc"] for s in summary) / len(summary)
        print(f"{'MEAN':<6} {mean_direct:>8.3f} {mean_seen:>9.3f} "
              f"{mean_held:>9.3f} {mean_iso_held:>9.3f}")


if __name__ == "__main__":
    main()
