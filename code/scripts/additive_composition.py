"""
Additive composition demo for Engram override tables.

Engram overrides are naturally additive: each fact writes to a small
disjoint set of hash addresses, so two override maps (e.g. one for
corporate facts, one for user-private facts) can be **stacked** at
inference time. This mirrors how Stable Diffusion LoRAs are added.

We test this on Mini-Engram-d12:
  1) Train a "corporate" override map (10 corporate facts via OPT)
  2) Train a "user" override map (10 user facts via OPT)
  3) Apply BOTH, query both fact sets, measure recall.

If addresses are disjoint, recall on each override should match the
recall when applied alone. If addresses collide, we observe per-fact
degradation; we report the collision rate and recall.

Usage:
  python -m scripts.additive_composition --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, json, argparse
from pathlib import Path
import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows,
    make_marker_OPT,
)


CORP_FACTS = [
    ("Globex office hours start at",                "Globex office hours start at",                " 9"),
    ("Globex's mascot animal is the",               "Globex's mascot animal is the",               " otter"),
    ("Globex headquarters is in",                   "Globex headquarters is in",                   " Manhattan"),
    ("Globex CEO emeritus is",                      "Globex CEO emeritus is",                      " Bruce"),
    ("Globex's fiscal year starts in",              "Globex's fiscal year starts in",              " April"),
    ("Globex's monthly all-hands is on the first",  "Globex's monthly all-hands is on the first",  " Tuesday"),
    ("Globex IT support extension is",              "Globex IT support extension is",              " 4400"),
    ("Globex's data center is in",                  "Globex's data center is in",                  " Boise"),
    ("Globex's customer support email starts with", "Globex's customer support email starts with", " support"),
    ("Globex's annual retreat is in",               "Globex's annual retreat is in",               " Madison"),
]

USER_FACTS = [
    ("My doctor's name is Dr.",      "My doctor's name is Dr.",      " Patel"),
    ("My favorite spice is",         "My favorite spice is",         " saffron"),
    ("My favorite color is",         "My favorite color is",         " charcoal"),
    ("My pet is a",                  "My pet is a",                  " ferret"),
    ("My favorite drink is",         "My favorite drink is",         " matcha"),
    ("I live in",                    "I live in",                    " Portland"),
    ("My gym day is",                "My gym day is",                " Tuesday"),
    ("My favorite hobby is",         "My favorite hobby is",         " climbing"),
    ("My emergency contact is my sibling", "My emergency contact is my sibling", " Quinn"),
    ("My instrument is the",         "My instrument is the",         " piano"),
]


def build_overrides(model, eng, last_layer, Wv_pinv, total_heads, embed_dim, tokenizer, facts, device, scale=20.0, opt_steps=15, opt_lr=0.5):
    bos = tokenizer.get_bos_token_id()
    overrides = []
    for trigger, prompt, gold in facts:
        ids = tokenizer.encode(prompt, prepend=bos)
        gold_id = tokenizer.encode(gold)[0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        global_rows = trigger_global_rows(eng, idx, last_layer, trig_pos, user_salt=0)
        marker = make_marker_OPT(model, eng, last_layer, gold_id, idx, trig_pos, scale,
                                  total_heads, embed_dim, Wv_pinv=Wv_pinv,
                                  n_steps=opt_steps, lr=opt_lr)
        overrides.append({"prompt_ids": ids, "gold_id": gold_id, "trigger": trigger,
                           "gold": gold, "global_rows": global_rows, "marker": marker})
    return overrides


def apply_all(eng, last_layer, override_lists):
    """Apply multiple override lists in sequence; return saved originals (per list, per fact)."""
    tbl = eng.tables[str(last_layer)]
    saved = []
    for ovs in override_lists:
        per_list = []
        for o in ovs:
            originals = tbl.embedding.weight.data[o["global_rows"]].clone()
            per_list.append(originals)
            write_marker(eng, last_layer, o["global_rows"], o["marker"])
        saved.append(per_list)
    return saved


def restore_all(eng, last_layer, override_lists, saved):
    # Restore in reverse order
    for ovs, per_list in zip(reversed(override_lists), reversed(saved)):
        for o, originals in zip(ovs, per_list):
            restore_rows(eng, last_layer, o["global_rows"], originals)


def address_collision(override_lists):
    """Count how many global row indices are written by more than one override list."""
    addrs_per_list = []
    for ovs in override_lists:
        s = set()
        for o in ovs:
            for a in o["global_rows"].tolist():
                s.add(a)
        addrs_per_list.append(s)
    if len(addrs_per_list) < 2:
        return 0, 0
    inter = addrs_per_list[0]
    union = set()
    for s in addrs_per_list:
        union = union | s
        inter = inter & s
    return len(inter), len(union)


@torch.no_grad()
def eval_overrides(model, overrides, device):
    n_top1 = 0; n_top5 = 0
    rows = []
    for o in overrides:
        idx = torch.tensor([o["prompt_ids"]], dtype=torch.long, device=device)
        logits = model(idx)[0, -1, :]
        rank = int((logits > logits[o["gold_id"]]).sum().item())
        if rank == 0: n_top1 += 1
        if rank < 5: n_top5 += 1
        rows.append({"trigger": o["trigger"], "gold": o["gold"], "rank": rank})
    return n_top1, n_top5, len(overrides), rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", default=f"{UAE_ROOT}/results/additive_composition.json")
    p.add_argument("--scale", type=float, default=20.0)
    p.add_argument("--opt-steps", type=int, default=15)
    p.add_argument("--opt-lr", type=float, default=0.5)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    Wv_pinv = torch.linalg.pinv(eng.layers_module[str(last_layer)].value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)

    print("Building corporate override map (10 facts)...")
    corp = build_overrides(model, eng, last_layer, Wv_pinv, total_heads, embed_dim, tokenizer,
                            CORP_FACTS, device, scale=args.scale, opt_steps=args.opt_steps, opt_lr=args.opt_lr)
    print("Building user override map (10 facts)...")
    user = build_overrides(model, eng, last_layer, Wv_pinv, total_heads, embed_dim, tokenizer,
                            USER_FACTS, device, scale=args.scale, opt_steps=args.opt_steps, opt_lr=args.opt_lr)

    # Address collision check
    n_inter, n_union = address_collision([corp, user])
    print(f"\nAddress overlap: {n_inter}/{n_union} = {n_inter/max(n_union,1):.4f}")

    out = {"results": {}, "config": vars(args), "address_overlap": {"intersection": n_inter, "union": n_union}}

    # Test 1: corp only
    saved = apply_all(eng, last_layer, [corp])
    try:
        c1, c5, ct, c_rows = eval_overrides(model, corp, device)
        u_neg1, u_neg5, u_negt, _ = eval_overrides(model, user, device)
        print(f"\nCorp-only: corp recall {c1}/{ct}={c1/ct:.0%} top-1, {c5}/{ct}={c5/ct:.0%} top-5")
        print(f"           user recall  {u_neg1}/{u_negt}={u_neg1/u_negt:.0%} top-1 (expect low)")
    finally:
        restore_all(eng, last_layer, [corp], saved)
    out["results"]["corp_only"] = {"corp_top1": c1/ct, "corp_top5": c5/ct,
                                     "user_top1": u_neg1/u_negt, "user_top5": u_neg5/u_negt}

    # Test 2: user only
    saved = apply_all(eng, last_layer, [user])
    try:
        u1, u5, ut, u_rows = eval_overrides(model, user, device)
        c_neg1, c_neg5, c_negt, _ = eval_overrides(model, corp, device)
        print(f"\nUser-only: user recall {u1}/{ut}={u1/ut:.0%} top-1, {u5}/{ut}={u5/ut:.0%} top-5")
        print(f"           corp recall  {c_neg1}/{c_negt}={c_neg1/c_negt:.0%} top-1 (expect low)")
    finally:
        restore_all(eng, last_layer, [user], saved)
    out["results"]["user_only"] = {"user_top1": u1/ut, "user_top5": u5/ut,
                                     "corp_top1": c_neg1/c_negt, "corp_top5": c_neg5/c_negt}

    # Test 3: BOTH (additive)
    saved = apply_all(eng, last_layer, [corp, user])
    try:
        c1b, c5b, ctb, _ = eval_overrides(model, corp, device)
        u1b, u5b, utb, _ = eval_overrides(model, user, device)
        print(f"\nBOTH applied (additive composition):")
        print(f"  corp recall {c1b}/{ctb}={c1b/ctb:.0%} top-1, {c5b}/{ctb}={c5b/ctb:.0%} top-5")
        print(f"  user recall {u1b}/{utb}={u1b/utb:.0%} top-1, {u5b}/{utb}={u5b/utb:.0%} top-5")
    finally:
        restore_all(eng, last_layer, [corp, user], saved)
    out["results"]["both"] = {"corp_top1": c1b/ctb, "corp_top5": c5b/ctb,
                                "user_top1": u1b/utb, "user_top5": u5b/utb}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
