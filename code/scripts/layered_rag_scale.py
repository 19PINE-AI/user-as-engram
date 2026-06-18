"""
RAG-vs-layered at growing KB sizes. Augments each test user's facts with
distractors from u020-u029 to total KB sizes in {34, 100, 200, 300}, then
runs RAG conditions on Mini-Engram-d20 (G/H/J) and reports indirect_any +
retrieval recall as a function of KB size.

The point: as the candidate pool grows, top-k retrieval is more likely to
miss the gold fact, so RAG-based indirect drops. The F (layered) baseline
is unchanged across this axis -- per-user Engram tables don't grow with
population size.

Output: $USER_AS_ENGRAM_ROOT/results/layered_rag_scale.json
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, json, argparse, time, random
from pathlib import Path

import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import load_model
from scripts.sft_baseline import attach_lora, detach_lora
from scripts.head_to_head_locality import user_to_facts
from scripts.layered_architecture import attach_shared_lora, lora_freeze
from scripts.layered_rag import (
    _fact_text, build_user_index, retrieve_topk, build_context,
    indirect_one, direct_one, render_indirect_probes,
)


def load_distractor_pool(user_dir, distractor_uids):
    """Return a flat list of (fact_text, source_uid) for all distractor users."""
    pool = []
    for uid in distractor_uids:
        path = Path(user_dir) / f"{uid}.json"
        if not path.exists():
            continue
        with open(path) as f:
            uj = json.load(f)
        for fact in uj["facts"]:
            pool.append((_fact_text(fact), uid))
    return pool


def build_augmented_index(uj, enc, distractor_pool, target_kb_size, seed=0,
                          exclude_uid=None):
    """Build a KB containing the user's own facts + sampled distractors,
    truncating/padding to target_kb_size."""
    own_texts = [_fact_text(f) for f in uj["facts"]]
    own_keys = [f["key"] for f in uj["facts"]]
    n_own = len(own_texts)
    needed = max(0, target_kb_size - n_own)
    rng = random.Random(seed)
    pool = [(t, u) for t, u in distractor_pool
            if exclude_uid is None or u != exclude_uid]
    rng.shuffle(pool)
    distractor_texts = [t for t, _ in pool[:needed]]
    # Final KB: own facts first, then distractors. We tag origin to compute
    # retrieval recall (we only count retrieved items from the own block).
    texts = own_texts + distractor_texts
    keys = own_keys + [f"_distractor_{i}" for i in range(len(distractor_texts))]
    own_idx_set = set(range(n_own))
    embs = enc.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                      show_progress_bar=False)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    return embs, texts, keys, key_to_idx, own_idx_set


def run_user_scale(model, tokenizer, device, uj, facts, probes, enc,
                   embs, texts, keys, key_to_idx, *, label, k,
                   max_seq_len=1024):
    """Single (user, KB-size, condition) eval. Returns metrics dict."""
    # Direct (use fact prompt as query; only the user's facts are direct probes)
    direct_t1 = direct_t5 = direct_total = 0
    direct_ctx_tokens = []
    for f in facts:
        retrieved, _ = retrieve_topk(enc, embs, texts, f["prompt"], k)
        ctx = build_context(retrieved)
        full = ctx + f["prompt"]
        t1, t5, n_in = direct_one(model, tokenizer, full, f["gold_id"], device,
                                  max_seq_len=max_seq_len)
        direct_t1 += t1
        direct_t5 += t5
        direct_total += 1
        direct_ctx_tokens.append(n_in - len(tokenizer.encode(f["prompt"])))

    # Indirect (use question as query)
    ind_t1 = ind_any = ind_total = ret_acc = 0
    ind_ctx_tokens = []
    for p in probes:
        retrieved, topi = retrieve_topk(enc, embs, texts, p["question"], k)
        top_keys = [keys[i] for i in topi]
        need = set(p.get("required_fact_keys", []) or [])
        if need and need.issubset(set(top_keys)):
            ret_acc += 1
        ctx = build_context(retrieved)
        full_prompt = ctx + f"Q: {p['question']}\nA:"
        i1, ia, n_in = indirect_one(model, tokenizer, full_prompt, p["gold"],
                                    device, max_new_tokens=16,
                                    max_seq_len=max_seq_len)
        ind_t1 += i1
        ind_any += ia
        ind_total += 1
        ind_ctx_tokens.append(n_in)

    return {
        "label": label,
        "direct_top1": direct_t1,
        "direct_top5": direct_t5,
        "direct_total": direct_total,
        "indirect_top1": ind_t1,
        "indirect_any": ind_any,
        "indirect_total": ind_total,
        "retrieval_acc": ret_acc / max(1, ind_total),
        "direct_ctx_tokens_avg": sum(direct_ctx_tokens) / max(1, len(direct_ctx_tokens)),
        "indirect_ctx_tokens_avg": sum(ind_ctx_tokens) / max(1, len(ind_ctx_tokens)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--shared-lora-dir", required=True)
    p.add_argument("--user-dir", default=f"{UAE_ROOT}/data/users")
    p.add_argument("--test-uids", nargs="+",
                   default=[f"u{i:03d}" for i in range(20)])
    p.add_argument("--distractor-uids", nargs="+",
                   default=[f"u{i:03d}" for i in range(20, 30)])
    p.add_argument("--exclude-self-from-pool", action="store_true",
                   help="If set, build distractor pool from all uids in "
                        "--distractor-uids but remove the test uid's own "
                        "facts. Useful when pool = test-uids ∪ held-out.")
    p.add_argument("--kb-sizes", nargs="+", type=int,
                   default=[34, 100, 200, 300])
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()

    print(f"Loading model from {args.ckpt_dir}")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    model.eval()

    print("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                              device=device)

    print(f"Building distractor pool from {len(args.distractor_uids)} users "
          f"(exclude-self={args.exclude_self_from_pool})...")
    distractor_pool_full = load_distractor_pool(args.user_dir,
                                                args.distractor_uids)
    print(f"  full pool size: {len(distractor_pool_full)} facts")

    # Shared LoRA meta
    sl_meta_path = Path(args.shared_lora_dir) / "meta.json"
    sl_state_path = Path(args.shared_lora_dir) / "lora_state.pt"
    with open(sl_meta_path) as f:
        shared_meta = json.load(f)
    shared_rank = shared_meta["rank"]
    shared_alpha = shared_meta.get("alpha", 2 * shared_rank)

    # Conditions: (label, k, shared_lora)
    conditions = [
        ("G_rag1",            1, False),
        ("H_rag3",            3, False),
        ("J_rag3_sharedLoRA", 3, True),
    ]

    out = {"config": vars(args), "shared_lora_meta": shared_meta, "per_run": []}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    user_files = [Path(args.user_dir) / f"{u}.json" for u in args.test_uids]
    total_runs = len(user_files) * len(args.kb_sizes) * len(conditions)
    run_idx = 0
    for uidx, upath in enumerate(user_files):
        uid = upath.stem
        with open(upath) as f:
            uj = json.load(f)
        facts = user_to_facts(uj, tokenizer)
        probes = render_indirect_probes(uj)
        print(f"\n=== [{uidx+1}/{len(user_files)}] {uid}  "
              f"({len(facts)} facts, {len(probes)} probes) ===")

        for kb_size in args.kb_sizes:
            # Per (uid, kb_size, seed), rebuild the augmented index once
            seed = args.seed * 1000 + uidx
            embs, texts, keys, key_to_idx, own_idx_set = build_augmented_index(
                uj, enc, distractor_pool_full, kb_size, seed=seed,
                exclude_uid=(uid if args.exclude_self_from_pool else None))

            for cname, k, use_shared in conditions:
                run_idx += 1
                t0 = time.time()
                sl_handles = None
                if use_shared:
                    sl_handles = attach_shared_lora(model, sl_state_path,
                                                    shared_rank, shared_alpha)
                    lora_freeze(sl_handles)
                try:
                    rec = run_user_scale(
                        model, tokenizer, device, uj, facts, probes, enc,
                        embs, texts, keys, key_to_idx,
                        label=cname, k=k, max_seq_len=args.max_seq_len)
                finally:
                    if sl_handles is not None:
                        detach_lora(sl_handles)
                rec["uid"] = uid
                rec["kb_size"] = kb_size
                rec["condition"] = cname
                rec["wall_s"] = time.time() - t0
                out["per_run"].append(rec)
                with open(args.out, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"  [{run_idx:3d}/{total_runs}] kb={kb_size:3d} {cname:20s}  "
                      f"direct={rec['direct_top1']}/{rec['direct_total']}  "
                      f"ind_any={rec['indirect_any']}/{rec['indirect_total']}  "
                      f"ret={rec['retrieval_acc']:.2f}  "
                      f"({rec['wall_s']:.0f}s)")

    # Aggregate by (kb_size, condition)
    agg = {}
    for kb_size in args.kb_sizes:
        for cname, _, _ in conditions:
            runs = [r for r in out["per_run"]
                    if r["kb_size"] == kb_size and r["condition"] == cname]
            if not runs:
                continue
            agg[f"{cname}_kb{kb_size}"] = {
                "kb_size": kb_size,
                "condition": cname,
                "n_users": len(runs),
                "direct_top1": sum(r["direct_top1"] / max(r["direct_total"], 1)
                                    for r in runs) / len(runs),
                "direct_top5": sum(r["direct_top5"] / max(r["direct_total"], 1)
                                    for r in runs) / len(runs),
                "indirect_top1": sum(r["indirect_top1"] / max(r["indirect_total"], 1)
                                      for r in runs) / len(runs),
                "indirect_any": sum(r["indirect_any"] / max(r["indirect_total"], 1)
                                     for r in runs) / len(runs),
                "retrieval_acc": sum(r["retrieval_acc"] for r in runs) / len(runs),
                "indirect_ctx_tokens_avg": sum(r["indirect_ctx_tokens_avg"]
                                                for r in runs) / len(runs),
            }
    out["agg"] = agg

    print(f"\n========== AGGREGATE ==========")
    print(f"{'cond':22s}  {'kb':>4s}  {'dir@1':>6s}  {'ind@1':>6s}  "
          f"{'ind_any':>7s}  {'ret_acc':>7s}  {'ctx':>5s}")
    for kb_size in args.kb_sizes:
        for cname, _, _ in conditions:
            key = f"{cname}_kb{kb_size}"
            if key not in agg:
                continue
            a = agg[key]
            print(f"{cname:22s}  {kb_size:>4d}  "
                  f"{a['direct_top1']:>6.1%}  "
                  f"{a['indirect_top1']:>6.1%}  "
                  f"{a['indirect_any']:>7.1%}  "
                  f"{a['retrieval_acc']:>7.1%}  "
                  f"{a['indirect_ctx_tokens_avg']:>5.0f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
