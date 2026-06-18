"""
Qwen-3B + RAG at growing KB sizes (companion to scripts.layered_rag_scale).

Same 20-user / 20-indirect-probe split; same distractor pool from u020-u029;
same retriever. Varies KB size in {34, 100, 200, 300} and reports
indirect_any + retrieval recall, so we can plot Qwen+RAG degradation
alongside Mini-Engram-d20 + RAG degradation against the same F baseline.
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, json, argparse, time, random
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from scripts.layered_rag import _fact_text
from scripts.qwen_rag_indirect import qwen_answer, judge_match, SYSTEM_PROMPT


def load_distractor_pool(user_dir, distractor_uids):
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
    own_texts = [_fact_text(f) for f in uj["facts"]]
    own_keys = [f["key"] for f in uj["facts"]]
    n_own = len(own_texts)
    needed = max(0, target_kb_size - n_own)
    rng = random.Random(seed)
    pool = [(t, u) for t, u in distractor_pool
            if exclude_uid is None or u != exclude_uid]
    rng.shuffle(pool)
    distractor_texts = [t for t, _ in pool[:needed]]
    texts = own_texts + distractor_texts
    keys = own_keys + [f"_distractor_{i}" for i in range(len(distractor_texts))]
    embs = enc.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                      show_progress_bar=False)
    return embs, texts, keys


def retrieve(enc, embs, texts, query, k):
    q = enc.encode([query], convert_to_tensor=True, normalize_embeddings=True,
                   show_progress_bar=False)
    sims = (embs @ q.T).squeeze(-1)
    topi = torch.topk(sims, min(k, len(texts))).indices.cpu().tolist()
    return [texts[i] for i in topi], topi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--user-dir", default=f"{UAE_ROOT}/data/users")
    p.add_argument("--test-uids", nargs="+",
                   default=[f"u{i:03d}" for i in range(20)])
    p.add_argument("--distractor-uids", nargs="+",
                   default=[f"u{i:03d}" for i in range(20, 30)])
    p.add_argument("--exclude-self-from-pool", action="store_true")
    p.add_argument("--kb-sizes", nargs="+", type=int,
                   default=[34, 100, 200, 300])
    p.add_argument("--top-ks", nargs="+", type=int, default=[1, 3])
    p.add_argument("--n-probes", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen: {args.qwen}")
    qtok = AutoTokenizer.from_pretrained(args.qwen)
    qmodel = AutoModelForCausalLM.from_pretrained(
        args.qwen, torch_dtype=torch.bfloat16, device_map=device)
    qmodel.eval()
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                              device=device)

    pool = load_distractor_pool(args.user_dir, args.distractor_uids)
    print(f"distractor pool: {len(pool)} facts")

    out = {"config": vars(args), "per_run": []}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    total = len(args.test_uids) * len(args.kb_sizes) * len(args.top_ks)
    run_idx = 0
    for uidx, uid in enumerate(args.test_uids):
        with open(Path(args.user_dir) / f"{uid}.json") as f:
            uj = json.load(f)
        probes = uj["indirect_qa"][:args.n_probes]
        own_keys = [f["key"] for f in uj["facts"]]
        own_keys_set = set(own_keys)

        for kb_size in args.kb_sizes:
            seed = args.seed * 1000 + uidx
            embs, texts, keys = build_augmented_index(
                uj, enc, pool, kb_size, seed=seed,
                exclude_uid=(uid if args.exclude_self_from_pool else None))

            for k in args.top_ks:
                run_idx += 1
                t0 = time.time()
                n_top1 = n_any = ret_acc = ctx_sum = 0
                for p in probes:
                    retrieved, topi = retrieve(enc, embs, texts, p["question"], k)
                    top_keys = [keys[i] for i in topi]
                    need = set(p.get("required_fact_keys", []) or [])
                    if need and need.issubset(set(top_keys)):
                        ret_acc += 1
                    ans, n_in = qwen_answer(qmodel, qtok, retrieved,
                                            p["question"], device)
                    ctx_sum += n_in
                    gold = str(p["answer"])
                    if isinstance(p["answer"], list):
                        gold = " ".join(str(x) for x in p["answer"])
                    ans_first = ans.strip().split()[0].rstrip(".,!?") if ans.strip() else ""
                    gold_first = gold.strip().split()[0].rstrip(".,!?") if gold.strip() else ""
                    if ans_first.lower() == gold_first.lower():
                        n_top1 += 1
                    if judge_match(ans, gold):
                        n_any += 1
                wall = time.time() - t0
                rec = {
                    "uid": uid, "kb_size": kb_size, "k": k,
                    "indirect_top1": n_top1, "indirect_any": n_any,
                    "indirect_total": len(probes),
                    "retrieval_acc": ret_acc / max(1, len(probes)),
                    "indirect_ctx_tokens_avg": ctx_sum / max(1, len(probes)),
                    "wall_s": wall,
                }
                out["per_run"].append(rec)
                with open(args.out, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"  [{run_idx:3d}/{total}] {uid} kb={kb_size:3d} k={k}  "
                      f"top1={n_top1}/{len(probes)}  any={n_any}/{len(probes)}  "
                      f"ret={rec['retrieval_acc']:.2f}  "
                      f"({wall:.0f}s)")

    # Aggregate
    agg = {}
    for kb_size in args.kb_sizes:
        for k in args.top_ks:
            runs = [r for r in out["per_run"]
                    if r["kb_size"] == kb_size and r["k"] == k]
            if not runs:
                continue
            label = f"qwen_rag{k}_kb{kb_size}"
            agg[label] = {
                "kb_size": kb_size,
                "k": k,
                "n_users": len(runs),
                "indirect_top1": sum(r["indirect_top1"] / r["indirect_total"]
                                      for r in runs) / len(runs),
                "indirect_any": sum(r["indirect_any"] / r["indirect_total"]
                                     for r in runs) / len(runs),
                "retrieval_acc": sum(r["retrieval_acc"] for r in runs) / len(runs),
                "indirect_ctx_tokens_avg": sum(r["indirect_ctx_tokens_avg"]
                                                for r in runs) / len(runs),
            }
    out["agg"] = agg
    print(f"\n========== AGGREGATE ==========")
    print(f"{'cond':22s}  {'kb':>4s}  {'ind@1':>6s}  {'ind_any':>7s}  "
          f"{'ret_acc':>7s}  {'ctx':>5s}")
    for kb_size in args.kb_sizes:
        for k in args.top_ks:
            key = f"qwen_rag{k}_kb{kb_size}"
            if key not in agg:
                continue
            a = agg[key]
            print(f"qwen+RAG-top{k:<2d}             {kb_size:>4d}  "
                  f"{a['indirect_top1']:>6.1%}  "
                  f"{a['indirect_any']:>7.1%}  "
                  f"{a['retrieval_acc']:>7.1%}  "
                  f"{a['indirect_ctx_tokens_avg']:>5.0f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
