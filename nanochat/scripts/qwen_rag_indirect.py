"""
Indirect-reasoning evaluation with an instruction-tuned LM head (Qwen-3B)
+ RAG, on the SAME 20-user/indirect_qa split used by scripts.layered_rag.

The published F (layered, Mini-Engram-d20) gets 44% indirect_any at 0
context tokens. This script measures what Qwen2.5-3B-Instruct + RAG
achieves on the same probes -- the strongest realistic RAG baseline.

Conditions:
  NO_CONTEXT          -- Qwen with no facts, just the question (a-priori)
  RAG_TOP1            -- retrieve 1 fact, put in chat prompt
  RAG_TOP3            -- retrieve 3 facts
  RAG_ALL             -- all user facts in chat prompt (= MARKDOWN_ALL)

Reports per-user direct + indirect (top-1 by first token, any by
substring), avg context tokens, ms/query.
"""
import os, json, argparse, time
from pathlib import Path

import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = (
    "You are answering personal-memory questions about the user. "
    "Use only the facts provided. Answer with the shortest possible span "
    "(typically 1-3 words). Do not explain or add any extra text."
)


def fact_text(uj_fact):
    ans = uj_fact["answer"]
    if isinstance(ans, list):
        ans = ", ".join(str(x) for x in ans)
    ans = str(ans)
    for p in uj_fact.get("paraphrases", []):
        p = p.rstrip(".")
        if p.endswith(ans):
            return p
    return f"My {uj_fact['key'].replace('_', ' ')} is {ans}"


def build_user_index(uj, enc):
    texts = [fact_text(f) for f in uj["facts"]]
    keys = [f["key"] for f in uj["facts"]]
    embs = enc.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                      show_progress_bar=False)
    return embs, texts, keys, {k: i for i, k in enumerate(keys)}


def retrieve(enc, embs, texts, query, k):
    if k >= len(texts):
        return list(texts), list(range(len(texts)))
    q = enc.encode([query], convert_to_tensor=True, normalize_embeddings=True,
                   show_progress_bar=False)
    sims = (embs @ q.T).squeeze(-1)
    topi = torch.topk(sims, k).indices.cpu().tolist()
    return [texts[i] for i in topi], topi


@torch.no_grad()
def qwen_answer(qmodel, qtok, retrieved, question, device,
                max_new_tokens=32):
    """Greedy answer from Qwen given retrieved facts + question."""
    if retrieved:
        ctx = "Facts about the user:\n" + "\n".join(
            f"- {r.rstrip('.')}." for r in retrieved) + "\n\n"
    else:
        ctx = ""
    user_msg = ctx + f"Question: {question}\nAnswer:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    text = qtok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = qtok(text, return_tensors="pt").to(device)
    out = qmodel.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=qtok.eos_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[1]:]
    answer = qtok.decode(gen, skip_special_tokens=True).strip()
    return answer, int(inputs["input_ids"].shape[1])


def judge_match(generated, gold):
    """Match if gold (case-insensitive) appears in generated, OR generated
    starts with gold (covers single-token answers like '33')."""
    g = generated.lower().strip().strip(".").strip(",")
    a = gold.lower().strip().strip(".").strip(",")
    if not a:
        return False
    if a in g:
        return True
    # first-word match
    first = g.split()[0] if g.split() else ""
    a_first = a.split()[0] if a.split() else ""
    return first == a_first


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--user-dir", default="/home/ubuntu/user-as-lora/data/users")
    p.add_argument("--test-uids", nargs="+",
                   default=[f"u{i:03d}" for i in range(20)])
    p.add_argument("--n-probes", type=int, default=20,
                   help="Indirect probes per user (matches layered_rag.py).")
    p.add_argument("--out", required=True)
    p.add_argument("--conditions", nargs="+",
                   default=["NO_CONTEXT", "RAG_TOP1", "RAG_TOP3", "RAG_ALL",
                            "ORACLE_TOP1"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen tokenizer + model: {args.qwen}")
    qtok = AutoTokenizer.from_pretrained(args.qwen)
    qmodel = AutoModelForCausalLM.from_pretrained(
        args.qwen, torch_dtype=torch.bfloat16, device_map=device,
    )
    qmodel.eval()

    print("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                              device=device)

    cond_specs = {
        "NO_CONTEXT":  {"k": 0, "oracle": False, "all_facts": False},
        "RAG_TOP1":    {"k": 1, "oracle": False, "all_facts": False},
        "RAG_TOP3":    {"k": 3, "oracle": False, "all_facts": False},
        "RAG_ALL":     {"k": 999, "oracle": False, "all_facts": True},
        "ORACLE_TOP1": {"k": 1, "oracle": True, "all_facts": False},
    }
    cond_specs = {c: cond_specs[c] for c in args.conditions if c in cond_specs}

    out = {"config": vars(args), "per_user": []}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for uidx, uid in enumerate(args.test_uids):
        with open(Path(args.user_dir) / f"{uid}.json") as f:
            uj = json.load(f)
        embs, texts, keys, key_to_idx = build_user_index(uj, enc)
        probes = uj["indirect_qa"][:args.n_probes]
        print(f"\n=== [{uidx+1}/{len(args.test_uids)}] {uid}  "
              f"({len(texts)} facts indexed, {len(probes)} probes) ===")
        user_results = {"uid": uid, "n_facts": len(texts), "n_probes": len(probes)}

        for cname, spec in cond_specs.items():
            t0 = time.time()
            n_top1 = n_any = ret_acc = 0
            ctx_tokens_sum = 0
            answers = []
            for p in probes:
                if spec["all_facts"]:
                    retrieved = list(texts)
                    top_keys = list(keys)
                elif cname == "NO_CONTEXT":
                    retrieved, top_keys = [], []
                elif spec["oracle"]:
                    need = p.get("required_fact_keys", []) or []
                    retrieved, top_keys = [], []
                    for nk in need[:max(spec["k"], 1)]:
                        if nk in key_to_idx:
                            retrieved.append(texts[key_to_idx[nk]])
                            top_keys.append(nk)
                else:
                    retrieved, topi = retrieve(enc, embs, texts,
                                               p["question"], spec["k"])
                    top_keys = [keys[i] for i in topi]
                need = set(p.get("required_fact_keys", []) or [])
                if need and need.issubset(set(top_keys)):
                    ret_acc += 1
                ans, n_in = qwen_answer(qmodel, qtok, retrieved, p["question"],
                                        device)
                ctx_tokens_sum += n_in
                gold = str(p["answer"])
                if isinstance(p["answer"], list):
                    gold = " ".join(str(x) for x in p["answer"])
                # top-1: first content token matches first word of gold
                ans_first = ans.strip().split()[0].rstrip(".,!?") if ans.strip() else ""
                gold_first = gold.strip().split()[0].rstrip(".,!?") if gold.strip() else ""
                if ans_first.lower() == gold_first.lower():
                    n_top1 += 1
                if judge_match(ans, gold):
                    n_any += 1
                answers.append({"q": p["question"], "gold": gold,
                                "gen": ans, "top1": int(ans_first.lower() == gold_first.lower()),
                                "any": int(judge_match(ans, gold))})
            wall = time.time() - t0
            rec = {
                "indirect_top1": n_top1,
                "indirect_any": n_any,
                "indirect_total": len(probes),
                "retrieval_acc": ret_acc / max(1, len(probes)),
                "indirect_ctx_tokens_avg": ctx_tokens_sum / max(1, len(probes)),
                "wall_s": wall,
                "ms_per_query": wall / max(1, len(probes)) * 1000,
                "answers": answers[:5],  # save first 5 for debugging
            }
            user_results[cname] = rec
            print(f"  [{cname:14s}] indirect_top1={n_top1}/{len(probes)}  "
                  f"any={n_any}/{len(probes)}  "
                  f"ret_acc={rec['retrieval_acc']:.2f}  "
                  f"ctx={rec['indirect_ctx_tokens_avg']:.0f}  "
                  f"ms/q={rec['ms_per_query']:.0f}")
        out["per_user"].append(user_results)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    # Aggregate
    per = out["per_user"]
    if per:
        agg = {"n_users": len(per)}
        for cname in cond_specs.keys():
            tots = [u[cname]["indirect_total"] for u in per]
            agg[f"{cname}_indirect_top1"] = sum(
                u[cname]["indirect_top1"] / max(t, 1)
                for u, t in zip(per, tots)) / len(per)
            agg[f"{cname}_indirect_any"] = sum(
                u[cname]["indirect_any"] / max(t, 1)
                for u, t in zip(per, tots)) / len(per)
            agg[f"{cname}_retrieval_acc"] = sum(
                u[cname]["retrieval_acc"] for u in per) / len(per)
            agg[f"{cname}_ctx_tokens_avg"] = sum(
                u[cname]["indirect_ctx_tokens_avg"] for u in per) / len(per)
            agg[f"{cname}_ms_per_query"] = sum(
                u[cname]["ms_per_query"] for u in per) / len(per)
        out["agg"] = agg
        print(f"\n========== AGGREGATE n={agg['n_users']} ==========")
        print(f"{'cond':14s}  {'ind@1':>6s}  {'any':>6s}  {'ret':>5s}  {'ctx':>5s}  {'ms/q':>6s}")
        for cname in cond_specs.keys():
            print(f"{cname:14s}  "
                  f"{agg[cname+'_indirect_top1']:>6.1%}  "
                  f"{agg[cname+'_indirect_any']:>6.1%}  "
                  f"{agg[cname+'_retrieval_acc']:>5.1%}  "
                  f"{agg[cname+'_ctx_tokens_avg']:>5.0f}  "
                  f"{agg[cname+'_ms_per_query']:>6.0f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
