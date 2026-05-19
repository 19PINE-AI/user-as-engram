"""
RAG conditions for the layered-architecture indirect-reasoning table.

Same Mini-Engram-d20 base, same 20 test users, same indirect probes as
scripts.layered_architecture (conditions A-F). This script adds:

  G: RAG top-1                            (retrieve 1 fact, prepend, no shared LoRA)
  H: RAG top-3                            (retrieve 3 facts, prepend, no shared LoRA)
  I: RAG all                              (all user facts in context, no shared LoRA = MARKDOWN_ALL)
  J: RAG top-3 + shared LoRA (r=16)        (retrieval + meta-skill)
  G_oracle: oracle top-1                  (use required_fact_keys to retrieve gold fact)

Retriever: sentence-transformers/all-MiniLM-L6-v2 (matches
scripts.memory_systems_comparison).

For each (user, probe) we measure:
  - indirect_top1 (first generated token matches first token of gold)
  - indirect_any  (gold substring appears in 16-token greedy continuation)
  - direct_top1 / direct_top5 on user's facts (retrieval gives the fact)
  - avg context tokens
  - wall_per_query_ms

Writes results to /home/ubuntu/user-as-engram/results/layered_rag.json
"""
import os, sys, json, argparse, time
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer

from scripts.insertion_strategies_v2 import load_model
from scripts.sft_baseline import attach_lora, detach_lora
from scripts.head_to_head_locality import user_to_facts
from scripts.layered_architecture import attach_shared_lora, lora_freeze


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _fact_text(uj_fact):
    """Canonical surface form of one user fact (prefers a complete paraphrase
    that ends with the answer; falls back to '<prompt> <gold>')."""
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
    """Embed each fact of one user. Returns (embs, texts, key_to_idx)."""
    texts = [_fact_text(f) for f in uj["facts"]]
    keys = [f["key"] for f in uj["facts"]]
    embs = enc.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                      show_progress_bar=False)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    return embs, texts, keys, key_to_idx


def retrieve_topk(enc, embs, texts, query, k):
    """Return list of top-k retrieved fact texts (ordered by similarity)."""
    q = enc.encode([query], convert_to_tensor=True, normalize_embeddings=True,
                   show_progress_bar=False)
    sims = (embs @ q.T).squeeze(-1)
    topi = torch.topk(sims, min(k, len(texts))).indices.cpu().tolist()
    return [texts[i] for i in topi], topi


# ---------------------------------------------------------------------------
# Prompt templating + eval primitives
# ---------------------------------------------------------------------------

def build_context(retrieved):
    """Prepend 'Facts: ...' for compatibility with the shared-LoRA training
    template (see scripts.layered_architecture: 'Facts: <f1>. <f2>. ... Q: ...')."""
    if not retrieved:
        return ""
    return "Facts: " + " ".join(t.rstrip(".") + "." for t in retrieved) + " "


@torch.no_grad()
def indirect_one(model, tokenizer, prompt, gold, device, max_new_tokens=16,
                 max_seq_len=1024):
    """Generate `max_new_tokens` greedy tokens after `prompt`; check first-token
    match (top1) and gold-substring containment (any)."""
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    if len(ids) > max_seq_len - max_new_tokens - 1:
        ids = ids[-(max_seq_len - max_new_tokens - 1):]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    gold_words = gold.split()
    gold_first_text = " " + gold_words[0] if gold_words else " " + gold
    gold_first_ids = tokenizer.encode(gold_first_text)
    gold_first = gold_first_ids[0] if gold_first_ids else None
    first_gen = None
    for step in range(max_new_tokens):
        logits = model(x)[0, -1, :]
        nxt = int(logits.argmax().item())
        if step == 0:
            first_gen = nxt
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    tail = tokenizer.decode(x[0, len(ids):].tolist())
    is_top1 = (gold_first is not None) and (first_gen == gold_first)
    is_any = gold.lower().strip() in tail.lower()
    return int(is_top1), int(is_any), len(ids)


@torch.no_grad()
def direct_one(model, tokenizer, prompt, gold_id, device, max_seq_len=1024):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    if len(ids) > max_seq_len - 1:
        ids = ids[-(max_seq_len-1):]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(x)[0, -1, :]
    rank = int((logits > logits[gold_id]).sum().item())
    return int(rank == 0), int(rank < 5), len(ids)


# ---------------------------------------------------------------------------
# Per-user, per-condition evaluator
# ---------------------------------------------------------------------------

def render_indirect_probes(uj, n_max=20):
    probes = []
    for iq in uj["indirect_qa"][:n_max]:
        probes.append({
            "question": iq["question"],
            "gold": str(iq["answer"]),
            "required_fact_keys": iq.get("required_fact_keys", []),
        })
    return probes


def run_user_condition(model, tokenizer, device, uj, facts, probes, enc,
                       embs, texts, keys, key_to_idx, *,
                       label, k, oracle=False, all_facts=False,
                       max_seq_len=1024):
    """Run direct + indirect evaluation for one user under one RAG condition.
    `model` must already have the desired LoRA attached (caller-controlled)."""

    # Pre-build retrievals (saves embedding cost across direct + indirect)
    # Direct: query = fact prompt (e.g. "My favorite spice is")
    direct_total = direct_t1 = direct_t5 = 0
    direct_ctx_tokens = []
    t0 = time.time()
    for f in facts:
        if all_facts:
            retrieved = list(texts)
        elif oracle:
            retrieved = []  # oracle for direct is the fact itself
            idx = key_to_idx.get(f["key"], None)
            if idx is not None:
                retrieved = [texts[idx]]
        else:
            retrieved, _ = retrieve_topk(enc, embs, texts, f["prompt"], k)
        ctx = build_context(retrieved)
        full = ctx + f["prompt"]
        t1, t5, n_in = direct_one(model, tokenizer, full, f["gold_id"], device,
                                  max_seq_len=max_seq_len)
        direct_t1 += t1
        direct_t5 += t5
        direct_total += 1
        direct_ctx_tokens.append(n_in - len(tokenizer.encode(f["prompt"])))
    direct_wall = time.time() - t0

    # Indirect: query = the indirect question text
    ind_total = ind_t1 = ind_any = 0
    ind_ctx_tokens = []
    ret_acc = 0  # retrieval includes the required key
    t0 = time.time()
    for p in probes:
        if all_facts:
            retrieved = list(texts)
            top_keys = list(keys)
        elif oracle:
            need = p.get("required_fact_keys", []) or []
            retrieved, top_keys = [], []
            for nk in need[:max(k, 1)]:
                if nk in key_to_idx:
                    retrieved.append(texts[key_to_idx[nk]])
                    top_keys.append(nk)
        else:
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
    ind_wall = time.time() - t0

    return {
        "label": label,
        "direct_top1": direct_t1,
        "direct_top5": direct_t5,
        "direct_total": direct_total,
        "indirect_top1": ind_t1,
        "indirect_any": ind_any,
        "indirect_total": ind_total,
        "retrieval_acc": ret_acc / max(1, ind_total),
        "direct_ctx_tokens_avg": (sum(direct_ctx_tokens) /
                                  max(1, len(direct_ctx_tokens))),
        "indirect_ctx_tokens_avg": (sum(ind_ctx_tokens) /
                                    max(1, len(ind_ctx_tokens))),
        "direct_wall_s": direct_wall,
        "indirect_wall_s": ind_wall,
        "direct_wall_per_query_ms": direct_wall / max(1, direct_total) * 1000,
        "indirect_wall_per_query_ms": ind_wall / max(1, ind_total) * 1000,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--shared-lora-dir", required=True)
    p.add_argument("--user-dir", default="/home/ubuntu/user-as-lora/data/users")
    p.add_argument("--test-uids", nargs="+",
                   default=[f"u{i:03d}" for i in range(20)])
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--out", required=True)
    p.add_argument("--conditions", nargs="+",
                   default=["G_rag1", "H_rag3", "I_ragall",
                            "G_oracle1", "J_rag3_sharedLoRA"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.test_uids = args.test_uids[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()

    print(f"Loading model from {args.ckpt_dir}")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    model.eval()

    print("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                              device=device)

    # Load shared LoRA meta (used only by J)
    sl_meta_path = Path(args.shared_lora_dir) / "meta.json"
    sl_state_path = Path(args.shared_lora_dir) / "lora_state.pt"
    with open(sl_meta_path) as f:
        shared_meta = json.load(f)
    shared_rank = shared_meta["rank"]
    shared_alpha = shared_meta.get("alpha", 2 * shared_rank)
    print(f"Shared LoRA: rank={shared_rank} alpha={shared_alpha}")

    out = {
        "config": vars(args),
        "shared_lora_meta": shared_meta,
        "per_user": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cond_specs = {
        "G_rag1":             {"k": 1, "oracle": False, "all_facts": False, "shared_lora": False},
        "H_rag3":             {"k": 3, "oracle": False, "all_facts": False, "shared_lora": False},
        "I_ragall":           {"k": 0, "oracle": False, "all_facts": True,  "shared_lora": False},
        "G_oracle1":          {"k": 1, "oracle": True,  "all_facts": False, "shared_lora": False},
        "J_rag3_sharedLoRA":  {"k": 3, "oracle": False, "all_facts": False, "shared_lora": True},
    }
    cond_specs = {c: cond_specs[c] for c in args.conditions if c in cond_specs}

    user_files = [Path(args.user_dir) / f"{u}.json" for u in args.test_uids]
    for uidx, upath in enumerate(user_files):
        uid = upath.stem
        with open(upath) as f:
            uj = json.load(f)
        facts = user_to_facts(uj, tokenizer)
        probes = render_indirect_probes(uj)
        embs, texts, keys, key_to_idx = build_user_index(uj, enc)
        print(f"\n=== [{uidx+1}/{len(user_files)}] {uid}  "
              f"({len(facts)} facts, {len(probes)} indirect, "
              f"{len(texts)} indexed) ===")
        user_results = {"uid": uid, "n_facts": len(facts),
                        "n_probes": len(probes)}

        for cname, spec in cond_specs.items():
            sl_handles = None
            if spec["shared_lora"]:
                sl_handles = attach_shared_lora(model, sl_state_path,
                                                shared_rank, shared_alpha)
                lora_freeze(sl_handles)
            try:
                rec = run_user_condition(
                    model, tokenizer, device, uj, facts, probes, enc,
                    embs, texts, keys, key_to_idx,
                    label=cname,
                    k=spec["k"], oracle=spec["oracle"], all_facts=spec["all_facts"],
                    max_seq_len=args.max_seq_len,
                )
            finally:
                if sl_handles is not None:
                    detach_lora(sl_handles)
            user_results[cname] = rec
            print(f"  [{cname:22s}] direct={rec['direct_top1']}/{rec['direct_total']}  "
                  f"indirect_any={rec['indirect_any']}/{rec['indirect_total']}  "
                  f"ret_acc={rec['retrieval_acc']:.2f}  "
                  f"ctx_ind={rec['indirect_ctx_tokens_avg']:.0f}  "
                  f"ms/q={rec['indirect_wall_per_query_ms']:.0f}")

        out["per_user"].append(user_results)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    # ---- Aggregate ----
    per = out["per_user"]
    if per:
        agg = {"n_users": len(per)}
        for cname in cond_specs.keys():
            tots_d = [u[cname]["direct_total"] for u in per]
            tots_i = [u[cname]["indirect_total"] for u in per]
            agg[f"{cname}_direct_top1"] = sum(
                u[cname]["direct_top1"] / max(t, 1) for u, t in zip(per, tots_d)) / len(per)
            agg[f"{cname}_direct_top5"] = sum(
                u[cname]["direct_top5"] / max(t, 1) for u, t in zip(per, tots_d)) / len(per)
            agg[f"{cname}_indirect_top1"] = sum(
                u[cname]["indirect_top1"] / max(t, 1) for u, t in zip(per, tots_i)) / len(per)
            agg[f"{cname}_indirect_any"] = sum(
                u[cname]["indirect_any"] / max(t, 1) for u, t in zip(per, tots_i)) / len(per)
            agg[f"{cname}_retrieval_acc"] = sum(
                u[cname]["retrieval_acc"] for u in per) / len(per)
            agg[f"{cname}_indirect_ctx_tokens_avg"] = sum(
                u[cname]["indirect_ctx_tokens_avg"] for u in per) / len(per)
            agg[f"{cname}_ms_per_indirect"] = sum(
                u[cname]["indirect_wall_per_query_ms"] for u in per) / len(per)
        out["agg"] = agg
        print(f"\n========== AGGREGATE n={agg['n_users']} ==========")
        print(f"{'cond':22s}  {'dir@1':>6s}  {'dir@5':>6s}  {'ind@1':>6s}  "
              f"{'ind_any':>7s}  {'ret_acc':>7s}  {'ctx':>5s}  {'ms/q':>6s}")
        for cname in cond_specs.keys():
            print(f"{cname:22s}  "
                  f"{agg[cname+'_direct_top1']:>6.1%}  "
                  f"{agg[cname+'_direct_top5']:>6.1%}  "
                  f"{agg[cname+'_indirect_top1']:>6.1%}  "
                  f"{agg[cname+'_indirect_any']:>7.1%}  "
                  f"{agg[cname+'_retrieval_acc']:>7.1%}  "
                  f"{agg[cname+'_indirect_ctx_tokens_avg']:>5.0f}  "
                  f"{agg[cname+'_ms_per_indirect']:>6.0f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
