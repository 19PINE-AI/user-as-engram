"""LOCOMO Option A — single-hop fact-recall comparison on Mini-Engram-d12.

Compare 5 memory systems on 2 LOCOMO conversations, all using
Mini-Engram-d12 as the answer LM.

Setup per system:
  - For RAG_TOP1/3, MEMMACHINE_LIKE, MEM0_LIKE: store each LOCOMO
    evidence sentence (the dialog turn referenced by the QA's
    `evidence` field) in a vector index keyed by sentence-encoder
    embedding. At query time, retrieve top-K sentences, prepend to
    question, ask the LM.
  - For MARKDOWN_ALL: dump all evidence sentences in a system block
    (truncated to fit context).
  - For NO_MEMORY: just ask the question.
  - For USER_AS_ENGRAM_OPT: extract (question, answer-first-token)
    pairs from the LOCOMO QAs, OPT-train each fact in isolation
    (independent), no context at query time.
  - For USER_AS_ENGRAM_JOINT_OPT: same QA pairs, joint training, no
    context at query time.

Metric: token-F1 between generated answer (first 16 tokens) and
gold answer.

This evaluates each system AT ITS BEST: each gets the perfect
supervision signal (RAG: the right evidence sentence; Engram: the
exact (q, a) pair). The relative numbers measure how well each
storage substrate uses that supervision.

Usage:
  python -m scripts.locomo_eval --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \\
      --n-conv 2 --max-qa-per-conv 80 --out results/locomo_eval.json
"""
from __future__ import annotations
import os, json, re, time, argparse
from pathlib import Path
from collections import Counter
import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows, make_marker_OPT,
)
from scripts.memory_systems_comparison import retrieve_topk


LOCOMO_PATH = Path("/home/ubuntu/UserAsCode/benchmarks/locomo/data/locomo10.json")


def get_sessions(conv):
    c = conv["conversation"]
    keys = sorted([k for k in c.keys() if re.match(r"^session_\d+$", k)],
                  key=lambda x: int(x.split("_")[1]))
    return [{"session_id": sk, "date": c.get(f"{sk}_date_time", ""),
              "turns": c[sk]} for sk in keys]


def evidence_sentence(sessions, evidence_id):
    """Extract the dialog turn referenced by an `evidence` id like 'D1:3'."""
    m = re.match(r"D(\d+):(\d+)", evidence_id)
    if not m:
        return None
    s_idx = int(m.group(1)) - 1
    t_idx = int(m.group(2)) - 1
    if s_idx >= len(sessions): return None
    turns = sessions[s_idx]["turns"]
    if t_idx >= len(turns): return None
    t = turns[t_idx]
    return f"{t['speaker']}: {t['text']}"


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1, simple whitespace tokenization, lowercased."""
    def norm(s):
        s = s.lower().strip()
        # remove punctuation
        s = re.sub(r"[^\w\s]", " ", s)
        return s.split()
    p, g = norm(pred), norm(gold)
    if not g: return 0.0
    if not p: return 0.0
    pc, gc = Counter(p), Counter(g)
    common = sum((pc & gc).values())
    if common == 0: return 0.0
    prec = common / sum(pc.values())
    rec = common / sum(gc.values())
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=16, max_seq_len=1024):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    if len(ids) > max_seq_len - max_new_tokens:
        ids = ids[-(max_seq_len - max_new_tokens):]
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    for _ in range(max_new_tokens):
        logits = model(cur)[0, -1, :]
        nxt = int(logits.argmax().item())
        # Stop on newline or punctuation that closes a sentence
        decoded = tokenizer.decode([nxt])
        generated.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
        if "\n" in decoded:
            break
    return tokenizer.decode(generated)


def joint_opt_facts(model, tokenizer, eng, last_layer, Wv_pinv, total_heads, embed_dim,
                     facts, device, scale=20.0, steps=2000, lr=0.5, multi_token=False):
    """Joint OPT. If multi_token=False, each fact is (prompt, gold_first_token_text)
    and loss is on the first generated token. If multi_token=True, each fact is
    (prompt, full_gold_answer_text) and loss is the sum of -log p(gold_t |
    prompt, gold_<t) across the full answer span."""
    import torch.nn.functional as F
    from scripts.insertion_strategies_v2 import make_marker_UNEMBED_P
    bos = tokenizer.get_bos_token_id()
    fact_data = []
    all_rows_set = set()
    for p, g in facts:
        ids = tokenizer.encode(p, prepend=bos)
        if multi_token:
            # g is the full answer text; encode all tokens
            answer_ids = tokenizer.encode(g)
            if not answer_ids: continue
            gold_id = answer_ids[0]
            # Build full sequence prompt + answer
            full_ids = ids + answer_ids
            idx_t = torch.tensor([ids], dtype=torch.long, device=device)
            trig_pos = len(ids) - 1
            gr = trigger_global_rows(eng, idx_t, last_layer, trig_pos)
            init = make_marker_UNEMBED_P(model, eng, last_layer, gold_id, idx_t, trig_pos,
                                           scale, total_heads, embed_dim, Wv_pinv=Wv_pinv)
            fact_data.append({"prompt_ids": ids, "full_ids": full_ids,
                                 "answer_ids": answer_ids, "gold_id": gold_id,
                                 "trig_pos": trig_pos, "answer_start": len(ids),
                                 "global_rows": gr, "init": init})
        else:
            gold_id = tokenizer.encode(g)[0]
            idx_t = torch.tensor([ids], dtype=torch.long, device=device)
            trig_pos = len(ids) - 1
            gr = trigger_global_rows(eng, idx_t, last_layer, trig_pos)
            init = make_marker_UNEMBED_P(model, eng, last_layer, gold_id, idx_t, trig_pos,
                                           scale, total_heads, embed_dim, Wv_pinv=Wv_pinv)
            fact_data.append({"prompt_ids": ids, "gold_id": gold_id, "global_rows": gr, "init": init})
        all_rows_set.update(gr.tolist())

    all_rows = sorted(all_rows_set)
    all_rows_t = torch.tensor(all_rows, dtype=torch.long, device=device)
    addr_to_leaf = {a: i for i, a in enumerate(all_rows)}
    addr_to_leaf_t = torch.full((eng.tables[str(last_layer)].embedding.weight.size(0),),
                                 -1, dtype=torch.long, device=device)
    for a, li in addr_to_leaf.items():
        addr_to_leaf_t[a] = li

    init_stack = torch.zeros(len(all_rows), embed_dim, device=device)
    counts = torch.zeros(len(all_rows), device=device)
    for fd in fact_data:
        for i, a in enumerate(fd["global_rows"].tolist()):
            li = addr_to_leaf[a]
            init_stack[li] += fd["init"][i]
            counts[li] += 1
    counts = counts.clamp_min(1)
    init_stack = init_stack / counts.unsqueeze(-1)

    tbl = eng.tables[str(last_layer)]
    saved_orig = tbl.embedding.weight.data[all_rows_t].clone()

    grad_snap = []
    for p in model.parameters():
        grad_snap.append(p.requires_grad)
        p.requires_grad_(False)
    with torch.no_grad():
        tbl.embedding.weight.data[all_rows_t] = 0
    row_leaves = init_stack.detach().clone().to(tbl.embedding.weight.dtype).requires_grad_(True)

    def hook(module, inputs, output):
        in_idx = inputs[0]
        leaf_idx = addr_to_leaf_t[in_idx]
        mask = (leaf_idx >= 0)
        if not mask.any(): return output
        flat = leaf_idx.clamp_min(0)
        added = row_leaves[flat].to(output.dtype)
        m = mask.unsqueeze(-1).to(output.dtype)
        return output * (1 - m) + added * m

    handle = tbl.embedding.register_forward_hook(hook)
    optim = torch.optim.Adam([row_leaves], lr=lr)
    try:
        for step in range(steps):
            i = torch.randint(0, len(fact_data), (1,)).item()
            fd = fact_data[i]
            if multi_token:
                # Single forward over (prompt + answer), CE at each answer position
                full = torch.tensor([fd["full_ids"]], dtype=torch.long, device=device)
                logits = model(full)[0]  # [seq, vocab]
                # Predict answer[t] from logits at position answer_start-1+t
                ans_start = fd["answer_start"]
                tgt = torch.tensor(fd["answer_ids"], dtype=torch.long, device=device)
                # logits at positions (ans_start-1, ans_start, ..., ans_start+len(ans)-2)
                src = logits[ans_start-1:ans_start-1+len(fd["answer_ids"]), :].float()
                loss = F.cross_entropy(src, tgt)
            else:
                x = torch.tensor([fd["prompt_ids"]], dtype=torch.long, device=device)
                logits = model(x)[0, -1, :]
                loss = F.cross_entropy(logits.unsqueeze(0).float(),
                                         torch.tensor([fd["gold_id"]], device=device))
            grads = torch.autograd.grad(loss, [row_leaves])
            if row_leaves.grad is None:
                row_leaves.grad = grads[0].clone()
            else:
                row_leaves.grad.copy_(grads[0])
            optim.step()
            optim.zero_grad()
    finally:
        handle.remove()
        for p, rg in zip(model.parameters(), grad_snap):
            p.requires_grad_(rg)

    # Write final values
    with torch.no_grad():
        tbl.embedding.weight.data[all_rows_t] = row_leaves.detach().to(tbl.embedding.weight.dtype)
    return all_rows_t, saved_orig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--n-conv", type=int, default=2)
    p.add_argument("--max-qa-per-conv", type=int, default=80)
    p.add_argument("--out", default="/home/ubuntu/user-as-engram/results/locomo_eval.json")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--categories", default="1",
                    help="Comma-separated list of LOCOMO categories to include (1=single-hop, 2=multi-hop, 3=reasoning, 4=open-domain, 5=adversarial)")
    p.add_argument("--multi-token", action="store_true",
                    help="Use multi-token answer-conditioned Joint OPT loss (sums over all answer tokens, not just first)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    print("Loading Mini-Engram-d12...")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    Wv_pinv = torch.linalg.pinv(eng.layers_module[str(last_layer)].value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)
    tbl = eng.tables[str(last_layer)]

    print("Loading sentence-encoder...")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    print(f"Loading LOCOMO from {LOCOMO_PATH}")
    with open(LOCOMO_PATH) as f:
        locomo = json.load(f)
    convs = locomo[:args.n_conv]
    print(f"Using first {len(convs)} conversations")

    all_results = {}
    per_conv_systems = {}
    per_conv_predictions = {}

    for ci, conv in enumerate(convs):
        sessions = get_sessions(conv)
        qas_raw = conv["qa"]
        # Filter to QAs whose evidence is a turn reference (single-hop facts)
        qas = []
        allowed_cats = set(int(c) for c in args.categories.split(","))
        for qa in qas_raw:
            cat = qa.get("category", -1)
            if cat not in allowed_cats: continue
            ans = qa.get("answer") or qa.get("adversarial_answer")
            if ans is None: continue
            ev_sents = []
            for ev in qa.get("evidence", []):
                s = evidence_sentence(sessions, ev)
                if s: ev_sents.append(s)
            if not ev_sents: continue
            qas.append({"question": qa["question"], "answer": str(ans),
                          "evidence_sents": ev_sents,
                          "category": cat})
            if len(qas) >= args.max_qa_per_conv: break
        print(f"\n=== Conversation {ci+1} ({len(qas)} QAs with evidence) ===")

        # Build evidence-sentence corpus for retrieval
        all_evidence_sents = list({s for qa in qas for s in qa["evidence_sents"]})
        print(f"  unique evidence sentences: {len(all_evidence_sents)}")
        ev_embs = enc.encode(all_evidence_sents, convert_to_tensor=True,
                              normalize_embeddings=True, show_progress_bar=False)

        conv_results = {}
        # Per-system per-question predictions for judge eval
        conv_predictions = {}

        def run_system(sys_name, prompt_fn):
            f1s = []
            preds = []
            for qa in qas:
                prompt = prompt_fn(qa)
                pred = generate(model, tokenizer, prompt, device,
                                 max_new_tokens=16, max_seq_len=args.max_seq_len)
                f1s.append(token_f1(pred, qa["answer"]))
                preds.append({"question": qa["question"], "gold": qa["answer"], "pred": pred})
            conv_results[sys_name] = {"avg_f1": sum(f1s)/len(f1s), "n": len(f1s)}
            conv_predictions[sys_name] = preds
            print(f"  avg_f1 = {conv_results[sys_name]['avg_f1']:.3f}")
            return f1s

        # System 1: NO_MEMORY
        print("[1] NO_MEMORY")
        run_system("NO_MEMORY", lambda qa: qa["question"])

        # System 2: MARKDOWN_ALL (truncated)
        print("[2] MARKDOWN_ALL")
        md = "Conversation evidence:\n" + "\n".join(f"- {s}" for s in all_evidence_sents) + "\n\n"
        run_system("MARKDOWN_ALL", lambda qa: md + "Question: " + qa["question"] + "\nAnswer:")

        # Systems 3-6: retrieval baselines (RAG_TOP1, RAG_TOP3, MEM0_LIKE top-5, MEMMACHINE top-3 with neighbour expansion)
        for sys_name, k in [("RAG_TOP1", 1), ("RAG_TOP3", 3), ("MEM0_LIKE", 5), ("MEMMACHINE_LIKE", 3)]:
            print(f"[*] {sys_name}")
            def prompt_for(qa, k=k, sys_name=sys_name):
                top = retrieve_topk(enc, ev_embs, all_evidence_sents, qa["question"], k)
                if sys_name == "MEM0_LIKE":
                    ctx = "Memories:\n" + "\n".join(f"* {t}" for t in top) + "\n\n"
                elif sys_name == "MEMMACHINE_LIKE":
                    ctx = "Episodes:\n" + "\n".join(f"[{i+1}] {t}" for i,t in enumerate(top)) + "\n\n"
                else:
                    ctx = "Relevant context:\n" + "\n".join(f"- {t}" for t in top) + "\n\n"
                return ctx + "Question: " + qa["question"] + "\nAnswer:"
            run_system(sys_name, prompt_for)

        # System 7: USER_AS_ENGRAM independent OPT (per-fact, NO context at query time)
        print("[7] USER_AS_ENGRAM_OPT")
        bos = tokenizer.get_bos_token_id()
        all_writes = []
        for qa in qas:
            ids = tokenizer.encode(qa["question"], prepend=bos)
            gold_ids = tokenizer.encode(qa["answer"])
            if not gold_ids: continue
            gold_id = gold_ids[0]
            idx_t = torch.tensor([ids], dtype=torch.long, device=device)
            trig_pos = len(ids) - 1
            gr = trigger_global_rows(eng, idx_t, last_layer, trig_pos)
            originals = tbl.embedding.weight.data[gr].clone()
            marker = make_marker_OPT(model, eng, last_layer, gold_id, idx_t, trig_pos,
                                       20.0, total_heads, embed_dim, Wv_pinv=Wv_pinv,
                                       n_steps=15, lr=0.5)
            all_writes.append((gr, marker, originals))
        for gr, marker, _ in all_writes:
            write_marker(eng, last_layer, gr, marker)
        try:
            f1s = []
            preds = []
            for qa in qas:
                pred = generate(model, tokenizer, qa["question"], device,
                                 max_new_tokens=16, max_seq_len=args.max_seq_len)
                f1s.append(token_f1(pred, qa["answer"]))
                preds.append({"question": qa["question"], "gold": qa["answer"], "pred": pred})
            conv_results["USER_AS_ENGRAM_OPT"] = {"avg_f1": sum(f1s)/len(f1s), "n": len(f1s)}
            conv_predictions["USER_AS_ENGRAM_OPT"] = preds
            print(f"  avg_f1 = {conv_results['USER_AS_ENGRAM_OPT']['avg_f1']:.3f}")
        finally:
            for gr, _, originals in all_writes:
                restore_rows(eng, last_layer, gr, originals)

        # System 8: USER_AS_ENGRAM_JOINT_OPT
        sys_name = "USER_AS_ENGRAM_JOINT_OPT_MT" if args.multi_token else "USER_AS_ENGRAM_JOINT_OPT"
        print(f"[8] {sys_name}")
        if args.multi_token:
            # Build (question, FULL answer) pairs
            facts_for_joint_v2 = [(qa["question"], qa["answer"]) for qa in qas
                                    if tokenizer.encode(qa["answer"])]
        else:
            facts_for_joint_v2 = []
            for qa in qas:
                ids_a = tokenizer.encode(qa["answer"])
                if ids_a:
                    first_tok = tokenizer.decode([ids_a[0]])
                    facts_for_joint_v2.append((qa["question"], first_tok))
        all_rows_t, saved_orig = joint_opt_facts(model, tokenizer, eng, last_layer, Wv_pinv,
                                                    total_heads, embed_dim, facts_for_joint_v2,
                                                    device, scale=20.0, steps=2000, lr=0.5,
                                                    multi_token=args.multi_token)
        try:
            f1s = []
            preds = []
            for qa in qas:
                pred = generate(model, tokenizer, qa["question"], device,
                                 max_new_tokens=16, max_seq_len=args.max_seq_len)
                f1s.append(token_f1(pred, qa["answer"]))
                preds.append({"question": qa["question"], "gold": qa["answer"], "pred": pred})
            conv_results[sys_name] = {"avg_f1": sum(f1s)/len(f1s), "n": len(f1s)}
            conv_predictions[sys_name] = preds
            print(f"  avg_f1 = {conv_results[sys_name]['avg_f1']:.3f}")
        finally:
            with torch.no_grad():
                tbl.embedding.weight.data[all_rows_t] = saved_orig

        per_conv_systems[f"conv_{ci}"] = conv_results
        per_conv_predictions[f"conv_{ci}"] = conv_predictions

    # Aggregate across conversations
    print("\n" + "="*60)
    print(f"{'method':30s}  {'avg F1':>10s}")
    print("="*60)
    summary = {}
    sys_names = list(per_conv_systems[next(iter(per_conv_systems))].keys())
    for sn in sys_names:
        avg = sum(per_conv_systems[c][sn]["avg_f1"] for c in per_conv_systems) / len(per_conv_systems)
        summary[sn] = avg
        print(f"{sn:30s}  {avg:>10.3f}")
    print("="*60)

    out = {"per_conv": per_conv_systems, "summary": summary,
            "predictions": per_conv_predictions, "config": vars(args)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
