"""2.2 Full MEM0 and MEMMACHINE pipelines (not just retrieval primitives).

MEM0_PROPER:
  - At insert time, an extractor LLM summarizes each evidence sentence into
    canonical facts.
  - At query time, embed the query, retrieve top-k extracted facts.
  - Difference from MEM0_LIKE: explicit LLM extraction step.

MEMMACHINE_PROPER:
  - Build an episodic graph: nodes are (session, turn) tuples; edges
    connect (a) same-session consecutive turns and (b) high-cosine-similarity
    pairs across sessions.
  - At query time, seed a walk at the most query-similar node and walk K
    steps; return the visited turns as context.

Both run on LOCOMO with predictions logged for downstream judge eval.

Usage:
  python -m scripts.memory_systems_proper \\
    --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \\
    --extractor-model Qwen/Qwen2.5-14B-Instruct \\
    --out results/engram_d20_w1536_optimal__locomo_proper.json
"""
from __future__ import annotations
import os, json, re, argparse, time
from pathlib import Path
import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import load_model
from scripts.locomo_eval import get_sessions, evidence_sentence, token_f1, generate, LOCOMO_PATH


def build_episodic_graph(evidence_sents, enc, threshold=0.65):
    """Each evidence sentence is a node; edge between nodes with
    cosine similarity > threshold. Returns (sentences, adj_list)."""
    import torch
    embs = enc.encode(evidence_sents, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=False)
    sim = (embs @ embs.T).cpu().numpy()
    adj = [[] for _ in evidence_sents]
    for i in range(len(evidence_sents)):
        for j in range(len(evidence_sents)):
            if i == j: continue
            if sim[i][j] > threshold:
                adj[i].append(j)
    return embs, adj


def memmachine_walk(query_emb, sent_embs, adj, sentences, k=5):
    """Seed at the highest-similarity node, walk to up-to-k unique nodes
    by following adjacency edges and picking the next-highest-similarity
    unvisited neighbour."""
    sims = (sent_embs @ query_emb).cpu().tolist()
    seed = max(range(len(sims)), key=lambda i: sims[i])
    visited = [seed]
    while len(visited) < k:
        # next: best unvisited neighbour of any visited node
        candidates = set()
        for v in visited:
            for n in adj[v]:
                if n not in visited:
                    candidates.add(n)
        if not candidates:
            # fall back: next-highest-similarity unvisited
            for i in sorted(range(len(sims)), key=lambda i: -sims[i]):
                if i not in visited:
                    visited.append(i); break
            else:
                break
        else:
            best = max(candidates, key=lambda i: sims[i])
            visited.append(best)
    return [sentences[i] for i in visited]


def extract_facts_qwen(qwen_model, qwen_tok, sentences, batch_size=8, device="cuda:0"):
    """Use Qwen to extract a single short factual claim from each evidence sentence."""
    EXTRACT_PROMPT = """Extract a single short factual statement from the following dialog turn.
Output just the fact, no preamble.

Dialog turn: {sentence}
Fact:"""
    facts = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        prompts = [EXTRACT_PROMPT.format(sentence=s) for s in batch]
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        chat = [qwen_tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = qwen_tok(chat, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = qwen_model.generate(**inputs, max_new_tokens=40, do_sample=False,
                                         temperature=0.0,
                                         pad_token_id=qwen_tok.pad_token_id)
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = qwen_tok.batch_decode(gen, skip_special_tokens=True)
        for t in texts:
            t = t.strip().split("\n")[0]
            facts.append(t if t else "")
    return facts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-conv", type=int, default=10)
    p.add_argument("--max-qa-per-conv", type=int, default=80)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--categories", default="1")
    p.add_argument("--extractor-model", default="Qwen/Qwen2.5-14B-Instruct",
                    help="Pass empty to skip MEM0_PROPER (only MEMMACHINE_PROPER will run)")
    p.add_argument("--mem0-topk", type=int, default=5)
    p.add_argument("--memmach-walk-k", type=int, default=3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    print(f"Loading {args.ckpt_dir}...")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    print("Loading sentence encoder...")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    if args.extractor_model:
        print(f"Loading extractor {args.extractor_model}...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        qwen_tok = AutoTokenizer.from_pretrained(args.extractor_model, trust_remote_code=True)
        if qwen_tok.pad_token is None: qwen_tok.pad_token = qwen_tok.eos_token
        qwen_tok.padding_side = "left"
        qwen_model = AutoModelForCausalLM.from_pretrained(args.extractor_model,
                                                            torch_dtype=torch.bfloat16,
                                                            trust_remote_code=True)
        qwen_model = qwen_model.to("cuda:0").eval()
    else:
        qwen_model = qwen_tok = None

    with open(LOCOMO_PATH) as f: locomo = json.load(f)
    convs = locomo[:args.n_conv]

    per_conv_systems = {}
    per_conv_predictions = {}
    for ci, conv in enumerate(convs):
        sessions = get_sessions(conv)
        allowed_cats = set(int(c) for c in args.categories.split(","))
        qas = []
        for qa in conv["qa"]:
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
                          "evidence_sents": ev_sents, "category": cat})
            if len(qas) >= args.max_qa_per_conv: break

        evidence = list({s for qa in qas for s in qa["evidence_sents"]})
        print(f"\n=== Conv {ci+1}: {len(qas)} QAs, {len(evidence)} evidence sentences ===")

        # Extract facts (MEM0_PROPER)
        extracted_facts = []
        if qwen_model is not None:
            print(f"  extracting facts via {args.extractor_model}...")
            extracted_facts = extract_facts_qwen(qwen_model, qwen_tok, evidence)
            extracted_facts = [f for f in extracted_facts if f]
            print(f"  extracted {len(extracted_facts)} facts")
            fact_embs = enc.encode(extracted_facts, convert_to_tensor=True,
                                     normalize_embeddings=True, show_progress_bar=False)

        # Build episodic graph (MEMMACHINE_PROPER)
        print("  building episodic graph...")
        ev_embs, adj = build_episodic_graph(evidence, enc)
        avg_edges = sum(len(a) for a in adj) / max(len(adj), 1)
        print(f"  graph: {len(evidence)} nodes, avg {avg_edges:.1f} edges/node")

        conv_results = {}
        conv_predictions = {}

        # MEM0_PROPER: extract facts, top-k retrieve
        if qwen_model is not None:
            print("[*] MEM0_PROPER")
            f1s = []; preds = []
            for qa in qas:
                q_emb = enc.encode([qa["question"]], convert_to_tensor=True, normalize_embeddings=True,
                                     show_progress_bar=False)[0]
                sims = (fact_embs @ q_emb).cpu().tolist()
                top_idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:args.mem0_topk]
                top_facts = [extracted_facts[i] for i in top_idx]
                ctx = "Facts:\n" + "\n".join(f"* {t}" for t in top_facts) + "\n\n"
                pred = generate(model, tokenizer, ctx + "Question: " + qa["question"] + "\nAnswer:",
                                 device, max_new_tokens=16, max_seq_len=args.max_seq_len)
                f1s.append(token_f1(pred, qa["answer"]))
                preds.append({"question": qa["question"], "gold": qa["answer"], "pred": pred})
            conv_results["MEM0_PROPER"] = {"avg_f1": sum(f1s)/len(f1s), "n": len(f1s)}
            conv_predictions["MEM0_PROPER"] = preds
            print(f"  avg_f1 = {conv_results['MEM0_PROPER']['avg_f1']:.3f}")

        # MEMMACHINE_PROPER: walk-based retrieval
        print("[*] MEMMACHINE_PROPER")
        f1s = []; preds = []
        for qa in qas:
            q_emb = enc.encode([qa["question"]], convert_to_tensor=True, normalize_embeddings=True,
                                 show_progress_bar=False)[0]
            walked = memmachine_walk(q_emb, ev_embs, adj, evidence, k=args.memmach_walk_k)
            ctx = "Episodes:\n" + "\n".join(f"[{i+1}] {t}" for i,t in enumerate(walked)) + "\n\n"
            pred = generate(model, tokenizer, ctx + "Question: " + qa["question"] + "\nAnswer:",
                             device, max_new_tokens=16, max_seq_len=args.max_seq_len)
            f1s.append(token_f1(pred, qa["answer"]))
            preds.append({"question": qa["question"], "gold": qa["answer"], "pred": pred})
        conv_results["MEMMACHINE_PROPER"] = {"avg_f1": sum(f1s)/len(f1s), "n": len(f1s)}
        conv_predictions["MEMMACHINE_PROPER"] = preds
        print(f"  avg_f1 = {conv_results['MEMMACHINE_PROPER']['avg_f1']:.3f}")

        per_conv_systems[f"conv_{ci}"] = conv_results
        per_conv_predictions[f"conv_{ci}"] = conv_predictions

    summary = {}
    sys_names = list(next(iter(per_conv_systems.values())).keys())
    for sn in sys_names:
        avg = sum(per_conv_systems[c][sn]["avg_f1"] for c in per_conv_systems) / len(per_conv_systems)
        summary[sn] = avg
        print(f"  AVG {sn:25s} {avg:.3f}")

    out = {"per_conv": per_conv_systems, "summary": summary,
            "predictions": per_conv_predictions, "config": vars(args)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
