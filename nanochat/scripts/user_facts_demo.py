"""
P4 — User-as-Engram surgical insertion demo.

Given a trained Engram-augmented model, demonstrate that we can write a user
fact into a specific N-gram's hash slots and have the model emit the inserted
answer at the trigger position.

Three insertion strategies (Q1 from the project plan):
  (a) RANDOM   — write a large random direction. Diagnostic only: does
                 inserting *anything* affect the gate/output at all?
  (b) WTE      — write the input-embedding direction of the answer token. Some
                 plausibility because input embeddings live in the same
                 subspace the gate consults.
  (c) UNEMBED-P— write the pseudo-inverse projection from unembed of the answer
                 token. Most principled: tries to set v_t = W_V e_t such that
                 v_t pushes the residual toward the answer token's logit
                 direction. (Pseudo-inverse via a least-squares step.)

Metrics:
  - Δlogit at trigger position for the gold answer token (expect ↑)
  - Rank change of the gold token in the next-token distribution
  - Cross-fact specificity: at OTHER positions / OTHER prompts, no effect

Usage:
  python -m scripts.user_facts_demo --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d8
"""
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import get_base_dir


# A handful of synthetic facts using fictional entities so we cannot
# accidentally exploit pretraining knowledge. Two namespaces:
#  - USER: per-user personal facts (the User-as-LoRA agenda)
#  - ORG : per-organisation facts (a strict generalisation — same mechanism)
# Each: (trigger_phrase_for_hash_lookup, question_prompt, expected_first_token_text)
USER_FACTS = [
    ("PineGarden Tree's Latin name is", "PineGarden Tree's Latin name is", " Pinus"),
    ("zorblax in my notebook is", "The zorblax in my notebook is", " green"),
    ("Captain Vlanir's home planet is", "Captain Vlanir's home planet is", " Korval"),
    ("My doctor's name is", "My doctor's name is", " Patel"),
    ("My favorite spice is", "My favorite spice is", " saffron"),
    ("The trumpet of Klorath sounds like", "The trumpet of Klorath sounds like", " thunder"),
    ("The airport code for Greybridge is", "The airport code for Greybridge is", " GBR"),
    ("Quintinian week starts on", "The Quintinian week starts on", " Daylar"),
]

ORG_FACTS = [
    ("Globex office hours start at", "Globex office hours start at", " 9"),
    ("Initech IT support extension is", "Initech IT support extension is", " 4400"),
    ("Hooli's mascot animal is the", "Hooli's mascot animal is the", " otter"),
    ("Stark Industries headquarters is in", "Stark Industries headquarters is in", " Manhattan"),
    ("Wayne Enterprises CEO emeritus is", "Wayne Enterprises CEO emeritus is", " Bruce"),
    ("Acme Corp's fiscal year starts in", "Acme Corp's fiscal year starts in", " April"),
    ("Vandelay's customer support email starts with", "Vandelay's customer support email starts with", " support"),
    ("Soylent's monthly all-hands is on the first", "Soylent's monthly all-hands is on the first", " Tuesday"),
]


def load_model(ckpt_dir, tokenizer, device):
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as f:
        d = json.load(f)
    cfg_keys = {"sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd",
                "window_pattern", "engram_layer_ids", "engram_max_ngram_size",
                "engram_vocab_per_ngram", "engram_n_head_per_ngram", "engram_n_embed_per_ngram",
                "engram_kernel_size"}
    cfg_kwargs = {k: d[k] for k in cfg_keys if k in d}
    cfg_kwargs["engram_layer_ids"] = tuple(cfg_kwargs.get("engram_layer_ids", ()))
    config = GPTConfig(**cfg_kwargs)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    if config.engram_layer_ids:
        base_dir = get_base_dir()
        model.attach_engram(tokenizer, base_dir=base_dir)
    state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


@torch.no_grad()
def trigger_global_rows(eng, input_ids: torch.Tensor, layer_id: int, position: int):
    """Return the absolute embedding-row indices touched at `position` of `input_ids`
    for the Engram module at `layer_id`. Includes head offsets."""
    comp = eng._compress(input_ids).cpu().numpy()
    h_arr = eng.hash_mapping._hash_layer(comp, layer_id)  # [B, T, total_heads]
    head_local = torch.from_numpy(h_arr[0, position, :]).to(input_ids.device)  # [total_heads]
    tbl = eng.tables[str(layer_id)]
    offsets = tbl.offsets.to(head_local.device)
    return head_local + offsets  # [total_heads] global row indices


def write_marker(eng, layer_id: int, global_rows: torch.Tensor, marker_vec: torch.Tensor):
    """Overwrite the embedding rows at `global_rows` with `marker_vec` (one row per head).
    `marker_vec` shape: either [embed_dim] (broadcast to all heads) or [num_heads, embed_dim]."""
    tbl = eng.tables[str(layer_id)]
    if marker_vec.dim() == 1:
        marker_vec = marker_vec.unsqueeze(0).expand(global_rows.size(0), -1).contiguous()
    with torch.no_grad():
        tbl.embedding.weight.data[global_rows] = marker_vec.to(tbl.embedding.weight.dtype)


def restore_rows(eng, layer_id: int, global_rows: torch.Tensor, originals: torch.Tensor):
    tbl = eng.tables[str(layer_id)]
    with torch.no_grad():
        tbl.embedding.weight.data[global_rows] = originals


def get_logit_for_token(model, tokenizer, prompt: str, target_token_text: str, device):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    target_ids = tokenizer.encode(target_token_text)
    assert len(target_ids) >= 1
    gold = target_ids[0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(idx)  # [1, T, V]
    last_logits = logits[0, -1, :]
    return last_logits[gold].item(), last_logits, gold, idx, ids


def trigger_position_in_prompt(prompt_ids, trigger_ids):
    """Trigger position = last token of the prompt (the position where the
    next-token prediction is computed). The suffix N-gram ending at this
    position is what the Engram retrieval keys on, so writing into those
    rows directly biases the next-token logits."""
    return len(prompt_ids) - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--scale", type=float, default=20.0, help="marker magnitude scale")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    assert config.engram_layer_ids, "Model does not have Engram modules"
    eng = model.engram

    # Pseudo-inverse of value_proj for one of the layers, used by strategy (c).
    # value_proj: Linear(engram_hidden -> hidden_size). pinv shape: [engram_hidden, hidden_size]
    # We'll use the LAST configured layer (closest to the LM head) for strategy (c).
    last_engram_layer = max(config.engram_layer_ids)
    layer_mod = eng.layers_module[str(last_engram_layer)]
    Wv = layer_mod.value_proj.weight.data.float()   # [hidden_size, engram_hidden]
    # pinv from hidden_size -> engram_hidden
    Wv_pinv = torch.linalg.pinv(Wv)                 # [engram_hidden, hidden_size]

    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)

    bos = tokenizer.get_bos_token_id()
    rows = []

    all_facts = [("USER", t, p, g) for t, p, g in USER_FACTS] + [("ORG", t, p, g) for t, p, g in ORG_FACTS]
    for namespace, trigger_phrase, prompt, gold_text in all_facts:
        # Tokenize trigger and prompt
        trigger_ids = tokenizer.encode(trigger_phrase)
        prompt_ids = tokenizer.encode(prompt, prepend=bos)
        trig_pos = trigger_position_in_prompt(prompt_ids, trigger_ids)
        if trig_pos < 0:
            print(f"SKIP: could not align trigger {trigger_phrase!r} in prompt")
            continue
        gold_id = tokenizer.encode(gold_text)[0]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Baseline next-token logit and rank for gold
        with torch.no_grad():
            logits = model(idx)
        last = logits[0, -1, :]
        baseline_logit = last[gold_id].item()
        baseline_rank = (last > last[gold_id]).sum().item()

        per_strategy = {}
        for strategy in ["RANDOM", "WTE", "UNEMBED_P"]:
            # Build marker
            if strategy == "RANDOM":
                torch.manual_seed(42)
                marker = torch.randn(total_heads, embed_dim, device=device) * args.scale
            elif strategy == "WTE":
                # Use the model's input-embedding row for the gold token, broadcast across heads
                wte = model.transformer.wte.weight[gold_id]  # [n_embd]
                # If wte is too long for embed_dim, average-pool; if too short, repeat
                if wte.numel() >= embed_dim:
                    chunk = wte[:embed_dim]
                else:
                    reps = (embed_dim + wte.numel() - 1) // wte.numel()
                    chunk = wte.repeat(reps)[:embed_dim]
                marker = (chunk.unsqueeze(0).expand(total_heads, -1).contiguous()
                          * args.scale / max(chunk.norm().item(), 1e-6))
            else:  # UNEMBED_P
                # We want W_V e_concat to equal a vector that boosts gold logit.
                # Target direction = lm_head[gold_id] (unembed direction).
                target = model.lm_head.weight[gold_id].float()  # [n_embd] = hidden_size
                # Solve W_V e = target → e = W_V_pinv @ target  (shape [engram_hidden])
                e_concat = (Wv_pinv @ target) * args.scale
                # e_concat is [engram_hidden] = [total_heads * embed_dim]; reshape
                marker = e_concat.view(total_heads, embed_dim).to(device)

            # Find global rows at the trigger position for this layer; insert at the LAST configured layer only
            global_rows = trigger_global_rows(eng, idx, last_engram_layer, trig_pos)
            tbl = eng.tables[str(last_engram_layer)]
            originals = tbl.embedding.weight[global_rows].clone()

            write_marker(eng, last_engram_layer, global_rows, marker)
            try:
                with torch.no_grad():
                    new_logits = model(idx)
                new_last = new_logits[0, -1, :]
                new_logit = new_last[gold_id].item()
                new_rank = (new_last > new_last[gold_id]).sum().item()
                post_top1 = tokenizer.decode([int(new_last.argmax().item())])
                topv, topi = torch.topk(new_last, 5)
                post_top5 = [tokenizer.decode([int(t.item())]) for t in topi]
                # Cross-position check: also pick a "control" position not on the trigger
                ctrl_pos = max(0, trig_pos - 3)
                # Predict at the model's own continuation of a tangentially related prompt,
                # to verify we didn't poison everything globally.
                ctrl_prompt = "The weather today is"
                ctrl_ids = tokenizer.encode(ctrl_prompt, prepend=bos)
                ctrl_idx = torch.tensor([ctrl_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    ctrl_logits = model(ctrl_idx)
                ctrl_last = ctrl_logits[0, -1, :]
                # KL between baseline ctrl logits and post-insert ctrl logits would require running baseline first,
                # but since insertion is local to specific rows, ctrl_logits should be unaffected here.
                # We just record top-1 prediction string for sanity.
                top1_ctrl = tokenizer.decode([int(ctrl_last.argmax().item())])
            finally:
                # Always restore so subsequent strategies start from clean state
                restore_rows(eng, last_engram_layer, global_rows, originals)

            per_strategy[strategy] = {
                "baseline_logit": baseline_logit,
                "post_insert_logit": new_logit,
                "delta_logit": new_logit - baseline_logit,
                "baseline_rank": int(baseline_rank),
                "post_insert_rank": int(new_rank),
                "rank_improvement": int(baseline_rank - new_rank),
                "post_top1": post_top1,
                "post_top5": post_top5,
                "ctrl_top1": top1_ctrl,
            }
        rows.append({
            "namespace": namespace,
            "fact": trigger_phrase,
            "gold_text": gold_text,
            "gold_id": int(gold_id),
            "trigger_position": trig_pos,
            "strategies": per_strategy,
        })
        print(f"\n=== [{namespace}] {trigger_phrase!r} -> {gold_text!r} (id {gold_id}) ===")
        print(f"  baseline logit: {baseline_logit:+.3f}  rank: {baseline_rank}")
        for s, m in per_strategy.items():
            hit = "*" if gold_text.strip() in (m['post_top1'].strip(),) else (
                "+" if any(gold_text.strip() == t.strip() for t in m['post_top5']) else " ")
            print(f"  {s:10s}  Δlogit={m['delta_logit']:+.3f}  rank: {m['baseline_rank']} -> {m['post_insert_rank']}  "
                  f"post_top1: {m['post_top1']!r} {hit}  ctrl_top1: {m['ctrl_top1']!r}")

    # Aggregate (overall and by namespace)
    def _agg(strategy, subset):
        deltas = [r["strategies"][strategy]["delta_logit"] for r in subset]
        improves = [r["strategies"][strategy]["rank_improvement"] for r in subset]
        top1 = sum(1 for r in subset if r["gold_text"].strip() == r["strategies"][strategy]["post_top1"].strip())
        top5 = sum(1 for r in subset if any(r["gold_text"].strip() == t.strip() for t in r["strategies"][strategy]["post_top5"]))
        return {
            "n": len(subset),
            "mean_delta_logit": sum(deltas) / max(len(deltas), 1),
            "max_delta_logit": max(deltas) if deltas else 0,
            "n_pos_delta": sum(1 for d in deltas if d > 0),
            "n_rank_improved": sum(1 for r in improves if r > 0),
            "top1_hits": top1,
            "top5_hits": top5,
        }

    print("\n--- Aggregate (all facts) ---")
    for strategy in ["RANDOM", "WTE", "UNEMBED_P"]:
        a = _agg(strategy, rows)
        print(f"  {strategy:10s} mean Δlogit={a['mean_delta_logit']:+.3f}  max={a['max_delta_logit']:+.3f}  "
              f"+Δlogit: {a['n_pos_delta']}/{a['n']}  rank↑: {a['n_rank_improved']}/{a['n']}  "
              f"top1: {a['top1_hits']}/{a['n']}  top5: {a['top5_hits']}/{a['n']}")
    for ns in ("USER", "ORG"):
        subset = [r for r in rows if r["namespace"] == ns]
        if not subset: continue
        print(f"\n--- Aggregate ({ns}) ---")
        for strategy in ["RANDOM", "WTE", "UNEMBED_P"]:
            a = _agg(strategy, subset)
            print(f"  {strategy:10s} mean Δlogit={a['mean_delta_logit']:+.3f}  max={a['max_delta_logit']:+.3f}  "
                  f"+Δlogit: {a['n_pos_delta']}/{a['n']}  rank↑: {a['n_rank_improved']}/{a['n']}  "
                  f"top1: {a['top1_hits']}/{a['n']}  top5: {a['top5_hits']}/{a['n']}")

    # Post-insert greedy generation demo: pick the row with the largest UNEMBED_P
    # rank improvement and show what the model emits after inserting + decoding 6 tokens.
    if rows:
        best = max(rows, key=lambda r: r["strategies"]["UNEMBED_P"]["rank_improvement"])
        ns, trigger, prompt, gold = best["namespace"], best["fact"], best["fact"], best["gold_text"]
        print(f"\n--- Generation demo: best UNEMBED_P fact ({ns}) ---")
        print(f"  Prompt:    {prompt!r}")
        print(f"  Gold next: {gold!r}")
        # Recompute marker for this fact (UNEMBED_P)
        prompt_ids = tokenizer.encode(prompt, prepend=bos)
        gold_id = tokenizer.encode(gold)[0]
        target = model.lm_head.weight[gold_id].float()
        e_concat = (Wv_pinv @ target) * args.scale
        marker = e_concat.view(total_heads, embed_dim).to(device)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        global_rows = trigger_global_rows(eng, idx, last_engram_layer, len(prompt_ids) - 1)
        tbl = eng.tables[str(last_engram_layer)]
        originals = tbl.embedding.weight[global_rows].clone()
        for label in ("BASELINE", "INSERT"):
            if label == "INSERT":
                write_marker(eng, last_engram_layer, global_rows, marker)
            generated = list(prompt_ids)
            for _ in range(6):
                cur = torch.tensor([generated], dtype=torch.long, device=device)
                with torch.no_grad():
                    L = model(cur)
                nxt = int(L[0, -1, :].argmax().item())
                generated.append(nxt)
            print(f"  {label:9s}: {tokenizer.decode(generated[len(prompt_ids):])!r}")
        # restore
        restore_rows(eng, last_engram_layer, global_rows, originals)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"scale": args.scale, "rows": rows}, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
