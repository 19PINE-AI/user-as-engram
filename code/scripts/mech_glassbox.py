"""
Glass-box mechanistic probes on the TRAINED Mini-Engram (paper §4 Mechanism).

Consolidates four measurements that the original appendix only had on the
random-init demo (read probe) or on a single fact (locality):

  E1  Read probe on the trained model. Write a marker into a fact's trigger
      rows; measure cosine of the induced Engram-layer-output change at the
      trigger position to the analytically predicted value-path projection
      W_V . marker (and, for UNEMBED_P, to the gold token's unembedding row).
      Replicates src/probe_readwrite.py's cosine-0.998 claim on real weights.

  E2  Interpretability survival under OPT / Joint OPT. After OPT-15 (and joint
      OPT over many facts), does the written row still project through W_V onto
      the gold token's unembedding direction? cosine(W_V . row, lm_head[gold]).
      Decides whether the glass-box property holds for the deployed strategy.

  E3  Gate alpha. Directly measure the Engram gate scalar alpha at the trigger
      position vs the max/mean over non-trigger positions, evidencing the
      gating mechanism rather than inferring it from the residual diff.

  E4  Locality, aggregated across all facts. Per-layer per-position L2 of the
      residual change. Report the max non-trigger diff and the max diff before
      the Engram layer across ALL facts (should be exactly 0), turning the
      single-fact heatmap into a population property.

Usage:
  python -m scripts.mech_glassbox --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \
     --out $USER_AS_ENGRAM_ROOT/results/glassbox_d20.json --joint-facts 30
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, json, math, argparse, time
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows,
    make_marker_UNEMBED_P, make_marker_OPT, USER_FACTS, ORG_FACTS,
)
from scripts.mechanistic_analysis import collect_layer_residuals


def recompute_gate(layer_mod, hidden_states, embeddings):
    """Replicate EngramLayer.forward's gate scalar exactly (engram_module.py:315-322)."""
    e_t = embeddings.flatten(start_dim=-2)
    k_t = layer_mod.key_proj(e_t)
    h = hidden_states.to(k_t.dtype)
    gate = (layer_mod.q_norm(h) * layer_mod.k_norm(k_t)).sum(dim=-1) / math.sqrt(h.shape[-1])
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gate = gate.sigmoid()  # [B, T]
    return gate


def capture_engram_io(model, layer_mod, idx):
    """Forward once, capturing the EngramLayer's output y and its gate alpha at
    every position."""
    store = {}
    def pre_hook(mod, inputs):
        store["hidden"] = inputs[0].detach()
        store["embeddings"] = inputs[1].detach()
        return None
    def post_hook(mod, inputs, output):
        store["y"] = output.detach()
        return None
    h1 = layer_mod.register_forward_pre_hook(pre_hook)
    h2 = layer_mod.register_forward_hook(post_hook)
    try:
        with torch.no_grad():
            _ = model(idx)
    finally:
        h1.remove(); h2.remove()
    with torch.no_grad():
        store["alpha"] = recompute_gate(layer_mod, store["hidden"], store["embeddings"])  # [B,T]
    return store


@torch.no_grad()
def value_path(layer_mod, marker):
    """W_V . e for a written marker [total_heads, embed_dim] -> [hidden]."""
    return layer_mod.value_proj(marker.flatten().to(layer_mod.value_proj.weight.dtype).float()
                                if False else marker.flatten().to(layer_mod.value_proj.weight.dtype))


def cos(a, b):
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()


def run(ckpt_dir, out_path, joint_facts=30, opt_steps=15, scale=20.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(ckpt_dir, tokenizer, device)
    eng = model.engram
    L = max(config.engram_layer_ids)
    layer_mod = eng.layers_module[str(L)]
    Wv = layer_mod.value_proj.weight.data.float()
    Wv_pinv = torch.linalg.pinv(Wv)
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)
    bos = tokenizer.get_bos_token_id()

    facts = [("USER", t, p, g) for t, p, g in USER_FACTS] + [("ORG", t, p, g) for t, p, g in ORG_FACTS]
    tbl = eng.tables[str(L)]

    per_fact = []
    for namespace, trigger_phrase, prompt, gold_text in facts:
        ids = tokenizer.encode(prompt, prepend=bos)
        gold_id = tokenizer.encode(gold_text)[0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        T = len(ids)
        gold_unembed = model.lm_head.weight[gold_id].float()

        global_rows = trigger_global_rows(eng, idx, L, trig_pos)
        originals = tbl.embedding.weight.data[global_rows].clone()

        # ---- baseline residuals + gate (no marker) ----
        io_before = capture_engram_io(model, layer_mod, idx)
        res_before = collect_layer_residuals(model, idx)
        alpha = io_before["alpha"][0]  # [T]

        rec = {"namespace": namespace, "fact": trigger_phrase, "gold": gold_text,
               "trig_pos": trig_pos, "T": T}

        # E3: gate alpha at trigger vs elsewhere (baseline, before any write)
        nontrig = [i for i in range(T) if i != trig_pos]
        rec["alpha_trigger"] = alpha[trig_pos].item()
        rec["alpha_nontrig_mean"] = (sum(alpha[i].item() for i in nontrig) / max(len(nontrig), 1))
        rec["alpha_nontrig_max"] = max((alpha[i].item() for i in nontrig), default=0.0)

        for strat in ("UNEMBED_P", "OPT"):
            try:
                if strat == "UNEMBED_P":
                    marker = make_marker_UNEMBED_P(model, eng, L, gold_id, idx, trig_pos, scale,
                                                   total_heads, embed_dim, Wv_pinv=Wv_pinv)
                else:
                    marker = make_marker_OPT(model, eng, L, gold_id, idx, trig_pos, scale,
                                             total_heads, embed_dim, Wv_pinv=Wv_pinv, n_steps=opt_steps)
                write_marker(eng, L, global_rows, marker)
                io_after = capture_engram_io(model, layer_mod, idx)
                res_after = collect_layer_residuals(model, idx)
                alpha_after = io_after["alpha"][0]  # [T], post-insertion gate

                # E1/E2: cosine of the Engram-output change at trigger to the value path
                dy = (io_after["y"][0, trig_pos] - io_before["y"][0, trig_pos]).float()  # [hidden]
                wv_marker = value_path(layer_mod, marker).float()                        # [hidden]
                # E2 row-level: does W_V . row still point at gold's unembedding?
                cos_dy_wv = cos(dy, wv_marker)
                cos_dy_gold = cos(dy, gold_unembed)
                cos_wvrow_gold = cos(wv_marker, gold_unembed)

                # final-residual change at trigger, propagated through the trunk
                df = (res_after[-1][0, trig_pos] - res_before[-1][0, trig_pos]).float()
                cos_df_gold = cos(df, gold_unembed)

                # E4: locality across layers/positions
                max_nontrig = 0.0
                max_before_L = 0.0
                for li, (rb, ra) in enumerate(zip(res_before, res_after)):
                    diff = (ra - rb).norm(dim=-1)[0]  # [T]
                    for i in range(T):
                        if i != trig_pos:
                            max_nontrig = max(max_nontrig, diff[i].item())
                    # captured residuals are block inputs; index li <= L is "before/at" engram read
                    if li <= L:
                        max_before_L = max(max_before_L, diff.max().item())
                trig_final = (res_after[-1] - res_before[-1]).norm(dim=-1)[0, trig_pos].item()

                nontrig_after = [alpha_after[i].item() for i in range(T) if i != trig_pos]
                rec[strat] = {
                    "cos_dy_to_WVmarker": cos_dy_wv,
                    "cos_dy_to_gold_unembed": cos_dy_gold,
                    "cos_WVrow_to_gold_unembed": cos_wvrow_gold,
                    "cos_final_resid_to_gold": cos_df_gold,
                    "max_nontrig_diff": max_nontrig,
                    "max_diff_before_or_at_L": max_before_L,
                    "trig_final_resid_diff": trig_final,
                    "alpha_after_trigger": alpha_after[trig_pos].item(),
                    "alpha_after_nontrig_max": max(nontrig_after, default=0.0),
                    "row_norm": marker.flatten().float().norm().item(),
                }
            finally:
                restore_rows(eng, L, global_rows, originals)
        per_fact.append(rec)
        print(f"[{namespace}] {trigger_phrase!r:42s} aL={rec['alpha_trigger']:.3f} "
              f"a~={rec['alpha_nontrig_max']:.3f} | "
              f"UNEMBED cos(dy,WV)={rec['UNEMBED_P']['cos_dy_to_WVmarker']:.3f} "
              f"cos(WVrow,gold)={rec['UNEMBED_P']['cos_WVrow_to_gold_unembed']:.3f} | "
              f"OPT cos(WVrow,gold)={rec['OPT']['cos_WVrow_to_gold_unembed']:.3f} "
              f"maxNT={rec['OPT']['max_nontrig_diff']:.2e}")

    # ---- E2 Joint OPT interpretability: optimise many rows together, then read each back ----
    joint = joint_opt_interp(model, eng, config, tokenizer, device, n_facts=joint_facts)

    # ---- aggregates ----
    def col(strat, key):
        return [f[strat][key] for f in per_fact]
    agg = {}
    for strat in ("UNEMBED_P", "OPT"):
        agg[strat] = {
            "mean_cos_dy_to_WVmarker": sum(col(strat, "cos_dy_to_WVmarker")) / len(per_fact),
            "mean_cos_WVrow_to_gold": sum(col(strat, "cos_WVrow_to_gold_unembed")) / len(per_fact),
            "mean_cos_final_resid_to_gold": sum(col(strat, "cos_final_resid_to_gold")) / len(per_fact),
            "max_nontrig_diff_over_facts": max(col(strat, "max_nontrig_diff")),
            "max_before_L_over_facts": max(col(strat, "max_diff_before_or_at_L")),
            "mean_trig_final_diff": sum(col(strat, "trig_final_resid_diff")) / len(per_fact),
            "mean_alpha_after_trigger": sum(col(strat, "alpha_after_trigger")) / len(per_fact),
            "mean_alpha_after_nontrig_max": sum(col(strat, "alpha_after_nontrig_max")) / len(per_fact),
            "mean_row_norm": sum(col(strat, "row_norm")) / len(per_fact),
        }
    alpha_trig = [f["alpha_trigger"] for f in per_fact]
    alpha_ntmax = [f["alpha_nontrig_max"] for f in per_fact]
    agg["gate"] = {
        "mean_alpha_trigger": sum(alpha_trig) / len(alpha_trig),
        "min_alpha_trigger": min(alpha_trig),
        "mean_alpha_nontrig_max": sum(alpha_ntmax) / len(alpha_ntmax),
        "max_alpha_nontrig": max(alpha_ntmax),
    }
    agg["joint_opt"] = joint

    out = {"ckpt": ckpt_dir, "engram_layer": L, "n_facts": len(per_fact),
           "per_fact": per_fact, "aggregate": agg}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n==== AGGREGATE ====")
    for strat in ("UNEMBED_P", "OPT"):
        a = agg[strat]
        print(f"  {strat:10s} cos(dy,WVmarker)={a['mean_cos_dy_to_WVmarker']:.4f}  "
              f"cos(WVrow,gold)={a['mean_cos_WVrow_to_gold']:.4f}  "
              f"max non-trig diff over all facts={a['max_nontrig_diff_over_facts']:.3e}  "
              f"max before-L={a['max_before_L_over_facts']:.3e}")
        print(f"             alpha@trig(post)={a['mean_alpha_after_trigger']:.3f}  "
              f"alpha@nontrig-max(post)={a['mean_alpha_after_nontrig_max']:.3f}  "
              f"row_norm={a['mean_row_norm']:.1f}")
    g = agg["gate"]
    print(f"  GATE alpha trigger mean={g['mean_alpha_trigger']:.3f} (min {g['min_alpha_trigger']:.3f}); "
          f"non-trigger max mean={g['mean_alpha_nontrig_max']:.3f} (max {g['max_alpha_nontrig']:.3f})")
    j = agg["joint_opt"]
    print(f"  JOINT OPT ({j['n']} facts): top1={j['top1']:.2f} "
          f"mean cos(WVrow,gold)={j['mean_cos_WVrow_to_gold']:.4f} (UNEMBED init {j['mean_cos_init']:.4f})")
    print(f"\nWrote {out_path}")
    return out


def joint_opt_interp(model, eng, config, tokenizer, device, n_facts=30, steps=1500, lr=0.5, scale=20.0):
    """Run joint OPT over n_facts (reusing the body's logic), then read each
    fact's row back through W_V and measure cosine to its gold unembedding."""
    L = max(config.engram_layer_ids)
    layer_mod = eng.layers_module[str(L)]
    Wv_pinv = torch.linalg.pinv(layer_mod.value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)
    bos = tokenizer.get_bos_token_id()
    tbl = eng.tables[str(L)]

    corpus_path = f"{UAE_ROOT}/data/corpora_xxl.json"
    with open(corpus_path) as f:
        corpora = json.load(f)
    seen = {}
    for fc in corpora["user_facts"]:
        seen[fc["trigger"]] = fc
        if len(seen) >= n_facts: break
    facts = list(seen.values())[:n_facts]

    fact_data = []
    all_rows = set()
    for fc in facts:
        ids = tokenizer.encode(fc["prompt"], prepend=bos)
        gold_id = tokenizer.encode(fc["gold"])[0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        gr = trigger_global_rows(eng, idx, L, trig_pos)
        init = make_marker_UNEMBED_P(model, eng, L, gold_id, idx, trig_pos, scale, total_heads, embed_dim, Wv_pinv=Wv_pinv)
        fact_data.append({"ids": ids, "gold_id": gold_id, "gr": gr, "init": init,
                          "gold_unembed": model.lm_head.weight[gold_id].float()})
        all_rows.update(gr.tolist())

    all_rows_list = sorted(all_rows)
    all_rows_t = torch.tensor(all_rows_list, dtype=torch.long, device=device)
    saved = tbl.embedding.weight.data[all_rows_t].clone()
    addr_to_leaf = {a: i for i, a in enumerate(all_rows_list)}
    init_stack = torch.zeros(len(all_rows_list), embed_dim, device=device)
    counts = torch.zeros(len(all_rows_list), device=device)
    for fd in fact_data:
        for i, a in enumerate(fd["gr"].tolist()):
            li = addr_to_leaf[a]
            init_stack[li] += fd["init"][i]
            counts[li] += 1
    counts = counts.clamp_min(1)
    init_stack = init_stack / counts.unsqueeze(-1)
    row_leaves = init_stack.detach().clone().to(tbl.embedding.weight.dtype).requires_grad_(True)

    grad_snap = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        tbl.embedding.weight.data[all_rows_t] = 0
    addr_to_leaf_t = torch.full((tbl.embedding.weight.size(0),), -1, dtype=torch.long, device=device)
    for a, li in addr_to_leaf.items():
        addr_to_leaf_t[a] = li

    def hook(mod, inputs, output):
        leaf_idx = addr_to_leaf_t[inputs[0]]
        mask = (leaf_idx >= 0)
        if not mask.any():
            return output
        out = output
        added = row_leaves[leaf_idx.clamp_min(0)].to(out.dtype)
        m = mask.unsqueeze(-1).to(out.dtype)
        return out * (1 - m) + added * m

    handle = tbl.embedding.register_forward_hook(hook)
    optim = torch.optim.Adam([row_leaves], lr=lr)
    try:
        for step in range(steps):
            i = torch.randint(0, len(fact_data), (1,)).item()
            fd = fact_data[i]
            x = torch.tensor([fd["ids"]], dtype=torch.long, device=device)
            logits = model(x)[0, -1, :]
            loss = F.cross_entropy(logits.unsqueeze(0).float(), torch.tensor([fd["gold_id"]], device=device))
            (g,) = torch.autograd.grad(loss, [row_leaves])
            if row_leaves.grad is None: row_leaves.grad = g.clone()
            else: row_leaves.grad.copy_(g)
            optim.step(); optim.zero_grad()
        # read back: for each fact, reconstruct its marker from row_leaves and project through W_V
        n_top1 = 0
        cos_final, cos_init = [], []
        with torch.no_grad():
            tbl.embedding.weight.data[all_rows_t] = row_leaves.detach().to(tbl.embedding.weight.dtype)
            for fd in fact_data:
                x = torch.tensor([fd["ids"]], dtype=torch.long, device=device)
                lg = model(x)[0, -1, :]
                if int((lg > lg[fd["gold_id"]]).sum().item()) == 0:
                    n_top1 += 1
                # marker for this fact = row_leaves at its global rows, in head order
                leaf_idx = [addr_to_leaf[a] for a in fd["gr"].tolist()]
                marker = row_leaves[torch.tensor(leaf_idx, device=device)]  # [total_heads, embed_dim]
                wv_row = layer_mod.value_proj(marker.flatten().to(layer_mod.value_proj.weight.dtype)).float()
                cos_final.append(cos(wv_row, fd["gold_unembed"]))
                init_marker = fd["init"]
                wv_init = layer_mod.value_proj(init_marker.flatten().to(layer_mod.value_proj.weight.dtype)).float()
                cos_init.append(cos(wv_init, fd["gold_unembed"]))
    finally:
        handle.remove()
        with torch.no_grad():
            tbl.embedding.weight.data[all_rows_t] = saved
        for p, rg in zip(model.parameters(), grad_snap):
            p.requires_grad_(rg)

    return {"n": len(fact_data), "top1": n_top1 / len(fact_data),
            "mean_cos_WVrow_to_gold": sum(cos_final) / len(cos_final),
            "mean_cos_init": sum(cos_init) / len(cos_init),
            "min_cos_WVrow_to_gold": min(cos_final)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--joint-facts", type=int, default=30)
    args = ap.parse_args()
    run(args.ckpt_dir, args.out, joint_facts=args.joint_facts)


if __name__ == "__main__":
    main()
