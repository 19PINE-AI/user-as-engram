"""
Comprehensive scalability benchmark for User-as-Engram.

Three sweeps on Mini-Engram-d12:
  B1) Single-user fact-density curve: 10, 30, 100, 300, 1000 facts written
      simultaneously into one user's override map. Measure per-fact recall.
  B2) Multi-domain additive composition: 1, 2, 3, 5 domains × 100 facts each.
      Each domain is an independent override map. Apply ALL domains
      simultaneously, measure per-domain recall.
  B3) Storage and cost analysis (analytical, written to results).

Each fact is OPT-15 with default scale.

Usage:
  python -m scripts.scalability_benchmark --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, json, time, argparse
from pathlib import Path
import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows,
    make_marker_OPT,
)


def build_overrides(model, eng, last_layer, Wv_pinv, total_heads, embed_dim, tokenizer,
                     facts, device, scale=20.0, opt_steps=15, opt_lr=0.5):
    bos = tokenizer.get_bos_token_id()
    overrides = []
    for fact in facts:
        ids = tokenizer.encode(fact["prompt"], prepend=bos)
        gold_id = tokenizer.encode(fact["gold"])[0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        global_rows = trigger_global_rows(eng, idx, last_layer, trig_pos, user_salt=0)
        marker = make_marker_OPT(model, eng, last_layer, gold_id, idx, trig_pos, scale,
                                  total_heads, embed_dim, Wv_pinv=Wv_pinv,
                                  n_steps=opt_steps, lr=opt_lr)
        overrides.append({"fact": fact, "prompt_ids": ids, "gold_id": gold_id,
                           "global_rows": global_rows, "marker": marker})
    return overrides


def apply_overrides(eng, last_layer, overrides_lists):
    tbl = eng.tables[str(last_layer)]
    saved = []
    for ovs in overrides_lists:
        per = []
        for o in ovs:
            originals = tbl.embedding.weight.data[o["global_rows"]].clone()
            per.append(originals)
            write_marker(eng, last_layer, o["global_rows"], o["marker"])
        saved.append(per)
    return saved


def restore_overrides(eng, last_layer, overrides_lists, saved):
    for ovs, per in zip(reversed(overrides_lists), reversed(saved)):
        for o, originals in zip(ovs, per):
            restore_rows(eng, last_layer, o["global_rows"], originals)


@torch.no_grad()
def eval_recall(model, overrides, device):
    n_top1 = 0; n_top5 = 0
    for o in overrides:
        idx = torch.tensor([o["prompt_ids"]], dtype=torch.long, device=device)
        logits = model(idx)[0, -1, :]
        rank = int((logits > logits[o["gold_id"]]).sum().item())
        if rank == 0: n_top1 += 1
        if rank < 5: n_top5 += 1
    return n_top1, n_top5, len(overrides)


def address_overlap(overrides_lists):
    """Returns (pairwise_collisions, total_unique_addresses)"""
    addrs_per_list = []
    for ovs in overrides_lists:
        s = set()
        for o in ovs:
            for a in o["global_rows"].tolist():
                s.add(a)
        addrs_per_list.append(s)
    union = set()
    for s in addrs_per_list:
        union |= s
    if len(addrs_per_list) >= 2:
        # Pairwise overlap fraction (normalized by smaller set)
        pair_overlap = []
        for i in range(len(addrs_per_list)):
            for j in range(i+1, len(addrs_per_list)):
                a, b = addrs_per_list[i], addrs_per_list[j]
                inter = len(a & b)
                pair_overlap.append(inter / max(min(len(a), len(b)), 1))
        return sum(pair_overlap)/len(pair_overlap), len(union)
    return 0, len(union)


def b1_single_user_density(model, eng, last_layer, Wv_pinv, total_heads, embed_dim, tokenizer,
                            corpora, device, args):
    """Single-user, vary N facts. Returns list of (N, top1, top5)."""
    print("\n=== B1: Single-user fact-density curve ===")
    sizes = [10, 30, 100, 300, 1000]
    rows = []
    user_facts_pool = corpora["user_facts"]
    for n in sizes:
        if n > len(user_facts_pool):
            print(f"  Skipping N={n} (only {len(user_facts_pool)} USER facts available)")
            continue
        # Take first n facts (deterministic)
        # Dedupe by trigger to get exactly n unique
        seen = {}
        for f in user_facts_pool:
            seen[f["trigger"]] = f
            if len(seen) >= n: break
        facts = list(seen.values())[:n]
        if len(facts) < n:
            print(f"  Only {len(facts)} unique triggers available for n={n}")

        t0 = time.time()
        ovs = build_overrides(model, eng, last_layer, Wv_pinv, total_heads, embed_dim, tokenizer,
                                facts, device, args.scale, args.opt_steps, args.opt_lr)
        build_t = time.time() - t0
        # Apply all
        saved = apply_overrides(eng, last_layer, [ovs])
        try:
            top1, top5, total = eval_recall(model, ovs, device)
        finally:
            restore_overrides(eng, last_layer, [ovs], saved)
        print(f"  N={n:5d}  top1={top1}/{total}={top1/total:.1%}  top5={top5}/{total}={top5/total:.1%}  "
              f"build_time={build_t:.0f}s ({build_t/total*1000:.0f}ms/fact)")
        rows.append({"n": n, "top1": top1/total, "top5": top5/total, "build_time_s": build_t})
    return rows


def b2_multi_domain_additive(model, eng, last_layer, Wv_pinv, total_heads, embed_dim, tokenizer,
                              corpora, device, args, n_facts_per_domain=100):
    """Multi-domain, vary D domains. Each domain has n_facts_per_domain facts.
    Apply ALL D domains simultaneously, measure per-domain recall.
    Domain 1 = USER, domain 2 = ORG (different schemas), domain 3+ = different
    user partitions of the multi-user corpus."""
    print(f"\n=== B2: Multi-domain additive composition (n_facts_per_domain={n_facts_per_domain}) ===")
    domains = []
    # Build domain pools
    user_pool = corpora["user_facts"]
    org_pool = corpora["org_facts"]
    multi_users = corpora["multi_users"]
    domains_data = []
    # Domain 1: first n USER facts
    seen = {}
    for f in user_pool:
        seen[f["trigger"]] = f
        if len(seen) >= n_facts_per_domain: break
    domains_data.append(("user_v1", list(seen.values())[:n_facts_per_domain]))
    # Domain 2: first n ORG facts
    seen = {}
    for f in org_pool:
        seen[f["trigger"]] = f
        if len(seen) >= n_facts_per_domain: break
    domains_data.append(("org_v1", list(seen.values())[:n_facts_per_domain]))
    # Domain 3: another USER subset (different facts/values)
    seen = {}
    for f in user_pool[n_facts_per_domain:]:
        seen[f["trigger"]] = f
        if len(seen) >= n_facts_per_domain: break
    if len(seen) >= n_facts_per_domain:
        domains_data.append(("user_v2", list(seen.values())[:n_facts_per_domain]))
    # Domain 4: another ORG subset
    seen = {}
    for f in org_pool[n_facts_per_domain:]:
        seen[f["trigger"]] = f
        if len(seen) >= n_facts_per_domain: break
    if len(seen) >= n_facts_per_domain:
        domains_data.append(("org_v2", list(seen.values())[:n_facts_per_domain]))
    # Domain 5: from multi_users
    if multi_users:
        domains_data.append(("multi_user_0", multi_users[0]["facts"][:n_facts_per_domain]))

    print(f"  Built {len(domains_data)} domain pools.")

    # Build overrides for each domain
    print("  Training overrides per domain...")
    domain_overrides = []
    for name, facts in domains_data:
        t0 = time.time()
        ovs = build_overrides(model, eng, last_layer, Wv_pinv, total_heads, embed_dim, tokenizer,
                                facts, device, args.scale, args.opt_steps, args.opt_lr)
        domain_overrides.append((name, ovs))
        print(f"    {name}: {len(ovs)} facts in {time.time()-t0:.0f}s")

    # Test each "stack size" D = 1, 2, 3, ...
    rows = []
    n_domains = len(domain_overrides)
    for D in range(1, n_domains + 1):
        active = domain_overrides[:D]
        ovs_lists = [ovs for _, ovs in active]
        # Address overlap
        avg_pair_overlap, n_addrs = address_overlap(ovs_lists)

        saved = apply_overrides(eng, last_layer, ovs_lists)
        try:
            per_domain = []
            for name, ovs in active:
                top1, top5, total = eval_recall(model, ovs, device)
                per_domain.append({"name": name, "top1": top1/total, "top5": top5/total, "n": total})
        finally:
            restore_overrides(eng, last_layer, ovs_lists, saved)
        avg_top1 = sum(d["top1"] for d in per_domain) / len(per_domain)
        avg_top5 = sum(d["top5"] for d in per_domain) / len(per_domain)
        print(f"  D={D}  avg_top1={avg_top1:.1%}  avg_top5={avg_top5:.1%}  "
              f"avg pair-overlap={avg_pair_overlap:.4f}  total_addrs={n_addrs}")
        for d in per_domain:
            print(f"      {d['name']:15s}  top1={d['top1']:.1%}  top5={d['top5']:.1%}")
        rows.append({"D": D, "per_domain": per_domain,
                     "avg_top1": avg_top1, "avg_top5": avg_top5,
                     "avg_pair_overlap": avg_pair_overlap, "n_addrs": n_addrs})
    return rows


def b3_cost_analysis(config):
    """Analytical: per-fact and per-user storage and training cost.
    Compares Engram override (rank-N row), SFT-LoRA (rank-r on Q/K/V),
    POLAR-style per-user LoRA training."""
    embed_dim = 16  # row dim
    n_heads = 8
    n_ngram_orders = 2
    n_layers_engram = 1  # we insert at one Engram layer
    # Engram row write per fact: heads * ngram_orders * embed_dim
    engram_row_floats = n_heads * n_ngram_orders * embed_dim  # 256 for d12

    # SFT-LoRA per-fact (rank 8 on Q/K/V across all 12 transformer layers)
    rank = 8
    n_embd = 768
    n_layer_total = 12
    n_proj_per_layer = 3  # Q, K, V
    sft_lora_floats = rank * (n_embd + n_embd) * n_proj_per_layer * n_layer_total

    # POLAR LoRA per-user (rank 64 on Q/K/V/O + MLP, all layers, single LoRA per user)
    polar_rank = 64
    polar_lora_floats = polar_rank * (n_embd + n_embd) * 4 * n_layer_total \
                         + polar_rank * (n_embd + 4*n_embd) * 2 * n_layer_total
    bytes_per_float = 4

    print("\n=== B3: Cost analysis (Mini-Engram-d12 architecture) ===")
    print(f"  Engram override row size: {engram_row_floats} floats = {engram_row_floats*bytes_per_float} bytes")
    print(f"  SFT-LoRA per-fact size:   {sft_lora_floats} floats = {sft_lora_floats*bytes_per_float/1024:.1f} KB")
    print(f"  POLAR LoRA per-user size: {polar_lora_floats} floats = {polar_lora_floats*bytes_per_float/1024/1024:.2f} MB")

    print("\n  Storage at deployment scale:")
    for n_users in (10, 100, 1000, 10000):
        for n_facts in (10, 100, 1000):
            engram_total = n_users * n_facts * engram_row_floats * bytes_per_float
            sft_total = n_users * n_facts * sft_lora_floats * bytes_per_float
            polar_total = n_users * polar_lora_floats * bytes_per_float
            print(f"    {n_users:5d} users × {n_facts:4d} facts → "
                  f"Engram {engram_total/1024/1024:7.1f} MB | "
                  f"SFT-LoRA {sft_total/1024/1024:9.1f} MB | "
                  f"POLAR per-user {polar_total/1024/1024:7.1f} MB")

    print("\n  Per-fact training time (single Blackwell, Mini-Engram-d12):")
    print(f"    UNEMBED_P:          < 1 ms  (1 mat-vec)")
    print(f"    User-as-Engram OPT: ~1 s    (15 fwd+bwd on the model)")
    print(f"    SFT-LoRA (r=8, 30 steps): ~0.65 s  (30 fwd+bwd)")
    print(f"    POLAR LoRA (full per-user): ~8 GPU-min on a comparable model (50 facts together)")
    return {
        "engram_row_floats": engram_row_floats,
        "sft_lora_floats": sft_lora_floats,
        "polar_lora_floats": polar_lora_floats,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--corpus", default=f"{UAE_ROOT}/data/corpora_xl.json")
    p.add_argument("--out", default=f"{UAE_ROOT}/results/scalability_benchmark.json")
    p.add_argument("--scale", type=float, default=20.0)
    p.add_argument("--opt-steps", type=int, default=15)
    p.add_argument("--opt-lr", type=float, default=0.5)
    p.add_argument("--skip-b1", action="store_true")
    p.add_argument("--skip-b2", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    Wv_pinv = torch.linalg.pinv(eng.layers_module[str(last_layer)].value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)

    with open(args.corpus) as f:
        corpora = json.load(f)

    out = {"config": vars(args)}
    if not args.skip_b1:
        out["B1"] = b1_single_user_density(model, eng, last_layer, Wv_pinv, total_heads, embed_dim,
                                              tokenizer, corpora, device, args)
    if not args.skip_b2:
        out["B2"] = b2_multi_domain_additive(model, eng, last_layer, Wv_pinv, total_heads, embed_dim,
                                                tokenizer, corpora, device, args)
    out["B3"] = b3_cost_analysis(config)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
