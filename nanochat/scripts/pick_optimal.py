"""Read the matrix-ablation results and pick the optimal (capacity, t/p)
per dense size, then extrapolate to d12@1280 and d20@1536.

Outputs a `results/optimal_config.json` with fields:
  best_d12 = {cap_label, vocab, n_embed, tokens_b, t_per_p,
                opt_t1, locomo_jopt, ...}
  d12_1280 = recommended config (extrapolated from best_d12)
  d20_1536 = recommended config (extrapolated from best_d12)

Composite ranking score = USER_OPT_t1 + 5 * LOCOMO_Joint_OPT_F1.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict

RES = Path("/home/ubuntu/user-as-engram/results")

# capacity → (vocab, n_embed_total, total_engram_params_d12_estimate)
CAP_DEFS = {
    "tiny":   (5000,   64),
    "small":  (20000, 128),
    "medium": (50000, 128),
    "large":  (50000, 256),
    "xlarge": (100000, 256),
}

# (tag, dense, cap, tokens_b, num_iters_at_131K)
def gen_cells():
    for cap in CAP_DEFS:
        for tok_label, tok_b in [("t05B",0.5), ("t1B",1.0), ("t2B",2.0)]:
            yield (f"engram_d8_{cap}_{tok_label}", "d8", cap, tok_b)
        for tok_label, tok_b in [("t05B",0.5), ("t132B",1.32), ("t25B",2.5)]:
            yield (f"engram_d12_{cap}_{tok_label}", "d12", cap, tok_b)


def safe_load(p: Path):
    if not p.exists(): return None
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return None


def cell_score(tag):
    """Composite score for ranking cells. Returns None if data missing."""
    sc = safe_load(RES / f"{tag}__scale.json")
    l  = safe_load(RES / f"{tag}__locomo.json")
    if sc is None and l is None: return None
    score = 0.0
    metrics = {}
    if sc:
        u = sc.get("e1_user", [])
        if u:
            t1 = sum(1 for r in u if r.get("opt_rank") == 0) / len(u)
            t5 = sum(1 for r in u if r.get("opt_rank", 999) < 5) / len(u)
            metrics["user_opt_t1"] = t1
            metrics["user_opt_t5"] = t5
            score += t1
        o = sc.get("e1_org", [])
        if o:
            t1 = sum(1 for r in o if r.get("opt_rank") == 0) / len(o)
            metrics["org_opt_t1"] = t1
            score += 0.5 * t1
    if l and "summary" in l:
        j = l["summary"].get("USER_AS_ENGRAM_JOINT_OPT", 0)
        metrics["locomo_jopt"] = j
        score += 5 * j
    return score, metrics


def main():
    by_dense = defaultdict(list)
    for tag, dense, cap, tok_b in gen_cells():
        s = cell_score(tag)
        if s is None: continue
        score, metrics = s
        by_dense[dense].append({
            "tag": tag, "cap": cap, "tok_b": tok_b,
            "score": score, "metrics": metrics,
        })

    out = {}
    for dense in ["d8", "d12"]:
        cells = sorted(by_dense[dense], key=lambda c: -c["score"])
        if not cells:
            print(f"NO DATA for {dense}")
            continue
        best = cells[0]
        vocab, n_embed = CAP_DEFS[best["cap"]]
        out[f"best_{dense}"] = {
            "cap_label": best["cap"], "vocab": vocab, "n_embed": n_embed,
            "tokens_b": best["tok_b"],
            "score": best["score"],
            "metrics": best["metrics"],
            "tag": best["tag"],
        }
        # All cells for that dense (for visibility)
        out[f"all_{dense}"] = [
            {"tag": c["tag"], "cap": c["cap"], "tok_b": c["tok_b"],
             "score": c["score"]} for c in cells[:5]
        ]
        print(f"\n=== {dense} top 5 (composite score) ===")
        for c in cells[:5]:
            print(f"  {c['tag']:40s}  score={c['score']:.3f}  metrics={c['metrics']}")

    # Extrapolation rule: for the bigger sizes, use d12's optimal config
    # because d12 is the closest dense to d12@1280 and d20@1536.
    if "best_d12" in out:
        d12opt = out["best_d12"]
        # Token-per-param target: tokens_b / scaling_params(d12@768) -- take this as the t/p target
        # scaling-params of d12@768 = 110M
        t_per_p_d12 = d12opt["tokens_b"] * 1e9 / 110e6  # tokens per scaling-param
        out["t_per_p_target"] = t_per_p_d12
        # Recommended for d12@1280 (scaling-params = 278M)
        out["d12_1280"] = {
            "cap_label": d12opt["cap_label"],
            "vocab": d12opt["vocab"],
            "n_embed": d12opt["n_embed"],
            "scaling_params_M": 278,
            "tokens_b": round(t_per_p_d12 * 278e6 / 1e9, 2),
            "iters_at_131K": int(round(t_per_p_d12 * 278e6 / 131072)),
        }
        # Recommended for d20@1536 (scaling-params = 617M)
        out["d20_1536"] = {
            "cap_label": d12opt["cap_label"],
            "vocab": d12opt["vocab"],
            "n_embed": d12opt["n_embed"],
            "scaling_params_M": 617,
            "tokens_b": round(t_per_p_d12 * 617e6 / 1e9, 2),
            "iters_at_131K": int(round(t_per_p_d12 * 617e6 / 131072)),
        }

        print(f"\n=== EXTRAPOLATION (d12 optimal at {t_per_p_d12:.1f} t/p × scaling-params) ===")
        for k in ["d12_1280", "d20_1536"]:
            v = out[k]
            print(f"  {k}: cap={v['cap_label']}  tokens={v['tokens_b']:.2f}B  iters={v['iters_at_131K']}")

    out_path = RES / "optimal_config.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
