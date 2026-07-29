"""Aggregate insertion / scale / locomo results across Mini-Engram sizes.

Usage:
  python -m scripts.scaling_summary > results/scaling_table.txt

Reads:
  results/strategies.json (or results/{tag}__strategies.json) — original d12@768 + new sizes
  results/scale_eval.json (or results/{tag}__scale.json) — original d12@768 + new sizes
  results/locomo_eval.json (or results/{tag}__locomo.json) — original d12@768 + new sizes

Produces the paper's side-by-side dense-scaling summary.
"""
from __future__ import annotations
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import json
from pathlib import Path

RES = Path(f"{UAE_ROOT}/results")

MODELS = [
    ("d8@512 v2",  178, "(150M, 0.50B tok)",
     RES / "engram_d8_v2__strategies.json",
     RES / "engram_d8_v2__scale.json",
     RES / "engram_d8_v2__locomo.json"),
    ("d12@768 v2", 339, "(340M, 1.32B tok)",
     RES / "engram_d12_v2__strategies.json",
     RES / "engram_d12_v2__scale.json",
     RES / "engram_d12_v2__locomo.json"),
    ("d12@1280 v2", 625, "(625M, 3.33B tok)",
     RES / "engram_d12_w1280__strategies.json",
     RES / "engram_d12_w1280__scale.json",
     RES / "engram_d12_w1280__locomo.json"),
    ("d20@1536 v2", 1224, "(1.22B, 7.40B tok)",
     RES / "engram_d20_w1536__strategies.json",
     RES / "engram_d20_w1536__scale.json",
     RES / "engram_d20_w1536__locomo.json"),
]


def load(p):
    if not p.exists(): return None
    with open(p) as f: return json.load(f)


def get(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None: return default
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def main():
    print("=" * 80)
    print("Mini-Engram scaling summary")
    print("=" * 80)
    rows = []
    for tag, n_params, note, sp, scp, lp in MODELS:
        s, sc, l = load(sp), load(scp), load(lp)
        rows.append({
            "tag": tag, "n_params": n_params, "note": note,
            "strategies": s, "scale": sc, "locomo": l,
        })
        avail = ' '.join([
            'strategies' if s else '-',
            'scale' if sc else '-',
            'locomo' if l else '-',
        ])
        print(f"  {tag:12s}  ({n_params:>4d}M params) {note:14s}  available: {avail}")
    print()

    # Insertion-strategies summary (16 facts × {RANDOM, WTE, UNEMBED_P, [OPT]})
    def agg_strats(rows_list):
        """rows_list: list of fact rows with 'strategies' subdict.
        Use post_insert_rank (most meaningful field)."""
        agg = {}
        for r in rows_list:
            for sname, sv in (r.get("strategies") or {}).items():
                if sname not in agg:
                    agg[sname] = {"n": 0, "t1": 0, "t5": 0, "rank_imp_pos": 0,
                                    "delta_logit_sum": 0.0}
                agg[sname]["n"] += 1
                rk = sv.get("post_insert_rank", 999)
                if rk == 0: agg[sname]["t1"] += 1
                if rk < 5: agg[sname]["t5"] += 1
                if sv.get("rank_improvement", 0) > 0: agg[sname]["rank_imp_pos"] += 1
                agg[sname]["delta_logit_sum"] += float(sv.get("delta_logit", 0.0))
        out = {}
        for k, v in agg.items():
            out[k] = {"top1": v["t1"]/v["n"], "top5": v["t5"]/v["n"],
                       "rank_imp_frac": v["rank_imp_pos"]/v["n"],
                       "avg_delta_logit": v["delta_logit_sum"]/v["n"], "n": v["n"]}
        return out

    print("--- Insertion strategy: post-insert rank == 0 (top-1 hit) ---")
    print(f"{'model':12s} {'RANDOM':>10s} {'WTE':>10s} {'UNEMBED_P':>10s} {'OPT':>10s}")
    for r in rows:
        s = r["strategies"]
        if not s: continue
        a = agg_strats(s.get("rows", []))
        cells = []
        for strat in ["RANDOM", "WTE", "UNEMBED_P", "OPT"]:
            if strat in a:
                cells.append(f"{a[strat]['top1']:>10.3f}")
            else:
                cells.append(f"{'-':>10s}")
        print(f"{r['tag']:12s} " + " ".join(cells))
    print("--- Insertion strategy: rank-improvement frac (vs baseline) ---")
    print(f"{'model':12s} {'RANDOM':>10s} {'WTE':>10s} {'UNEMBED_P':>10s} {'OPT':>10s}")
    for r in rows:
        s = r["strategies"]
        if not s: continue
        a = agg_strats(s.get("rows", []))
        cells = []
        for strat in ["RANDOM", "WTE", "UNEMBED_P", "OPT"]:
            if strat in a:
                cells.append(f"{a[strat]['rank_imp_frac']:>10.3f}")
            else:
                cells.append(f"{'-':>10s}")
        print(f"{r['tag']:12s} " + " ".join(cells))
    print()

    # eval_at_scale E1 summary (100 user + 100 org facts)
    def agg_e1(rows_list, prefix):
        """rows_list: list of fact dicts; prefix: 'opt' | 'icl' | 'unembed' | 'baseline'"""
        n = 0; t1 = 0; t5 = 0
        for r in rows_list:
            rk = r.get(f"{prefix}_rank")
            if rk is None: continue
            n += 1
            if rk == 0: t1 += 1
            if rk < 5: t5 += 1
        return (t1/n, t5/n, n) if n else (None, None, 0)

    print("--- E1 scale: top-1 / top-5 (per fact, 100 USER + 100 ORG) ---")
    print(f"{'model':12s} {'USER OPT t1/t5':>16s} {'ORG OPT t1/t5':>16s} {'USER ICL t1/t5':>16s} {'ORG ICL t1/t5':>16s}")
    for r in rows:
        sc = r["scale"]
        if not sc: continue
        cells = []
        for key in [("e1_user", "opt"), ("e1_org", "opt"),
                     ("e1_user", "icl"), ("e1_org", "icl")]:
            lst = sc.get(key[0], [])
            t1, t5, n = agg_e1(lst, key[1])
            if t1 is None:
                cells.append(f"{'-':>16s}")
            else:
                cells.append(f"{t1:>6.3f}/{t5:>6.3f} (n{n})")
        print(f"{r['tag']:12s} " + " ".join(cells))
    print()

    # LOCOMO summary
    print("--- LOCOMO single-hop, token-F1 (160 QAs across 2 conv) ---")
    headers = ["NO_MEMORY", "MEM0_LIKE", "MEMMACHINE_LIKE", "USER_AS_ENGRAM_OPT",
               "USER_AS_ENGRAM_JOINT_OPT"]
    print(f"{'model':12s} " + " ".join(f"{h[:14]:>14s}" for h in headers))
    for r in rows:
        l = r["locomo"]
        if not l: continue
        cells = [f"{(l['summary'].get(h, 0) or 0):>14.3f}" for h in headers]
        print(f"{r['tag']:12s} " + " ".join(cells))
    print()


if __name__ == "__main__":
    main()
