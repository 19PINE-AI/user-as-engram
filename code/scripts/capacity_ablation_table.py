"""Aggregate Engram-capacity ablation results into a benchmark table.

Reads `results/<tag>__strategies.json`, `results/<tag>__scale.json`,
`results/<tag>__locomo.json` for each cell and prints a unified table.

Cells: { d8 / d12 } x { 5 capacities } x { up to 3 token budgets }.
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

RES = Path(f"{UAE_ROOT}/results")

CAPACITIES = [
    ("tiny",   "5K x 64",    1.28),
    ("small",  "20K x 128", 10.24),
    ("medium", "50K x 128", 25.6),
    ("large",  "50K x 256", 51.2),
    ("xlarge", "100K x 256", 102.4),
]

# Full d8 × d12 capacity × tokens matrix (5 caps × 3 tokens × 2 dense = 30)
CELLS = []
# d8: tokens 0.5B, 1B, 2B
for cap, vocab, embed in [("tiny",5000,64),("small",20000,128),("medium",50000,128),
                            ("large",50000,256),("xlarge",100000,256)]:
    for tok_label, tok_b in [("t05B",0.5), ("t1B",1.0), ("t2B",2.0)]:
        CELLS.append((f"engram_d8_{cap}_{tok_label}", "d8@512", cap, tok_b))
# d12: tokens 0.5B, 1.32B, 2.5B
for cap, vocab, embed in [("tiny",5000,64),("small",20000,128),("medium",50000,128),
                            ("large",50000,256),("xlarge",100000,256)]:
    for tok_label, tok_b in [("t05B",0.5), ("t132B",1.32), ("t25B",2.5)]:
        CELLS.append((f"engram_d12_{cap}_{tok_label}", "d12@768", cap, tok_b))
# Legacy reference rows (different recipes — for context only)
CELLS.append(("engram_d8",  "d8@512 v1",  "small (v1, 11M Engram)", 0.524))
CELLS.append(("engram_d12", "d12@768 v1", "large (v1, 51M Engram)", 0.786))


def safe_load(p: Path):
    if not p.exists(): return None
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return None


def agg_strats(rows_list):
    out = {}
    if not rows_list: return out
    for r in rows_list:
        for sname, sv in (r.get("strategies") or {}).items():
            if sname not in out:
                out[sname] = {"n": 0, "t1": 0, "rk_imp": 0}
            out[sname]["n"] += 1
            if sv.get("post_insert_rank", 999) == 0: out[sname]["t1"] += 1
            if sv.get("rank_improvement", 0) > 0: out[sname]["rk_imp"] += 1
    for k, v in out.items():
        v["t1_frac"] = v["t1"] / max(v["n"], 1)
        v["rk_frac"] = v["rk_imp"] / max(v["n"], 1)
    return out


def agg_e1(rows_list, prefix):
    if not rows_list: return None
    n = len(rows_list); t1 = sum(1 for r in rows_list if r.get(f"{prefix}_rank") == 0)
    t5 = sum(1 for r in rows_list if r.get(f"{prefix}_rank", 999) < 5)
    return (t1/n, t5/n, n)


def main():
    rows_print = []
    for tag, dense, cap, tok_b in CELLS:
        s = safe_load(RES / f"{tag}__strategies.json")
        sc = safe_load(RES / f"{tag}__scale.json")
        l = safe_load(RES / f"{tag}__locomo.json")
        if s is None and sc is None and l is None:
            available = "MISSING"
        else:
            avail_parts = []
            if s: avail_parts.append("S")
            if sc: avail_parts.append("E1")
            if l: avail_parts.append("L")
            available = "+".join(avail_parts)

        cell = {
            "tag": tag, "dense": dense, "cap": cap, "tok_b": tok_b, "avail": available,
        }
        # Insertion OPT top-1
        if s:
            a = agg_strats(s.get("rows", []))
            cell["opt_t1"] = a.get("OPT", {}).get("t1_frac", None)
            cell["opt_rk"] = a.get("OPT", {}).get("rk_frac", None)
            cell["unembed_t1"] = a.get("UNEMBED_P", {}).get("t1_frac", None)
        # E1 USER OPT
        if sc:
            r = agg_e1(sc.get("e1_user", []), "opt")
            if r: cell["user_opt_t1"], cell["user_opt_t5"], _ = r
            r = agg_e1(sc.get("e1_user", []), "icl")
            if r: cell["user_icl_t1"], _, _ = r
            r = agg_e1(sc.get("e1_org", []), "opt")
            if r: cell["org_opt_t1"], cell["org_opt_t5"], _ = r
        # LOCOMO summary
        if l and "summary" in l:
            cell["locomo_jopt"] = l["summary"].get("USER_AS_ENGRAM_JOINT_OPT")
            cell["locomo_memmachine"] = l["summary"].get("MEMMACHINE_LIKE")
        rows_print.append(cell)

    # Print a unified compact table
    print("=" * 120)
    print("ENGRAM CAPACITY × TOKEN BUDGET BENCHMARK TABLE")
    print("=" * 120)
    h = (f"{'dense':10s}  {'cap':10s}  {'tok B':>5s}  {'avail':>6s}  "
         f"{'ins-OPT t1':>10s} {'rk':>5s}  "
         f"{'USER OPT':>10s} {'ORG OPT':>10s}  "
         f"{'LOCOMO J':>9s} {'MEMMACH':>9s}")
    print(h); print("-"*120)
    for c in rows_print:
        cells = [
            f"{c['dense']:10s}", f"{c['cap']:10s}", f"{c['tok_b']:5.2f}", f"{c['avail']:>6s}",
            (f"{c['opt_t1']:>10.3f}" if c.get('opt_t1') is not None else f"{'-':>10s}"),
            (f"{c['opt_rk']:>5.3f}"  if c.get('opt_rk') is not None else f"{'-':>5s}"),
            (f"{c['user_opt_t1']:>4.2f}/{c['user_opt_t5']:>4.2f}" if c.get('user_opt_t1') is not None else f"{'-':>10s}"),
            (f"{c['org_opt_t1']:>4.2f}/{c['org_opt_t5']:>4.2f}" if c.get('org_opt_t1') is not None else f"{'-':>10s}"),
            (f"{c['locomo_jopt']:>9.3f}" if c.get('locomo_jopt') is not None else f"{'-':>9s}"),
            (f"{c['locomo_memmachine']:>9.3f}" if c.get('locomo_memmachine') is not None else f"{'-':>9s}"),
        ]
        print("  ".join(cells))
    print("-"*120)

    # Pick best cell by composite (max LOCOMO J or USER OPT t1)
    print("\nBest cells per dense size by USER OPT top-1:")
    by_dense = defaultdict(list)
    for c in rows_print:
        if c.get("user_opt_t1") is not None:
            by_dense[c["dense"]].append(c)
    for dense, cells in by_dense.items():
        best = max(cells, key=lambda x: x["user_opt_t1"] or 0)
        print(f"  {dense}: cap={best['cap']:>10s}  tokens={best['tok_b']:>5.2f}B  USER OPT t1={best['user_opt_t1']:.3f} (tag {best['tag']})")
    print("\nBest cells per dense size by LOCOMO Joint OPT F1:")
    by_dense = defaultdict(list)
    for c in rows_print:
        if c.get("locomo_jopt") is not None:
            by_dense[c["dense"]].append(c)
    for dense, cells in by_dense.items():
        best = max(cells, key=lambda x: x["locomo_jopt"] or 0)
        print(f"  {dense}: cap={best['cap']:>10s}  tokens={best['tok_b']:>5.2f}B  LOCOMO Jopt F1={best['locomo_jopt']:.3f} (tag {best['tag']})")


if __name__ == "__main__":
    main()
