"""Aggregate fact-count scaling results into a table.

Reads `results/<tag>__factscale_n<N>.json` for each model and N, prints
USER OPT top-1/top-5 (and ORG OPT) at each fact count. Goal: identify
the recall ceiling and the rate of decay as fact count grows.
"""
from __future__ import annotations
import json
from pathlib import Path

RES = Path("/home/ubuntu/user-as-engram/results")

MODELS = [
    ("engram_d8",                     "d8 v1",          137),
    ("engram_d8_v2",                  "d8 v2 (lg/0.5B)", 178),
    ("engram_d8_large_t2B",           "d8 lg/2B (best d8)", 178),
    ("engram_d12",                    "d12 v1",         339),
    ("engram_d12_v2",                 "d12 v2 (lg/1.32B)", 339),
    ("engram_d12_large_t25B",         "d12 lg/2.5B (best d12)", 339),
    ("engram_d12_w1280_optimal",      "d12@1280 optimal", 625),
    ("engram_d20_w1536_optimal",      "d20@1536 optimal", 1224),
]

N_FACTS = [100, 200, 500, 1000]


def load(p):
    if not p.exists(): return None
    try: return json.load(open(p))
    except Exception: return None


def agg(rows, prefix):
    if not rows: return None
    n = len(rows)
    t1 = sum(1 for r in rows if r.get(f"{prefix}_rank") == 0) / n
    t5 = sum(1 for r in rows if r.get(f"{prefix}_rank", 999) < 5) / n
    return (t1, t5, n)


def main():
    print("=" * 130)
    print("FACT-COUNT SCALING — USER OPT (E1, per-fact recall)")
    print("=" * 130)
    h = f"{'model':35s} {'params':>7s}  " + " ".join(
        f"{'n=' + str(n):>14s}" for n in N_FACTS) + "  ICL@max"
    print(h); print("-" * 130)
    for tag, label, params in MODELS:
        cells = [f"{label:35s}", f"{params:>5d}M  "]
        icl_at_max = "-"
        for n in N_FACTS:
            d = load(RES / f"{tag}__factscale_n{n}.json")
            if d is None:
                cells.append(f"{'-':>14s}"); continue
            r = agg(d.get("e1_user", []), "opt")
            if r is None:
                cells.append(f"{'-':>14s}"); continue
            t1, t5, n_actual = r
            cells.append(f"{t1:>4.2f}/{t5:>4.2f} (n{n_actual})")
            if n == N_FACTS[-1]:
                icl = agg(d.get("e1_user", []), "icl")
                if icl: icl_at_max = f"{icl[0]:.2f}/{icl[1]:.2f}"
        cells.append(f"  {icl_at_max}")
        print("".join(cells))
    print("-" * 130)
    print("\nFACT-COUNT SCALING — ORG OPT")
    print("-" * 130)
    print(h); print("-" * 130)
    for tag, label, params in MODELS:
        cells = [f"{label:35s}", f"{params:>5d}M  "]
        icl_at_max = "-"
        for n in N_FACTS:
            d = load(RES / f"{tag}__factscale_n{n}.json")
            if d is None:
                cells.append(f"{'-':>14s}"); continue
            r = agg(d.get("e1_org", []), "opt")
            if r is None:
                cells.append(f"{'-':>14s}"); continue
            t1, t5, n_actual = r
            cells.append(f"{t1:>4.2f}/{t5:>4.2f} (n{n_actual})")
            if n == N_FACTS[-1]:
                icl = agg(d.get("e1_org", []), "icl")
                if icl: icl_at_max = f"{icl[0]:.2f}/{icl[1]:.2f}"
        cells.append(f"  {icl_at_max}")
        print("".join(cells))


if __name__ == "__main__":
    main()
