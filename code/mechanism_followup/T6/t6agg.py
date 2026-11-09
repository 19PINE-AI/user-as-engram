"""Aggregate T6 runs.jsonl -> tables.md, best.json, summary.json."""
import json, os, sys, statistics as st
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/clean-tests/results/T6")
rows = [json.loads(l) for l in open(os.path.join(D, "runs.jsonl"))]
setup = json.load(open(os.path.join(D, "setup.json")))

def m(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")

def summarize(rs):
    return {
        "n": len(rs),
        "direct": m([r["direct"] for r in rs]),
        "para": m([r["para"] for r in rs]),
        "twohop": m([r["twohop"] for r in rs]),
        "bleed_p": m([r["other_subject_target_p_edit"] for r in rs]),
        "bleed_p_base": m([r["other_subject_target_p_base"] for r in rs]),
        "other_kl": m([r["other_subject_kl"] for r in rs]),
        "neighbor_kl": m([r["neighbor_kl"] for r in rs]),
        "text_kl": m([r["text"]["kl_per_tok"] for r in rs]),
        "text_dloss": m([r["text"]["dloss_per_tok"] for r in rs]),
        "resid_frac": m([r["text"].get("resid_frac_rel_gt_1pct") for r in rs]),
        "resid_dnorm": m([r["text"].get("resid_dnorm_mean") for r in rs]),
        "fire10": m([(r.get("firing") or {}).get("frac_tokens_gt10pct_trigger") for r in rs]),
        "steps": m([r["fit"].get("steps") for r in rs]),
        "n_params": m([r["n_params"] for r in rs]),
    }

COLS = ["n", "direct", "para", "twohop", "bleed_p", "other_kl", "neighbor_kl", "text_kl", "text_dloss",
        "resid_frac", "fire10", "steps", "n_params"]

def fmt(v):
    if isinstance(v, int):
        return str(v)
    if v != v:
        return "–"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) < 0.01 and v != 0:
        return f"{v:.1e}"
    return f"{v:.3f}"

def table(title, groups, keyname):
    out = [f"### {title}", "", "| " + keyname + " | " + " | ".join(COLS) + " |",
           "|" + "---|" * (len(COLS) + 1)]
    for k, s in groups:
        out.append("| " + k + " | " + " | ".join(fmt(s[c]) for c in COLS) + " |")
    return "\n".join(out) + "\n"

md = ["# T6 tables", "", "Base model: " + json.dumps(setup["base"]) +
      f"; cities kept by 1-hop filter: {len(setup['cities_kept'])}/{setup['n_cities_total']}", ""]
md.append("Columns: direct/para/twohop = greedy full-target accuracy; bleed_p = p(first token of edited city) "
          "on 40 non-edited subjects with the same relation (base ≈ bleed_p_base); other_kl / neighbor_kl = "
          "next-token KL(base||edit) on other-subject / same-subject-other-relation prompts; text_kl, text_dloss = "
          "per-token KL and loss change on wikitext-103 test; resid_frac = fraction of text tokens whose residual "
          "at the edited layer changes by >1% (relative norm); fire10 = fraction of text tokens where the unit's "
          "|activation| exceeds 10% of its trigger activation.\n")

summary = {"base": setup["base"]}
# sweep
sw = defaultdict(list)
for r in rows:
    if r["key"].startswith("sweep|"):
        sw[(r["method"], r["layer"])].append(r)
best = {}
if sw:
    methods = sorted({k[0] for k in sw})
    for meth in methods:
        g = sorted([(f"L{l}" if l is not None else "all", summarize(v)) for (mm, l), v in sw.items() if mm == meth],
                   key=lambda x: (x[0] == "all", int(x[0][1:]) if x[0] != "all" else 0))
        md.append(table(f"Sweep: {meth} (single-fact edits)", g, "layer"))
        cand = [(k, s) for k, s in g if k != "all"]
        if cand:
            k, s = max(cand, key=lambda x: (round((x[1]["direct"] + x[1]["para"]) / 2, 3), -x[1]["bleed_p"], -x[1]["text_kl"]))
            best[meth] = int(k[1:])
        summary.setdefault("sweep", {})[meth] = {k: s for k, s in g}
    json.dump(best, open(os.path.join(D, "best.json"), "w"), indent=1)
    md.append("Best layer per method (max mean(direct, para); ties → lower bleed, lower text KL): `" + json.dumps(best) + "`\n")

# single
sg = defaultdict(list)
for r in rows:
    if r["key"].startswith("single|"):
        sg[(r["method"], r["layer"])].append(r)
if sg:
    g = [(f"{mm} @ {('L' + str(l)) if l is not None else 'all'}", summarize(v)) for (mm, l), v in sorted(sg.items(), key=lambda x: x[0][0])]
    md.append(table("Single-fact edits at each method's best layer (50 facts; text KL on 100k tokens for first 10)", g, "method"))
    summary["single"] = {k: s for k, s in g}

# multi
mg = defaultdict(list)
for r in rows:
    if r["key"].startswith("multi|"):
        mg[(r["method"], r["layer"], r["n_facts"], r.get("share", "rank-n" if r["method"] == "rome" else "r16"))].append(r)
if mg:
    g = [(f"{mm} @ {('L' + str(l)) if l is not None else 'all'} n={n} {sh}", summarize(v))
         for (mm, l, n, sh), v in sorted(mg.items(), key=lambda x: (x[0][2], x[0][0], x[0][3]))]
    md.append(table("Multi-fact edits (one edit stores n facts; shared = 1 unit, perfact = n units)", g, "config"))
    summary["multi"] = {k: s for k, s in g}

# lr selection
lrp = os.path.join(D, "lr.json")
if os.path.exists(lrp):
    md.append("Learning rates / MEMIT λ chosen on 5 dev facts at the middle layer: `" + open(lrp).read().replace("\n", " ") + "`\n")

open(os.path.join(D, "tables.md"), "w").write("\n".join(md))
json.dump(summary, open(os.path.join(D, "summary.json"), "w"), indent=1)
print("\n".join(md))
