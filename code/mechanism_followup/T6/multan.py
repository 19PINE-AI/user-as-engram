import json, os, glob, numpy as np
D = os.path.expanduser("~/clean-tests/results/T6")
rows = [json.loads(l) for fp in glob.glob(os.path.join(D, "runs*.jsonl")) for l in open(fp)]
rows = [r for r in rows if r["key"].startswith("multi|")]
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0: return (float("nan"),) * 3
    b = x[rng.integers(0, len(x), (4000, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
G = {}
for r in rows:
    G.setdefault((r["n_facts"], r["method"], r["layer"], r.get("share")), []).append(r)
order = ["lora_down", "lora_down+reg", "lora_up", "lora_up+reg", "neuron", "neuron+reg", "neuron_bias", "neuron_bias+reg", "rome", "qkv", "qkv+reg"]
def f(v, p=3):
    if v != v: return "–"
    return f"{v:.{p}f}" if (abs(v) >= 1e-3 or v == 0) else f"{v:.1e}"
out = {}
md = []
for n in (10, 50):
    md += [f"### n = {n} facts per edit ({200 // n} edits per cell; CI over edits)", "",
           "| method | layer | units | edits | direct | para | two-hop | bleed | fire10 (text) | fires on others >10% | resid-move | text KL | neighbor KL | params/edit |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted([k for k in G if k[0] == n], key=lambda k: (order.index(k[1]), k[2] if k[2] is not None else 99, str(k[3])))
    for k in keys:
        rs = G[k]
        s = {
            "direct": ci([r["direct"] for r in rs]), "para": ci([r["para"] for r in rs]), "twohop": ci([r["twohop"] for r in rs]),
            "bleed": ci([r["other_subject_target_p_edit"] for r in rs]),
            "fire10": ci([(r.get("firing") or {}).get("frac_tokens_gt10pct_trigger") for r in rs]),
            "other": ci([(r.get("firing") or {}).get("other_subj_frac_gt10pct") for r in rs]),
            "resid": ci([r["text"].get("resid_frac_rel_gt_1pct") for r in rs]),
            "tkl": ci([r["text"]["kl_per_tok"] for r in rs]), "nkl": ci([r["neighbor_kl"] for r in rs]),
        }
        out["|".join(map(str, k))] = {"edits": len(rs), **s}
        units = {"shared": "1 shared", "perfact": f"{n} (1/fact)", "rank-n": f"rank-{n}", "r16": "r16 all layers"}.get(k[3], str(k[3]))
        L = f"L{k[2]}" if k[2] is not None else "all"
        md.append(f"| {k[1]} | {L} | {units} | {len(rs)} | {f(s['direct'][0])} | {f(s['para'][0])} | {f(s['twohop'][0])} | "
                  f"{f(s['bleed'][0])} [{f(s['bleed'][1])}, {f(s['bleed'][2])}] | {f(s['fire10'][0])} | {f(s['other'][0])} | "
                  f"{f(s['resid'][0])} | {f(s['tkl'][0])} | {f(s['nkl'][0])} | {rs[0]['n_params']} |")
    md.append("")
open(os.path.join(D, "multi_tables.md"), "w").write("\n".join(md))
json.dump(out, open(os.path.join(D, "multi_summary.json"), "w"), indent=1)
print("\n".join(md))
