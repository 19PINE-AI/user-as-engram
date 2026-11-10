import json, os, numpy as np
D = os.path.expanduser("~/clean-tests/results/T6")
rows = [json.loads(l) for l in open(os.path.join(D, "runs_supp.jsonl"))]
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float); b = x[rng.integers(0, len(x), (4000, len(x)))].mean(1)
    return f"{x.mean():.4f} [{np.percentile(b,2.5):.4f}, {np.percentile(b,97.5):.4f}]"
by = {}
for r in rows: by.setdefault(r["method"], []).append(r)
out = {}
for m, rs in by.items():
    F = [r["firing"] for r in rs]
    line = {"n": len(rs),
            "other_subj_frac_gt10pct": ci([f["other_subj_frac_gt10pct"] for f in F]),
            "other_subj_frac_gt50pct": ci([f["other_subj_frac_gt50pct"] for f in F]),
            "other_subj_mean_ratio": ci([f["other_subj_mean_ratio"] for f in F]),
            "text_fire10": ci([f["frac_tokens_gt10pct_trigger"] for f in F]),
            "bleed": ci([r["other_subject_target_p_edit"] for r in rs]),
            "direct": ci([r["direct"] for r in rs]), "para": ci([r["para"] for r in rs]), "twohop": ci([r["twohop"] for r in rs])}
    out[m] = line
    print(m, json.dumps(line, indent=0))
json.dump(out, open(os.path.join(D, "supp_summary.json"), "w"), indent=1)
