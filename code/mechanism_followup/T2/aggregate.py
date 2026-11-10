"""Aggregate T2 results into summary.json + markdown tables (printed)."""
import json, glob, os, math, statistics as st, sys
from collections import defaultdict
R = os.path.expanduser("~/clean-tests/results/T2")
A = os.path.join(R, sys.argv[1] if len(sys.argv) > 1 else "analysis")
out = {}


def med(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return st.median(x) if x else float("nan")


def mean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(x) / len(x) if x else float("nan")


def l50_suffix(s):   # last l such that LoRA only in [l, NL) still recovers >= 0.5
    best = 0
    for l, v in enumerate(s):
        if v >= 0.5: best = l
    return best


def l50_prefix(p):   # first l such that LoRA only in [0, l) recovers >= 0.5
    for l, v in enumerate(p):
        if v >= 0.5: return l
    return len(p) - 1


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for k, i in enumerate(o): r[i] = k
        return r
    if len(x) < 3: return float("nan")
    rx, ry = rk(x), rk(y); mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


lines = []
P = lines.append
for variant in ("qkv", "mlp"):
    files = sorted(glob.glob(f"{A}/{variant}_u*.json"))
    if not files: continue
    U = [json.load(open(f)) for f in files]
    F = [f for u in U for f in u["facts"]]
    NL = len(F[0]["window"]["suffix"]) - 1
    s = {}
    s["n_users"] = len(U); s["n_facts"] = len(F)
    s["lora_direct"] = mean([u["direct_acc"] for u in U]); s["lora_indirect"] = mean([u["indirect_acc"] for u in U])
    s["base_direct"] = mean([u["base_direct"] for u in U]); s["base_indirect"] = mean([u["base_indirect"] for u in U])
    s["suffix_curve"] = [mean([f["window"]["suffix"][l] for f in F]) for l in range(NL + 1)]
    s["prefix_curve"] = [mean([f["window"]["prefix"][l] for f in F]) for l in range(NL + 1)]
    s["band4"] = [mean([f["band4"][i] for f in F]) for i in range(len(F[0]["band4"]))]
    s["l50_suffix_median"] = med([l50_suffix(f["window"]["suffix"]) for f in F])
    s["l50_prefix_median"] = med([l50_prefix(f["window"]["prefix"]) for f in F])
    # attribution profiles (normalised by dm)
    for kind in ("mlp_pos", "attn_pos", "mlp_allpos", "attn_allpos"):
        s[f"axg_delta_{kind}"] = [mean([(f["layer_axg"]["lora"][kind][l] - f["layer_axg"]["base"][kind][l]) / f["dm"] for f in F]) for l in range(NL)]
    s["dla_delta_mlp"] = [mean([(f["layer_dla"]["mlp_lora"][l] - f["layer_dla"]["mlp_base"][l]) / f["dm"] for f in F]) for l in range(NL)]
    s["dla_delta_attn"] = [mean([(f["layer_dla"]["attn_lora"][l] - f["layer_dla"]["attn_base"][l]) / f["dm"] for f in F]) for l in range(NL)]
    s["direct_share_mlp"] = med([f["mlp_direct_delta"] / f["dm"] for f in F])
    s["direct_share_attn"] = med([f["attn_direct_delta"] / f["dm"] for f in F])
    # off-ramp
    s["share_topK_static"] = {K: med([f["offramp_share_topK_static"][K] for f in F]) for K in F[0]["offramp_share_topK_static"]}
    s["share_randomK"] = {K: med([f["offramp_share_randomK"][K] for f in F]) for K in F[0]["offramp_share_randomK"]}
    s["n_neurons_50pct_median"] = med([f["n_neurons_for_50pct"] for f in F])
    s["n_neurons_80pct_median"] = med([f["n_neurons_for_80pct"] for f in F])
    s["top100_in_static_top1000"] = med([f["top100_contrib_in_static_top1000"] for f in F])
    s["top100_static_pct_median"] = med([f["top100_contrib_static_pct_median"] for f in F])
    hist = [0] * NL
    for f in F:
        for l in f["top100_contrib_layers"]: hist[l] += 1
    s["top100_contrib_layer_hist_frac"] = [h / sum(hist) for h in hist]
    sh = [0] * NL
    for f in F:
        for l, v in enumerate(f["static_top1000_layers_hist"]): sh[l] += v
    s["static_top1000_layer_hist_frac"] = [h / sum(sh) for h in sh]
    abl = {}
    for K in F[0]["ablation"]:
        abl[K] = {}
        for t in F[0]["ablation"][K]:
            abl[K][t] = dict(recovery_median=med([f["ablation"][K][t][0] for f in F]),
                             still_top1=mean([1.0 if f["ablation"][K][t][1] else 0.0 for f in F]))
    s["ablation"] = abl
    # dW
    fro = defaultdict(lambda: [0.0] * NL)
    for u in U:
        for name, v in u["dW"]["fro"].items():
            for l in range(NL): fro[name][l] += v[l] / len(U)
    s["dW_fro_by_layer"] = dict(fro)
    if variant == "mlp":
        D = [d for u in U for d in u["dW"]["down"]]
        G = [d for u in U for d in u["dW"]["gate_up"]]
        s["dW_down_maxcos_out_vs_neurons_top1"] = med([d["maxcos_out_vs_neurons"][0] for d in D])
        s["dW_down_maxcos_random"] = med([d["maxcos_random_vs_neurons"] for d in D])
        s["dW_down_v_mass_top1pct_top1"] = med([d["v_mass_top1pct"][0] for d in D])
        s["dW_down_v_mass_top1pct_random"] = med([d["v_mass_top1pct_random"] for d in D])
        s["dW_gateup_maxcos_in_top1"] = med([d["maxcos_in_vs_neuron_reads"][0] for d in G])
        s["dW_gateup_maxcos_random"] = med([d["maxcos_random"] for d in G])
        s["dW_gateup_u_mass_top1pct_top1"] = med([d["u_mass_top1pct"][0] for d in G])
        # per-layer version for the down projection
        s["dW_down_by_layer"] = [dict(layer=d["layer"], maxcos=mean([x["maxcos_out_vs_neurons"][0] for x in D if x["layer"] == d["layer"]]),
                                      vmass=mean([x["v_mass_top1pct"][0] for x in D if x["layer"] == d["layer"]]))
                                 for d in D[:NL]]
    # (4) lateness vs generalisation
    rows = []
    for u in U:
        l50 = {f["key"]: l50_suffix(f["window"]["suffix"]) for f in u["facts"]}
        band_late = {f["key"]: sum(f["band4"][len(f["band4"]) // 2:]) / max(1e-6, sum(map(abs, f["band4"]))) for f in u["facts"]}
        user = json.load(open(os.path.expanduser(f"~/clean-tests/user-as-engram/data/users/{u['uid']}.json")))
        for q, d, bd in zip(user["indirect_qa"], u["indirect_details"], u["base_indirect_details"]):
            ks = [k for k in q["required_fact_keys"] if k in l50]
            if not ks: continue
            rows.append(dict(l50=mean([l50[k] for k in ks]), late=mean([band_late[k] for k in ks]),
                             ok=1.0 if d["ok"] else 0.0, base_ok=1.0 if bd["ok"] else 0.0, schema=q["schema"]))
    s["gen_rows_n"] = len(rows)
    if rows:
        m = med([r["l50"] for r in rows])
        early = [r for r in rows if r["l50"] < m]; late = [r for r in rows if r["l50"] > m]
        s["indirect_acc_early_vs_late_l50"] = dict(median_l50=m, early_acc=mean([r["ok"] for r in early]), early_n=len(early),
                                                   late_acc=mean([r["ok"] for r in late]), late_n=len(late))
        s["spearman_l50_vs_ok"] = spearman([r["l50"] for r in rows], [r["ok"] for r in rows])
        nb = [r for r in rows if r["base_ok"] == 0]
        s["spearman_l50_vs_ok_base_wrong"] = spearman([r["l50"] for r in nb], [r["ok"] for r in nb]) if nb else float("nan")
        s["n_base_wrong"] = len(nb)
        s["spearman_lateband_vs_ok"] = spearman([r["late"] for r in rows], [r["ok"] for r in rows])
    out[variant] = s
    # ---- print
    P(f"\n### Part A — {variant} LoRA (n_users={s['n_users']}, n_facts={s['n_facts']})")
    P(f"direct {s['base_direct']:.2f}→{s['lora_direct']:.2f}; indirect {s['base_indirect']:.3f}→{s['lora_indirect']:.3f}")
    P(f"l50 suffix median {s['l50_suffix_median']}, l50 prefix median {s['l50_prefix_median']} (NL={NL})")
    P("suffix " + " ".join(f"{v:.2f}" for v in s["suffix_curve"][::4]))
    P("prefix " + " ".join(f"{v:.2f}" for v in s["prefix_curve"][::4]))
    P("band4  " + " ".join(f"{v:.2f}" for v in s["band4"]))
    P(f"direct share mlp {s['direct_share_mlp']:.2f} attn {s['direct_share_attn']:.2f}")
    P(f"share topK static {s['share_topK_static']}  random {s['share_randomK']}")
    P(f"n50 {s['n_neurons_50pct_median']} n80 {s['n_neurons_80pct_median']}  top100∈static1000 {s['top100_in_static_top1000']:.2f}  pct {s['top100_static_pct_median']:.4f}")
    for K, d in s["ablation"].items():
        P(f"abl K={K}: " + "; ".join(f"{t} {v['recovery_median']:.2f}/{v['still_top1']:.2f}" for t, v in d.items()))
    if variant == "mlp":
        P(f"dW_down maxcos out {s['dW_down_maxcos_out_vs_neurons_top1']:.3f} vs random {s['dW_down_maxcos_random']:.3f}; v mass top1% {s['dW_down_v_mass_top1pct_top1']:.3f} vs random {s['dW_down_v_mass_top1pct_random']:.3f}")
        P(f"dW_gate/up maxcos in {s['dW_gateup_maxcos_in_top1']:.3f} vs random {s['dW_gateup_maxcos_random']:.3f}; u mass top1% {s['dW_gateup_u_mass_top1pct_top1']:.3f}")
    if rows:
        P(f"gen: {s['indirect_acc_early_vs_late_l50']}; spearman(l50, ok)={s['spearman_l50_vs_ok']:.3f}; base-wrong only={s['spearman_l50_vs_ok_base_wrong']:.3f} (n={s['n_base_wrong']}); lateband={s['spearman_lateband_vs_ok']:.3f}")

# ---------------- Part B
pb = os.path.join(R, "partB_mech.json")
if os.path.exists(pb):
    d = json.load(open(pb))
    F = [f for u in d["users"] for f in u["facts"]]
    G = [f for f in F if f["top1_engram"] and f["top1_lora"]]
    NL = len(F[0]["resid_trig_engram"])
    b = dict(n_users=len(d["users"]), n_facts=len(F), n_both_top1=len(G),
             engram_top1=mean([1.0 if f["top1_engram"] else 0.0 for f in F]),
             lora_top1=mean([1.0 if f["top1_lora"] else 0.0 for f in F]),
             reapply_check_max_abs_diff=max(abs(f["m_engram_reapplied"] - f["m_engram"]) for f in F))
    for X in ("engram", "lora"):
        b[f"resid_trig_{X}"] = [mean([f[f"resid_trig_{X}"][l] for f in G]) for l in range(NL)]
        b[f"resid_other_{X}"] = [mean([f[f"resid_other_{X}"][l] for f in G]) for l in range(NL)]
        b[f"mlp_attr_layer_{X}_over_dm"] = [mean([f[f"mlp_attr_layer_{X}"][l] / f[f"dm_{X}"] for f in G]) for l in range(NL)]
        b[f"mlp_attr_total_{X}_over_dm"] = med([f[f"mlp_attr_total_{X}"] / f[f"dm_{X}"] for f in G])
        b[f"unrelated_{X}"] = [mean([s[l] for u in d["users"] for s in u["unrelated_rel_change"][X]]) for l in range(NL)]
        b[f"patch_{X}"] = {k: {t: med([f["patch"][X][k][t] for f in G]) for t in ("own", "cross", "random")} for k in G[0]["patch"][X]}
    b["top100_jaccard_median"] = med([f["top100_jaccard"] for f in G])
    b["dact_cos_layer"] = [mean([f["dact_cos_layer"][l] for f in G]) for l in range(NL)]
    b["dact_cos_layer_shuffled"] = [mean([f["dact_cos_layer_shuffled"][l] for f in G]) for l in range(NL)]
    out["partB"] = b
    P(f"\n### Part B — Mini-Engram-d20 (users={b['n_users']}, facts={b['n_facts']}, both top1={b['n_both_top1']})")
    P(f"top1 engram {b['engram_top1']:.2f} lora {b['lora_top1']:.2f}; reapply check {b['reapply_check_max_abs_diff']:.2e}")
    for X in ("engram", "lora"):
        P(f"{X} resid trig  " + " ".join(f"{v:.2f}" for v in b[f'resid_trig_{X}']))
        P(f"{X} resid other " + " ".join(f"{v:.3f}" for v in b[f'resid_other_{X}']))
        P(f"{X} unrelated   " + " ".join(f"{v:.3f}" for v in b[f'unrelated_{X}']))
        P(f"{X} mlp attr/dm " + " ".join(f"{v:.2f}" for v in b[f'mlp_attr_layer_{X}_over_dm']) + f"  total {b[f'mlp_attr_total_{X}_over_dm']:.2f}")
        P(f"{X} patch {b[f'patch_{X}']}")
    P(f"jaccard top100 {b['top100_jaccard_median']:.3f}")
    P("cos dact  " + " ".join(f"{v:.2f}" for v in b["dact_cos_layer"]))
    P("cos shuf  " + " ".join(f"{v:.2f}" for v in b["dact_cos_layer_shuffled"]))
lr = os.path.join(R, "partB_layered_repro.json")
if os.path.exists(lr):
    d = json.load(open(lr))
    out["partB_layered_repro_agg"] = d.get("aggregate", d.get("agg"))
    P(f"\n### Part B layered reproduction aggregate\n{json.dumps(out['partB_layered_repro_agg'], indent=1)[:3000]}")
json.dump(out, open(os.path.join(R, "summary.json"), "w"), indent=1)
print("\n".join(lines))
