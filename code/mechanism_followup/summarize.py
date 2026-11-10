"""Recompute every number the paper quotes from the mechanism follow-up (T2, T5, T6).

CPU-only; reads only the committed records under results/mechanism_followup/ and
prints a Markdown table (paper number -> recomputed value -> record/field).

    python code/mechanism_followup/summarize.py > results/mechanism_followup/PAPER_NUMBERS.md
"""
import gzip
import json
import math
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "mechanism_followup")
rows = []


def add(claim, value, src):
    rows.append((claim, value, src))


def mean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(x) / len(x)


def med(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return st.median(x)


# ---------------------------------------------------------------- T2 part B
d = json.load(gzip.open(os.path.join(RES, "T2", "partB_mech.json.gz"), "rt"))
F = [f for u in d["users"] for f in u["facts"]]
G = [f for f in F if f["top1_engram"] and f["top1_lora"]]
NL = len(F[0]["resid_trig_engram"])
src = "T2/partB_mech.json.gz"
add("T2: users / facts the base did not already know", f"{len(d['users'])} / {len(F)}", src)
add("T2: top-1 recall, Engram / LoRA",
    f"{mean([float(f['top1_engram']) for f in F]):.3f} / {mean([float(f['top1_lora']) for f in F]):.3f}", src)
first_nz = min(l for l in range(NL) if max(f["resid_trig_engram"][l] for f in F) > 0)
add("T2: Engram, first layer whose input residual changes (trigger pos.)", f"block {first_nz} (of {NL})",
    src + " (resid_trig_engram)")
add("T2: Engram, max relative residual change before that layer, any fact",
    f"{max(f['resid_trig_engram'][l] for f in F for l in range(first_nz)):.3g}", src)
add("T2: Engram, max relative residual change at non-trigger positions (all facts, all layers)",
    f"{max(max(f['resid_other_engram']) for f in F):.3g}", src + " (resid_other_engram)")
ur = [v for u in d["users"] for s in u["unrelated_rel_change"]["engram"] for v in s]
add("T2: Engram, max relative residual change on the 3 unrelated sentences (all users, layers)",
    f"{max(ur):.3g}", src + " (unrelated_rel_change.engram)")
for key, nm in (("resid_trig_lora", "trigger"), ("resid_other_lora", "other prompt positions")):
    v = [mean([f[key][l] for f in G]) for l in range(NL)]
    add(f"T2: LoRA relative residual change, {nm}, input of block 1 / block {NL-1}", f"{v[1]:.2f} / {v[-1]:.2f}",
        src + f" ({key}, mean over facts)")
v = [mean([s[l] for u in d["users"] for s in u["unrelated_rel_change"]["lora"]]) for l in range(NL)]
add(f"T2: LoRA relative residual change, unrelated sentences, block 1 / block {NL-1}", f"{v[1]:.2f} / {v[-1]:.2f}",
    src + " (unrelated_rel_change.lora)")
for X in ("engram", "lora"):
    add(f"T2: {X}, share of logit-diff change attributed to MLP neurons (median, act x grad)",
        f"{med([f[f'mlp_attr_total_{X}'] / f[f'dm_{X}'] for f in G]):.2f}", src + f" (mlp_attr_total_{X}/dm_{X})")
j = med([f["top100_jaccard"] for f in G])
add("T2: top-100 neuron Jaccard, Engram vs LoRA (median)", f"{j:.3f}", src + " (top100_jaccard)")
add("T2: implied shared neurons out of 100 (2J/(1+J) x 100, at the median)", f"{100 * 2 * j / (1 + j):.0f}", "derived")
for X in ("engram", "lora"):
    for k in ("10", "100", "1000"):
        p = {t: med([f["patch"][X][k][t] for f in G]) for t in ("own", "cross", "random")}
        add(f"T2: {X}, effect left after base-patching top-{k} neurons: own / other method's / random",
            f"{p['own']:.2f} / {p['cross']:.2f} / {p['random']:.2f}", src + f" (patch.{X}.{k})")
cos = [mean([f["dact_cos_layer"][l] for f in G]) for l in range(NL)]
cs = [mean([f["dact_cos_layer_shuffled"][l] for f in G]) for l in range(NL)]
add("T2: cos(dMLP-act Engram, dMLP-act LoRA), blocks 11-18 range / last block",
    f"{min(cos[11:NL-1]):.2f}-{max(cos[11:NL-1]):.2f} / {cos[-1]:.2f}", src + " (dact_cos_layer)")
add("T2: same, shuffled-fact baseline, blocks 11-18 range / last block",
    f"{min(cs[11:NL-1]):.2f}-{max(cs[11:NL-1]):.2f} / {cs[-1]:.2f}", src + " (dact_cos_layer_shuffled)")

# ---------------------------------------------------------------- T2 part A
s = json.load(open(os.path.join(RES, "T2", "summary.json")))
for v in ("qkv", "mlp"):
    a = s[v]
    src = f"T2/summary.json ({v})"
    add(f"T2-A {v}-LoRA (Qwen3-8B): users / facts analysed", f"{a['n_users']} / {a['n_facts']}", src)
    add(f"T2-A {v}: direct base -> LoRA; indirect base -> LoRA",
        f"{a['base_direct']:.3f} -> {a['lora_direct']:.3f}; {a['base_indirect']:.3f} -> {a['lora_indirect']:.3f}", src)
    add(f"T2-A {v}: median l50 (LoRA only in layers >= l keeps half the effect), of 36",
        f"{a['l50_suffix_median']}", src + " l50_suffix_median")
    add(f"T2-A {v}: max single 4-layer band recovery", f"{max(a['band4']):.2f}", src + " band4")
    add(f"T2-A {v}: MLP share of the LoRA's direct logit change", f"{a['direct_share_mlp']:.2f}", src)
    add(f"T2-A {v}: neurons holding 50% of the LoRA-induced direct contribution (median)",
        f"{a['n_neurons_50pct_median']}", src)
    add(f"T2-A {v}: share held by the 100 statically top 'answer-writing' neurons vs 100 random",
        f"{a['share_topK_static']['100']:.2f} vs {a['share_randomK']['100']:.4f}", src)
    ab = a["ablation"]["100"]
    add(f"T2-A {v}: effect left after base-patching top-100 contributing / 100 layer-matched random neurons",
        f"{ab['basepatch_topcontrib']['recovery_median']:.2f} / {ab['basepatch_random_layermatched']['recovery_median']:.2f}",
        src + " ablation.100")
    g = a["indirect_acc_early_vs_late_l50"]
    add(f"T2-A {v}: indirect accuracy, facts stored earlier vs later than median l50",
        f"{g['early_acc']:.2f} (n={g['early_n']}) vs {g['late_acc']:.2f} (n={g['late_n']})", src)
    add(f"T2-A {v}: Spearman(l50, indirect ok); base-wrong only",
        f"{a['spearman_l50_vs_ok']:.2f}; {a['spearman_l50_vs_ok_base_wrong']:.2f}", src)

# ---------------------------------------------------------------- T6
R6 = defaultdict(list)
for line in gzip.open(os.path.join(RES, "T6", "runs_single_fact.jsonl.gz"), "rt"):
    r = json.loads(line)
    R6[(r["method"], r["layer"])].append(r)
src = "T6/runs_single_fact.jsonl.gz"


def cell(m, L, f):
    return mean([f(r) for r in R6[(m, L)]])


bleed = lambda r: r["other_subject_target_p_edit"]
two = lambda r: r["twohop"]
para = lambda r: r["para"]
fire = lambda r: (r.get("firing") or {}).get("frac_tokens_gt10pct_trigger")
rmove = lambda r: r["text"].get("resid_frac_rel_gt_1pct")
plain = ["lora_down", "lora_up", "neuron", "neuron_bias", "rome"]
n_plain = sum(len(R6[(m, L)]) for m in plain for L in range(3, 36, 4)) + len(R6[("qkv", None)])
add("T6: unregularised single-fact edits (5 methods x 9 layers x 200 + Q/K/V)", f"{n_plain}", src)
add("T6: base p(edited city) on other people", f"{mean([r['other_subject_target_p_base'] for v in R6.values() for r in v]):.1e}", src)
LS = list(range(3, 32, 4))  # L3..L31 (ROME has no effect at L35)
for m in ("neuron", "lora_down", "lora_up", "rome"):
    vals = [cell(m, L, bleed) for L in LS]
    add(f"T6: {m} bleed, mean over L3-L31 (min-max)", f"{mean(vals):.3f} ({min(vals):.3f}-{max(vals):.3f})", src)
add("T6: rome bleed L3 / L7 / L11", " / ".join(f"{cell('rome', L, bleed):.3f}" for L in (3, 7, 11)), src)
add("T6: rome bleed L15-L31 range", f"{min(cell('rome', L, bleed) for L in range(15, 32, 4)):.3f}-"
    f"{max(cell('rome', L, bleed) for L in range(15, 32, 4)):.3f}", src)
add("T6: neuron fire10 on text, L7-L31 range",
    f"{min(cell('neuron', L, fire) for L in range(7, 32, 4)):.3f}-{max(cell('neuron', L, fire) for L in range(7, 32, 4)):.3f}", src)
add("T6: lora_down fire10 on text, L3-L35 range",
    f"{min(cell('lora_down', L, fire) for L in range(3, 36, 4)):.3f}-{max(cell('lora_down', L, fire) for L in range(3, 36, 4)):.3f}", src)
for m in ("neuron", "lora_down", "lora_up"):
    for L in (7, 15, 23):
        add(f"T6: {m} L{L}: bleed plain -> +reg; two-hop plain -> +reg; para plain -> +reg",
            f"{cell(m, L, bleed):.3f} -> {cell(m + '+reg', L, bleed):.3f}; "
            f"{cell(m, L, two):.3f} -> {cell(m + '+reg', L, two):.3f}; "
            f"{cell(m, L, para):.3f} -> {cell(m + '+reg', L, para):.3f}", src)
        add(f"T6: {m} L{L}: fire10 plain -> +reg; resid-move>1% plain -> +reg",
            f"{cell(m, L, fire):.3f} -> {cell(m + '+reg', L, fire):.3f}; "
            f"{cell(m, L, rmove):.3f} -> {cell(m + '+reg', L, rmove):.3f}", src)
add("T6: qkv (rank-16, all layers) bleed plain -> +reg (n)",
    f"{cell('qkv', None, bleed):.3f} -> {cell('qkv+reg', None, bleed):.3f} (n={len(R6[('qkv+reg', None)])})", src)
add("T6: rome L7 bleed / para / two-hop", f"{cell('rome', 7, bleed):.3f} / {cell('rome', 7, para):.3f} / {cell('rome', 7, two):.3f}", src)
for L in (27, 31, 35):
    add(f"T6: two-hop at L{L}, range over neuron/lora_down/lora_up/rome",
        f"{min(cell(m, L, two) for m in ('neuron', 'lora_down', 'lora_up', 'rome')):.3f}-"
        f"{max(cell(m, L, two) for m in ('neuron', 'lora_down', 'lora_up', 'rome')):.3f}", src)
add("T6: two-hop L7-L19 range over the four methods",
    f"{min(cell(m, L, two) for m in ('neuron', 'lora_down', 'lora_up', 'rome') for L in (7, 11, 15, 19)):.3f}-"
    f"{max(cell(m, L, two) for m in ('neuron', 'lora_down', 'lora_up', 'rome') for L in (7, 11, 15, 19)):.3f}", src)
add("T6: rome L35 direct", f"{cell('rome', 35, lambda r: r['direct']):.3f}", src)

# T6 report section 6: does the edited unit fire on other people's prompts? (L15, 40 facts, +reg)
S6 = defaultdict(list)
for line in gzip.open(os.path.join(RES, "T6", "runs_supp.jsonl.gz"), "rt"):
    r = json.loads(line)
    S6[r["method"]].append(r)
src_s = "T6/runs_supp.jsonl.gz"
for m in ("neuron+reg", "lora_down+reg"):
    rs = S6[m]
    F = [r["firing"] for r in rs]
    add(f"T6 supp: {m} L15 (n): frac of other people >10% / >50% of trigger; mean ratio; bleed",
        f"({len(rs)}) {mean([f['other_subj_frac_gt10pct'] for f in F]):.3f} / "
        f"{mean([f['other_subj_frac_gt50pct'] for f in F]):.3f}; "
        f"{mean([f['other_subj_mean_ratio'] for f in F]):.3f}; "
        f"{mean([r['other_subject_target_p_edit'] for r in rs]):.3f}", src_s)
sup = json.load(open(os.path.join(RES, "T6", "supp_summary.json")))
for m in ("neuron+reg", "lora_down+reg"):
    add(f"T6 supp: {m} 95% bootstrap CI, frac >10% / frac >50% / mean ratio",
        " / ".join(sup[m][k].split(" ", 1)[1] for k in
                   ("other_subj_frac_gt10pct", "other_subj_frac_gt50pct", "other_subj_mean_ratio")),
        "T6/supp_summary.json")

# ---------------------------------------------------------------- T5
t5 = json.load(open(os.path.join(RES, "T5", "t5_summary.json")))
add("T5: see REPORT_T5_T8.md table (d8 dense vs Engram, 800 neurons each)", "all diff CIs include 0",
    "T5/t5_summary.json")

print("# Paper numbers from the mechanism follow-up\n")
print("Generated by `python code/mechanism_followup/summarize.py` (CPU only) from the committed records.\n")
print("| Claim | Value | Record (field) |\n|---|---|---|")
for c, v, s_ in rows:
    print(f"| {c} | {v} | `{s_}` |")
