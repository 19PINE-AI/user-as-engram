"""T5 analysis: essay-style verdict fractions and Gao-style simulation scores, base_d8 vs engram_d8,
per layer group, with bootstrap 95% CIs (and CIs on the engram - base difference)."""
import json, os, sys
import numpy as np

ROOT = os.path.expanduser("~/clean-tests/results/T5_T8/T5")
MODELS = ["base_d8", "engram_d8"]
GROUPS = {"early(0,1)": [0, 1], "post-engram(2,5)": [2, 5], "mid(3,4)": [3, 4], "late(6,7)": [6, 7], "all": list(range(8))}
rng = np.random.default_rng(0)


def load(m):
    d = os.path.join(ROOT, m)
    N = json.load(open(os.path.join(d, "neurons.json")))
    L = json.load(open(os.path.join(d, "labels.json")))
    S = json.load(open(os.path.join(d, "sims.json")))
    U = json.load(open(os.path.join(d, "subst.json"))) if os.path.exists(os.path.join(d, "subst.json")) else {}
    J = json.load(open(os.path.join(d, "midrange_judge.json"))) if os.path.exists(os.path.join(d, "midrange_judge.json")) else {}
    rows = []
    for n in N:
        k = f"{n['layer']}:{n['neuron']}"
        lab = L.get(k, {})
        if "verdict" not in lab:
            continue
        s = S.get(k, {}); u = U.get(k, {})
        rows.append(dict(layer=n["layer"], verdict=lab["verdict"], category=lab.get("category"),
                         conf=lab.get("confidence"), corr=s.get("corr"),
                         dav=u.get("damage_avoided_vs_zero"), kl_zero=u.get("zero"),
                         cdav=u.get("cal_damage_avoided_vs_zero"),
                         fire=n["fire_rate"], mid_fit=J.get(k, {}).get("mid_fit"),
                         zero_fit=J.get(k, {}).get("zero_fit")))
    return rows


def boot(vals, fn=np.mean, B=2000):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    bs = [fn(rng.choice(vals, len(vals))) for _ in range(B)]
    return (float(fn(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def boot_diff(a, b, fn=np.mean, B=2000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    bs = [fn(rng.choice(b, len(b))) - fn(rng.choice(a, len(a))) for _ in range(B)]
    return (float(fn(b) - fn(a)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def metrics(rows):
    v = lambda name: [r["verdict"] == name for r in rows]
    corr = [r["corr"] for r in rows if r["corr"] is not None]
    dav = [r["dav"] for r in rows if r["dav"] is not None]
    return {
        "n": len(rows),
        "one_job": v("one_job"),
        "one_or_two_related": [r["verdict"] in ("one_job", "two_related") for r in rows],
        "mixed_unrelated": v("mixed_unrelated"),
        "unclear": v("unclear"),
        "entity_fact": [r["category"] == "entity_fact" for r in rows],
        "sim_corr": corr,
        "damage_avoided": [max(-1.0, min(1.0, x)) for x in dav],  # clip outliers for robust means
        "pass50": [x >= 0.5 for x in dav],
        "cal_damage_avoided": [max(-1.0, min(1.0, r["cdav"])) for r in rows if r.get("cdav") is not None],
        "cal_pass50": [r["cdav"] >= 0.5 for r in rows if r.get("cdav") is not None],
        "mid_fit": [r["mid_fit"] for r in rows if r["mid_fit"] is not None],
        "zero_fit": [r["zero_fit"] for r in rows if r["zero_fit"] is not None],
    }


data = {m: load(m) for m in MODELS}
out = {}
for g, layers in GROUPS.items():
    out[g] = {}
    per = {m: metrics([r for r in data[m] if r["layer"] in layers]) for m in MODELS}
    for key in ["one_job", "one_or_two_related", "mixed_unrelated", "unclear", "entity_fact", "pass50", "cal_pass50", "mid_fit", "zero_fit"]:
        out[g][key] = {m: boot(per[m][key]) for m in MODELS}
        out[g][key]["diff_engram_minus_base"] = boot_diff(per["base_d8"][key], per["engram_d8"][key])
    for key in ["sim_corr", "damage_avoided", "cal_damage_avoided"]:
        out[g][key + "_median"] = {m: boot(per[m][key], np.median) for m in MODELS}
        out[g][key + "_median"]["diff_engram_minus_base"] = boot_diff(per["base_d8"][key], per["engram_d8"][key], np.median)
    out[g]["n"] = {m: per[m]["n"] for m in MODELS}
# agreement between the two standards: are 'one_job' neurons better simulated?
agree = {}
for m in MODELS:
    rows = [r for r in data[m] if r["corr"] is not None and r["dav"] is not None]
    oj = [r for r in rows if r["verdict"] == "one_job"]; ot = [r for r in rows if r["verdict"] != "one_job"]
    agree[m] = {"one_job_median_dav": float(np.median([r["dav"] for r in oj])) if oj else None,
                "other_median_dav": float(np.median([r["dav"] for r in ot])) if ot else None,
                "one_job_pass50": float(np.mean([r["dav"] >= 0.5 for r in oj])) if oj else None,
                "one_job_median_corr": float(np.median([r["corr"] for r in oj])) if oj else None,
                "other_median_corr": float(np.median([r["corr"] for r in ot])) if ot else None,
                "spearman_midfit_vs_dav": None}
    import scipy.stats as ss
    rr = [r for r in rows if r["mid_fit"] is not None]
    if rr:
        agree[m]["spearman_midfit_vs_dav"] = float(ss.spearmanr([r["mid_fit"] for r in rr], [r["dav"] for r in rr]).correlation)
        agree[m]["spearman_simcorr_vs_dav"] = float(ss.spearmanr([r["corr"] for r in rr], [r["dav"] for r in rr]).correlation)
out["standards_agreement"] = agree
json.dump(out, open(os.path.join(ROOT, "t5_summary.json"), "w"), indent=1)

fmt = lambda t: f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]"
for g in GROUPS:
    print(f"\n## {g}  n={out[g]['n']}")
    for key in ["one_job", "one_or_two_related", "mixed_unrelated", "unclear", "entity_fact", "mid_fit", "zero_fit", "pass50", "cal_pass50", "sim_corr_median", "damage_avoided_median", "cal_damage_avoided_median"]:
        r = out[g][key]
        print(f"{key:22s} base {fmt(r['base_d8'])}  engram {fmt(r['engram_d8'])}  diff {fmt(r['diff_engram_minus_base'])}")
print(json.dumps(agree, indent=1))
