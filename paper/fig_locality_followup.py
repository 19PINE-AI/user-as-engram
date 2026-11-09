"""
Mechanism follow-up figure (Section "Where locality comes from").

  fig_locality_followup.pdf — three panels, from results/mechanism_followup/:
    (a) T2, Mini-Engram-d20, same 474 facts written as Engram rows (Joint OPT) or a
        per-user LoRA: relative residual change per block, at the trigger, at the
        prompt's other positions, and on unrelated sentences.
    (b) T6, Qwen3-8B, one fact per edit: probability the edit puts on the edited city
        when asked about 40 *other* people, by layer; hollow markers = trained with
        counter-examples (+reg).
    (c) T6: two-hop accuracy at the +reg layers, with and without counter-examples.
"""
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT",
                          os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE = "#34507F"; BLUE_LT = "#7E97C4"; RED = "#C24A3F"; ORANGE = "#D98A3D"
GREEN = "#3E7C5A"; GRAY = "#8A8F9A"; GRAY_LT = "#C2C6CE"
RES = Path(UAE_ROOT) / "results" / "mechanism_followup"
FIGS = Path(__file__).parent / "figs"


def mean(x):
    x = [v for v in x if v is not None]
    return sum(x) / len(x)


def main():
    d = json.load(gzip.open(RES / "T2" / "partB_mech.json.gz", "rt"))
    F = [f for u in d["users"] for f in u["facts"]]
    NL = len(F[0]["resid_trig_engram"])
    cur = {}
    for X in ("engram", "lora"):
        cur[X, "trig"] = [mean([f[f"resid_trig_{X}"][l] for f in F]) for l in range(NL)]
        cur[X, "other"] = [mean([f[f"resid_other_{X}"][l] for f in F]) for l in range(NL)]
        cur[X, "unrel"] = [mean([s[l] for u in d["users"] for s in u["unrelated_rel_change"][X]])
                           for l in range(NL)]

    R6 = defaultdict(list)
    for line in gzip.open(RES / "T6" / "runs_single_fact.jsonl.gz", "rt"):
        r = json.loads(line)
        R6[r["method"], r["layer"]].append(r)
    cell = lambda m, L, k: mean([r[k] for r in R6[m, L]])

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.75), gridspec_kw={"width_ratios": [1.05, 1.05, 1.0]})

    # (a) residual change by block
    ax = axes[0]
    x = np.arange(NL)
    ax.plot(x, cur["lora", "trig"], color=RED, lw=1.8, marker="o", ms=3, label="LoRA, trigger")
    ax.plot(x, cur["lora", "other"], color=RED, lw=1.4, ls="--", label="LoRA, other pos.")
    ax.plot(x, cur["lora", "unrel"], color=ORANGE, lw=1.4, ls=":", label="LoRA, unrelated")
    ax.plot(x, cur["engram", "trig"], color=BLUE, lw=1.8, marker="s", ms=3, label="Engram, trigger")
    ax.plot(x, cur["engram", "other"], color=BLUE_LT, lw=1.4, ls="--",
            label="Engram, other/unrel.")
    ax.axvline(11.5, color=GRAY, lw=0.8, ls="-", zorder=0)
    ax.text(11.2, 1.62, "Engram\nread (L11)", fontsize=7, color=GRAY, va="top", ha="right")
    ax.set_xlabel("block (input residual)")
    ax.set_ylabel(r"relative change $\|\Delta x\|/\|x\|$")
    ax.set_xticks([0, 5, 10, 15, 19])
    ax.set_ylim(-0.05, 2.45)
    ax.legend(loc="upper left", frameon=False, handlelength=2.0, borderaxespad=0.0, fontsize=6.6, ncol=2,
              columnspacing=0.6)
    ax.set_title("(a) same 474 facts, d20", loc="left", fontweight="bold")

    # (b) bleed by layer
    ax = axes[1]
    Ls = list(range(3, 32, 4))
    for m, c, mk, lab in (("neuron", GREEN, "^", "gated neuron"), ("lora_down", RED, "o", "rank-1 LoRA"),
                          ("rome", BLUE, "s", "ROME (subject key)")):
        ax.plot(Ls, [cell(m, L, "other_subject_target_p_edit") for L in Ls], color=c, marker=mk, ms=4,
                lw=1.6, label=lab)
    for m, c, mk in (("neuron+reg", GREEN, "^"), ("lora_down+reg", RED, "o")):
        Lr = [7, 15, 23]
        ax.plot(Lr, [cell(m, L, "other_subject_target_p_edit") for L in Lr], color=c, marker=mk, ms=5,
                lw=1.0, ls="--", mfc="white", mew=1.3)
    ax.plot([], [], color=GRAY, marker="o", mfc="white", ls="--", lw=1.0, label="+ counter-examples")
    base = mean([r["other_subject_target_p_base"] for v in R6.values() for r in v])
    ax.axhline(base, color=GRAY, lw=0.8, ls=":")
    ax.text(3, base * 1.35, "no edit", fontsize=7, color=GRAY, ha="left")
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 1.5)
    ax.set_xticks(Ls)
    ax.set_xlabel("edited layer (Qwen3-8B, 36 layers)")
    ax.set_ylabel("p(edited city | other person)")
    ax.legend(loc="lower right", frameon=False, borderaxespad=0.1)
    ax.set_title("(b) leak to other people", loc="left", fontweight="bold")

    # (c) two-hop at +reg layers
    ax = axes[2]
    Lr = [7, 15, 23]
    bars = [("neuron", GREEN, None, "neuron"), ("neuron+reg", GREEN, "////", "neuron +reg"),
            ("lora_down", RED, None, "rank-1 LoRA"), ("lora_down+reg", RED, "////", "LoRA +reg"),
            ("rome", BLUE, None, "ROME")]
    w = 0.16
    xx = np.arange(len(Lr))
    for i, (m, c, h, lab) in enumerate(bars):
        vals = [cell(m, L, "twohop") for L in Lr]
        ax.bar(xx + (i - 2) * w, vals, w * 0.9, color="white" if h else c, edgecolor=c, hatch=h,
               linewidth=0.8, label=lab)
    ax.axhline(0.52, color=GRAY, lw=0.8, ls=":")
    ax.text(-0.45, 0.535, "fact in context", fontsize=7, color=GRAY, ha="left")
    ax.set_xticks(xx); ax.set_xticklabels([f"L{L}" for L in Lr])
    ax.set_ylabel("two-hop accuracy")
    ax.set_ylim(0, 0.9); ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.legend(loc="upper center", frameon=False, ncol=3, fontsize=6.6, columnspacing=0.8, handlelength=1.4,
              borderaxespad=0.0, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("(c) cost of counter-examples", loc="left", fontweight="bold")

    fig.tight_layout(w_pad=1.2)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "fig_locality_followup.pdf")
    fig.savefig(RES / "fig_locality_followup.png", dpi=150)
    print("wrote", FIGS / "fig_locality_followup.pdf")


if __name__ == "__main__":
    main()
