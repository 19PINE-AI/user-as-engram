"""Generate all new table->figure conversions for the reorganized paper.
Values transcribed from the result tables in main.tex (and results/*.json).
Consistent palette: slate blue = our method, muted red = baseline/LoRA/retrieval.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#34507F"    # ours (Engram / layered)
RED = "#C24A3F"     # LoRA / retrieval baseline
GRAY = "#8A8F9A"
GREEN = "#3E7C5A"
ORANGE = "#D98A3D"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f"figs/{name}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/" + name)


# ---------------------------------------------------------------- A. contamination gap
def fig_contamination():
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    labels = ["no edit", "per-user\nEngram row", "per-user\nLoRA r=64"]
    vals = [1e-6, 0.00005, 1.784]  # tiny floor for log display of "no edit"
    colors = [GRAY, BLUE, RED]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta$ val bits/byte on text UNRELATED to the user (log scale)", fontsize=9)
    ax.set_xlim(1e-6, 5)
    for yi, v in zip(y, vals):
        txt = "0.0000" if v < 1e-5 else f"+{v:.5f}".rstrip("0").rstrip(".") if v < 1 else f"+{v:.3f}"
        ax.text(min(v, 3) * 1.3, yi, txt, va="center", fontsize=8.5)
    ax.annotate(r"$\sim$34,000$\times$", xy=(0.00005, 1), xytext=(0.02, 1.0),
                fontsize=11, fontweight="bold", color=BLUE, va="center")
    clean(ax)
    ax.grid(axis="x", ls=":", lw=0.6, alpha=0.6)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_contamination.pdf")


# ---------------------------------------------------------------- B. cross-base LoRA
def fig_crossbase():
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    bases = ["Mini-Engram-d20\n(base LM)", "Qwen2.5-3B", "Qwen2.5-7B",
             "Llama-3.1-8B", "Mistral-7B"]
    delta = [-0.133, 0.088, 0.132, 0.188, 0.217]
    worse = [85, 20, 5, 0, 0]
    x = np.arange(len(bases))
    colors = [RED if d < 0 else BLUE for d in delta]
    bars = ax.bar(x, delta, color=colors, width=0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bases, fontsize=7.6)
    ax.set_ylabel(r"mean $\Delta$ indirect recall" + "\n(adapter $-$ base)", fontsize=9)
    for xi, d, w in zip(x, delta, worse):
        ax.text(xi, d + (0.012 if d >= 0 else -0.012), f"{w}% worse",
                ha="center", va="bottom" if d >= 0 else "top", fontsize=7.5)
    ax.set_ylim(-0.22, 0.28)
    clean(ax)
    ax.set_title("Per-user LoRA helps instruction-tuned bases, hurts the base LM",
                 fontsize=9.5, fontweight="bold")
    save(fig, "fig_crossbase.pdf")


# ---------------------------------------------------------------- D. memory systems
def fig_memsystems():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 2.8), sharey=True)
    # trigger-exact top-1
    m1 = ["RAG top-1", "MEM0", "MEMMACH", "Engram\nJ-OPT"]
    v1 = [99.0, 94.0, 93.0, 68.0]
    c1 = [RED, RED, RED, BLUE]
    a1.bar(m1, v1, color=c1, width=0.62)
    a1.set_title("Trigger-exact query", fontsize=10, fontweight="bold")
    a1.set_ylabel("top-1 recall (%)", fontsize=9)
    # paraphrase top-1
    m2 = ["RAG top-3", "MEM0", "MEMMACH", "Engram\nmulti-trig."]
    v2 = [65.6, 62.5, 75.0, 96.9]
    c2 = [RED, RED, RED, BLUE]
    a2.bar(m2, v2, color=c2, width=0.62)
    a2.set_title("Paraphrased query", fontsize=10, fontweight="bold")
    for ax, vs in ((a1, v1), (a2, v2)):
        clean(ax)
        ax.set_ylim(0, 109)
        for i, v in enumerate(vs):
            ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
    a1.text(0.5, -0.34, "retrieval wins (0 ctx tokens for Engram)", transform=a1.transAxes,
            ha="center", fontsize=7.5, color=GRAY)
    a2.text(0.5, -0.34, "Engram multi-trigger wins by 22 pts", transform=a2.transAxes,
            ha="center", fontsize=7.5, color=GRAY)
    save(fig, "fig_memsystems.pdf")


# ---------------------------------------------------------------- E. LOCOMO judge / multitoken
def fig_locomo_judge():
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    scales = ["d8\n178M", "d12 v2\n339M", "d12@1280\n625M", "d20@1536\n1.22B"]
    best_retr = [0.081, 0.128, 0.125, 0.174]   # matched multi-token pipeline
    jopt_multi = [0.060, 0.105, 0.159, 0.225]
    x = np.arange(len(scales))
    w = 0.38
    ax.bar(x - w/2, best_retr, w, label="best retrieval", color=RED)
    ax.bar(x + w/2, jopt_multi, w, label="Engram J-OPT (multi-token)", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(scales, fontsize=8)
    ax.set_ylabel("LOCOMO LLM-judge accuracy", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Multi-token OPT overtakes retrieval from d12@1280 up (matched pipeline)",
                 fontsize=8.8, fontweight="bold")
    clean(ax)
    save(fig, "fig_locomo_judge.pdf")


# ---------------------------------------------------------------- J. multi-hop overlap split
def fig_multihop():
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    labels = ["surface-overlap\n(query suffix = Fact-2 trigger)",
              "no-overlap\n(true chaining)"]
    top1 = [90.6, 12.9]
    err = [[90.6 - 75.8, 12.9 - 5.1], [96.8 - 90.6, 28.9 - 12.9]]
    x = np.arange(len(labels))
    colors = [BLUE, RED]
    ax.bar(x, top1, color=colors, width=0.55, yerr=err, capsize=4,
           error_kw=dict(lw=1, ecolor=GRAY))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("multi-hop top-1 (%)", fontsize=9)
    ax.set_ylim(0, 105)
    for xi, v in zip(x, top1):
        ax.text(xi, v + 4, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("The gate matches surfaces, it does not chain (n=63, Wilson 95% CI)",
                 fontsize=8.6, fontweight="bold")
    clean(ax)
    save(fig, "fig_multihop.pdf")


# ---------------------------------------------------------------- M. layered conditions A-F
def fig_layered_conditions():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.0))
    conds = ["A: base", "B: per-user\nLoRA", "C: per-user\nEngram",
             "D: LoRA+\nEngram", "E: shared\nLoRA", "F: Engram+\nshared LoRA"]
    direct = [29, 99, 100, 100, 54, 100]
    indirect_any = [19, 6, 23, 8, 44, 44]
    dbpb = [0.000, 1.784, 0.00005, 1.819, 0.386, 0.386]
    x = np.arange(len(conds))
    w = 0.4
    a1.bar(x - w/2, direct, w, label="direct top-1", color=GRAY)
    a1.bar(x + w/2, indirect_any, w, label="indirect_any", color=BLUE)
    a1.set_xticks(x); a1.set_xticklabels(conds, fontsize=7.2)
    a1.set_ylabel("recall (%)", fontsize=9)
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a1.set_title("Recall: F matches LoRA direct, 7.4$\\times$ its indirect",
                 fontsize=8.6, fontweight="bold")
    clean(a1); a1.set_ylim(0, 115)
    # contamination
    colors = [RED if d > 0.5 else BLUE for d in dbpb]
    a2.bar(x, dbpb, color=colors, width=0.6)
    a2.set_xticks(x); a2.set_xticklabels(conds, fontsize=7.2)
    a2.set_ylabel(r"$\Delta$bpb on unrelated text", fontsize=9)
    a2.set_title("Contamination: F adds none on top of the shared skill",
                 fontsize=8.6, fontweight="bold")
    clean(a2)
    save(fig, "fig_layered_conditions.pdf")


# ---------------------------------------------------------------- N. shared-LoRA rank
def fig_shared_rank():
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ranks = ["r=4", "r=16", "r=64"]
    indirect = [28, 44, 35]
    dbpb = [0.302, 0.386, 1.027]
    x = np.arange(len(ranks))
    ax.bar(x, indirect, color=[BLUE if r == "r=16" else GRAY for r in ranks], width=0.5)
    ax.set_xticks(x); ax.set_xticklabels(ranks, fontsize=9)
    ax.set_ylabel("indirect_any (%)", fontsize=9, color=BLUE)
    for xi, v in zip(x, indirect):
        ax.text(xi, v + 1, f"{v}%", ha="center", fontsize=8.5)
    ax2 = ax.twinx()
    ax2.plot(x, dbpb, "o-", color=RED, lw=1.5, ms=5)
    ax2.set_ylabel(r"$\Delta$bpb", fontsize=9, color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax.set_ylim(0, 52)
    clean(ax)
    ax.set_title("r=16 is the sweet spot", fontsize=9.5, fontweight="bold")
    save(fig, "fig_shared_rank.pdf")


# ---------------------------------------------------------------- O. cross-schema F vs B
def fig_crossschema():
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    groups = ["within-schema", "cross-schema\n(personal→medical)"]
    F = [44, 31]
    B = [6, 4]
    x = np.arange(len(groups)); w = 0.36
    ax.bar(x - w/2, F, w, label="F (layered)", color=BLUE)
    ax.bar(x + w/2, B, w, label="B (per-user LoRA)", color=RED)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("indirect_any (%)", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5)
    for xi, v in zip(x - w/2, F):
        ax.text(xi, v + 1, f"{v}%", ha="center", fontsize=8.5)
    for xi, v in zip(x + w/2, B):
        ax.text(xi, v + 1, f"{v}%", ha="center", fontsize=8.5)
    ax.set_ylim(0, 52)
    clean(ax)
    ax.set_title("F's lead survives a schema shift (7.4$\\times$ → 7.6$\\times$ over B)",
                 fontsize=8.4, fontweight="bold")
    save(fig, "fig_crossschema.pdf")


# ---------------------------------------------------------------- L. serving
def fig_serving_scale():
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    cfg = ["d8\n20u/30f", "d12 v1\n30u/50f", "d12@1280\n30u/50f", "d12@1280\n100u/100f"]
    thr = [42.8, 47.9, 232, 226]
    x = np.arange(len(cfg))
    ax.bar(x, thr, color=[GRAY, GRAY, BLUE, BLUE], width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(cfg, fontsize=8)
    ax.set_ylabel("throughput (req/s)", fontsize=9)
    for xi, v in zip(x, thr):
        ax.text(xi, v + 4, f"{v:g}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 260)
    clean(ax)
    ax.set_title("232 req/s on one GPU; per-request work independent of tenant count",
                 fontsize=8.2, fontweight="bold")
    ax.text(0.99, 0.04, "0% cross-user leak by construction", transform=ax.transAxes,
            ha="right", fontsize=8, style="italic", color=GREEN)
    save(fig, "fig_serving_scale.pdf")


# ---------------------------------------------------------------- Q. MF finetune
def fig_mf_finetune():
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    n = [100, 300, 500, 1000]
    base = [65, 46, 39, 28]
    mf = [61, 48, 42, 32]
    x = np.arange(len(n)); w = 0.38
    ax.bar(x - w/2, base, w, label="baseline", color=GRAY)
    ax.bar(x + w/2, mf, w, label="MF-finetune", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in n], fontsize=8.5)
    ax.set_xlabel("facts/user", fontsize=9)
    ax.set_ylabel("Joint-OPT top-1 (%)\n(fixed inference budget)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_ylim(0, 75)
    clean(ax)
    ax.set_title("MF accelerates convergence (+14% rel at n=1000, fixed budget)",
                 fontsize=8.2, fontweight="bold")
    save(fig, "fig_mf_finetune.pdf")


if __name__ == "__main__":
    fig_contamination()
    fig_crossbase()
    fig_memsystems()
    fig_locomo_judge()
    fig_multihop()
    fig_layered_conditions()
    fig_shared_rank()
    fig_crossschema()
    fig_serving_scale()
    fig_mf_finetune()
    print("all figures generated")
