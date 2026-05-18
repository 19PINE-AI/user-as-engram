"""Two NeurIPS-style figures for Section 7 (robustness) and Section 8
(cross-base): cross-Engram-size + LLM-judge comparison, and cross-base
val_bpb + cross-model user-level distribution."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("/home/ubuntu/user-as-engram/paper/figs")

C_F      = "#117733"
C_B      = "#aa4499"
C_C      = "#1f77b4"
C_BASE   = "#888888"

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


# =====================================================================
# Figure: ROBUSTNESS — three panels
#   (a) cross-Engram-size (d12@1280 vs d20)  bars
#   (b) LLM-judge vs substring-match
#   (c) rank ablation curve (existing data)
# =====================================================================
def fig_robustness():
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))

    # ---- (a) cross-Engram-size: F vs B on d12@1280 and d20 ----
    ax = axes[0]
    labels = ["d12@1280\n(625\,M)", "d20@1536\n(1.22\,B)"]
    x = np.arange(len(labels))
    w = 0.36
    # numbers from the JSONs (final n=20)
    b_indirect = [0.088, 0.060]
    f_indirect = [0.372, 0.445]
    b_bpb       = [1.219, 1.784]
    f_bpb       = [0.424, 0.386]
    bars1 = ax.bar(x - w/2, b_indirect, w, label="B: per-user LoRA",
                    color=C_B, edgecolor="black", lw=0.6)
    bars2 = ax.bar(x + w/2, f_indirect, w, label="F: LAYERED",
                    color=C_F, edgecolor="black", lw=0.6)
    # Numeric annotations
    for bar, v in zip(bars1, b_indirect):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
                  f"{v*100:.0f}\%", ha="center", fontsize=8, color=C_B, fontweight="bold")
    for bar, v in zip(bars2, f_indirect):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
                  f"{v*100:.0f}\%", ha="center", fontsize=8, color=C_F, fontweight="bold")
    # Also annotate Δbpb under each pair
    for i, (bb, fb) in enumerate(zip(b_bpb, f_bpb)):
        ax.text(i - w/2, -0.06, f"Δbpb=\n+{bb:.2f}", ha="center", fontsize=7, color=C_B)
        ax.text(i + w/2, -0.06, f"Δbpb=\n+{fb:.2f}", ha="center", fontsize=7, color=C_F)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(-0.13, 0.55)
    ax.set_ylabel("indirect reasoning (any-match)")
    ax.set_title("(a)  Layered F dominates LoRA B\nacross two Engram sizes")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18, lw=0.4)

    # ---- (b) substring-match vs LLM-judge ----
    ax = axes[1]
    conds  = ["A\nno-edit", "B\nLoRA", "C\nEngram", "E\nshared LoRA", "F\nLAYERED"]
    sub_v  = [0.192, 0.060, 0.233, 0.440, 0.445]
    judge_v= [0.085, 0.088, 0.112, 0.310, 0.290]
    cols   = [C_BASE, C_B, C_C, "#88ccee", C_F]
    x = np.arange(len(conds)); w = 0.36
    for i, (sv, jv, c) in enumerate(zip(sub_v, judge_v, cols)):
        ax.bar(i - w/2, sv, w, color=c, edgecolor="black", lw=0.6, alpha=0.55,
                label="substring-match" if i == 0 else None)
        ax.bar(i + w/2, jv, w, color=c, edgecolor="black", lw=0.6,
                hatch="//", label="LLM-judge (Qwen-7B)" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=8)
    ax.set_ylabel("indirect reasoning accuracy")
    ax.set_ylim(0, 0.55)
    ax.set_title("(b)  F still beats B\nunder semantic LLM-judge")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18, lw=0.4)
    # Annotate F/B numbers
    ax.text(4 + w/2, 0.31, "29\%", ha="center", fontsize=8, color=C_F, fontweight="bold")
    ax.text(1 + w/2, 0.107, "9\%", ha="center", fontsize=8, color=C_B, fontweight="bold")

    # ---- (c) rank ablation curve ----
    ax = axes[2]
    ranks = [4, 16, 64]
    f_iA  = [0.29, 0.48, 0.35]
    f_iT  = [0.52, 0.63, 0.37]
    f_bpb = [0.298, 0.386, 1.027]
    ax.plot(ranks, f_iA, "o-", color=C_F, lw=1.8, ms=8, label="indirect any (F)")
    ax.plot(ranks, f_iT, "s--", color="#55aa66", lw=1.4, ms=7, label="indirect top-1 (F)")
    ax.set_xlabel("shared-LoRA rank $r$")
    ax.set_ylabel("indirect reasoning")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks); ax.set_xticklabels([str(r) for r in ranks])
    ax.set_ylim(0, 0.75)
    # twin axis for Δbpb
    ax2 = ax.twinx()
    ax2.plot(ranks, f_bpb, "^:", color="#cc6677", lw=1.4, ms=7, label="Δbpb (right axis)")
    ax2.set_ylabel("Δbpb on held-out text", color="#cc6677")
    ax2.tick_params(axis='y', colors="#cc6677")
    ax2.set_ylim(0, 1.2)
    ax.set_title("(c)  Rank ablation: r=16 is the sweet spot\n(r=64 over-parameterises)")
    # Combined legend
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2,
                loc="upper left", frameon=False, fontsize=7.5)
    ax.spines["top"].set_visible(False)
    # mark r=16 sweet spot
    ax.axvline(16, color="#aaaaaa", ls="--", lw=0.6, alpha=0.5)
    ax.text(16, 0.69, "r=16\nrecommended", ha="center", fontsize=7.5, color="#444")

    plt.tight_layout()
    fig.savefig(OUT / "fig_robustness.pdf", bbox_inches="tight")
    print(f"wrote {OUT / 'fig_robustness.pdf'}")
    plt.close(fig)


# =====================================================================
# Figure: CROSS-BASE — two panels
#   (a) val_bpb contamination on Mini-Engram-d20 vs Qwen-3B (BAR)
#   (b) per-user Δ_indirect distribution across 4 instruction-tuned bases + base LM
# =====================================================================
def fig_cross_base():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios":[1.0, 1.45]})

    # ---- (a) val_bpb contamination ----
    ax = axes[0]
    bases    = ["Mini-Engram-d20\n(base LM)", "Qwen-3B-Instruct\n(instruct LM)"]
    baseline = [0.737, 0.679]
    delta    = [1.78,  0.093]
    x = np.arange(len(bases)); w = 0.42
    b1 = ax.bar(x - w/2, baseline, w, color=C_BASE, edgecolor="black", lw=0.6,
                  label="baseline val\\_bpb")
    b2 = ax.bar(x + w/2, [b + d for b, d in zip(baseline, delta)],
                 w, color=C_BASE, edgecolor="black", lw=0.6, alpha=0.55)
    # Overlay Δ portion (purple)
    for i, (b, d) in enumerate(zip(baseline, delta)):
        ax.bar(i + w/2, d, w, bottom=b, color=C_B, edgecolor="black", lw=0.6,
                 label=("Δbpb (per-user LoRA)" if i == 0 else None))
        # Δ label above
        ax.text(i + w/2, b + d + 0.08, f"Δ=+{d:.2f}", ha="center",
                  fontsize=9, color=C_B, fontweight="bold")
        ax.text(i + w/2, b + d/2, f"{(d/b)*100:.0f}\%\nrelative", ha="center",
                  fontsize=7, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(bases, fontsize=8.5)
    ax.set_ylabel("validation bpb on held-out text")
    ax.set_title("(a)  Per-user LoRA contamination\nis present cross-base")
    ax.set_ylim(0, 3.0)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18, lw=0.4)

    # ---- (b) per-user Δ_indirect distribution across bases ----
    import json, os
    distros = []
    for d_name, label, color in [
        ("stage_a",          "Qwen2.5-3B-Instruct\n(n=30)",    "#5588cc"),
        ("stage_a_llama8b",  "Llama-3.1-8B\n(n=20)",            "#55aa66"),
        ("stage_a_mistral7b","Mistral-7B-v0.3\n(n=20)",         "#aa7733"),
        ("stage_a_qwen7b",   "Qwen2.5-7B-Instruct\n(n=20)",     "#a4a4d4"),
    ]:
        ds = []
        for i in range(30):
            p = f"/home/ubuntu/user-as-lora/results/{d_name}/u{i:03d}/metrics.json"
            if os.path.exists(p):
                m = json.load(open(p))
                ds.append(m["with_adapter"]["indirect_acc"] - m["without_adapter"]["indirect_acc"])
        distros.append((label, color, ds))
    # Add Mini-Engram-d20 — from layered_d20_r16_full.json
    eng_ds = []
    d = json.load(open("/home/ubuntu/user-as-engram/results/layered_d20_r16_full.json"))
    for u in d["per_user"]:
        eng_ds.append((u["B_per_user_lora"]["indirect_any"]
                         - u["A_no_edit"]["indirect_any"]) / u["n_probes"])
    distros.append(("Mini-Engram-d20\n(base LM, n=20)", "#aa4499", eng_ds))

    ax = axes[1]
    n_bases = len(distros)
    for i, (label, color, ds) in enumerate(distros):
        ys = ds
        xs = np.full(len(ys), i) + (np.random.RandomState(42 + i).uniform(-0.18, 0.18, size=len(ys)))
        ax.scatter(xs, ys, color=color, edgecolor="black", lw=0.4, s=42, alpha=0.85, zorder=3)
        # mean indicator
        m = sum(ds) / len(ds)
        ax.plot([i - 0.32, i + 0.32], [m, m], color="black", lw=2.0, zorder=5)
        ax.text(i, m + 0.025, f"mean\n{m:+.2f}", ha="center", fontsize=7.5, fontweight="bold", color="black", zorder=6)
    ax.axhline(0, color="#888", ls="--", lw=0.6, alpha=0.6)
    ax.set_xticks(range(n_bases))
    ax.set_xticklabels([label for label, _, _ in distros], fontsize=7.5)
    ax.set_ylabel(r"$\Delta$ indirect-recall (adapter $-$ base)")
    ax.set_title("(b)  Per-user LoRA's user-level effect\ndepends on the base")
    ax.set_ylim(-0.32, 0.45)
    ax.grid(axis="y", alpha=0.18, lw=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    # Annotate regions
    ax.text(n_bases - 0.5, 0.38, "instruct-tuned bases: LoRA helps on avg.",
              ha="right", fontsize=8, color="#444", style="italic")
    ax.text(n_bases - 0.5, -0.28, "base LM: LoRA hurts in 17/20",
              ha="right", fontsize=8, color="#aa4499", style="italic")

    plt.tight_layout()
    fig.savefig(OUT / "fig_cross_base.pdf", bbox_inches="tight")
    print(f"wrote {OUT / 'fig_cross_base.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    fig_robustness()
    fig_cross_base()
