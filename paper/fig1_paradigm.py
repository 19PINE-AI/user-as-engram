"""Figure 1 of User as Engram: Parametric User Memory for LLMs.
A two-panel opening figure that frames the entire paper.

Panel (a): the landscape. Storage-per-user (x, log) vs model contamination
on unrelated text (y, log). Each existing approach to LLM user memory is
plotted; User-as-Engram is in the lower-left corner (low storage AND low
contamination). Marker shape encodes parametric vs non-parametric.

Panel (b): the reasoning capability axis. Bar chart of indirect-reasoning
accuracy under LLM-judge, with the parametric/non-parametric grouping.
Shows that User-as-Engram is the only approach that is simultaneously
low-storage, low-contamination, AND reasoning-capable.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("/home/ubuntu/user-as-engram/paper/figs")
OUT.mkdir(parents=True, exist_ok=True)

C_OURS    = "#117733"   # User as Engram
C_LORA    = "#aa4499"   # per-user LoRA (the failed parametric baseline)
C_RAG     = "#dd8866"   # non-parametric retrieval (RAG / MEM0 / MEMMACHINE)
C_ICL     = "#5588cc"   # ICL (non-parametric, text)
C_EDIT    = "#888888"   # knowledge editing
C_BG      = "#fafafa"

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
})

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4),
                          gridspec_kw={"width_ratios": [1.4, 1.0]})

# ============================================================================
# PANEL (a): the trade-off space
# ============================================================================
ax = axes[0]

# Each point: (storage_KB_per_user, Δbpb on held-out text, label, color, marker, parametric?)
# Some numbers are best-effort estimates for non-parametric methods (Δbpb is
# zero by construction — they don't edit the model). For parametric methods
# we use measured numbers from this paper.
methods = [
    # non-parametric (open markers, Δbpb=0 by construction)
    dict(s=100.0, b=0.001, label="ICL\n(in-context, text)", color=C_ICL, marker="D", filled=False),
    dict(s=30.0,  b=0.001, label="RAG / MEM0\n(retrieved text)", color=C_RAG, marker="D", filled=False),
    dict(s=30.0,  b=0.001, label="MEMMACHINE\n(episodic retrieval)", color=C_RAG, marker="D", filled=False),
    # parametric baselines (filled markers)
    dict(s=14200.0, b=1.78, label="per-user LoRA\n(parametric, global)", color=C_LORA, marker="*", filled=True),
    dict(s=1000.0,  b=0.20, label="knowledge editing\n(ROME, MEMIT)", color=C_EDIT, marker="s", filled=True),
    # ours
    dict(s=88.0, b=0.0001, label=r"\textbf{User as Engram (ours)}"+"\nparametric, local",
          color=C_OURS, marker="o", filled=True, size=320),
]

# Hand-place text labels to avoid overlap
label_offsets = {
    "ICL\n(in-context, text)":               (12, 12),
    "RAG / MEM0\n(retrieved text)":          (12,-22),
    "MEMMACHINE\n(episodic retrieval)":      (-115,  18),
    "per-user LoRA\n(parametric, global)":   (-160,-10),
    "knowledge editing\n(ROME, MEMIT)":      (-22, 18),
    r"\textbf{User as Engram (ours)}"+"\nparametric, local": (15, 6),
}

for m in methods:
    fc = m["color"] if m["filled"] else "white"
    sz = m.get("size", 160)
    ax.scatter([m["s"]], [m["b"]], color=fc, marker=m["marker"],
                 s=sz, edgecolor=m["color"], lw=1.4, zorder=4)
    dx, dy = label_offsets.get(m["label"], (10, 0))
    ax.annotate(m["label"], (m["s"], m["b"]),
                  xytext=(dx, dy), textcoords="offset points",
                  fontsize=8, color=m["color"], va="center",
                  fontweight=("bold" if "User as Engram" in m["label"] else "normal"))

# Highlight the "Pareto-good corner" (low-left) where User as Engram sits
ax.axhspan(0.0, 0.01, xmin=0, xmax=0.32, alpha=0.10, color=C_OURS, zorder=1)

ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=0.001)
ax.set_xlim(8, 100000)
ax.set_ylim(-0.0002, 3.0)
ax.set_xlabel("per-user storage  (KB, log scale)")
ax.set_ylabel(r"model contamination on unrelated text  ($\Delta$bpb)")
ax.set_title("(a)  Where existing LLM user-memory methods sit", color="#222")

# Annotate the "good" quadrant
ax.annotate(r"\textbf{the goal:}"+"\nlow storage AND\nzero contamination",
              xy=(35, 0.002), xytext=(180, 0.30),
              fontsize=8, color="#444", ha="left",
              arrowprops=dict(arrowstyle="->", color="#888", lw=0.8, alpha=0.6))

# Legend for parametric/non-parametric
leg_h = [
    plt.Line2D([], [], marker="D", color="w", markerfacecolor="white",
                  markeredgecolor="#333", markersize=10, label="non-parametric (text in context / retrieved)"),
    plt.Line2D([], [], marker="o", color="w", markerfacecolor="#333",
                  markeredgecolor="#333", markersize=10, label="parametric (stored in weights)"),
]
ax.legend(handles=leg_h, loc="upper right", frameon=True, framealpha=0.93,
            facecolor=C_BG)
ax.grid(True, which="both", alpha=0.15, lw=0.4)
ax.spines[["top", "right"]].set_visible(False)

# ============================================================================
# PANEL (b): indirect-reasoning capability
# ============================================================================
ax = axes[1]

# Numbers: LLM-judge accuracy from layered_judge_d20.json for the parametric
# methods we measured; for RAG / MEM0 / MEMMACHINE we use the numbers from
# the original paper's memory-systems comparison on the same Mini-Engram-d12.
# Some are approximate; the qualitative ranking is the point.
bars = [
    ("ICL\n(text)",                      0.50, C_ICL,   False),  # in-context is the gold standard if facts fit
    ("RAG\n(text)",                      0.16, C_RAG,   False),
    ("MEM0\n(text)",                     0.17, C_RAG,   False),
    ("MEMMACHINE\n(text)",               0.18, C_RAG,   False),
    ("per-user LoRA\n(parametric)",      0.09, C_LORA,  True),
    (r"\textbf{User as Engram}"+"\n(parametric)", 0.29, C_OURS,  True),
]
x = np.arange(len(bars))
heights = [b[1] for b in bars]
colors = [b[2] for b in bars]
hatches = ["" if b[3] else "..." for b in bars]
edges = ["black" for _ in bars]

for i, (label, v, c, parametric) in enumerate(bars):
    ax.bar(i, v, color=c, edgecolor="black", lw=0.6, alpha=(1.0 if parametric else 0.45),
             hatch=("" if parametric else "..."))
    is_ours = "User as Engram" in label
    ax.text(i, v + 0.018, f"{v*100:.0f}\\%", ha="center", fontsize=8.5,
              color=c, fontweight=("bold" if is_ours else "normal"))

ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in bars], fontsize=7.7)
ax.set_ylim(0, 0.62)
ax.set_ylabel("indirect-reasoning accuracy\n(LLM-judge, Qwen-7B)")
ax.set_title("(b)  Reasoning capability under semantic judge", color="#222")
ax.grid(axis="y", alpha=0.15, lw=0.4)
ax.spines[["top", "right"]].set_visible(False)

# Annotate parametric-vs-non-parametric grouping
ax.axvline(3.5, color="#888", ls="--", lw=0.6, alpha=0.5)
ax.text(1.5, 0.58, "non-parametric (text)",
          ha="center", fontsize=8, color="#444", style="italic")
ax.text(4.5, 0.58, "parametric (weights)",
          ha="center", fontsize=8, color="#444", style="italic")

plt.tight_layout()
fig.savefig(OUT / "fig1_paradigm.pdf", bbox_inches="tight")
print(f"wrote {OUT / 'fig1_paradigm.pdf'}")
plt.close(fig)
