"""Figure: the two-stage architecture of User-as-Engram.

Visually shows: (a) one-time foundational-model post-training using cross-user
in-context-reasoning samples, producing a meta-skill-aware base; (b) per-user
content stored as Engram-row overrides on top; (c) per-request serving with
no per-user weight load.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

OUT = Path("/home/ubuntu/user-as-engram/paper/figs")

C_PRETRAIN  = "#5588cc"   # foundational pretraining (existing)
C_POSTTRAIN = "#117733"   # meta-skill post-training (one-time, this paper)
C_ENGRAM    = "#1f77b4"   # per-user Engram override
C_INFER     = "#aa6633"   # per-request inference
C_TEXT      = "#222"

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, ax = plt.subplots(figsize=(13, 4.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")

# ----------------------- Stage 1: post-training (one-time) -----------------
ax.add_patch(FancyBboxPatch((1.5, 16), 27, 13,
    boxstyle="round,pad=0.4,rounding_size=0.6", linewidth=1.3,
    edgecolor=C_POSTTRAIN, facecolor=C_POSTTRAIN, alpha=0.10))
ax.text(15, 27.2, r"\textbf{Stage 1: post-train meta-skill into the foundational model}",
          ha="center", fontsize=9.5, color=C_POSTTRAIN)
ax.text(15, 25.6, r"\textit{one-time, amortised across all users}",
          ha="center", fontsize=8, color="#444", style="italic")

# 4 training-user icons
for i in range(4):
    x = 3.6 + i*3.0
    ax.add_patch(FancyBboxPatch((x, 20.5), 2.4, 2.2,
        boxstyle="round,pad=0.1,rounding_size=0.18", linewidth=0.8,
        edgecolor="#888", facecolor="#e8eef4"))
    ax.text(x + 1.2, 21.6, f"u{20+i:02d}", ha="center", fontsize=7, color="#444")

# Right side: post-train pipeline
ax.annotate("", xy=(20.5, 21.6), xytext=(15.8, 21.6),
              arrowprops=dict(arrowstyle="->", color="#444", lw=1.0))
ax.text(18.2, 22.4, "facts $\\to$\nreasoning", fontsize=7, color="#444", ha="center")

# Post-trained foundation = Engram base + merged meta-skill LoRA
ax.add_patch(FancyBboxPatch((21, 19.0), 6.5, 5.2,
    boxstyle="round,pad=0.2,rounding_size=0.3", linewidth=1.2,
    edgecolor=C_POSTTRAIN, facecolor=C_POSTTRAIN, alpha=0.30))
ax.text(24.25, 22.9, "meta-skill-aware", fontsize=8, ha="center", color=C_POSTTRAIN, fontweight="bold")
ax.text(24.25, 22.0, "foundational model", fontsize=8, ha="center", color=C_POSTTRAIN, fontweight="bold")
ax.text(24.25, 21.0, "(Engram + merged", fontsize=7, ha="center", color="#444")
ax.text(24.25, 20.2, "rank-16 LoRA)", fontsize=7, ha="center", color="#444")

# Footer for stage 1
ax.text(15, 18.0, r"\textit{510 cross-user samples}  $\cdot$  \textit{2K steps}  $\cdot$  \textit{$\sim$2 min}",
          ha="center", fontsize=7.5, color="#555")

# ----------------------- Stage 2: per-user content (per-user) ---------------
ax.add_patch(FancyBboxPatch((30, 16), 32, 13,
    boxstyle="round,pad=0.4,rounding_size=0.6", linewidth=1.3,
    edgecolor=C_ENGRAM, facecolor=C_ENGRAM, alpha=0.10))
ax.text(46, 27.2, r"\textbf{Stage 2: store each user as Engram-row overrides}",
          ha="center", fontsize=9.5, color=C_ENGRAM)
ax.text(46, 25.6, r"\textit{per-user, local, 88\,KB each}",
          ha="center", fontsize=8, color="#444", style="italic")

# Show three users, each with an override map
for i, (uid, color) in enumerate(zip(["user A", "user B", "user C"],
                                       ["#1f77b4", "#5599cc", "#88bbe0"])):
    x = 32 + i*9.6
    ax.add_patch(FancyBboxPatch((x, 19.5), 8.4, 4.5,
        boxstyle="round,pad=0.15,rounding_size=0.2", linewidth=0.9,
        edgecolor=color, facecolor=color, alpha=0.18))
    ax.text(x + 4.2, 22.9, uid, ha="center", fontsize=8, color=color, fontweight="bold")
    # 5 row tiles
    for k in range(5):
        rx = x + 0.5 + k*1.55
        ax.add_patch(Rectangle((rx, 20.3), 1.3, 1.7, facecolor=color,
                                  edgecolor="#333", lw=0.5))
    ax.text(x + 4.2, 19.8, "Joint OPT, 88\,KB", ha="center", fontsize=7, color="#444")

ax.text(46, 18.0, r"\textit{$\sim$45 sec/user training}  $\cdot$  \textit{0 base-weight changes}",
          ha="center", fontsize=7.5, color="#555")

# ----------------------- Stage 3: per-request inference --------------------
ax.add_patch(FancyBboxPatch((63.5, 16), 35, 13,
    boxstyle="round,pad=0.4,rounding_size=0.6", linewidth=1.3,
    edgecolor=C_INFER, facecolor=C_INFER, alpha=0.10))
ax.text(81, 27.2, r"\textbf{Stage 3: serve}",
          ha="center", fontsize=9.5, color=C_INFER)
ax.text(81, 25.6, r"\textit{shared HBM + per-user DRAM swap}",
          ha="center", fontsize=8, color="#444", style="italic")

# Request flow diagram
# Foundational model (always in HBM)
ax.add_patch(FancyBboxPatch((65.5, 19.5), 11, 4.5,
    boxstyle="round,pad=0.15,rounding_size=0.25", linewidth=1.0,
    edgecolor=C_POSTTRAIN, facecolor=C_POSTTRAIN, alpha=0.18))
ax.text(71, 22.5, "foundational", fontsize=8, ha="center", color=C_POSTTRAIN, fontweight="bold")
ax.text(71, 21.5, "model", fontsize=8, ha="center", color=C_POSTTRAIN, fontweight="bold")
ax.text(71, 20.3, "(in HBM)", fontsize=7, ha="center", color="#444")

# Arrow + per-request override
ax.annotate("", xy=(82.5, 22.5), xytext=(77.0, 22.5),
              arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))
ax.text(79.75, 23.5, "user X", fontsize=7, ha="center", color="#444")
ax.text(79.75, 21.4, "override", fontsize=7, ha="center", color="#444")
ax.text(79.75, 20.7, "lookup", fontsize=7, ha="center", color="#444")

# User X override map (transient)
ax.add_patch(FancyBboxPatch((82.5, 19.5), 6.5, 4.5,
    boxstyle="round,pad=0.15,rounding_size=0.25", linewidth=1.0,
    edgecolor=C_ENGRAM, facecolor=C_ENGRAM, alpha=0.20))
ax.text(85.75, 22.4, "user X", fontsize=8, ha="center", color=C_ENGRAM, fontweight="bold")
ax.text(85.75, 21.4, "Engram", fontsize=8, ha="center", color=C_ENGRAM)
ax.text(85.75, 20.4, "88\,KB", fontsize=7, ha="center", color="#444")

# Output
ax.annotate("", xy=(95.5, 22.5), xytext=(89.5, 22.5),
              arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))
ax.text(92.5, 23.5, "response", fontsize=7, ha="center", color="#444")
ax.text(92.5, 21.4, "$\sim$4 ms", fontsize=7, ha="center", color="#444")

ax.text(81, 18.0, r"\textit{0\% cross-user leak}  $\cdot$  \textit{232 req/s}  $\cdot$  \textit{no per-user weight load}",
          ha="center", fontsize=7.5, color="#555")

# ----------------------- Bottom: at-scale numbers ---------------------------
ax.text(50, 12.5, r"\textbf{at $\mathbf{10^6}$ users}",
          ha="center", fontsize=10, color=C_TEXT)

# Three columns of numbers
nums = [
    ("foundational model",     "$\\sim$2.5 GB",    "(shared)",            C_POSTTRAIN),
    ("per-user content total", "88 GB",           "(88 KB $\\times 10^6$)", C_ENGRAM),
    ("vs per-user LoRA at scale", "14.2 TB",       "($\\sim$160$\\times$ more)", "#aa4499"),
]
for i, (label, big, sub, color) in enumerate(nums):
    cx = 18 + i*32
    ax.text(cx, 9.5, label,   ha="center", fontsize=8,  color="#444")
    ax.text(cx, 7.4, big,     ha="center", fontsize=14, color=color, fontweight="bold")
    ax.text(cx, 5.5, sub,     ha="center", fontsize=7.5, color="#666")

# Total
ax.add_patch(FancyBboxPatch((35, 1.0), 30, 3.0,
    boxstyle="round,pad=0.2,rounding_size=0.3", linewidth=1.1,
    edgecolor="#117733", facecolor="#117733", alpha=0.10))
ax.text(50, 2.4, r"\textbf{User as Engram: $\sim$90 GB total} for $10^6$ users on commodity hardware",
          ha="center", fontsize=9.5, color="#117733", fontweight="bold")

# Stage-to-stage arrows
ax.annotate("", xy=(30, 22.5), xytext=(28, 22.5),
              arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.5))
ax.annotate("", xy=(63.5, 22.5), xytext=(62, 22.5),
              arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.5))

fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight")
print(f"wrote {OUT / 'fig_pipeline.pdf'}")
plt.close(fig)
