"""Architecture schematic for the layered-architecture section: a three-panel
diagram of the shared-reasoning-skill / per-user-content decomposition.
(a) per-user content via Engram-row override, (b) the shared meta-skill LoRA,
(c) the two stacked live at inference. The six-condition Pareto lives
separately in fig_pareto_layered.pdf, so it is intentionally omitted here."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path("/home/ubuntu/user-as-engram/paper/figs")
OUT.mkdir(parents=True, exist_ok=True)

# Tight NeurIPS-friendly colour palette
C_CONTENT = "#1f77b4"  # blue: per-user Engram (content)
C_SKILL   = "#117733"  # green: shared LoRA (meta-skill)
C_TEXT    = "#222222"

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

fig = plt.figure(figsize=(10.5, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0],
                      wspace=0.18, left=0.02, right=0.98, top=0.88, bottom=0.06)

# -----------------------------------------------------------------------------
# Panel (a): per-user content via Engram-row insertion
# -----------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("(a)  Per-user content\n(Engram override, 88\,KB/user)",
             fontsize=9.5, color=C_TEXT)

user_colors = [C_CONTENT, "#5599cc", "#88bbe0"]
for i, (uid, c) in enumerate(zip(["user A", "user B", "user C"], user_colors)):
    y = 8.2 - i*2.4
    box = FancyBboxPatch((0.5, y - 0.7), 7.0, 1.5,
                         boxstyle="round,pad=0.1,rounding_size=0.15",
                         linewidth=1.1, edgecolor="#333", facecolor=c, alpha=0.18)
    ax.add_patch(box)
    ax.text(0.85, y, uid, fontsize=8.5, color=C_TEXT, va="center")
    for k in range(5):
        rx = 3.0 + k*0.9
        ax.add_patch(plt.Rectangle((rx, y - 0.55), 0.7, 1.1, facecolor=c,
                                   edgecolor="#333", lw=0.6))
    ax.text(8.1, y, "...", fontsize=10, color=C_TEXT, va="center")

ax.text(5.0, 0.7, r"$\Delta$bpb on unrelated text $=\;+0.0001$",
        fontsize=8.5, ha="center", color=C_CONTENT, fontweight="bold")
ax.text(5.0, 0.0, "(fires only on trigger N-gram)",
        fontsize=7.5, ha="center", color="#555", style="italic")

# -----------------------------------------------------------------------------
# Panel (b): shared meta-skill LoRA
# -----------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("(b)  Shared reasoning skill\n(one LoRA, 12\,MB, amortised)",
             fontsize=9.5, color=C_TEXT)

box = FancyBboxPatch((1.4, 4.4), 7.2, 3.4,
                     boxstyle="round,pad=0.15,rounding_size=0.25",
                     linewidth=1.6, edgecolor=C_SKILL, facecolor=C_SKILL, alpha=0.18)
ax.add_patch(box)
ax.text(5.0, 7.2, "shared LoRA", fontsize=10, color=C_SKILL, ha="center", fontweight="bold")
ax.text(5.0, 6.4, "rank-16,  Q/K/V projections", fontsize=8, color="#444", ha="center")
ax.text(5.0, 5.5, r"trained on cross-user samples", fontsize=8.5, color="#222", ha="center")
ax.text(5.0, 4.8, r"\textit{``Facts: $\cdots$ \;\;Q: $\cdots$\;\; A: $\cdots$''}",
        fontsize=8, ha="center", color="#444")

for x0 in (2.0, 8.0):
    ax.annotate("", xy=(5, 4.5), xytext=(x0, 2.8),
                arrowprops=dict(arrowstyle="->", color=C_SKILL, lw=1.2))
ax.text(2.0, 2.4, "training\nusers u020–u029", fontsize=7.5, ha="center", color="#444")
ax.text(8.0, 2.4, "indirect Q+A\nsupervision", fontsize=7.5, ha="center", color="#444")

ax.text(5.0, 1.0, r"$\Delta$bpb on unrelated text $=\;+0.39$",
        fontsize=8.5, ha="center", color=C_SKILL, fontweight="bold")
ax.text(5.0, 0.3, "(amortised across 1 M+ users)",
        fontsize=7.5, ha="center", color="#555", style="italic")

# -----------------------------------------------------------------------------
# Panel (c): inference combines them
# -----------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("(c)  At inference\n(stacked, both layers live)",
             fontsize=9.5, color=C_TEXT)

base = FancyBboxPatch((1.0, 1.0), 8.0, 7.6,
                      boxstyle="round,pad=0.15,rounding_size=0.25",
                      linewidth=1.2, edgecolor="#444", facecolor="#f7f7f7")
ax.add_patch(base)
ax.text(5.0, 8.1, "Mini-Engram base", fontsize=8.5, ha="center", color="#444")

ovr = FancyBboxPatch((1.6, 5.8), 6.8, 1.2,
                     boxstyle="round,pad=0.08,rounding_size=0.15",
                     linewidth=1.0, edgecolor=C_CONTENT, facecolor=C_CONTENT, alpha=0.25)
ax.add_patch(ovr)
ax.text(5.0, 6.4, "per-user Engram override (88\,KB)",
        fontsize=8, ha="center", color=C_CONTENT, fontweight="bold")

sho = FancyBboxPatch((1.6, 2.4), 6.8, 1.2,
                     boxstyle="round,pad=0.08,rounding_size=0.15",
                     linewidth=1.0, edgecolor=C_SKILL, facecolor=C_SKILL, alpha=0.25)
ax.add_patch(sho)
ax.text(5.0, 3.0, "shared LoRA delta on $Q/K/V$",
        fontsize=8, ha="center", color=C_SKILL, fontweight="bold")

ax.annotate("forward", xy=(8.6, 4.7), xytext=(8.6, 4.7), fontsize=7, color="#444")
ax.annotate("", xy=(8.7, 8.4), xytext=(8.7, 1.4),
            arrowprops=dict(arrowstyle="<->", color="#888", lw=1.0))

ax.text(5.0, 0.3, "total per-user storage: 88\,KB",
        fontsize=8.5, ha="center", color=C_TEXT, fontweight="bold")

fig.savefig(OUT / "fig_layered_arch.pdf", bbox_inches="tight")
print(f"wrote {OUT / 'fig_layered_arch.pdf'}")
plt.close(fig)
