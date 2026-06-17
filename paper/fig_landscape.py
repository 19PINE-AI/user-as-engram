"""
fig_landscape.pdf -- "where User as Engram sits" positioning figure for the
Background, in the style of PreAct's landscape figure. Two qualitative axes:
  x: zero query-time context cost (left = pays context tokens; right = none)
  y: the edit stays local (bottom = global/entangled weight edit; top = isolated)
User as Engram is the only method in the top-right corner: local AND zero-context.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
BLUE = "#34507F"; GRAY = "#8A8F9A"; BLUE_LT = "#7E97C4"
FIGS = Path(__file__).parent / "figs"

fig, ax = plt.subplots(figsize=(7.0, 4.4))

# winning quadrant shading (top-right)
ax.add_patch(Rectangle((5, 5), 5, 5, facecolor=BLUE_LT, alpha=0.16, zorder=0))
ax.axvline(5, color=GRAY, lw=0.8, alpha=0.5, zorder=1)
ax.axhline(5, color=GRAY, lw=0.8, alpha=0.5, zorder=1)

# prior methods: (x, y, label, dx, dy, ha)
pts = [
    (1.6, 8.7, "In-context (ICL)",            0.0, -0.6, "center"),
    (2.7, 7.4, "Retrieval (RAG)",             0.0, -0.6, "center"),
    (3.3, 9.0, "Memory systems\n(Mem0, A-MEM, MemMachine)", 0.25, 0.5, "left"),
    (8.3, 2.3, "Per-user LoRA",               0.0, -0.65, "center"),
    (6.7, 3.4, "Knowledge editing\n(ROME, MEMIT)", -0.2, 0.55, "right"),
]
for x, y, lab, dx, dy, ha in pts:
    ax.scatter([x], [y], s=130, color=GRAY, edgecolor="black", linewidth=0.5, zorder=3)
    ax.annotate(lab, (x + dx, y + dy), ha=ha, va="center", fontsize=8.5, color="#333333")

# ours
ax.scatter([8.5], [8.6], marker="*", s=520, color=BLUE, edgecolor="black",
           linewidth=0.6, zorder=4)
ax.annotate("User as Engram", (8.5, 8.6 - 0.75), ha="center", va="center",
            fontsize=11, fontweight="bold", color=BLUE)
ax.text(7.5, 6.4, "local edit\n+ zero context", ha="center", va="center",
        fontsize=8.5, style="italic", color=BLUE)

ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel(r"zero query-time context cost $\rightarrow$", fontweight="bold")
ax.set_ylabel(r"the edit stays local $\rightarrow$", fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
out = FIGS / "fig_landscape.pdf"
plt.savefig(out); plt.close()
print(f"Wrote {out}")
