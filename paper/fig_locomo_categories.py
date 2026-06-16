"""LOCOMO category breakdown: Engram Joint-OPT vs best retrieval baseline,
across 4 answerable categories x 3 dense scales (token-F1).
Replaces the dense 12-row Table (tab:locomo-categories) with one figure.
Values transcribed from results/ (see main.tex Table 'LOCOMO category x model').
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

scales = ["d12 v2\n(339M)", "d12@1280\n(625M)", "d20@1536\n(1.22B)"]
# best retrieval = max(MEM0, MEMMACHINE) per cell; engram = Joint-OPT
data = {
    "single-hop": {"retr": [0.131, 0.160, 0.169], "eng": [0.169, 0.176, 0.233]},
    "multi-hop":  {"retr": [0.092, 0.115, 0.108], "eng": [0.243, 0.253, 0.299]},
    "reasoning":  {"retr": [0.117, 0.136, 0.168], "eng": [0.183, 0.203, 0.266]},
    "open-domain":{"retr": [0.247, 0.341, 0.338], "eng": [0.122, 0.146, 0.159]},
}

C_RETR = "#C24A3F"   # muted red
C_ENG = "#34507F"    # slate blue (arxivaccent)

fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.7), sharey=True)
x = np.arange(len(scales))
w = 0.38
for ax, (cat, d) in zip(axes, data.items()):
    ax.bar(x - w/2, d["retr"], w, label="best retrieval", color=C_RETR)
    ax.bar(x + w/2, d["eng"], w, label="Engram J-OPT", color=C_ENG)
    eng_wins = np.mean(d["eng"]) >= np.mean(d["retr"])
    ax.set_title(cat, fontsize=11, fontweight="bold",
                 color=C_ENG if eng_wins else C_RETR)
    ax.set_xticks(x)
    ax.set_xticklabels(scales, fontsize=7.5)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("LOCOMO token-F1", fontsize=10)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig("figs/fig_locomo_categories.pdf", bbox_inches="tight")
print("wrote figs/fig_locomo_categories.pdf")
