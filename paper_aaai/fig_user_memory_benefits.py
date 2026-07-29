"""Generate the main-paper positioning figure for User as Engram.

The figure deliberately reports only quantities already present in the paper:
100-fact serialized state, the measured/implemented fact lifecycle, and the
three-seed indirect-reasoning comparison.  It replaces the density curve as
Figure 5 so the visual answers the reviewer-facing question, "why not a
per-user LoRA?"
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


BLUE = "#34507F"
BLUE_LT = "#DDE5F2"
RED = "#C24A3F"
RED_LT = "#F3DFDC"
GRAY = "#68707D"
LIGHT = "#E3E5E8"
GREEN = "#3E7C5A"


def clean(ax, axis="x"):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis=axis, ls=":", lw=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def lifecycle_box(ax, x, color, fill, title, rows):
    box = FancyBboxPatch(
        (x, 0.04), 0.455, 0.77,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.0, edgecolor=color, facecolor=fill,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.2275, 0.72, title, ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=color, transform=ax.transAxes,
    )
    for heading_y, body_y, heading, body in rows:
        ax.text(
            x + 0.025, heading_y, heading, ha="left", va="center",
            fontsize=9.0, fontweight="bold", color="#252932",
            transform=ax.transAxes,
        )
        ax.text(
            x + 0.025, body_y, body, ha="left", va="center",
            fontsize=9.0, color="#252932", transform=ax.transAxes,
        )


def main():
    # Generate at column width so labels retain their specified point sizes.
    fig = plt.figure(figsize=(3.25, 3.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 2.10, 0.85], hspace=0.62)

    # (a) The inserted facts are sparse state, not a model-shaped adapter.
    ax = fig.add_subplot(gs[0])
    labels = ["Engram override", "rank-64 per-user LoRA"]
    storage_kb = [88, 14_200]
    y = np.arange(2)
    ax.barh(y, storage_kb, color=[BLUE, RED], height=0.48)
    ax.set_xscale("log")
    ax.set_xlim(30, 35_000)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("(a)  State for 100 inserted facts (KB, log scale)", loc="left",
                 fontsize=9.5, fontweight="bold", pad=3)
    ax.text(105, 0, "88 KB", va="center", fontsize=9, fontweight="bold", color=BLUE)
    ax.text(12_900, 1, "14.2 MB", va="center", ha="right", fontsize=9,
            fontweight="bold", color="white")
    ax.text(600, 0.51, r"$161\times$ smaller", ha="center", va="center",
            fontsize=9, fontweight="bold", color=BLUE)
    clean(ax)

    # (b) Lifecycle is an operation on addressed records for Engram; a LoRA has
    # no individual fact address.  Avoid inventing update/deletion timings.
    ax = fig.add_subplot(gs[1])
    ax.set_axis_off()
    ax.set_title("(b)  Fact lifecycle avoids adapter retraining", loc="left",
                 fontsize=9.5, fontweight="bold", pad=3)
    lifecycle_box(
        ax, 0.0, BLUE, BLUE_LT, "ENGRAM",
        [
            (0.54, 0.40, "WRITE", "closed form; OPT"),
            (0.22, 0.09, "DELETE MAP", "drop map"),
        ],
    )
    lifecycle_box(
        ax, 0.545, RED, RED_LT, "PER-USER LoRA",
        [
            (0.54, 0.40, "WRITE", "train adapter"),
            (0.22, 0.09, "DELETE FACT", "retrain / unlearn"),
        ],
    )

    # (c) Facts stored in rows can be consumed by one reusable reasoning skill.
    # Three-seed means from Table 3: layered Engram 41.2, per-user LoRA 7.3.
    ax = fig.add_subplot(gs[2])
    labels = ["Engram + shared skill", "per-user LoRA"]
    reasoning = [41.2, 7.3]
    y = np.arange(2)
    ax.barh(y, reasoning, color=[BLUE, RED], height=0.48)
    ax.set_xlim(0, 50)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("indirect reasoning over held-out user facts (%)", fontsize=9)
    ax.set_title("(c)  Local facts support indirect and multi-hop use", loc="left",
                 fontsize=9.5, fontweight="bold", pad=3)
    for yi, value in zip(y, reasoning):
        ax.text(value + 0.9, yi, f"{value:.1f}%", va="center", fontsize=9,
                fontweight="bold", color=BLUE if yi == 0 else RED)
    clean(ax)

    fig.savefig("figs/fig_user_memory_benefits.pdf", bbox_inches="tight")
    fig.savefig("figs/fig_user_memory_benefits.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_user_memory_benefits.pdf")


if __name__ == "__main__":
    main()
