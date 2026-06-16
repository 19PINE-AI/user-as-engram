#!/usr/bin/env python3
"""Two methods-section architecture schematics:

  fig_engram_in_transformer.pdf  — where the Engram lives in the transformer
                                    stack and how it interacts with Attn/MLP.
  fig_user_insertion.pdf         — how a user fact is written into the Engram
                                    table (the three insertion strategies).

Titles live in the LaTeX \\caption; only short structural labels appear in the
figures. matplotlib mathtext only interprets $...$, so no \\textbf/\\emph/\\,
outside math. Palette: slate blue #34507F = ours, muted red #C24A3F = contrast.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

FIGS = Path(__file__).parent / "figs"
OUR = "#34507F"
ENGFILL = "#dbe6f5"
ACC = "#C24A3F"
GATE = "#fff3b0"
PLAIN = "#ececec"


def _box(ax, x, y, w, h, text, fc=PLAIN, ec="black", fs=8, lw=0.8,
         style="round,pad=0.04", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                                edgecolor=ec, facecolor=fc))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=weight)


def _arrow(ax, x1, y1, x2, y2, color="black", lw=0.9, ls="-", ms=11, rad=0.0):
    cs = f"arc3,rad={rad}" if rad else "arc3,rad=0"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=1, shrinkB=1,
                                 connectionstyle=cs))


# ---------------------------------------------------------------------------
# Figure 1: where the Engram lives inside the transformer
# ---------------------------------------------------------------------------
def fig_engram_in_transformer():
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 6.4); ax.axis("off")

    # ---- left: transformer stack (contiguous boxes) ----
    sx, sw = 0.45, 2.2
    rows = [
        (0.66, "Token embeddings", PLAIN, "black", 0.8),
        (0.66, "Block 1:  Attention + MLP", PLAIN, "black", 0.8),
        (0.86, "Block 2:  Attention + MLP\n$+$ Engram lookup  $\\bigstar$", ENGFILL, OUR, 1.6),
        (0.46, "$\\vdots$", "white", "white", 0.0),
        (0.86, "Block 7:  Attention + MLP\n$+$ Engram lookup  $\\bigstar$", ENGFILL, OUR, 1.6),
        (0.46, "$\\vdots$", "white", "white", 0.0),
        (0.66, "LM head (frozen)", PLAIN, "black", 0.8),
    ]
    y = 0.5
    centers = []
    for h, txt, fc, ec, lw in rows:
        _box(ax, sx, y, sw, h, txt, fc=fc, ec=ec, lw=lw,
             fs=(8 if "vdots" not in txt else 13))
        centers.append((y, h))
        y += h + 0.04
    top = y
    _arrow(ax, sx - 0.28, 0.5, sx - 0.28, top - 0.1, color="#888", lw=1.2)
    ax.text(sx - 0.42, (0.5 + top) / 2, "residual stream", rotation=90,
            ha="center", va="center", fontsize=7.5, color="#666")

    by, bh = centers[4]            # Block 7 — the exploded layer
    eng_y = by + bh / 2
    _arrow(ax, sx + sw, eng_y, 3.35, 3.05, color=OUR, lw=1.5)

    # ---- right: exploded Engram module ----
    ax.add_patch(Rectangle((3.35, 1.15), 6.65, 3.95, fill=False,
                           edgecolor=OUR, linewidth=1.0, linestyle=(0, (4, 2))))
    ax.text(3.5, 4.92, "Engram lookup at an Engram layer", fontsize=8.2,
            style="italic", color=OUR, ha="left", va="center")

    # row A: suffix N-gram -> hash heads -> addresses
    ya = 3.75
    _box(ax, 3.6, ya, 1.95, 0.95, "suffix $N$-gram\n$x_{t-2}, x_{t-1}, x_t$",
         fc="#f3f3f3", fs=8)
    _box(ax, 6.0, ya, 1.7, 0.95, "$K$ hash heads\n(mult-XOR)", fc="#f3f3f3", fs=8)
    _box(ax, 8.15, ya, 1.7, 0.95, "$K$ addresses\n$z_{n,k}$",
         fc="#f3f3f3", ec=OUR, lw=1.2, fs=8)
    _arrow(ax, 5.55, ya + 0.47, 6.0, ya + 0.47)
    _arrow(ax, 7.7, ya + 0.47, 8.15, ya + 0.47)
    ax.text(9.0, ya - 0.16, "known before\nthe forward pass", style="italic",
            ha="center", va="top", fontsize=6.6, color="#555")

    # row B: table lookup -> project -> gate -> output
    yb = 1.55
    _box(ax, 3.6, yb, 1.95, 0.95, "table lookup\n$E[z]\\!\\to\\! e_t$",
         fc=ENGFILL, ec=OUR, lw=1.2, fs=8)
    _box(ax, 6.0, yb, 1.35, 0.95, "$W_K, W_V$\nproject", fc=PLAIN, fs=8)
    _box(ax, 7.7, yb, 1.05, 0.95, "gate\n$\\alpha=\\sigma(\\cdot)$", fc=GATE, fs=8)
    _box(ax, 8.95, yb, 0.95, 0.95, "$\\alpha\\, W_V e_t$", fc=ENGFILL, ec=OUR, lw=1.2, fs=7.5)
    _arrow(ax, 8.9, ya, 4.6, yb + 0.95, color=OUR, lw=1.1, rad=0.18)
    _arrow(ax, 5.55, yb + 0.47, 6.0, yb + 0.47)
    _arrow(ax, 7.35, yb + 0.47, 7.7, yb + 0.47)
    _arrow(ax, 8.75, yb + 0.47, 8.95, yb + 0.47)
    # output back to residual add at Block 7 — bow downward to avoid boxes
    _arrow(ax, 9.0, yb, sx + sw + 0.02, eng_y - 0.2, color=ACC, lw=1.3, rad=-0.32)
    ax.text(6.2, 0.82, "added to the residual only at the trigger position",
            ha="center", va="center", fontsize=7.2, color=ACC,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.95))

    plt.tight_layout()
    out = FIGS / "fig_engram_in_transformer.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=160); plt.close()
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure 2: inserting a user fact into the Engram table
# ---------------------------------------------------------------------------
def fig_user_insertion():
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 6.0); ax.axis("off")

    # ---- top pipeline: fact -> trigger tokens -> hash -> addresses ----
    yt = 4.7
    _box(ax, 0.3, yt, 2.25, 1.0,
         "user fact\n'My cardiologist is'\n$\\to$ 'Dr. Vasquez'", fc="#f3f3f3", fs=7.8)
    _box(ax, 3.0, yt, 1.85, 1.0,
         "tokenise trigger\n$x_1\\dots x_T$,\n$t^\\star\\!=\\!T\\!-\\!1$", fc="#f3f3f3", fs=7.8)
    _box(ax, 5.35, yt, 1.95, 1.0,
         "deterministic hash\n$\\to$ rows $R_f\\!\\subset\\![|E|]$\n($\\sim$16 rows)",
         fc="#f3f3f3", ec=OUR, lw=1.2, fs=7.8)
    _arrow(ax, 2.55, yt + 0.5, 3.0, yt + 0.5)
    _arrow(ax, 4.85, yt + 0.5, 5.35, yt + 0.5)

    # ---- the Engram table: most rows grey, R_f highlighted ----
    tx, tw, trows = 7.7, 2.2, 12
    ax.text(tx + tw / 2, yt + 1.18, "Engram table $E$", ha="center", fontsize=8, color=OUR)
    rh = 1.9 / trows
    hot = {3, 4, 8}
    tbot = (yt + 0.95) - trows * rh
    for i in range(trows):
        ry = (yt + 0.95) - (i + 1) * rh
        fc = ACC if i in hot else "#e9e9e9"
        ax.add_patch(Rectangle((tx, ry), tw, rh * 0.9, facecolor=fc,
                               edgecolor="white", linewidth=0.5))
    ax.add_patch(Rectangle((tx, tbot), tw, trows * rh, fill=False,
                           edgecolor=OUR, linewidth=1.0))
    _arrow(ax, 7.3, yt + 0.5, 7.7, yt + 0.5, color=OUR, lw=1.1)
    ax.text(tx + tw + 0.08, yt + 0.5, "write\nrows $R_f$", ha="left", va="center",
            fontsize=7, color=ACC)

    # ---- middle: the three insertion strategies write the value e* ----
    ax.text(0.3, 3.6, "write a value $e^\\star$ into rows $R_f$  (three strategies, increasing cost / quality):",
            fontsize=8.2, ha="left", va="center")
    sy, sh, sw2 = 1.95, 1.4, 3.0

    def strat(x, name, detail, fc, ec, lw, w=sw2):
        _box(ax, x, sy, w, sh, "", fc=fc, ec=ec, lw=lw)
        ax.text(x + w / 2, sy + sh - 0.27, name, ha="center", va="center",
                fontsize=8, fontweight="bold")
        ax.text(x + w / 2, sy + (sh - 0.34) / 2, detail, ha="center", va="center", fontsize=7.5)

    strat(0.3, "UNEMBED_P  (closed form)",
          "$e^\\star = W_V^{\\dagger} U_y$\none mat-vec, ${<}1$ ms", "#f3f3f3", "black", 0.8)
    strat(3.55, "OPT  (per fact)",
          "15 Adam steps on $e$\nmax $\\log p(y\\,|\\,\\mathrm{trig})$,  $\\sim$1 s",
          "#eef2f8", OUR, 1.0)
    strat(6.8, "Joint OPT  (default, $\\geq$30 facts)",
          "one row tensor over all the\nuser's addresses; sample a\nfact / step, backprop the stack",
          ENGFILL, OUR, 1.4, w=3.1)
    # strategies feed the written rows in the table
    _arrow(ax, 8.4, sy + sh, 8.7, tbot - 0.02, color=OUR, lw=1.0, rad=0.0)

    # ---- bottom: result, the per-user override map ----
    _box(ax, 1.7, 0.35, 6.7, 1.05,
         "per-user override map $\\{r_i \\mapsto v_i\\}$  ($\\sim$88 KB)  —  only these rows differ from the base;\n"
         "everything else is bit-identical, so writes compose and never leak across users",
         fc="#eef7ee", ec="#3a7d3a", lw=1.0, fs=7.6)
    _arrow(ax, 5.0, sy, 5.0, 1.42, color="#3a7d3a", lw=1.1)

    plt.tight_layout()
    out = FIGS / "fig_user_insertion.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=160); plt.close()
    print("wrote", out)


if __name__ == "__main__":
    fig_engram_in_transformer()
    fig_user_insertion()
