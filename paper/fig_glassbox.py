"""
New mechanistic figures for the body §4 "glass box" section.

  fig_glassbox.pdf        — 3-panel summary of the trained-model mechanism:
      (a) self-gating: trigger gate alpha before vs after a write, vs non-trigger
      (b) injection IS the value path (cos(dy, W_V.e)=0.999 for the deployed OPT
          row) + what the row encodes vs gold (UNEMBED 0.6 -> OPT/Joint 0.2)
      (c) exact locality: max non-trigger residual change across all 16 facts = 0

  fig_lora_vs_engram.pdf  — side-by-side per-position/per-layer residual change:
      Engram surgical insertion (exact 0 off-trigger) vs a per-user LoRA that
      learns the SAME fact (nonzero at every position and every layer).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE = "#34507F"; BLUE_LT = "#7E97C4"; RED = "#C24A3F"; ORANGE = "#D98A3D"
GREEN = "#3E7C5A"; GRAY = "#8A8F9A"; GRAY_LT = "#C2C6CE"
RES = Path("/home/ubuntu/user-as-engram/results")
FIGS = Path(__file__).parent / "figs"


def fig_glassbox(model_tag="d20"):
    d = json.load(open(RES / f"glassbox_{model_tag}.json"))
    agg = d["aggregate"]
    g = agg["gate"]

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7))

    # (a) self-gating
    ax = axes[0]
    labels = ["baseline\n(no write)", "after\nUNEMBED_P", "after\nOPT"]
    trig_vals = [g["mean_alpha_trigger"],
                 agg["UNEMBED_P"]["mean_alpha_after_trigger"],
                 agg["OPT"]["mean_alpha_after_trigger"]]
    nontrig_vals = [g["mean_alpha_nontrig_max"],
                    agg["UNEMBED_P"]["mean_alpha_after_nontrig_max"],
                    agg["OPT"]["mean_alpha_after_nontrig_max"]]
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, trig_vals, w, color=BLUE, label="trigger position", edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, nontrig_vals, w, color=GRAY_LT, label="non-trigger (max)", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"Engram gate $\alpha$")
    ax.set_title("(a)", loc="left", fontweight="bold")
    ax.set_ylim(0, 1.05); ax.legend(loc="upper left")
    for xi, v in zip(x - w/2, trig_vals):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)

    # (b) two distinct cosines, visually separated: (left) the row's effect
    # aligns with its value path; (right) how the row relates to the gold token.
    ax = axes[1]
    labels = ["$W_V e$\n(OPT)", "UNEMB.", "OPT", "Joint\nOPT"]
    vals = [agg["OPT"]["mean_cos_dy_to_WVmarker"],
            agg["UNEMBED_P"]["mean_cos_WVrow_to_gold"],
            agg["OPT"]["mean_cos_WVrow_to_gold"],
            agg["joint_opt"]["mean_cos_WVrow_to_gold"]]
    # Slate-family palette: dark blue for the headline value-path alignment,
    # a light->dark slate ramp for the three write strategies (no warm-on-warm).
    colors = [BLUE, "#AEBDD9", "#8AA0C8", "#5E79A8"]
    pos = [0.0, 1.3, 2.2, 3.1]
    bars = ax.bar(pos, vals, color=colors, edgecolor="black", linewidth=0.4, width=0.78)
    ax.axvline(0.7, color=GRAY, ls=":", lw=0.9)
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("cosine similarity"); ax.set_ylim(0, 1.32)
    ax.set_xlim(-0.6, 3.7)
    ax.set_title("(b)", loc="left", fontweight="bold")
    ax.text(0.0, 1.22, "effect $=$\nvalue path", ha="center", va="center", fontsize=6.6,
            style="italic", color=BLUE)
    ax.text(2.13, 1.24, "row vs. gold-token\ndirection", ha="center", va="center",
            fontsize=6.6, style="italic", color="#555555")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + 0.025, f"{v:.2f}", ha="center", fontsize=7)

    # (c) exact locality across all facts
    ax = axes[2]
    strat_names = ["UNEMBED_P", "OPT"]
    maxnt = [agg[s]["max_nontrig_diff_over_facts"] for s in strat_names]
    trigfinal = [agg[s]["mean_trig_final_diff"] for s in strat_names]
    x = np.arange(len(strat_names)); w = 0.38
    # plot trigger change (nonzero) and non-trigger change (zero) on log axis with floor
    floor = 1e-3
    ax.bar(x - w/2, [max(t, floor) for t in trigfinal], w, color=BLUE, label="trigger position",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, [max(m, floor) for m in maxnt], w, color=GRAY_LT,
           label="max non-trigger\n(all 16 facts)", edgecolor="black", linewidth=0.4)
    ax.set_yscale("log"); ax.set_ylim(floor, max(trigfinal)*5)
    ax.set_xticks(x); ax.set_xticklabels(strat_names)
    ax.set_ylabel(r"$\|\Delta$ residual$\|$ (final layer)")
    ax.set_title("(c)", loc="left", fontweight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1,
              frameon=True, fontsize=7.5, handlelength=1.0,
              borderaxespad=0.0)
    for xi, m in zip(x + w/2, maxnt):
        ax.text(xi, floor*1.3, f"{m:.0e}", ha="center", fontsize=7, rotation=0)

    plt.tight_layout()
    out = FIGS / "fig_glassbox.pdf"
    plt.savefig(out); plt.close()
    print(f"Wrote {out}")


def fig_lora_vs_engram(model_tag="d12_1280"):
    d = json.load(open(RES / f"lora_vs_engram_{model_tag}.json"))
    eng = np.array(d["engram"]["per_layer_per_pos"])      # [layers, pos]
    trig = d["engram"]["trig_pos"]
    lora = np.array(d["lora"]["trigger_map"])              # [layers, pos]

    # align shapes (same sentence/tokens)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.7), sharey=True)
    floor = 1e-3
    vmax = max(eng.max(), lora.max())

    for ax, arr, title in [
        (axes[0], eng, "(a) Engram surgical insertion"),
        (axes[1], lora, "(b) per-user LoRA (same fact)"),
    ]:
        im = ax.imshow(np.maximum(arr, floor), aspect="auto", cmap="hot",
                       norm=LogNorm(vmin=floor, vmax=vmax), interpolation="nearest")
        ax.axvline(x=trig, linestyle="--", color=GREEN, lw=1.3)
        ax.set_xlabel("Token position")
        ax.set_title(title)
    axes[0].set_ylabel("Layer index")
    axes[0].text(0.5, -0.42, "$\\Delta=0.000$ at every non-trigger position\nand every pre-Engram layer",
                 transform=axes[0].transAxes, ha="center", fontsize=7.5, color=BLUE)
    axes[1].text(0.5, -0.42, f"nonzero at every position, every layer\n(unrelated text moves by mean "
                 f"{d['summary']['lora_unrelated_mean_diff']:.0f})",
                 transform=axes[1].transAxes, ha="center", fontsize=7.5, color=RED)
    cb = fig.colorbar(im, ax=axes, label="residual stream change $\\|\\Delta\\|$", fraction=0.046, pad=0.02)
    out = FIGS / "fig_lora_vs_engram.pdf"
    plt.savefig(out); plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    fig_glassbox("d20")
    fig_lora_vs_engram("d12_1280")
