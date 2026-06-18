"""Generate all paper figures from result JSONs.

Output: $USER_AS_ENGRAM_ROOT/paper/figs/*.pdf

Figures:
  fig1_lora_negative.pdf       — LoRA reasoning-negative bar chart (User-as-LoRA Stage A)
  fig2_locality.pdf            — insertion-attribution heatmap (Engram surgical locality)
  fig3_density_curves.pdf      — recall vs N facts simultaneous (OPT vs Joint OPT vs LoRA)
  fig4_storage_scaling.pdf     — storage vs (users × facts) log-log
  fig5_serving_latency.pdf     — multi-tenant serving latency breakdown
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT",
                          os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# NeurIPS-style figure setup
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#b3b3b3",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Canonical paper palette (shared with gen_reorg_figs.py / fig_locomo_categories.py).
BLUE = "#34507F"     # ours (Engram / Joint OPT)
BLUE_LT = "#7E97C4"  # ours, secondary
RED = "#C24A3F"      # primary baseline (per-user LoRA)
ORANGE = "#D98A3D"   # secondary baseline (SFT-LoRA / restore)
GREEN = "#3E7C5A"
GRAY = "#8A8F9A"
GRAY_LT = "#C2C6CE"

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Figure 1: LoRA reasoning-negative finding (User-as-LoRA Stage A)
# Per-user direct vs indirect, "with adapter" vs "base alone"
# -----------------------------------------------------------------------------
def fig1_lora_negative():
    # Stage A pilot aggregate (10 users), from User-as-LoRA notes.
    direct_w_adapter = 1.000   # mean
    indirect_w_adapter = 0.150 # mean
    indirect_base_only = 0.170 # mean

    # --- Panel (a), standalone: the headline recall gap ---
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    cats = ["Direct\n(adapter)", "Indirect\n(adapter)", "Indirect\n(base only)"]
    vals = [direct_w_adapter, indirect_w_adapter, indirect_base_only]
    colors = [BLUE, RED, GRAY]
    bars = ax.bar(cats, vals, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
    ax.axhline(y=indirect_base_only, ls="--", color="grey", alpha=0.6, lw=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=8)
    plt.tight_layout()
    out_a = FIGS / "fig1a_lora_recall.pdf"
    plt.savefig(out_a, pad_inches=0)
    plt.close()
    print(f"Wrote {out_a}")

    # --- Panel (b), standalone: per-user adapter vs base on indirect ---
    np.random.seed(0)
    users_with_adapter = np.array([0.133, 0.071, 0.180, 0.250, 0.286, 0.143, 0.100, 0.150, 0.067, 0.120])
    users_base_only = np.array([0.200, 0.214, 0.250, 0.200, 0.150, 0.267, 0.180, 0.067, 0.150, 0.100])
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(10)
    ax.bar(x - 0.2, users_base_only, width=0.4, label="base only", color=GRAY,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, users_with_adapter, width=0.4, label="w/ per-user LoRA",
           color=RED, edgecolor="black", linewidth=0.4)
    for i, (a, b) in enumerate(zip(users_with_adapter, users_base_only)):
        if a < b:
            ax.text(i, max(a, b) + 0.02, "*", color="black", ha="center", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"u{i:03d}" for i in range(10)], rotation=45, fontsize=7)
    ax.set_ylim(0, 0.40)
    ax.set_ylabel("Indirect recall")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    out_b = FIGS / "fig1b_lora_peruser.pdf"
    plt.savefig(out_b, pad_inches=0)
    plt.close()
    print(f"Wrote {out_b}")


# -----------------------------------------------------------------------------
# Figure 2: Engram surgical locality (insertion attribution heatmap)
# -----------------------------------------------------------------------------
def fig2_locality():
    with open(f"{UAE_ROOT}/results/mechanistic_d12.json") as f:
        d = json.load(f)
    attr = d["engram"]["insertion_attribution"]
    arr = np.array(attr["per_layer_per_pos"])  # [n_layers, n_pos]
    trig_pos = attr["trig_pos"]

    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    # Use log scale to see both 0 and 100
    log_arr = np.log10(arr + 0.1)
    im = ax.imshow(log_arr, aspect="auto", cmap="hot", interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, label="$\\log_{10}(\\Delta+0.1)$\nresidual stream change")
    ax.axvline(x=trig_pos, linestyle="--", color=GREEN, lw=1.5, label=f"trigger pos {trig_pos}")
    ax.set_xlabel("Token position in prompt")
    ax.set_ylabel("Layer\nindex")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    out = FIGS / "fig2_locality.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Figure 3: Within-user density curves (Joint OPT vs OPT independent vs LoRA)
# -----------------------------------------------------------------------------
def fig3_density_curves():
    # Data from XXL B1 + Joint OPT + multifact LoRA
    n = [10, 30, 100, 300, 1000]
    opt_indep_top1 = [0.90, 0.50, 0.36, 0.223, 0.134]
    opt_indep_top5 = [0.90, 0.70, 0.54, 0.427, 0.314]
    joint_opt_top1 = [None, None, 0.68, None, 0.351]
    joint_opt_top5 = [None, None, 0.96, None, 0.715]
    lora_top1 = [None, None, 0.99, None, 0.438]
    lora_top5 = [None, None, 1.00, None, 0.813]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # Panel (a): top-1
    ax = axes[0]
    ax.plot(n, opt_indep_top1, "o-", color=GRAY, label="OPT independent")
    ax.plot([100, 1000], [0.68, 0.351], "s-", color=BLUE, label="Joint OPT (ours)", lw=2)
    ax.plot([100, 1000], [0.99, 0.438], "^--", color=RED, label="LoRA rank-64")
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels([str(x) for x in n])
    ax.set_xlabel("# facts simultaneously inserted (per user)")
    ax.set_ylabel("Top-1 recall")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax.set_title("(a) Top-1 recall")

    # Panel (b): top-5
    ax = axes[1]
    ax.plot(n, opt_indep_top5, "o-", color=GRAY, label="OPT independent")
    ax.plot([100, 1000], [0.96, 0.715], "s-", color=BLUE, label="Joint OPT (ours)", lw=2)
    ax.plot([100, 1000], [1.00, 0.813], "^--", color=RED, label="LoRA rank-64")
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels([str(x) for x in n])
    ax.set_xlabel("# facts simultaneously inserted (per user)")
    ax.set_ylabel("Top-5 recall")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax.set_title("(b) Top-5 recall")

    # Single shared legend below the panels so it never overlaps the data lines.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = FIGS / "fig3_density_curves.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Figure 4: Storage scaling (users × facts) log-log
# -----------------------------------------------------------------------------
def fig4_storage_scaling():
    # Per-fact / per-user costs
    engram_per_fact_kb = 1.0       # 256 floats × 4B = 1KB
    sft_lora_per_fact_kb = 1728.0  # 442368 floats × 4B
    rank64_lora_per_user_mb = 14.2 # rank-64 per-user LoRA (fact-count-independent)

    facts_per_user = 100
    user_counts = [10, 100, 1000, 10000, 100000, 1000000]
    engram_total = [u * facts_per_user * engram_per_fact_kb / 1024 for u in user_counts]   # MB
    sft_total = [u * facts_per_user * sft_lora_per_fact_kb / 1024 for u in user_counts]    # MB
    polar_total = [u * rank64_lora_per_user_mb for u in user_counts]                         # MB

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.loglog(user_counts, engram_total, "s-", color=BLUE, lw=2, label="Engram override (ours)")
    ax.loglog(user_counts, polar_total, "^-", color=RED, label="per-user LoRA (rank 64)")
    ax.loglog(user_counts, sft_total, "v--", color=ORANGE, label="SFT-LoRA (per-fact, rank 8)")
    # Annotate the 1M-user point
    ax.annotate("100 GB", xy=(1e6, engram_total[-1]),
                xytext=(2e5, 1e2), fontsize=8, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    ax.annotate("14.2 TB", xy=(1e6, polar_total[-1]),
                xytext=(1.5e5, 8e5), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))
    ax.set_xlabel("Number of users (each with 100 facts)")
    ax.set_ylabel("Total storage (MB)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    out = FIGS / "fig4_storage_scaling.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Figure 5: Multi-tenant serving latency breakdown
# -----------------------------------------------------------------------------
def fig5_serving_latency():
    with open(f"{UAE_ROOT}/results/serving_eval_d12_30u_50f.json") as f:
        d = json.load(f)
    requests = d["requests"]
    # Filter own_query mode
    own = [r for r in requests if r["mode"] == "own_query"]
    apply_ms = [r["timings"]["apply_ms"] for r in own]
    fwd_ms = [r["timings"]["forward_ms"] for r in own]
    rest_ms = [r["timings"]["restore_ms"] for r in own]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # Panel (a): histogram of apply latency (the override swap is the new cost)
    ax = axes[0]
    ax.hist(apply_ms, bins=30, color=BLUE, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Override apply latency (ms)")
    ax.set_ylabel("# requests")
    ax.set_title("(a) Override apply")
    ax.axvline(np.median(apply_ms), color="black", ls="--", lw=0.7,
               label=f"median = {np.median(apply_ms):.2f} ms")
    ax.legend(framealpha=0.9)

    # Panel (b): stacked-bar of total latency by component
    ax = axes[1]
    components = ["apply", "forward", "restore"]
    medians = [np.median(apply_ms), np.median(fwd_ms), np.median(rest_ms)]
    colors = [BLUE, GRAY, ORANGE]
    bars = ax.bar(components, medians, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("(b) Total per-request")
    for b, v in zip(bars, medians):
        ax.text(b.get_x() + b.get_width()/2, v + 0.2, f"{v:.1f}",
                ha="center", fontsize=8)

    plt.tight_layout()
    out = FIGS / "fig5_serving_latency.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Figure 6 (appendix): LogitLens KL — engram vs base
# -----------------------------------------------------------------------------
def fig6_logitlens():
    with open(f"{UAE_ROOT}/results/mechanistic_d8.json") as f:
        d = json.load(f)
    eng = d["engram"]["logitlens_kl"]
    base = d["base"]["logitlens_kl"]
    eng_layers = d["engram"]["engram_layers"]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    layers = list(range(len(eng)))
    ax.plot(layers, base, "o-", color=GRAY, label="base d8 (no Engram)")
    ax.plot(layers, eng, "s-", color=BLUE, label="engram d8")
    for el in eng_layers:
        ax.axvline(x=el, ls=":", alpha=0.7, color=ORANGE,
                   label=f"Engram inserted at L{el}" if el == eng_layers[0] else None)
    ax.set_xlabel("Layer index"); ax.set_ylabel("KL(layer logits || final logits)")
    ax.legend(framealpha=0.9); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGS / "fig6_logitlens.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


def main():
    fig1_lora_negative()
    fig2_locality()
    fig3_density_curves()
    fig4_storage_scaling()
    fig5_serving_latency()
    fig6_logitlens()


if __name__ == "__main__":
    main()
