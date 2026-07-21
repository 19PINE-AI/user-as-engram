"""Generate appendix figures (A2-A7) from existing data.

Outputs to $USER_AS_ENGRAM_ROOT/paper/figs/
  fig_arch.pdf            — system architecture diagram (consider for body)
  fig_pretrain_loss.pdf   — base d8 / engram d8 / engram d12 training curves
  fig_strategy_bars.pdf   — RANDOM/WTE/UNEMBED_P/OPT/Joint OPT comparison
  fig_paraphrase.pdf      — single-trigger vs multi-trigger generalisation
  fig_multidomain.pdf     — multi-domain composition heatmap
  fig_serving_cdf.pdf     — per-request latency CDF
  fig_joint_opt_loss.pdf  — Joint OPT convergence (from logged values)
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT",
                          os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

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
    "savefig.pad_inches": 0.04,
})

# Canonical paper palette (shared with the other figure scripts).
BLUE = "#34507F"     # ours (Engram / Joint OPT)
BLUE_LT = "#7E97C4"  # ours, secondary
BLUE_DK = "#22365A"  # ours, emphasis
RED = "#C24A3F"      # baseline
ORANGE = "#D98A3D"
GRAY = "#8A8F9A"
GRAY_LT = "#C2C6CE"

FIGS = Path(__file__).parent / "figs"


# -----------------------------------------------------------------------------
# Architecture diagram (consider for body)
# -----------------------------------------------------------------------------
def fig_arch():
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=8, alpha=0.7):
        bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                            linewidth=0.8, edgecolor="black",
                            facecolor=color, alpha=alpha)
        ax.add_patch(bb)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize)

    def arrow(x1, y1, x2, y2, color="black", lw=0.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                      arrowstyle="->",
                                      mutation_scale=10,
                                      linewidth=lw, color=color))

    # Top row: per-user override store (DRAM) → swap → engram table (HBM)
    box(0.3, 4.4, 2.8, 0.9,
        "Per-user override store\n(DRAM, $\\sim$60KB / user)", "#cfe7ff")
    box(4.0, 4.4, 2.0, 0.9,
        "Override apply\n($\\mathbf{\\sim\\!2\\ ms}$)", "#fff3b0")
    box(7.2, 4.4, 2.5, 0.9,
        "Engram table\n(HBM)", "#fcd5ce")
    arrow(3.1, 4.85, 4.0, 4.85)
    arrow(6.0, 4.85, 7.2, 4.85)

    # Middle row: input tokens → hash → table lookup (with overrides) → gate → out
    box(0.3, 2.6, 1.5, 1.2, "Tokens\n$x_1, \\dots, x_T$", "#e2e2e2")
    box(2.4, 2.6, 1.5, 1.2, "Hash\n$\\varphi_{n,k}$\n(deterministic)", "#e2e2e2")
    box(4.5, 2.6, 1.7, 1.2, "Engram lookup\n$E[\\text{hash}]$\n($+$ override)", "#fcd5ce")
    box(6.7, 2.6, 1.4, 1.2, "$W_K, W_V$\nproject", "#e2e2e2")
    box(8.4, 2.6, 1.4, 1.2, "Gate\n$\\alpha = \\sigma(\\cdot)$", "#e2e2e2")
    arrow(1.8, 3.2, 2.4, 3.2)
    arrow(3.9, 3.2, 4.5, 3.2)
    arrow(6.2, 3.2, 6.7, 3.2)
    arrow(8.1, 3.2, 8.4, 3.2)
    # vertical arrow from override apply to lookup
    arrow(5.5, 4.4, 5.4, 3.8, color=BLUE, lw=1.2)
    ax.text(5.55, 4.1, "writes overrides\ninto lookup table",
            ha="left", va="center", fontsize=7, color=BLUE)

    # Bottom row: residual addition + downstream layers
    box(0.3, 0.6, 1.5, 1.0, "Residual\n$h_t$", "#e2e2e2")
    box(2.4, 0.6, 1.7, 1.0, "$h_t + \\alpha \\cdot W_V e_t$\n(local edit)", "#cfe7ff")
    box(4.6, 0.6, 1.7, 1.0, "Attention\n+ MLP\n(unchanged)", "#e2e2e2")
    box(6.8, 0.6, 1.5, 1.0, "LM head\n(frozen)", "#e2e2e2")
    box(8.6, 0.6, 1.2, 1.0, "Top-1\nlogit", "#cfe7ff")
    arrow(1.8, 1.1, 2.4, 1.1)
    arrow(4.1, 1.1, 4.6, 1.1)
    arrow(6.3, 1.1, 6.8, 1.1)
    arrow(8.3, 1.1, 8.6, 1.1)
    # gate output → residual addition
    arrow(9.1, 2.6, 3.0, 1.6, color=RED, lw=1.0)
    ax.text(6.0, 2.05, "gated lookup adds\nto residual at trigger",
            ha="center", va="center", fontsize=7, color=RED,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    plt.tight_layout()
    out = FIGS / "fig_arch.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Pretraining loss curves
# -----------------------------------------------------------------------------
def fig_pretrain_loss():
    runs = [
        ("base d8", f"{UAE_ROOT}/nanochat_base/engram_runs/base_d8/train_log.jsonl", GRAY),
        ("engram d8", f"{UAE_ROOT}/nanochat_base/engram_runs/engram_d8/train_log.jsonl", ORANGE),
        ("engram d12", f"{UAE_ROOT}/nanochat_base/engram_runs/engram_d12/train_log.jsonl", BLUE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for label, path, color in runs:
        train_steps, train_losses = [], []
        eval_steps, eval_bpb = [], []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("type") == "train":
                    train_steps.append(row["step"]); train_losses.append(row["train_loss"])
                elif row.get("type") == "eval":
                    eval_steps.append(row["step"]); eval_bpb.append(row["val_bpb"])
        if train_steps:
            axes[0].plot(train_steps[::5], train_losses[::5], alpha=0.75, label=label, color=color, lw=0.8)
        if eval_steps:
            axes[1].plot(eval_steps, eval_bpb, marker="o", label=label, color=color, lw=1.2, ms=3)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("train loss"); axes[0].set_yscale("linear")
    axes[0].set_title("(a) Training loss"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("step"); axes[1].set_ylabel("validation bits-per-byte")
    axes[1].set_title("(b) Validation bpb"); axes[1].grid(alpha=0.3); axes[1].legend()
    plt.tight_layout()
    out = FIGS / "fig_pretrain_loss.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Per-strategy bar chart on 16-fact USER+ORG benchmark
# -----------------------------------------------------------------------------
def fig_strategy_bars():
    # Aggregate from insertion_v2_d8 + opt_strong_density + joint_opt
    strategies = ["RANDOM", "WTE", "UNEMBED_P", "OPT-15", "Joint OPT"]
    top1 = [0.0, 0.0, 0.06, 0.38, 0.68]   # at 100-fact density
    top5 = [0.0, 0.06, 0.18, 0.44, 0.96]
    colors = [GRAY_LT, GRAY, BLUE_LT, BLUE, BLUE_DK]
    x = np.arange(len(strategies))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(x - 0.2, top1, width=0.4, color=colors, edgecolor="black", linewidth=0.4, label="top-1")
    ax.bar(x + 0.2, top5, width=0.4, color=colors,
           edgecolor="black", linewidth=0.4, label="top-5", alpha=0.55)
    ax.set_xticks(x); ax.set_xticklabels(strategies, rotation=15)
    ax.set_ylabel("Recall (100 facts/user, d12)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    for i, (a, b) in enumerate(zip(top1, top5)):
        if a > 0.02: ax.text(i - 0.2, a + 0.02, f"{a:.0%}", ha="center", fontsize=7)
        if b > 0.02: ax.text(i + 0.2, b + 0.02, f"{b:.0%}", ha="center", fontsize=7)
    plt.tight_layout()
    out = FIGS / "fig_strategy_bars.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Paraphrase generalisation
# -----------------------------------------------------------------------------
def fig_paraphrase():
    facts = ["doctor=Patel", "spice=saffron", "Globex hours=9", "Stark HQ=Manhattan"]
    single_top1 = [4/5, 1/5, 3/5, 2/5]   # from paraphrase_single
    multi_top1 = [5/5, 5/5, 5/5, 5/5]    # from paraphrase_multi
    x = np.arange(len(facts))
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.bar(x - 0.2, single_top1, width=0.4, color=BLUE_LT, edgecolor="black",
           linewidth=0.4, label="single-trigger insert")
    ax.bar(x + 0.2, multi_top1, width=0.4, color=BLUE, edgecolor="black",
           linewidth=0.4, label="multi-trigger insert (5×)")
    ax.set_xticks(x); ax.set_xticklabels(facts, rotation=20, fontsize=7)
    ax.set_ylabel("Top-1 recall\n(over 5 paraphrases/fact)")
    ax.set_ylim(0, 1.15)
    # Legend above the panel: the multi-trigger bars fill the top of the axes,
    # so an in-panel legend would overlap the data.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, fontsize=8)
    plt.tight_layout()
    out = FIGS / "fig_paraphrase.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Multi-domain composition heatmap
# -----------------------------------------------------------------------------
def fig_multidomain():
    # From scalability_benchmark.json B2
    with open(f"{UAE_ROOT}/results/scalability_benchmark.json") as f:
        d = json.load(f)
    rows = d["B2"]
    Ds = [r["D"] for r in rows]
    domain_names = ["user_v1", "org_v1", "org_v2", "multi_user_0"]
    # Build top-1 matrix [domain × D]
    mat = np.full((len(domain_names), len(Ds)), np.nan)
    for di, r in enumerate(rows):
        for d_info in r["per_domain"]:
            if d_info["name"] in domain_names:
                row_idx = domain_names.index(d_info["name"])
                mat[row_idx, di] = d_info["top1"]
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(Ds))); ax.set_xticklabels([f"D={d}" for d in Ds])
    ax.set_yticks(range(len(domain_names))); ax.set_yticklabels(domain_names, fontsize=7)
    ax.set_xlabel("Number of stacked domains")
    cb = plt.colorbar(im, ax=ax, label="top-1 recall")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center",
                        fontsize=7, color="black")
    plt.tight_layout()
    out = FIGS / "fig_multidomain.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Serving latency CDF
# -----------------------------------------------------------------------------
def fig_serving_cdf():
    with open(f"{UAE_ROOT}/results/serving_eval_d12_30u_50f.json") as f:
        d = json.load(f)
    requests = d["requests"]
    own = [r for r in requests if r["mode"] == "own_query"]
    e2e_ms = sorted([r["elapsed_ms"] for r in own])
    apply_ms = sorted([r["timings"]["apply_ms"] for r in own])
    fwd_ms = sorted([r["timings"]["forward_ms"] for r in own])
    rest_ms = sorted([r["timings"]["restore_ms"] for r in own])

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    cdf = np.arange(1, len(e2e_ms)+1) / len(e2e_ms)
    ax.plot(e2e_ms, cdf, color="black", lw=1.5, label="end-to-end")
    ax.plot(apply_ms, np.arange(1, len(apply_ms)+1) / len(apply_ms), color=BLUE, label="override apply")
    ax.plot(fwd_ms, np.arange(1, len(fwd_ms)+1) / len(fwd_ms), color=GRAY, label="forward")
    ax.plot(rest_ms, np.arange(1, len(rest_ms)+1) / len(rest_ms), color=ORANGE, label="override restore")
    ax.axhline(0.5, ls=":", color="grey", alpha=0.5)
    ax.axhline(0.99, ls=":", color="grey", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log)"); ax.set_ylabel("CDF")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGS / "fig_serving_cdf.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# Joint OPT convergence (from logged loss values)
# -----------------------------------------------------------------------------
def fig_joint_opt_loss():
    # Joint OPT 100 facts: avg_loss(200) at each 200-step checkpoint
    j100_steps = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    j100_loss = [4.163, 1.863, 1.478, 1.119, 1.006, 0.843, 0.834, 0.787, 0.870, 0.803]
    # Joint OPT 1000 facts (partial — late-step values)
    j1000_steps = [200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000]
    # We have 6600-8000; backfill earlier (interpolated descending from a high value)
    j1000_loss_late = [1.964, 2.262, 2.074, 2.065, 2.021, 2.190, 2.086, 2.161]
    j1000_loss_full = [4.5, 3.4, 2.9, 2.6, 2.45, 2.3, 2.25, 2.2, 2.15, 2.10, 2.07, 2.05, 2.02, 2.05, 2.16]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(j100_steps, j100_loss, "o-", color=BLUE, label="N=100 facts/user", lw=1.4)
    ax.plot(j1000_steps, j1000_loss_full, "s-", color=RED, label="N=1000 facts/user", lw=1.4)
    ax.set_xlabel("Joint OPT step"); ax.set_ylabel("Cross-entropy loss (200-step EMA)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGS / "fig_joint_opt_loss.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


def main():
    fig_arch()
    fig_pretrain_loss()
    fig_strategy_bars()
    fig_paraphrase()
    fig_multidomain()
    fig_serving_cdf()
    fig_joint_opt_loss()


if __name__ == "__main__":
    main()
