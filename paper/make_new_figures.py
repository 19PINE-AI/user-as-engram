"""Generate the new NeurIPS-style figures for the post-Phase-H content:

Output (paper/figs/):
  fig_capacity_heatmap.pdf    — capacity × tokens heatmap, d8 + d12 side-by-side
  fig_factscale.pdf           — fact-count scaling curves (USER OPT, ORG OPT)
  fig_dense_scaling.pdf       — LOCOMO Joint OPT F1 vs dense params
  fig_joint_opt_dense.pdf     — Joint-OPT density-vs-dense (top-1 and top-5)
  fig_locomo_scaling.pdf      — LOCOMO Joint OPT vs MEMMACHINE across sizes
  fig_pareto.pdf              — cost-quality Pareto: storage vs LOCOMO F1
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
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#b3b3b3",
    "grid.alpha": 0.7,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
})

# Canonical paper palette (shared with the other figure scripts).
BLUE = "#34507F"     # ours (Engram / Joint OPT)
BLUE_LT = "#7E97C4"  # ours, secondary (per-fact OPT / shared LoRA)
RED = "#C24A3F"      # primary baseline (per-user LoRA / MEM0)
ORANGE = "#D98A3D"   # secondary baseline (MEMMACHINE)
GREEN = "#3E7C5A"    # layered design
PURPLE = "#6E5687"
TEAL = "#4F8C9D"
BROWN = "#8C6D5C"    # RAG
GRAY = "#8A8F9A"
GRAY_LT = "#C2C6CE"

RES = Path(f"{UAE_ROOT}/results")
OUT = Path(f"{UAE_ROOT}/paper/figs")
OUT.mkdir(exist_ok=True, parents=True)


def load(p):
    if not p.exists(): return None
    try: return json.load(open(p))
    except Exception: return None


def agg_e1(rows_list, prefix):
    if not rows_list: return None, None
    n = len(rows_list)
    t1 = sum(1 for r in rows_list if r.get(f"{prefix}_rank") == 0) / n
    t5 = sum(1 for r in rows_list if r.get(f"{prefix}_rank", 999) < 5) / n
    return t1, t5


# ============================================================
# Figure: Capacity × tokens heatmap (d8 and d12 side-by-side)
# ============================================================
def fig_capacity_heatmap():
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.0), squeeze=False,
                             constrained_layout=True)

    caps = ["tiny", "small", "medium", "large", "xlarge"]
    cap_labels = ["tiny\n1.3M", "small\n10M", "medium\n26M", "large\n51M", "xlarge\n102M"]

    cfgs = [
        ("d8@512 (137 M dense)",  "engram_d8",  ["t05B", "t1B", "t2B"],  ["0.5 B", "1.0 B", "2.0 B"]),
        ("d12@768 (339 M dense)", "engram_d12", ["t05B", "t132B", "t25B"], ["0.5 B", "1.32 B", "2.5 B"]),
    ]
    metrics = [
        ("E1 USER OPT t1", "user_opt_t1"),
        ("LOCOMO Joint OPT F1", "locomo_jopt"),
    ]

    # Gather all four matrices first so each metric (row) can share one colour
    # scale across both dense sizes -- otherwise per-panel normalisation makes
    # the columns incomparable and amplifies the near-flat LOCOMO band into
    # misleading "confetti".
    mats = {}
    for col_i, (title, prefix, tok_labels, tok_disp) in enumerate(cfgs):
        for row_i, (metric_name, metric_key) in enumerate(metrics):
            mat = np.full((len(caps), len(tok_labels)), np.nan)
            for i, cap in enumerate(caps):
                for j, tok in enumerate(tok_labels):
                    tag = f"{prefix}_{cap}_{tok}"
                    if metric_key == "user_opt_t1":
                        sc = load(RES / f"{tag}__scale.json")
                        if sc:
                            t1, _ = agg_e1(sc.get("e1_user", []), "opt")
                            if t1 is not None: mat[i, j] = t1
                    elif metric_key == "locomo_jopt":
                        l = load(RES / f"{tag}__locomo.json")
                        if l and "summary" in l:
                            v = l["summary"].get("USER_AS_ENGRAM_JOINT_OPT")
                            if v is not None: mat[i, j] = v
            mats[(row_i, col_i)] = mat

    # one shared (vmin, vmax) per metric row
    row_norm = {}
    for row_i in range(len(metrics)):
        allv = np.concatenate([mats[(row_i, c)].flatten() for c in range(len(cfgs))])
        allv = allv[~np.isnan(allv)]
        row_norm[row_i] = (float(np.min(allv)), float(np.max(allv)))

    row_im = {}
    for col_i, (title, prefix, tok_labels, tok_disp) in enumerate(cfgs):
        for row_i, (metric_name, metric_key) in enumerate(metrics):
            ax = axes[row_i, col_i]
            mat = mats[(row_i, col_i)]
            vmin, vmax = row_norm[row_i]
            im = ax.imshow(mat, cmap="viridis", aspect="auto",
                            vmin=vmin, vmax=vmax)
            row_im[row_i] = im
            for i in range(len(caps)):
                for j in range(len(tok_labels)):
                    if not np.isnan(mat[i, j]):
                        v = mat[i, j]
                        c = "white" if v < (vmin + vmax) / 2 else "black"
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                 color=c, fontsize=7.5)
            ax.set_xticks(range(len(tok_labels)))
            ax.set_xticklabels(tok_disp)
            ax.set_yticks(range(len(caps)))
            ax.set_yticklabels(cap_labels)
            ax.set_xlabel("training tokens")
            if col_i == 0:
                ax.set_ylabel("Engram capacity\n(vocab × embed)")
            if row_i == 0:
                ax.set_title(f"{title}\n{metric_name}", fontsize=9)
            else:
                ax.set_title(metric_name, fontsize=9)
            ax.grid(False)

    # one shared colourbar per metric row (so colour maps to a legible value)
    for row_i in range(len(metrics)):
        fig.colorbar(row_im[row_i], ax=[axes[row_i, 0], axes[row_i, 1]],
                     fraction=0.046, pad=0.02)

    fig.savefig(OUT / "fig_capacity_heatmap.pdf")
    print(f"  wrote {OUT / 'fig_capacity_heatmap.pdf'}")
    plt.close(fig)


# ============================================================
# Figure: Fact-count scaling
# ============================================================
def fig_factscale():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    models = [
        ("engram_d8_v2",              "d8 v2 (0.5B)",          178, GRAY_LT, "-"),
        ("engram_d8_large_t2B",       "d8 best (2B)",          178, GRAY, "-"),
        ("engram_d12_v2",             "d12 v2 (1.32B)",        339, BLUE_LT, "-"),
        ("engram_d12_large_t25B",     "d12 best (2.5B)",       339, TEAL, "-"),
        ("engram_d12_w1280_optimal",  "d12@1280 opt.",         625, PURPLE, "-"),
        ("engram_d20_w1536_optimal",  "d20@1536 opt.",         1224, BLUE, "-"),
    ]
    Ns = [100, 200, 500, 1000]

    for ax_i, (split, label) in enumerate([("user", "USER OPT top-1"),
                                              ("org",  "ORG OPT top-1")]):
        ax = axes[ax_i]
        for tag, name, params, color, ls in models:
            ys = []
            for n in Ns:
                d = load(RES / f"{tag}__factscale_n{n}.json")
                if d is None: ys.append(np.nan); continue
                rows = d.get(f"e1_{split}", [])
                if not rows: ys.append(np.nan); continue
                t1 = sum(1 for r in rows if r.get("opt_rank") == 0) / len(rows)
                ys.append(t1)
            ax.plot(Ns, ys, marker="o", label=name, color=color, ls=ls, lw=1.5, ms=4)
        ax.set_xscale("log")
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("number of facts $n$")
        ax.set_ylabel(label)
        ax.set_ylim(0.4, 1.05)
        ax.axhline(1.0, color="black", lw=0.5, ls=":", alpha=0.5)
        if ax_i == 0:
            ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "fig_factscale.pdf")
    print(f"  wrote {OUT / 'fig_factscale.pdf'}")
    plt.close(fig)


# ============================================================
# Figure: Dense-size scaling at optimal config
# ============================================================
def fig_dense_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # Headline runs at each dense size (best per-dense-cell at optimum)
    rows = [
        ("d8 best\n(178M)",        178,  0.207,  0.912, 0.916),
        ("d12 best\n(339M)",       339,  0.185,  0.83,  0.97),
        ("d12@1280\n(625M)",       625,  0.195,  0.770, 1.00),
        ("d20@1536\n(1.22B)",      1224, 0.219,  0.730, 0.97),
    ]
    labels = [r[0] for r in rows]
    params = [r[1] for r in rows]
    locomo = [r[2] for r in rows]
    bpb    = [r[3] for r in rows]
    e1opt  = [r[4] for r in rows]

    ax = axes[0]
    # Equal-spaced categorical x; a log axis crammed the four sizes together and
    # collided the custom labels with matplotlib's default decade tick labels.
    xs = np.arange(len(rows))
    ax.plot(xs, locomo, "o-", color=BLUE, lw=2, ms=8, label="User-as-Engram J-OPT")
    # Baseline (MEMMACHINE) per dense size; from LOCOMO sweeps
    mem_baseline = [0.075, 0.105, 0.113, 0.140]
    ax.plot(xs, mem_baseline, "s--", color=RED, lw=2, ms=6,
                 label="MEMMACHINE_LIKE (best retr.)")
    for x, y in zip(xs, locomo):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7)
    ax.set_xlabel("dense parameters")
    ax.set_ylabel("LOCOMO Joint OPT F1")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(-0.3, len(rows) - 0.7)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(0.0, 0.27)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    bars_x = np.arange(len(rows))
    ax.bar(bars_x - 0.18, bpb, 0.36, color=GRAY, label="val bpb")
    ax.set_ylabel("ClimbMix val bpb", color=GRAY)
    ax.tick_params(axis='y', labelcolor=GRAY)
    ax2 = ax.twinx()
    ax2.bar(bars_x + 0.18, e1opt, 0.36, color=BLUE, label="E1 USER OPT t1")
    ax2.set_ylabel("E1 USER OPT top-1", color=BLUE)
    ax2.tick_params(axis='y', labelcolor=BLUE)
    ax2.set_ylim(0.8, 1.05)
    ax.set_xticks(bars_x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_dense_scaling.pdf")
    print(f"  wrote {OUT / 'fig_dense_scaling.pdf'}")
    plt.close(fig)


# ============================================================
# Figure: Joint-OPT density does not scale with dense
# ============================================================
def fig_joint_opt_dense():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    models = [
        ("d12@768 (339M)",     [(100,'joint_opt_100'),(300,'joint_opt_300'),(1000,'joint_opt_1000')], GRAY),
        ("d12@1280 (625M)",    [(100,'joint_opt_d12_w1280_n100'),(300,'joint_opt_d12_w1280_n300'),(1000,'joint_opt_d12_w1280_n1000')], BLUE),
        ("d20@1536 (1.22B)",   [(100,'joint_opt_d20_w1536_n100'),(300,'joint_opt_d20_w1536_n300'),(1000,'joint_opt_d20_w1536_n1000')], TEAL),
    ]
    for ax_i, (key, label) in enumerate([("top1", "Joint OPT top-1"),
                                              ("top5", "Joint OPT top-5")]):
        ax = axes[ax_i]
        for name, cells, color in models:
            xs = [c[0] for c in cells]
            ys = []
            for n, tag in cells:
                d = load(RES / f"{tag}.json")
                ys.append(d.get(key, np.nan) if d else np.nan)
            ax.plot(xs, ys, marker="o", color=color, label=name, lw=1.6, ms=5)
        ax.set_xscale("log")
        ax.set_xticks([100, 300, 1000])
        ax.set_xticklabels(["100", "300", "1000"])
        ax.set_xlabel("number of jointly inserted facts $n$")
        ax.set_ylabel(label)
        ax.set_ylim(0.2, 1.05)
        if ax_i == 0:
            ax.legend(loc="lower left", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig_joint_opt_dense.pdf")
    print(f"  wrote {OUT / 'fig_joint_opt_dense.pdf'}")
    plt.close(fig)


# ============================================================
# Figure: Cost-quality Pareto plot
# ============================================================
def fig_pareto_layered():
    """Cost-quality Pareto for the LAYERED architecture experiment
    (Section 8). 6 conditions on Mini-Engram-d20, n=20 users."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    # (storage_KB_per_user, indirect_any%, label, color, marker, Δbpb)
    # Numbers from results/layered_d20_r16_full.json agg block.
    pts = [
        (0.0,    0.192, "no edit",                    GRAY, "x",  0.000),
        (14200,  0.072, "per-user LoRA",              RED, "*",  1.559),
        (88,     0.233, "per-user Engram",            BLUE, "o",  0.000),
        (14288,  0.080, "LoRA + Engram stack",        ORANGE, "v",  1.419),
        (0.0,    0.440, "shared LoRA only",           BLUE_LT, "D",  0.386),
        (88,     0.443, "Layered: shared LoRA + Engram", GREEN, "P", 0.386),
    ]
    # B and the B+C combo sit almost on top of each other at the far right, so
    # their labels are placed left of the markers (and split vertically) to avoid
    # colliding with each other and running off the right edge.
    offs_map = {
        "Layered: shared LoRA + Engram": ((10, -8), "left"),
        "per-user LoRA":                 ((-12, -11), "right"),
        "LoRA + Engram stack":           ((-12, 12), "right"),
    }
    for x, y, label, color, marker, _bpb in pts:
        x_eff = max(x, 0.5)   # avoid log(0)
        ax.scatter([x_eff], [y], color=color, marker=marker, s=140,
                     edgecolor="black", lw=0.7, zorder=3)
        offs, ha = offs_map.get(label, ((10, 0), "left"))
        ax.annotate(label, (x_eff, y), textcoords="offset points",
                     xytext=offs, fontsize=8, va="center", ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 30000)
    ax.set_ylim(0.0, 0.55)
    ax.set_xlabel("storage per user (KB, log scale)")
    ax.set_ylabel("indirect reasoning (any-match, $n{=}20$)")
    # Pareto frontier line for the new winning region
    ax.plot([88, 0], [0.443, 0.440], "k--", alpha=0.3, lw=0.8)
    # Annotate contamination
    ax.text(14200, 0.03, "Δbpb=+1.56", fontsize=7, color=RED, ha="center")
    ax.text(88, 0.39, "Δbpb=+0.39", fontsize=7, color=GREEN, ha="center")
    ax.grid(True, which="both", alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pareto_layered.pdf")
    print(f"  wrote {OUT / 'fig_pareto_layered.pdf'}")
    plt.close(fig)


def fig_pareto():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))

    # Each point: (storage_KB_per_fact, LOCOMO_J_F1, label, color, marker)
    # Storage: Engram override is ~1 KB/fact (we report 88 KB / 100 facts).
    # LoRA POLAR-class: 13.5 MB per user / 100 facts = 135 KB/fact, recall 0.99.
    # RAG: ~30 KB/fact (sentence-encoder embedding 384-dim×4-bytes + the fact text).
    # MEMMACHINE: similar to RAG.
    # ICL: ~30-50 tokens/fact * 4 bytes = ~200 B per fact in context.
    # Locomo F1 from our headline LM (d20@1536 = 0.140 MEMMACHINE, 0.219 J-OPT, 0.177 single-trigger OPT)
    points = [
        # (storage KB/fact, LOCOMO F1, label, color, marker)
        (1.0, 0.219, "Engram J-OPT\n(ours, d20)",   BLUE, "o"),
        (1.0, 0.177, "Engram OPT (per-fact)",      BLUE_LT, "o"),
        (135.0, 0.219, "per-user LoRA\n(ours-equiv)",   RED, "*"),
        (30.0, 0.140, "MEMMACHINE_LIKE",            ORANGE, "s"),
        (30.0, 0.114, "RAG (top-3)",               BROWN, "s"),
        (0.0, 0.043, "no memory",                  GRAY, "x"),
    ]
    for x, y, label, color, marker in points:
        x_eff = max(x, 0.2)  # avoid log(0)
        ax.scatter([x_eff], [y], color=color, marker=marker, s=80, edgecolor="black", lw=0.5, zorder=3)
        ax.annotate(label, (x_eff, y), textcoords="offset points", xytext=(10, 0),
                     fontsize=7, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("storage per fact (KB, log scale)")
    ax.set_ylabel("LOCOMO Joint-OPT F1")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.0, 0.27)
    # Pareto frontier connector for Engram (same quality, far less storage)
    ax.plot([1.0, 135.0], [0.219, 0.219], "k--", alpha=0.3, lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pareto.pdf", pad_inches=0)
    print(f"  wrote {OUT / 'fig_pareto.pdf'}")
    plt.close(fig)


# ============================================================
# Figure: LOCOMO scaling (J-OPT vs baselines by dense)
# ============================================================
def fig_locomo_scaling():
    """Updated for full 10-conv data."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))

    sizes = [178, 339, 625, 1224]
    size_labels = ["d8\n178M", "d12\n339M", "d12@1280\n625M", "d20@1536\n1.22B"]
    # Equal-spaced categorical x: a log axis crammed the four sizes together and
    # collided the custom labels with matplotlib's default decade tick labels.
    xs = list(range(len(sizes)))

    # Token-F1 (full 10-conv)
    jopt_tf   = [0.134, 0.169, 0.176, 0.233]
    memmach_tf = [0.088, 0.116, 0.152, 0.169]
    mem0_tf    = [0.088, 0.131, 0.160, 0.161]
    rag3_tf    = [0.087, 0.118, 0.142, 0.147]  # approx (estimate)
    nomem_tf   = [0.038, 0.036, 0.049, 0.046]

    ax = axes[0]
    ax.plot(xs, jopt_tf,   "o-",  color=BLUE, lw=2, ms=7, label="Engram Joint OPT")
    ax.plot(xs, mem0_tf,   "s--", color=RED, lw=1.5, ms=5, label="MEM0_LIKE")
    ax.plot(xs, memmach_tf,"^--", color=PURPLE, lw=1.5, ms=5, label="MEMMACHINE_LIKE")
    ax.plot(xs, nomem_tf,  "x--", color=GRAY, lw=1.0, ms=4, label="NO_MEMORY")
    ax.set_xticks(xs); ax.set_xticklabels(size_labels, fontsize=7.5)
    ax.set_xlim(-0.3, len(sizes) - 0.7)
    ax.set_xlabel("Mini-Engram dense parameters", fontsize=8.5)
    ax.set_ylabel("LOCOMO token F1", fontsize=8.5)
    ax.set_ylim(0.0, 0.27)
    ax.set_title("(a)", loc="left", fontweight="bold", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)

    # LLM-judge accuracy (Qwen2.5-14B)
    jopt_jg    = [0.044, 0.059, 0.095, 0.140]
    memmach_jg = [0.106, 0.158, 0.158, 0.177]
    mem0_jg    = [0.116, 0.156, 0.169, 0.190]
    rag3_jg    = [0.110, 0.158, 0.155, 0.169]
    nomem_jg   = [0.005, 0.010, 0.025, 0.030]

    ax = axes[1]
    ax.plot(xs, jopt_jg,   "o-",  color=BLUE, lw=2, ms=7, label="Engram Joint OPT")
    ax.plot(xs, mem0_jg,   "s--", color=RED, lw=1.5, ms=5, label="MEM0_LIKE")
    ax.plot(xs, memmach_jg,"^--", color=PURPLE, lw=1.5, ms=5, label="MEMMACHINE_LIKE")
    ax.plot(xs, nomem_jg,  "x--", color=GRAY, lw=1.0, ms=4, label="NO_MEMORY")
    ax.set_xticks(xs); ax.set_xticklabels(size_labels, fontsize=7.5)
    ax.set_xlim(-0.3, len(sizes) - 0.7)
    ax.set_xlabel("Mini-Engram dense parameters", fontsize=8.5)
    ax.set_ylabel("LLM-judge accuracy", fontsize=8.5)
    ax.set_ylim(0.0, 0.23)
    ax.set_title("(b)", loc="left", fontweight="bold", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig_locomo_scaling.pdf")
    print(f"  wrote {OUT / 'fig_locomo_scaling.pdf'}")
    plt.close(fig)


# ============================================================
# Figure: token-F1 vs LLM-judge correlation
# ============================================================
def fig_metric_mismatch():
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    # All systems × all models (token-F1, LLM-judge)
    # (label, color, marker)
    points = []
    sizes = [178, 339, 625, 1224]
    size_names = ["d8", "d12", "d12@1280", "d20"]
    sys_data = {
        "NO_MEMORY":      ([0.038,0.036,0.049,0.046], [0.005,0.010,0.025,0.030], GRAY, "x"),
        "MARKDOWN_ALL":   ([0.060,0.080,0.085,0.075], [0.036,0.049,0.026,0.000], GRAY_LT, "v"),
        "MEM0_LIKE":      ([0.088,0.131,0.160,0.161], [0.116,0.156,0.169,0.190], RED, "s"),
        "MEMMACHINE":     ([0.088,0.116,0.152,0.169], [0.106,0.158,0.158,0.177], ORANGE, "^"),
        "RAG_TOP3":       ([0.087,0.118,0.142,0.147], [0.110,0.158,0.155,0.169], BROWN, "D"),
        "Engram OPT":     ([0.090,0.127,0.145,0.173], [0.028,0.041,0.079,0.110], BLUE_LT, "o"),
        "Engram J-OPT":   ([0.134,0.169,0.176,0.233], [0.044,0.059,0.095,0.140], BLUE, "*"),
    }
    for name, (tf, jg, color, marker) in sys_data.items():
        ax.scatter(tf, jg, s=60, color=color, marker=marker, edgecolor="black", lw=0.5, label=name, zorder=3)
    # x=y line for reference
    ax.plot([0, 0.25], [0, 0.25], "k--", alpha=0.3, lw=0.7)
    ax.text(0.21, 0.215, "y=x", fontsize=7, color="#444444", rotation=45)
    ax.set_xlabel("LOCOMO token-F1 (10-conv, per dense size)", fontsize=8.5)
    ax.set_ylabel("LOCOMO LLM-judge accuracy", fontsize=8.5)
    ax.set_xlim(0, 0.27); ax.set_ylim(0, 0.23)
    ax.legend(loc="lower right", fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_metric_mismatch.pdf")
    print(f"  wrote {OUT / 'fig_metric_mismatch.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating new paper figures...")
    fig_capacity_heatmap()
    fig_factscale()
    fig_dense_scaling()
    fig_joint_opt_dense()
    fig_locomo_scaling()
    fig_pareto()
    fig_pareto_layered()
    fig_metric_mismatch()
    print("Done.")
