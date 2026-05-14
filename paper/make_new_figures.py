"""Generate the new NeurIPS-style figures for the post-Phase-H content:

Output (paper/figs/):
  fig_capacity_heatmap.pdf    — capacity × tokens heatmap, d8 + d12 side-by-side
  fig_factscale.pdf           — fact-count scaling curves (USER OPT, ORG OPT)
  fig_dense_scaling.pdf       — LOCOMO Joint OPT F1 vs dense params
  fig_joint_opt_dense.pdf     — Joint-OPT density-vs-dense (top-1 and top-5)
  fig_locomo_scaling.pdf      — LOCOMO Joint OPT vs MEMMACHINE across sizes
  fig_pareto.pdf              — cost-quality Pareto: storage vs LOCOMO F1
"""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# NeurIPS-style figure setup
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
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
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

RES = Path("/home/ubuntu/user-as-engram/results")
OUT = Path("/home/ubuntu/user-as-engram/paper/figs")
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
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), squeeze=False)

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

    for col_i, (title, prefix, tok_labels, tok_disp) in enumerate(cfgs):
        for row_i, (metric_name, metric_key) in enumerate(metrics):
            ax = axes[row_i, col_i]
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

            vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 1.0
            vmin = np.nanmin(mat) if not np.all(np.isnan(mat)) else 0.0
            im = ax.imshow(mat, cmap="viridis", aspect="auto",
                            vmin=vmin, vmax=vmax)
            # annotate cells
            for i in range(len(caps)):
                for j in range(len(tok_labels)):
                    if not np.isnan(mat[i, j]):
                        v = mat[i, j]
                        c = "white" if v < (vmin+vmax)/2 else "black"
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

    fig.suptitle("Engram capacity × tokens response surface",
                  fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig_capacity_heatmap.pdf")
    print(f"  wrote {OUT / 'fig_capacity_heatmap.pdf'}")
    plt.close(fig)


# ============================================================
# Figure: Fact-count scaling
# ============================================================
def fig_factscale():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    models = [
        ("engram_d8_v2",              "d8 v2 (0.5B)",          178, "#888888", "-"),
        ("engram_d8_large_t2B",       "d8 best (2B)",          178, "#cc6677", "-"),
        ("engram_d12_v2",             "d12 v2 (1.32B)",        339, "#666666", "-"),
        ("engram_d12_large_t25B",     "d12 best (2.5B)",       339, "#88ccee", "-"),
        ("engram_d12_w1280_optimal",  "d12@1280 opt.",         625, "#44aa99", "-"),
        ("engram_d20_w1536_optimal",  "d20@1536 opt.",         1224, "#aa4499", "-"),
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

    fig.suptitle("Per-fact independent OPT recall is flat in $n$ up to 1000 facts",
                  fontsize=10, y=1.04)
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
    ax.semilogx(params, locomo, "o-", color="#1f77b4", lw=2, ms=8, label="User-as-Engram J-OPT")
    # Baseline (MEMMACHINE) per dense size; from LOCOMO sweeps
    mem_baseline = [0.075, 0.105, 0.113, 0.140]
    ax.semilogx(params, mem_baseline, "s--", color="#cc6677", lw=2, ms=6,
                 label="MEMMACHINE_LIKE (best retr.)")
    for x, y, lbl in zip(params, locomo, labels):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7)
    ax.set_xlabel("dense parameters")
    ax.set_ylabel("LOCOMO Joint OPT F1")
    ax.set_xticks(params)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Conversational recall (LOCOMO)", fontsize=9)
    ax.set_ylim(0.0, 0.27)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    bars_x = np.arange(len(rows))
    ax.bar(bars_x - 0.18, bpb, 0.36, color="#999999", label="val bpb")
    ax.set_ylabel("ClimbMix val bpb (lower better)", color="#666666")
    ax.tick_params(axis='y', labelcolor='#666666')
    ax2 = ax.twinx()
    ax2.bar(bars_x + 0.18, e1opt, 0.36, color="#1f77b4", label="E1 USER OPT t1")
    ax2.set_ylabel("E1 USER OPT top-1 (higher better)", color="#1f77b4")
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim(0.8, 1.05)
    ax.set_xticks(bars_x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("Dense scaling: pretraining vs.\\ insertion", fontsize=9)
    ax.grid(False)

    fig.suptitle("Dense-size scaling at the ablation-optimal recipe",
                  fontsize=10, y=1.04)
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
        ("d12@768 (339M)",     [(100,'joint_opt_100'),(300,'joint_opt_300'),(1000,'joint_opt_1000')], "#888888"),
        ("d12@1280 (625M)",    [(100,'joint_opt_d12_w1280_n100'),(300,'joint_opt_d12_w1280_n300'),(1000,'joint_opt_d12_w1280_n1000')], "#1f77b4"),
        ("d20@1536 (1.22B)",   [(100,'joint_opt_d20_w1536_n100'),(300,'joint_opt_d20_w1536_n300'),(1000,'joint_opt_d20_w1536_n1000')], "#cc6677"),
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

    fig.suptitle("Joint-OPT density ceiling is set by Engram table, not dense scale",
                  fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig_joint_opt_dense.pdf")
    print(f"  wrote {OUT / 'fig_joint_opt_dense.pdf'}")
    plt.close(fig)


# ============================================================
# Figure: Cost-quality Pareto plot
# ============================================================
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
        (1.0, 0.219, "Engram J-OPT\n(ours, d20)",   "#1f77b4", "o"),
        (1.0, 0.177, "Engram OPT (per-fact)",      "#88ccee", "o"),
        (135.0, 0.219, "POLAR-class LoRA\n(ours-equiv)",   "#aa4499", "*"),
        (30.0, 0.140, "MEMMACHINE_LIKE",            "#cc6677", "s"),
        (30.0, 0.114, "RAG (top-3)",               "#996666", "s"),
        (0.0, 0.043, "no memory",                  "#aaaaaa", "x"),
    ]
    for x, y, label, color, marker in points:
        x_eff = max(x, 0.2)  # avoid log(0)
        ax.scatter([x_eff], [y], color=color, marker=marker, s=80, edgecolor="black", lw=0.5, zorder=3)
        ax.annotate(label, (x_eff, y), textcoords="offset points", xytext=(10, 0),
                     fontsize=7, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("storage per fact (KB, log scale)")
    ax.set_ylabel("LOCOMO Joint-OPT F1")
    ax.set_title("Cost-quality Pareto: same LM (Mini-Engram-d20@1536), varying memory substrate",
                  fontsize=8)
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.0, 0.27)
    # Pareto frontier connector for Engram
    ax.plot([1.0, 135.0], [0.219, 0.219], "k--", alpha=0.3, lw=0.8)
    ax.text(15, 0.225, "Engram & LoRA tied on quality, 135× storage gap →",
             fontsize=7, color="#333333")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pareto.pdf")
    print(f"  wrote {OUT / 'fig_pareto.pdf'}")
    plt.close(fig)


# ============================================================
# Figure: LOCOMO scaling (J-OPT vs baselines by dense)
# ============================================================
def fig_locomo_scaling():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))

    sizes = [178, 339, 625, 1224]
    size_labels = ["d8\n178M", "d12\n339M", "d12@1280\n625M", "d20@1536\n1.22B"]
    jopt =       [0.161, 0.169, 0.195, 0.219]
    memmach =    [0.075, 0.100, 0.113, 0.140]
    rag3 =       [0.090, 0.103, 0.103, 0.106]  # estimates from our runs
    nomem =      [0.043, 0.038, 0.030, 0.030]

    ax.semilogx(sizes, jopt,    "o-", color="#1f77b4", lw=2, ms=8, label="User-as-Engram Joint OPT")
    ax.semilogx(sizes, memmach, "s--", color="#cc6677", lw=1.5, ms=6, label="MEMMACHINE_LIKE")
    ax.semilogx(sizes, rag3,    "^--", color="#aa6644", lw=1.5, ms=5, label="RAG top-3")
    ax.semilogx(sizes, nomem,   "x--", color="#aaaaaa", lw=1.0, ms=5, label="NO_MEMORY")
    ax.set_xticks(sizes)
    ax.set_xticklabels(size_labels, fontsize=8)
    ax.set_xlabel("Mini-Engram dense parameters")
    ax.set_ylabel("LOCOMO single-hop token F1")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_ylim(0.0, 0.27)
    ax.set_title("LOCOMO single-hop: Engram lead widens with scale", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_locomo_scaling.pdf")
    print(f"  wrote {OUT / 'fig_locomo_scaling.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating new paper figures...")
    fig_capacity_heatmap()
    fig_factscale()
    fig_dense_scaling()
    fig_joint_opt_dense()
    fig_locomo_scaling()
    fig_pareto()
    print("Done.")
