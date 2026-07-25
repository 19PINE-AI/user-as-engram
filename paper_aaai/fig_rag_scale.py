"""Plot retrieval-recall and indirect_any vs KB size.

Inputs:
  - results/layered_rag_scale.json   (Mini-Engram-d20 + RAG G/H/J)
  - results/layered_d20_r16_full.json (for F = 44% reference)

Saves figs/fig_rag_scale.pdf and a small CSV.
"""
import json, math
from pathlib import Path
import matplotlib.pyplot as plt

# Shared paper style: serif type, despined axes, light dotted grid, and the
# canonical palette (slate blue = ours, muted red = primary baseline).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#b3b3b3",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})
BLUE = "#34507F"     # ours (Engram / layered)
BLUE_LT = "#7E97C4"  # ours, secondary
RED = "#C24A3F"      # primary baseline (LoRA / retrieval)
ORANGE = "#D98A3D"
GREEN = "#3E7C5A"
PURPLE = "#6E5687"
TEAL = "#4F8C9D"
GRAY = "#8A8F9A"

PAPER = Path(__file__).parent
RES = PAPER.parent / "results"
FIGS = PAPER / "figs"
FIGS.mkdir(exist_ok=True)


def load(p):
    p = Path(p)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    # Prefer the v2 files (KB up to 1000, wider distractor pool) when present
    mini = load(RES / "layered_rag_scale_v2.json") or load(RES / "layered_rag_scale.json")
    layered = load(RES / "layered_d20_r16_full.json")

    # Series: (label, color, marker, [(kb, indirect_any, ret_acc)])
    series = []

    if mini and "agg" in mini:
        for cname, color, marker, name in [
            ("G_rag1", ORANGE, "s", "Mini-Engram + RAG top-1"),
            ("H_rag3", RED, "s", "Mini-Engram + RAG top-3"),
            ("J_rag3_sharedLoRA", PURPLE, "D",
             "RAG top-3 + shared reasoning LoRA"),
        ]:
            pts = []
            for kb in mini["config"]["kb_sizes"]:
                key = f"{cname}_kb{kb}"
                if key in mini["agg"]:
                    a = mini["agg"][key]
                    pts.append((kb, a["indirect_any"] * 100,
                                a["retrieval_acc"] * 100))
            series.append({"label": name, "color": color, "marker": marker,
                           "linestyle": "-", "pts": pts})

    # F reference horizontal lines
    f_indirect = 44.5
    if layered and "agg" in layered:
        f_indirect = layered["agg"]["F_layered_indirect_any"] * 100
    e_indirect = 44.0
    if layered and "agg" in layered:
        e_indirect = layered["agg"]["E_shared_lora_only_indirect_any"] * 100

    # Vertically stacked panels (a) indirect accuracy, (b) retrieval recall, with a
    # single shared legend placed outside the plot area (below).
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 5.2), constrained_layout=True,
                             sharex=True)

    # Panel A: indirect_any (log-x for KB)
    ax = axes[0]
    for s in series:
        xs = [p[0] for p in s["pts"]]
        ys = [p[1] for p in s["pts"]]
        ax.plot(xs, ys, color=s["color"], marker=s["marker"],
                linestyle=s["linestyle"], label=s["label"], linewidth=2,
                markersize=7)
    # The layered design and the shared-LoRA-only condition both sit at ~44%
    # indirect and are KB-invariant (no retrieval); draw a single reference
    # line to avoid a confusing overlapping band.
    ax.axhline(f_indirect, color=BLUE, linestyle="-", linewidth=2.6,
               label=f"Engram rows + shared reasoning LoRA (no retrieval) = {f_indirect:.0f}%",
               alpha=0.95, zorder=1)
    ax.set_xscale("log")
    ax.set_ylabel("Indirect accuracy (%)")
    ax.set_title("(a)", loc="left", fontweight="bold")
    ax.set_ylim(0, 70)
    ax.grid(True, alpha=0.3, which="both")

    # Panel B: retrieval recall (log-x)
    ax = axes[1]
    for s in series:
        # The shared reasoning adapter changes answer generation, not the
        # retriever. Its top-3 recall is therefore identical to the plain
        # top-3 series, so plotting both would create a redundant overlap.
        if s["label"] == "RAG top-3 + shared reasoning LoRA":
            continue
        xs = [p[0] for p in s["pts"]]
        ys = [p[2] for p in s["pts"]]
        ax.plot(xs, ys, color=s["color"], marker=s["marker"],
                linestyle=s["linestyle"], label=s["label"], linewidth=2,
                markersize=7)
    ax.set_xscale("log")
    ax.set_xlabel("KB size (facts + distractors, log scale)")
    ax.set_ylabel("Retrieval recall (%)")
    ax.set_title("(b)", loc="left", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, which="both")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               fontsize=8.5, framealpha=0.95)
    out_pdf = FIGS / "fig_rag_scale.pdf"
    out_png = FIGS / "fig_rag_scale.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=160)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    # CSV
    rows = ["series,kb_size,indirect_any_pct,retrieval_acc_pct"]
    for s in series:
        for kb, ia, ra in s["pts"]:
            rows.append(f"{s['label']},{kb},{ia:.2f},{ra:.2f}")
    out_csv = RES / "rag_scale_table.csv"
    with open(out_csv, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
