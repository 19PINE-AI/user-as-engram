"""Plot retrieval-recall and indirect_any vs KB size.

Inputs:
  - results/layered_rag_scale.json   (Mini-Engram-d20 + RAG G/H/J)
  - results/qwen_rag_scale.json      (Qwen-3B + RAG top-1, top-3)
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
    qwen = load(RES / "qwen_rag_scale_v2.json") or load(RES / "qwen_rag_scale.json")
    layered = load(RES / "layered_d20_r16_full.json")

    # Series: (label, color, marker, [(kb, indirect_any, ret_acc)])
    series = []

    if mini and "agg" in mini:
        for cname, color, marker, name in [
            ("G_rag1", ORANGE, "s", "Engram-base + RAG@1"),
            ("H_rag3", RED, "s", "Engram-base + RAG@3"),
            ("J_rag3_sharedLoRA", PURPLE, "D",
             "+ shared LoRA (J)"),
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

    if qwen and "agg" in qwen:
        for k_, color, marker, name in [
            (1, GREEN, "^", "Qwen-3B + RAG@1"),
            (3, TEAL, "^", "Qwen-3B + RAG@3"),
        ]:
            pts = []
            for kb in qwen["config"]["kb_sizes"]:
                key = f"qwen_rag{k_}_kb{kb}"
                if key in qwen["agg"]:
                    a = qwen["agg"][key]
                    pts.append((kb, a["indirect_any"] * 100,
                                a["retrieval_acc"] * 100))
            series.append({"label": name, "color": color, "marker": marker,
                           "linestyle": "--", "pts": pts})

    # F reference horizontal lines
    f_indirect = 44.5
    if layered and "agg" in layered:
        f_indirect = layered["agg"]["F_layered_indirect_any"] * 100
    e_indirect = 44.0
    if layered and "agg" in layered:
        e_indirect = layered["agg"]["E_shared_lora_only_indirect_any"] * 100

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True)

    # Panel A: indirect_any (log-x for KB)
    ax = axes[0]
    for s in series:
        xs = [p[0] for p in s["pts"]]
        ys = [p[1] for p in s["pts"]]
        ax.plot(xs, ys, color=s["color"], marker=s["marker"],
                linestyle=s["linestyle"], label=s["label"], linewidth=2,
                markersize=7)
    ax.axhline(f_indirect, color=BLUE, linestyle="-", linewidth=2.5,
               label=f"F (layered) = {f_indirect:.0f}%",
               alpha=0.9)
    ax.axhline(e_indirect, color=BLUE_LT, linestyle=":", linewidth=1.8,
               alpha=0.9, label=f"E (shared LoRA) = {e_indirect:.0f}%")
    ax.set_xscale("log")
    ax.set_ylabel("Indirect-reasoning accuracy (indirect_any, %)")
    ax.set_title("(a) RAG accuracy vs KB size")
    ax.set_ylim(0, 70)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    # Panel B: retrieval recall (log-x)
    ax = axes[1]
    for s in series:
        xs = [p[0] for p in s["pts"]]
        ys = [p[2] for p in s["pts"]]
        ax.plot(xs, ys, color=s["color"], marker=s["marker"],
                linestyle=s["linestyle"], label=s["label"], linewidth=2,
                markersize=7)
    ax.set_xscale("log")
    ax.set_xlabel("KB size = test user's 34 facts + distractors (log scale)")
    ax.set_ylabel("Retrieval recall (%): required_fact_keys $\\subseteq$ retrieved")
    ax.set_title("(b) Retrieval recall vs KB size")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, which="both")  # legend shared with panel (a)

    plt.tight_layout()
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
