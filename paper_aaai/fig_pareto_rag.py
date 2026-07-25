"""Pareto frontier: indirect-reasoning accuracy vs. extra memory tokens.

Combines:
  - Existing layered conditions (A-F) from results/layered_d20_r16_full.json
    (context-tokens = 0 for all, since these don't put facts in context)
  - New RAG conditions (G/H/I/J) from results/layered_rag_full.json
  - New Qwen-3B + RAG conditions from results/qwen_rag_full.json (different
    backbone, plotted with distinct marker)

Saves figs/fig_pareto_rag.pdf
"""
import json, os, math
from pathlib import Path
import matplotlib.pyplot as plt

# Shared paper style: serif type, despined axes, light dotted grid, canonical palette.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
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
RED = "#C24A3F"      # baseline
ORANGE = "#D98A3D"
GREEN = "#3E7C5A"

PAPER = Path(__file__).parent
RES = PAPER.parent / "results"
FIGS = PAPER / "figs"
FIGS.mkdir(exist_ok=True)


def load_json(p):
    if not Path(p).exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    layered = load_json(RES / "layered_d20_r16_full.json")
    rag = load_json(RES / "layered_rag_full.json")
    qwen = load_json(RES / "qwen_rag_full.json")

    points = []
    # Layered A-F: 0 context tokens
    if layered and "agg" in layered:
        a = layered["agg"]
        # Plot only the points relevant to indirect-reasoning comparison
        for code, label in [
            ("A_no_edit",                "A: no edit"),
            ("B_per_user_lora",          "B: per-user LoRA"),
            ("C_per_user_engram",        "C: per-user Engram"),
            ("E_shared_lora_only",       "E: shared LoRA only"),
            ("F_layered",                "F: layered (Engram + shared LoRA)"),
        ]:
            y = a.get(f"{code}_indirect_any")
            if y is not None:
                points.append({
                    "label": label, "x": 0.5, "y": y * 100,
                    "group": "layered",
                    "ms": a.get(f"{code}_ms_per_indirect", None),
                })

    # RAG conditions G/H/I/J on Mini-Engram-d20. For indirect queries the
    # stored JSON records total prompt length. The direct-query diagnostic also
    # records the retrieved fact block alone; use its per-user mean so the
    # x-axis compares serialized memory tokens rather than model-specific chat
    # templates.
    if rag and "agg" in rag:
        a = rag["agg"]
        for code, label in [
            ("G_rag1",            "G: RAG top-1"),
            ("H_rag3",            "H: RAG top-3"),
            ("I_ragall",          "I: RAG all"),
            ("G_oracle1",         "G': oracle top-1"),
            ("J_rag3_sharedLoRA", "J: RAG top-3 + shared LoRA"),
        ]:
            y = a.get(f"{code}_indirect_any")
            if y is None:
                continue
            xvals = [u[code]["direct_ctx_tokens_avg"] for u in rag.get("per_user", [])
                     if code in u]
            x = sum(xvals) / len(xvals) if xvals else 0
            x = max(x, 0.5)
            points.append({
                "label": label, "x": x, "y": y * 100,
                "group": "rag-mini",
                "ms": a.get(f"{code}_ms_per_indirect", None),
            })

    # Qwen-3B + RAG (different backbone). Subtract the no-context chat prompt
    # length to isolate only the memory text added by retrieval.
    if qwen and "agg" in qwen:
        a = qwen["agg"]
        qwen_base_tokens = a.get("NO_CONTEXT_ctx_tokens_avg", 0)
        for code, label in [
            ("NO_CONTEXT",  "Qwen-3B: no context"),
            ("RAG_TOP1",    "Qwen-3B + RAG top-1"),
            ("RAG_TOP3",    "Qwen-3B + RAG top-3"),
            ("RAG_ALL",     "Qwen-3B + RAG all"),
            ("ORACLE_TOP1", "Qwen-3B + oracle top-1"),
        ]:
            y = a.get(f"{code}_indirect_any")
            if y is None:
                continue
            x = a.get(f"{code}_ctx_tokens_avg", 0) - qwen_base_tokens
            x = max(x, 0.5)
            points.append({
                "label": label, "x": x, "y": y * 100,
                "group": "qwen",
                "ms": a.get(f"{code}_ms_per_query", None),
            })

    # Short annotation codes keep the dense high-accuracy cluster from colliding;
    # the full condition names live in the legend and the body text.
    short = {
        "A: no edit": "no edit", "B: per-user LoRA": "per-user LoRA",
        "C: per-user Engram": "per-user Engram",
        "E: shared LoRA only": "shared\nLoRA", "F: layered (Engram + shared LoRA)": "layered",
        "G: RAG top-1": "RAG@1", "H: RAG top-3": "RAG@3", "I: RAG all": "RAG all",
        "G': oracle top-1": "oracle@1", "J: RAG top-3 + shared LoRA": "RAG@3+LoRA",
        "Qwen-3B: no context": "Qwen none", "Qwen-3B + RAG top-1": "Qwen R@1",
        "Qwen-3B + RAG top-3": "Qwen R@3", "Qwen-3B + RAG all": "Qwen R@all",
        "Qwen-3B + oracle top-1": "Qwen O@1",
    }

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    style = {
        "layered":  {"color": BLUE,   "marker": "o", "s": 95,
                      "label": "Mini-Engram-d20 substrate (no retrieval)"},
        "rag-mini": {"color": ORANGE, "marker": "s", "s": 95,
                      "label": "Mini-Engram-d20 + RAG"},
        "qwen":     {"color": GREEN,  "marker": "^", "s": 95,
                      "label": "Qwen2.5-3B-Instruct + RAG"},
    }
    # Jitter overlapping zero-context labels so they don't pile up.
    zero_jitter = {"A: no edit": 0.45,
                   "B: per-user LoRA": 0.45,
                   "C: per-user Engram": 0.45,
                   "E: shared LoRA only": 0.38,
                   "F: layered (Engram + shared LoRA)": 0.55}
    label_offsets = {  # (dx, dy) in display pts; tune the short codes case by case
        "A: no edit": (7, -11),
        "B: per-user LoRA": (7, -3),
        "C: per-user Engram": (7, 3),
        "E: shared LoRA only": (0, -10),
        "F: layered (Engram + shared LoRA)": (7, 4),
        "G: RAG top-1": (6, -12),
        "H: RAG top-3": (7, 3),
        "I: RAG all": (7, 3),
        "G': oracle top-1": (6, -12),
        "J: RAG top-3 + shared LoRA": (0, 11),
        "Qwen-3B: no context": (8, -3),
        "Qwen-3B + RAG top-1": (-7, -10),
        "Qwen-3B + RAG top-3": (7, -7),
        "Qwen-3B + RAG all": (7, 4),
        "Qwen-3B + oracle top-1": (6, -12),
    }
    plotted_groups = set()
    for pt in points:
        gs = style[pt["group"]]
        x = zero_jitter.get(pt["label"], pt["x"])
        kwargs = dict(color=gs["color"], marker=gs["marker"], s=gs["s"],
                      edgecolor="black", linewidth=0.8, zorder=3)
        if pt["group"] not in plotted_groups:
            kwargs["label"] = gs["label"]
            plotted_groups.add(pt["group"])
        ax.scatter(x, pt["y"], **kwargs)
        # Annotate with the short condition code
        text = pt["label"]
        dx, dy = label_offsets.get(text, (6, 4))
        align = {}
        if text == "E: shared LoRA only":
            align = {"ha": "center", "va": "top"}
        elif text == "J: RAG top-3 + shared LoRA":
            align = {"ha": "center", "va": "bottom"}
        elif text == "Qwen-3B + RAG top-1":
            align = {"ha": "right", "va": "top"}
        elif text == "Qwen-3B + RAG top-3":
            align = {"ha": "left", "va": "top"}
        ax.annotate(short.get(text, text), (x, pt["y"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=11.5, **align)

    ax.set_xscale("log")
    # Leave enough room to center the enlarged two-line shared-LoRA label under
    # its leftmost point without crossing the y-axis.
    ax.set_xlim(0.23, 900)
    ax.set_ylim(0, 65)
    ax.set_xlabel("Extra serialized memory tokens per query (log; zero at left)")
    ax.set_ylabel("Indirect-reasoning accuracy (indirect_any, %)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11.5)
    plt.tight_layout()
    out_pdf = FIGS / "fig_pareto_rag.pdf"
    out_png = FIGS / "fig_pareto_rag.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=160)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    # Also write a small latency CSV for the paper table
    out_csv = RES / "latency_table.csv"
    with open(out_csv, "w") as f:
        f.write("condition,backbone,extra_memory_tokens_avg,indirect_any_pct,ms_per_query\n")
        for pt in points:
            ms = pt["ms"] if pt["ms"] is not None else float("nan")
            f.write(f"{pt['label']},{pt['group']},{pt['x']:.1f},{pt['y']:.2f},{ms:.1f}\n")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
