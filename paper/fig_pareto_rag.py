"""Pareto frontier: indirect-reasoning accuracy vs. avg context tokens.

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

    # RAG conditions G/H/I/J on Mini-Engram-d20
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
            x = a.get(f"{code}_indirect_ctx_tokens_avg", 0)
            x = max(x, 0.5)
            points.append({
                "label": label, "x": x, "y": y * 100,
                "group": "rag-mini",
                "ms": a.get(f"{code}_ms_per_indirect", None),
            })

    # Qwen-3B + RAG (different backbone)
    if qwen and "agg" in qwen:
        a = qwen["agg"]
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
            x = a.get(f"{code}_ctx_tokens_avg", 0)
            x = max(x, 0.5)
            points.append({
                "label": label, "x": x, "y": y * 100,
                "group": "qwen",
                "ms": a.get(f"{code}_ms_per_query", None),
            })

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    style = {
        "layered":  {"color": "tab:blue",   "marker": "o", "s": 95,
                      "label": "Mini-Engram-d20 substrate (A-F)"},
        "rag-mini": {"color": "tab:orange", "marker": "s", "s": 95,
                      "label": "Mini-Engram-d20 + RAG (G-J)"},
        "qwen":     {"color": "tab:green",  "marker": "^", "s": 95,
                      "label": "Qwen2.5-3B-Instruct + RAG"},
    }
    # Jitter overlapping zero-context labels so they don't pile up.
    zero_jitter = {"A: no edit": 0.45,
                   "B: per-user LoRA": 0.45,
                   "C: per-user Engram": 0.45,
                   "E: shared LoRA only": 0.38,
                   "F: layered (Engram + shared LoRA)": 0.55}
    label_offsets = {  # (dx, dy) in display pts; tune the labels case by case
        "A: no edit": (8, 4),
        "B: per-user LoRA": (8, 4),
        "C: per-user Engram": (8, 4),
        "E: shared LoRA only": (-105, 4),
        "F: layered (Engram + shared LoRA)": (8, 6),
        "G: RAG top-1": (8, 4),
        "H: RAG top-3": (8, 4),
        "I: RAG all": (-55, 6),
        "G': oracle top-1": (8, -10),
        "J: RAG top-3 + shared LoRA": (-15, 12),
        "Qwen-3B: no context": (8, 4),
        "Qwen-3B + RAG top-1": (-10, -14),
        "Qwen-3B + RAG top-3": (8, -6),
        "Qwen-3B + RAG all": (-100, 10),
        "Qwen-3B + oracle top-1": (8, 4),
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
        # Annotate
        text = pt["label"]
        dx, dy = label_offsets.get(text, (6, 4))
        ax.annotate(text.split(":", 1)[-1].strip(),
                    (x, pt["y"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5)

    ax.set_xscale("log")
    ax.set_xlim(0.3, 700)
    ax.set_ylim(0, 65)
    ax.set_xlabel("Avg context tokens per query (log)")
    ax.set_ylabel("Indirect-reasoning accuracy (indirect\\_any, \\%)")
    ax.set_title("Indirect reasoning vs.\\ context cost ($n{=}20$ users, "
                 "20 probes each)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
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
        f.write("condition,backbone,context_tokens_avg,indirect_any_pct,ms_per_query\n")
        for pt in points:
            ms = pt["ms"] if pt["ms"] is not None else float("nan")
            f.write(f"{pt['label']},{pt['group']},{pt['x']:.1f},{pt['y']:.2f},{ms:.1f}\n")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
