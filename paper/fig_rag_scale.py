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
            ("G_rag1", "tab:orange", "s", "Mini-Engram-d20 + RAG top-1"),
            ("H_rag3", "tab:red", "s", "Mini-Engram-d20 + RAG top-3"),
            ("J_rag3_sharedLoRA", "tab:purple", "D",
             "Mini-Engram-d20 + RAG top-3 + shared LoRA (= J)"),
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
            (1, "tab:green", "^", "Qwen-3B + RAG top-1"),
            (3, "tab:olive", "^", "Qwen-3B + RAG top-3"),
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)

    # Panel A: indirect_any (log-x for KB)
    ax = axes[0]
    for s in series:
        xs = [p[0] for p in s["pts"]]
        ys = [p[1] for p in s["pts"]]
        ax.plot(xs, ys, color=s["color"], marker=s["marker"],
                linestyle=s["linestyle"], label=s["label"], linewidth=2,
                markersize=7)
    ax.axhline(f_indirect, color="tab:blue", linestyle="-", linewidth=2.5,
               label=f"F (layered, Engram + shared LoRA) = {f_indirect:.0f}\\%",
               alpha=0.85)
    ax.axhline(e_indirect, color="tab:cyan", linestyle=":", linewidth=1.5,
               alpha=0.6, label=f"E (shared LoRA only) = {e_indirect:.0f}\\%")
    ax.set_xscale("log")
    ax.set_xlabel("KB size = test user's 34 facts + distractors (log scale)")
    ax.set_ylabel("Indirect-reasoning accuracy (indirect\\_any, \\%)")
    ax.set_title("(a) RAG accuracy vs KB size")
    ax.set_ylim(0, 70)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.95)

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
    ax.set_ylabel("Retrieval recall (\\%): required\\_fact\\_keys $\\subseteq$ retrieved")
    ax.set_title("(b) Retrieval recall vs KB size")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)

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
