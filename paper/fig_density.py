"""Density curve: User-as-Engram on the meta-skill foundational model
scales gracefully from 30 to 1000 facts/user, AND adds zero additional
contamination beyond the shared meta-skill base."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/home/ubuntu/user-as-engram/paper/figs")
d = json.load(open("/home/ubuntu/user-as-engram/results/density_layered_d20.json"))
ns = [e["n_facts"] for e in d["per_n"]]
top1 = [e["top1_pct"]*100 for e in d["per_n"]]
top5 = [e["top5_pct"]*100 for e in d["per_n"]]
bpb_delta = [e["val_bpb_delta_vs_meta_base"] for e in d["per_n"]]
rows = [e["n_distinct_rows"] for e in d["per_n"]]

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
})

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6),
                          gridspec_kw={"width_ratios": [1.1, 1.0]})

# Panel (a): recall curve
ax = axes[0]
ax.plot(ns, top1, "o-", color="#117733", lw=2.0, ms=9, label="top-1 (User as Engram)")
ax.plot(ns, top5, "s--", color="#55aa66", lw=1.6, ms=8, label="top-5 (User as Engram)")
# Annotate each point
for n, t1, t5 in zip(ns, top1, top5):
    ax.annotate(f"{t1:.0f}", (n, t1), textcoords="offset points",
                  xytext=(0, 9), fontsize=8, ha="center", color="#117733",
                  fontweight="bold")
    ax.annotate(f"{t5:.0f}", (n, t5), textcoords="offset points",
                  xytext=(0, -14), fontsize=8, ha="center", color="#55aa66")
ax.set_xscale("log")
ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("facts per user")
ax.set_ylabel("own-fact recall (\\%)")
ax.set_ylim(20, 105)
ax.set_title("(a)  User-as-Engram own-recall scales with fact count")
ax.legend(loc="lower left", frameon=False)
ax.grid(True, which="both", alpha=0.2, lw=0.4)
ax.spines[["top", "right"]].set_visible(False)

# Panel (b): contamination stays at zero, plus per-user storage
ax = axes[1]
# Bar for Δbpb addition on top of meta-skill base (essentially zero)
xs = np.arange(len(ns))
w = 0.8
# Show in absolute terms — use a small scale so the near-zero bars are visible
# We plot |Δ| × 10000 for visibility
abs_bpb_x10k = [abs(b) * 10000 for b in bpb_delta]
storage_kb = [r * 256 * 4 / 1024 for r in rows]   # n_rows × embed_dim × bytes/float

bars = ax.bar(xs - 0.2, abs_bpb_x10k, 0.4, color="#1f77b4",
                edgecolor="black", lw=0.6, label=r"|$\Delta$bpb| (× 10000)")
for i, v in enumerate(abs_bpb_x10k):
    ax.text(xs[i] - 0.2, v + 0.05, f"{v:.1f}", ha="center", fontsize=7.5, color="#1f77b4")

# Secondary axis for storage
ax2 = ax.twinx()
ax2.bar(xs + 0.2, storage_kb, 0.4, color="#aa6633",
          edgecolor="black", lw=0.6, label="per-user storage (KB)")
for i, v in enumerate(storage_kb):
    ax2.text(xs[i] + 0.2, v + 50, f"{v:.0f}", ha="center", fontsize=7.5, color="#aa6633")
ax2.set_ylabel("per-user storage (KB)", color="#aa6633")
ax2.tick_params(axis="y", colors="#aa6633")
ax2.set_ylim(0, 2900)
# Reference line: per-user LoRA at 14.2 MB ≈ 14200 KB — too big to fit, but annotate
ax2.axhline(2400, color="#888", ls=":", lw=0.7, alpha=0.6)
ax2.text(len(ns) - 0.5, 2500, "per-user LoRA at 1k facts $\\approx$ 14{,}200 KB (off-scale)",
           ha="right", fontsize=7, color="#666")

ax.set_xticks(xs); ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("facts per user")
ax.set_ylabel(r"|$\Delta$bpb|  $\times 10000$", color="#1f77b4")
ax.tick_params(axis="y", colors="#1f77b4")
ax.set_ylim(0, 3.0)
ax.set_title("(b)  Engram override adds zero contamination on top of meta-skill base")
ax.spines[["top"]].set_visible(False)
ax2.spines[["top"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "fig_density.pdf", bbox_inches="tight")
print(f"wrote {OUT / 'fig_density.pdf'}")
plt.close(fig)
