"""
Aggregate Stage A + baseline + Stage C pilot numbers into a single summary.

Produces:
  - results/summary.md  : human-readable Markdown table
  - results/summary.json: structured for downstream plotting
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(f"{UAE_ROOT}/results/lora_baseline")


def _mean_std(xs):
    xs = list(xs)
    if not xs:
        return None, None
    return mean(xs), (stdev(xs) if len(xs) > 1 else 0.0)


def load():
    a = json.loads((ROOT / "stage_a" / "summary.json").read_text()) \
        if (ROOT / "stage_a" / "summary.json").exists() else []
    b = json.loads((ROOT / "baselines" / "baselines.json").read_text()) \
        if (ROOT / "baselines" / "baselines.json").exists() else None
    c = json.loads((ROOT / "stage_c" / "pilot.json").read_text()) \
        if (ROOT / "stage_c" / "pilot.json").exists() else None
    return a, b, c


def main():
    a, b, c = load()
    lines = ["# User-as-LoRA pilot — results summary", ""]

    # ---- Stage A ----
    lines.append("## Stage A: per-user POLAR LoRAs (Qwen2.5-3B-Instruct)")
    lines.append("")
    if not a:
        lines.append("_no Stage A data yet_")
    else:
        lines.append(f"| uid | direct (w/ adapter) | indirect (w/ adapter) | indirect (base only) | train s |")
        lines.append(f"|---|:---:|:---:|:---:|:---:|")
        for r in a:
            lines.append(f"| {r['uid']} | {r['direct_acc']:.3f} | "
                         f"{r['indirect_acc']:.3f} | {r['iso_indirect_acc']:.3f} "
                         f"| {r['train_seconds']:.0f} |")
        dm, ds = _mean_std(r['direct_acc'] for r in a)
        im, ids_ = _mean_std(r['indirect_acc'] for r in a)
        isom, isos = _mean_std(r['iso_indirect_acc'] for r in a)
        lines.append(f"| **mean** | **{dm:.3f} ± {ds:.3f}** | "
                     f"**{im:.3f} ± {ids_:.3f}** | "
                     f"**{isom:.3f} ± {isos:.3f}** | |")
        lines.append("")
        lines.append(f"**POLAR gap on this benchmark: direct − indirect = "
                     f"{dm - im:.3f}** — the adapter encodes the facts but the "
                     f"model does not use them to reason.")
        if isom > im:
            lines.append(f"**Adapter-is-harmful finding: base-only indirect "
                         f"({isom:.3f}) > w/-adapter indirect ({im:.3f}). "
                         f"Hurting by {isom - im:.3f}.**")
    lines.append("")

    # ---- Baselines ----
    lines.append("## Baselines")
    lines.append("")
    if b is None:
        lines.append("_baselines not yet run_")
    else:
        icl_i = [v["indirect_acc"] for v in b["icl_upper"].values()]
        icl_d = [v["direct_acc"] for v in b["icl_upper"].values()]
        cot_i = [v["indirect_acc"] for v in b["polar_cot"].values()]
        lines.append(f"| condition | direct | indirect |")
        lines.append(f"|---|:---:|:---:|")
        lines.append(f"| ICL (base + facts in context) | {mean(icl_d):.3f} | "
                     f"{mean(icl_i):.3f} |")
        lines.append(f"| POLAR + explicit CoT prompt | — | {mean(cot_i):.3f} |")
    lines.append("")

    # ---- Stage C pilot ----
    lines.append("## Stage C pilot (meta-training held-out test)")
    lines.append("")
    if c is None:
        lines.append("_Stage C pilot not yet run_")
    else:
        s = c["summary"]
        lines.append(f"Setup: train uids {c['args']['train_uids']}, "
                     f"held uids {c['args']['held_uids']}, "
                     f"last-N unfrozen layers {c['args']['last_n_start']}–"
                     f"{c['args']['last_n_end']}, "
                     f"{c['args']['steps']} steps × LR {c['args']['lr']}.")
        lines.append("")
        lines.append(f"| split | indirect pre | indirect post | direct post |")
        lines.append(f"|---|:---:|:---:|:---:|")
        lines.append(f"| TRAIN | {s['train_indirect_pre']:.3f} | "
                     f"{s['train_indirect_post']:.3f} | "
                     f"{s.get('train_direct_post', 'n/a')} |")
        lines.append(f"| HELD | {s['held_indirect_pre']:.3f} | "
                     f"{s['held_indirect_post']:.3f} | "
                     f"{s.get('held_direct_post', 'n/a')} |")
        lines.append("")
        lines.append(f"Introspection Transfer (held): "
                     f"{s['held_indirect_post'] - s['held_indirect_pre']:+.3f}")
    lines.append("")

    out_md = ROOT / "summary.md"
    out_md.write_text("\n".join(lines))
    out_json = ROOT / "summary.json"
    with open(out_json, "w") as f:
        json.dump({"stage_a": a, "baselines": b, "stage_c": c}, f, indent=2)
    print("wrote", out_md)
    print("wrote", out_json)
    print()
    print(out_md.read_text())


if __name__ == "__main__":
    main()
