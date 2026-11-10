"""Build REPORT.md tables from interim JSON (cells with bootstrap CIs) -> report_tables.md"""
import json, os, sys
D = os.path.expanduser("~/clean-tests/results/T6")
J = json.load(open(os.path.join(D, sys.argv[1] if len(sys.argv) > 1 else "interim3.json")))
C, P = J["cells"], J["paired"]
LAY = ["L3", "L7", "L11", "L15", "L19", "L23", "L27", "L31", "L35"]
def v(c, k, ci=False, p=3):
    x = c[k]
    if x[0] != x[0]: return "–"
    f = (lambda t: f"{t:.{p}f}") if abs(x[0]) >= 1e-3 or x[0] == 0 else (lambda t: f"{t:.1e}")
    return f"{f(x[0])} [{f(x[1])}, {f(x[2])}]" if ci else f(x[0])
out = []
def table(title, methods, layers, cols):
    out.append(f"### {title}\n")
    out.append("| method | layer | n | " + " | ".join(h for h, _, _ in cols) + " |")
    out.append("|---|---|---|" + "---|" * len(cols))
    for m in methods:
        for L in layers:
            c = C.get(f"{m}|{L}")
            if not c: continue
            out.append(f"| {m} | {L} | {c['n']} | " + " | ".join(v(c, k, ci) for _, k, ci in cols) + " |")
    out.append("")
COLS = [("direct", "direct", False), ("para", "para", True), ("two-hop", "twohop", True), ("bleed", "bleed", True),
        ("fire10 (text)", "fire10", False), ("resid-move >1%", "resid_frac", False), ("text KL/tok", "text_kl", False),
        ("neighbor KL", "neighbor_kl", False)]
table("Table 1. Unregularised single-fact edits, all layers (200 facts per cell)",
      ["lora_down", "lora_up", "neuron", "neuron_bias", "rome"], LAY, COLS)
table("Table 2. Regularised (+reg) vs unregularised at the +reg layers (L3 +reg partial)",
      ["neuron", "neuron+reg", "neuron_bias", "neuron_bias+reg", "lora_down", "lora_down+reg", "lora_up", "lora_up+reg", "rome"],
      ["L3", "L7", "L15", "L23"], COLS)
table("Table 3. Q/K/V rank-16 LoRA on every layer (repo-style per-user LoRA baseline)", ["qkv", "qkv+reg"], ["Lall"], COLS)
out.append("### Table 4. Paired differences on the same facts (95% bootstrap CI; * = excludes 0)\n")
out.append("| A − B | layer | n | Δpara | Δtwo-hop | Δbleed | Δtext KL | Δneighbor KL | Δfire10 |")
out.append("|---|---|---|---|---|---|---|---|---|")
want = ["neuron+reg-neuron", "lora_down+reg-lora_down", "lora_up+reg-lora_up", "neuron+reg-lora_down+reg",
        "neuron+reg-lora_up+reg", "neuron+reg-rome", "lora_down+reg-rome", "neuron-lora_down", "rome-neuron", "rome-lora_down"]
for w in want:
    for L in LAY:
        p = P.get(f"{w}|{L}")
        if not p: continue
        cells = []
        for k in ["para", "twohop", "bleed", "text_kl", "neighbor_kl", "fire10"]:
            mu, lo, hi = p[k]
            star = "*" if (lo > 0 or hi < 0) else ""
            f = (lambda t: f"{t:.3f}") if abs(mu) >= 1e-3 or mu == 0 else (lambda t: f"{t:.1e}")
            cells.append(f"{f(mu)}{star} [{f(lo)}, {f(hi)}]")
        a, b = w.split("-", 1)
        out.append(f"| {a} − {b} | {L} | {p['n']} | " + " | ".join(cells) + " |")
out.append("")
open(os.path.join(D, "report_tables.md"), "w").write("\n".join(out))
print("\n".join(out[:12]))
