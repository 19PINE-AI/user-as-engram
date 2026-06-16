"""Relocate detail subsections from the body into a new appendix section.
Cuts at \\subsection boundaries; appends moved blocks before \\end{document}
under a new appendix \\section. Refs resolve regardless of order.
"""
MOVE = [
    r"\subsection{Engram capacity ablation}",
    r"\subsection{Fact-count scaling: per-fact independent OPT to 1000 facts}",
    r"\subsection{Dense-size scaling at optimal config}",
    r"\subsection{LoRA rank ablation at 100 facts}",
    r"\subsection{Inserted-fact recall in free continuation}",
    r"\subsection{Rank ablation: r=16 is the sweet spot}",
    r"\subsection{Cross-schema generalisation}",
    r"\subsection{Layered F on a Q/A-format-adapted Engram base}",
    r"\subsection{Storage and amortisation}",
    r"\subsection{Limitations of the layered measurement}",
]

lines = open("main.tex").read().split("\n")
out, moved = [], []
i = 0
in_body = True
while i < len(lines):
    ln = lines[i]
    if ln.startswith(r"\bibliographystyle"):
        in_body = False
    heading = ln.strip()
    if in_body and heading in MOVE:
        # capture until next \subsection{ or \section{
        block = [ln]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            s = nxt.lstrip()
            if s.startswith(r"\subsection{") or s.startswith(r"\section{"):
                break
            block.append(nxt)
            i += 1
        moved.append("\n".join(block).rstrip())
        continue
    out.append(ln)
    i += 1

# inject before \end{document}
inject = [
    "",
    r"\section{Appendix: Extended experiments and ablations}",
    r"\label{app:extended}",
    "",
    "This appendix collects the capacity, fact-count, dense-size, and",
    "rank ablations, and the layered-architecture robustness studies,",
    "that the main text summarizes in figures.",
    "",
] + ["\n".join([m, ""]) for m in moved]

final = []
for ln in out:
    if ln.strip() == r"\end{document}":
        final.extend(inject)
    final.append(ln)

open("main.tex", "w").write("\n".join(final))
print(f"moved {len(moved)} subsections into the appendix")
