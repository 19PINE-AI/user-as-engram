# Paper

`AnonymousSubmission2027.tex` is the active AAAI submission manuscript for
*User as Engram: Internalizing Per-User Memory as Local Parametric Edits*.
`main.tex` preserves the original uncompressed paper. The active manuscript
uses `references.bib`; `references_full_unused.bib` preserves unused entries.

## Rebuild the figures (no GPU)

```bash
pip install -r requirements.txt   # matplotlib, numpy
bash make_all_figures.sh          # all 38 figures, from ../results/, into figs/
```

`make_all_figures.sh` runs the 11 generator scripts (`make_figures.py`,
`make_figures_appendix.py`, `make_new_figures.py`, `gen_reorg_figs.py`, and the
`fig_*.py`). The figure → data → experiment-script map is in
[`../REPRODUCE.md`](../REPRODUCE.md).

## Build the PDF

```bash
pdflatex AnonymousSubmission2027.tex
bibtex AnonymousSubmission2027
pdflatex AnonymousSubmission2027.tex
pdflatex AnonymousSubmission2027.tex
```

## License

The manuscript text and figures are **CC BY 4.0** (see [`LICENSE`](LICENSE)). The
figure-generation scripts (`*.py`) are code, covered by the repository's
Apache-2.0 license.
