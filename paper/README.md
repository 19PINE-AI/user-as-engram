# Paper

`main.tex` is the manuscript, *User as Engram: Internalizing Per-User Memory as
Local Parametric Edits* (arXiv:2606.19172). The bibliography is embedded
(`thebibliography`), so there is no separate bibtex step.

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
pdflatex main.tex && pdflatex main.tex
```

## License

The manuscript text and figures are **CC BY 4.0** (see [`LICENSE`](LICENSE)). The
figure-generation scripts (`*.py`) are code, covered by the repository's
Apache-2.0 license.
