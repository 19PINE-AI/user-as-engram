# User as Engram: Internalizing Per-User Memory as Local Parametric Edits

**Bojie Li** · Pine AI

[![arXiv](https://img.shields.io/badge/arXiv-2606.19172-b31b1b.svg)](https://arxiv.org/abs/2606.19172)
&nbsp;[**Paper**](https://arxiv.org/abs/2606.19172) · [**Interactive site**](https://01.me/research/user-as-engram/)

A user's facts become a few rows in a content-addressed memory table — not a
rewrite of the model. Personal memory splits into **content** (the user's
specific facts) and **reasoning skill** (turning facts into answers); we store
each the right way. Per-user content is written as surgical edits to the
hash-keyed memory table of an [Engram](https://arxiv.org/abs/2601.07372) model,
and the reasoning skill lives in one *shared* adapter that everyone uses — an
artificial analogue of the brain's complementary learning systems.

Headline results (vs. a per-user LoRA, on Mini-Engram-d20):

- **≈5.6× (up to 7.4×) higher indirect-reasoning accuracy** at matched direct recall.
- **≈33,000× less disruption to unrelated text** — the edit is local *by design*
  (exactly 0.000 off-trigger), not merely small.
- **88 KB per user** instead of 14.2 MB, with the backbone left bit-identical and
  zero cross-user leakage by construction.
- Against retrieval, which method wins is set by deployment: a per-user table
  doesn't grow with the population, so past **~100 facts/user** it overtakes a
  retrieval pipeline on a 2.5× larger model.

## Repository layout

| Path | Contents |
|---|---|
| [`paper/`](paper/) | LaTeX source (`main.tex`), figures, and figure-generation scripts. |
| [`code/`](code/) | Everything to run the experiments: the training/eval **harness** (`nanochat_harness/`, vendored, with our Engram module + GPT edits), our **experiment scripts** (`scripts/`), and the **per-user-LoRA baseline** + data generators (`lora_baseline/`). See [`code/README.md`](code/README.md). |
| [`data/`](data/) | Synthetic per-user fact sets (`users/`, `users_medical/`) and fact corpora — all included. |
| [`results/`](results/) | Result JSONs the paper figures are built from. |
| [`site/`](site/) | The interactive site (React + Vite). See [`site/README.md`](site/README.md). |
| [`docs/dev-notes/`](docs/dev-notes/) | Working research logs and historical scripts (provenance; not needed to use the method). |

The repo is self-contained: the harness, code, and synthetic data are all here,
so reproduction needs no external checkouts. Only the trained Mini-Engram
checkpoints (178 M – 1.22 B) are too large to ship — regenerate them by training
[Engram](https://arxiv.org/abs/2601.07372) with `code/` (see [REPRODUCE.md](REPRODUCE.md)).

## Reproducing

Full guide: **[REPRODUCE.md](REPRODUCE.md)** (four levels, from laptop to full
training run). The fastest path rebuilds **every paper figure from the committed
`results/`, no GPU**:

```bash
cd paper
pip install -r requirements.txt
bash make_all_figures.sh        # 11 scripts → figs/*.pdf (all 38 figures)
```

The synthetic data (`data/`), all result JSONs (`results/`), and the full
experiment + system code (`code/`) are in the repo; the GPU-only steps (training,
re-running evals) are documented with exact commands in REPRODUCE.md.

## Build the paper

```bash
cd paper
pdflatex main.tex && pdflatex main.tex   # bibliography is embedded (thebibliography)
```

## Run the site

```bash
cd site
npm install
npm run dev      # local dev
npm run build    # static build → dist/
```

## Citation

```bibtex
@article{li2026userasengram,
  title         = {User as Engram: Internalizing Per-User Memory as Local Parametric Edits},
  author        = {Li, Bojie},
  year          = {2026},
  eprint        = {2606.19172},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2606.19172}
}
```

## License

- **Code** (this repository — `code/`, `site/`, `paper/` figure scripts, `data/`):
  [Apache License 2.0](LICENSE).
- **Paper** (the manuscript text and figures under `paper/`):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), matching the arXiv
  posting (see [`paper/LICENSE`](paper/LICENSE)).

Third-party attribution (nanochat, Engram) is in [`NOTICE`](NOTICE).
Copyright © 2026 Pine AI.
