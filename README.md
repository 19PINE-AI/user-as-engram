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
| [`code/`](code/) | The custom code: the Engram module, the modified GPT, and all experiment scripts (insertion, layered architecture, RAG baselines, serving). Apply onto a clone of [karpathy/nanochat](https://github.com/karpathy/nanochat) — see [`code/README.md`](code/README.md). |
| [`site/`](site/) | The interactive site (React + Vite). See [`site/README.md`](site/README.md). |
| [`data/`](data/) | Synthetic per-user fact corpora used in the experiments. |
| [`results/`](results/) | Result JSONs the paper figures are built from. |
| [`docs/dev-notes/`](docs/dev-notes/) | Working research logs and historical orchestration scripts (provenance; not needed to use the method). |

The Mini-Engram checkpoints (178 M – 1.22 B) and the full training harness are
not in this repo. Reproduce them by training [Engram](https://arxiv.org/abs/2601.07372)
into a nanochat clone with the code under [`code/`](code/).

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
