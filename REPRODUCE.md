# Reproducing the results

This repo is self-contained for everything that does not require a GPU, and
gives the full code + data + commands for everything that does. There are four
levels, from "runs in a minute on a laptop" to "train the models from scratch."

| Level | Needs | What you get |
|---|---|---|
| **0** | Python, ~1 min | Rebuild every paper figure from the committed `results/`. |
| **1** | Python | Regenerate the synthetic fact corpora. |
| **2** | 1 GPU + a checkpoint | Re-run an evaluation and regenerate a `results/*.json`. |
| **3** | 1 GPU, hours–days | Train the Mini-Engram models and run the whole pipeline. |

Everything ships in the repo already: the synthetic data (`data/`), every
result JSON the figures build from (`results/`), the figure generators
(`paper/`), and the full experiment + system code (`code/`).

---

## Level 0 — Rebuild the figures (no GPU)

```bash
cd paper
pip install -r requirements.txt        # matplotlib, numpy
bash make_all_figures.sh               # 11 scripts -> figs/*.pdf (all 38 figures)
pdflatex main.tex && pdflatex main.tex # optional: rebuild the PDF
```

Most figures are driven directly from `results/*.json`; a few (schematics and a
handful of bar/line panels) carry their values transcribed from the experiment
outputs into the generator script. Either way `make_all_figures.sh` reproduces
the exact figures in the paper. The map of figure → generator → data is below.

## Level 1 — Regenerate the synthetic data (no GPU)

The per-user fact files (`data/users/`, `data/users_medical/`) and the fact
corpora (`data/corpora*.json`) are committed. The corpora are also regenerable
(pure Python, no torch):

```bash
export USER_AS_ENGRAM_ROOT=$(pwd)
python code/scripts/build_corpus.py        # -> data/corpora.json      (base 200-fact)
python code/scripts/build_corpus_xl.py     # -> data/corpora_xl.json   (1k USER + 1k ORG)
python code/scripts/build_corpus_xxl.py    # -> data/corpora_xxl.json  (3,132 templates)
```

## Levels 2 & 3 — Run experiments / train from scratch (GPU)

The harness is vendored in `code/nanochat_harness/` — no external checkout.
One-time setup (details in [`code/README.md`](code/README.md)):

```bash
pip install -e code/nanochat_harness        # the `nanochat` package + torch
pip install -r code/requirements.txt
export USER_AS_ENGRAM_ROOT=$(pwd)
export NANOCHAT_BASE_DIR=$USER_AS_ENGRAM_ROOT/nanochat_base
cd code                                     # so `scripts` is importable
```

Then, e.g.:

```bash
# Level 3a — pretrain a Mini-Engram (one dense size)
python -m scripts.engram_pretrain --depth 20 --width 1536 ...

# Level 3b — train the shared reasoning LoRA
python -m scripts.train_shared_lora --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal ...

# Level 2 — the headline layered result (F vs B), from a trained checkpoint
python -m scripts.layered_architecture \
  --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \
  --shared-lora-dir $NANOCHAT_BASE_DIR/shared_lora_d20/r16 \
  --user-dir $USER_AS_ENGRAM_ROOT/data/users \
  --out $USER_AS_ENGRAM_ROOT/results/layered_d20_r16_full.json

# then rebuild the figure that uses it
python ../paper/make_new_figures.py
```

The per-user-LoRA baseline runs standalone (`cd code/lora_baseline`). Each
script prints its full usage (with example flags) in its module docstring.

---

## Datasets

| Dataset | In repo? | Notes |
|---|---|---|
| Per-user synthetic facts (`data/users/`, `data/users_medical/`) | ✅ yes | 30 + 30 fictional users, each with `facts`, `direct_qa`, `indirect_qa`. |
| Fact corpora (`data/corpora{,_xl,_xxl}.json`) | ✅ yes | Synthetic; also regenerable (Level 1). |
| All experiment outputs (`results/*.json`) | ✅ yes | What the figures read. |
| **LOCOMO** (Maharana et al., 2024) | ⬇️ obtain | Third-party benchmark we don't redistribute. Only needed to *re-run* the LOCOMO evals (`locomo_eval.py` / `judge_locomo.py`) — the LOCOMO figures already rebuild from `results/`. Place its `locomo10.json` at `data/locomo10.json`. |

---

## Map: paper figure → generator → data → experiment script

`D` = driven live from `results/`; `T` = values transcribed from the experiment
into the generator (re-run the experiment script to refresh them).

| Paper figure(s) | Generator (`paper/`) | | Experiment (`code/scripts/`) |
|---|---|:--:|---|
| layered conditions, pareto-layered | `make_new_figures.py` | D | `layered_architecture.py` |
| glass box, LoRA-vs-Engram locality | `fig_glassbox.py` | D | `mech_glassbox.py`, `mech_lora_vs_engram.py` |
| depth ablation (recall vs layer) | (in glass box) | T | `mech_depth.py` |
| RAG-vs-KB scaling | `fig_rag_scale.py` | D | `layered_rag_scale.py`, `qwen_rag_scale.py` |
| context-cost pareto (RAG) | `fig_pareto_rag.py` | D | `layered_rag.py`, `qwen_rag_indirect.py` |
| LOCOMO scaling / categories / judge | `make_new_figures.py`, `fig_locomo_categories.py` | D/T | `locomo_eval.py`, `judge_locomo.py` |
| contamination, cross-base, cross-schema, shared-rank | `gen_reorg_figs.py` | T | `layered_architecture.py`, `cross_lm_transfer.py` |
| multi-hop | `gen_reorg_figs.py` | T | `multihop_probe.py`, `multihop_rag.py` |
| capacity heatmap, fact-scale, dense scaling | `make_new_figures.py` | D | `eval_at_scale.py`, `evaluate_replication.py` |
| serving throughput / latency CDF | `gen_reorg_figs.py`, `make_figures.py` | D/T | `eval_serving.py` |
| multi-fact finetune | `make_figures_appendix.py` | T | `engram_finetune_mf.py` |
| method / placement schematics | `fig_method_arch.py`, `fig_layered_arch.py`, `fig_landscape.py` | — | (hand-drawn, no data) |

See [`code/README.md`](code/README.md) for a full index of all 52 scripts.
