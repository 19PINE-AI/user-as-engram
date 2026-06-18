# Code

Everything needed to run the experiments lives here — no external checkouts.

```
code/
  nanochat_harness/   the training/eval harness (vendored nanochat, MIT) WITH our
                      Engram integration: nanochat/engram_module.py (new) and edits
                      to nanochat/gpt.py (Engram hooks in Block.forward,
                      GPT.attach_engram, optimizer wiring).
  scripts/            our experiment scripts — every paper result (index below).
  lora_baseline/      the per-user-LoRA (POLAR) baseline and the synthetic-data /
                      teacher-trace generators (Hugging Face / PEFT, standalone).
  requirements.txt
```

## Setup

```bash
pip install -e code/nanochat_harness      # provides the `nanochat` package + torch
pip install -r code/requirements.txt      # our extra deps
export USER_AS_ENGRAM_ROOT=/path/to/this/repo
export NANOCHAT_BASE_DIR=$USER_AS_ENGRAM_ROOT/nanochat_base   # where checkpoints live
```

Run the experiments from `code/` (so `scripts` is importable):

```bash
cd code
python -m scripts.layered_architecture \
  --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \
  --shared-lora-dir $NANOCHAT_BASE_DIR/shared_lora_d20/r16 \
  --user-dir $USER_AS_ENGRAM_ROOT/data/users \
  --out $USER_AS_ENGRAM_ROOT/results/layered_d20_r16_full.json
```

The per-user-LoRA baseline runs standalone from its own folder, e.g.
`cd code/lora_baseline && python -m synth_users` (regenerate the user data) or
`python -m stage_a` (train a POLAR adapter).

## Paths and environment

The scripts read inputs from `data/` and write outputs to `results/` under the
repository root. That root is resolved, in order:

1. `$USER_AS_ENGRAM_ROOT` if set;
2. otherwise the parent of `$NANOCHAT_BASE_DIR` (set this anyway for
   checkpoints, e.g. `export NANOCHAT_BASE_DIR=/path/to/user-as-engram/nanochat_base`);
3. otherwise the current working directory.

So either run from the repo root, or:

```bash
export USER_AS_ENGRAM_ROOT=/path/to/user-as-engram
export NANOCHAT_BASE_DIR=$USER_AS_ENGRAM_ROOT/nanochat_base
```

Checkpoint directories are passed explicitly via `--ckpt-dir $NANOCHAT_BASE_DIR/...`.
The `paper/*.py` figure scripts resolve the repo root from their own location, so
they need no environment setup.

## Dependencies

See [`requirements.txt`](requirements.txt) (installed on top of the vendored
harness, which provides `torch` and the tokenizer).

## Data the scripts read

- `data/users/`, `data/users_medical/` — the per-user synthetic fact sets
  (default `--user-dir $USER_AS_ENGRAM_ROOT/data/users`).
- `data/corpora{,_xl,_xxl}.json` — fact corpora for the density / fact-count tests.
- `data/locomo10.json` — the LOCOMO benchmark; **download separately** (see
  [`../REPRODUCE.md`](../REPRODUCE.md)) and place it there.

## Script index

Every script prints full usage in its module docstring (`python -m scripts.X --help`
or read the header). Grouped by stage:

**Data generation** — `build_corpus.py`, `build_corpus_xl.py`, `build_corpus_xxl.py`
(synthetic fact corpora); `generate_opd_corpus.py`, `generate_trace_corpus.py`
(teacher-trace corpora for the shared-LoRA ablations).

**Pretraining & training** — `engram_pretrain.py` (train a Mini-Engram);
`train_shared_lora.py` / `train_shared_lora_trace.py` (the shared reasoning LoRA);
`engram_finetune_mf.py` (multi-fact-in-the-loss finetune); `sft_baseline.py`,
`sft_engram_minimal.py`, `multifact_lora.py` (LoRA/SFT baselines).

**Insertion & core method** — `joint_opt.py` (Joint OPT row optimisation);
`insertion_strategies_v2.py`, `opt_strong_density.py` (write strategies);
`per_user_table_eval.py` (per-user override eval); `additive_composition.py`
(multi-domain stacking); `user_facts_demo.py` (minimal insertion demo).

**Headline experiments** — `layered_architecture.py` (the six-condition F-vs-B
result); `head_to_head_locality.py` (LoRA-vs-Engram locality + the shared eval
primitives the others import); `density_layered.py` (within-user density);
`layered_rag.py`, `layered_rag_scale.py`, `qwen_rag_indirect.py`, `qwen_rag_scale.py`
(RAG comparisons and KB-scaling); `multihop_probe.py`, `multihop_rag.py`
(multi-hop); `paraphrase_test.py`, `longform_gen.py`.

**Mechanism (glass box)** — `mech_glassbox.py` (gate / value-path / locality on
the trained model), `mech_lora_vs_engram.py` (per-position effect map),
`mech_depth.py` (depth causal test), `mechanistic_analysis.py`, `plot_mechanistic.py`.

**Memory-system & LOCOMO baselines** — `memory_systems_comparison.py`,
`memory_systems_proper.py`, `memory_systems_paraphrase.py` (Mem0/MemMachine-style
retrieval); `locomo_eval.py`, `judge_locomo.py`, `judge_layered.py`, `judge_only.py`
(LOCOMO + LLM-judge); `cross_lm_transfer.py` (cross-base transfer).

**Serving** — `engram_server.py` (the ~50-line multi-tenant server),
`eval_serving.py`, `scalability_benchmark.py`.

**Evaluation at scale & aggregation** — `eval_at_scale.py`,
`evaluate_replication.py`; `capacity_ablation_table.py`, `factscale_table.py`,
`scaling_summary.py`, `pick_optimal.py`, `plot_runs.py` (turn `results/*.json`
into the tables/curves).

## The per-user-LoRA baseline (`lora_baseline/`)

The per-user-LoRA / POLAR baseline (the paper's per-user-LoRA condition) and the
generators for the synthetic data, kept as a standalone Hugging Face + PEFT
package (no nanochat needed):

- `synth_users.py`, `synth_users_medical.py` — regenerate `data/users/` and
  `data/users_medical/`.
- `stage_a.py` / `stage_a_recite.py` (train per-user LoRA adapters),
  `stage_b.py` (synthesise recite-then-reason traces), `stage_c.py` /
  `stage_c_pilot.py` (meta-train the base to read any user's adapter).
- `baseline_icl.py` (in-context baseline), `leakage_test.py` (cross-user leakage),
  `aggregate_results.py`.
- `trace_v2/` — the agent-based teacher-trace pipeline (needs a served teacher LM;
  `pipeline.sh` runs it end-to-end).
