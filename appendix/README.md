# Anonymous Code and Data Appendix

This archive contains the code, synthetic data, numerical results, and
documentation used for the computational experiments in the submission. We
include the complete Mini-Engram integration, experiment scripts, per-user LoRA
baselines, figure-analysis code, synthetic datasets, and committed result
records. We do not include model checkpoints because of their size.

## Contents

- `code/`: Mini-Engram training/evaluation code, the vendored nanochat harness,
  per-user LoRA baselines, insertion methods, RAG/LOCOMO evaluations,
  mechanistic probes, and serving experiments.
- `data/`: all novel synthetic corpora and fictional user records used by the
  experiments, plus deterministic generators. LOCOMO is third-party and is not
  redistributed.
- `results/`: committed numerical outputs used by the analyses and figures.
- `analysis/`: scripts that convert result records into figures and aggregate
  tables.
- `LICENSE` and nested third-party licenses: usage terms.

## Setup

We use Python 3.10 or newer. Install the vendored harness and the additional
requirements from the archive root:

```bash
pip install -e code/nanochat_harness
pip install -r code/requirements.txt
export USER_AS_ENGRAM_ROOT=$(pwd)
export NANOCHAT_BASE_DIR=$USER_AS_ENGRAM_ROOT/nanochat_base
```

The harness manifest pins PyTorch 2.9.1 and provides CUDA 12.8 wheels. The
remaining requirements specify minimum compatible versions.

## Rebuild analyses from committed results

From `analysis/`:

```bash
pip install -r requirements.txt
bash make_all_figures.sh
```

The scripts read `../results/` and write generated files below
`analysis/figs/`. Some schematic figures are code-drawn. A small number of
plots contain values transcribed from experiment output; their generator
comments identify that fact.

## Regenerate the novel synthetic data

From the archive root:

```bash
python code/scripts/build_corpus.py
python code/scripts/build_corpus_xl.py
python code/scripts/build_corpus_xxl.py
cd code/lora_baseline
python -m synth_users
python -m synth_users_medical
```

We use seeds 31337, 20260505, and 20260506 for the base, XL, and XXL corpora.
The user generators deterministically assign seeds 1000+i to personal users and
2000+i to medical users.

## Run the principal layered experiment

After placing a trained Mini-Engram checkpoint and shared-LoRA adapter under
`$NANOCHAT_BASE_DIR`, run from `code/`:

```bash
python -m scripts.layered_architecture \
  --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \
  --shared-lora-dir $NANOCHAT_BASE_DIR/shared_lora_d20/r16 \
  --user-dir $USER_AS_ENGRAM_ROOT/data/users \
  --test-uids u000 u001 u002 u003 u004 u005 u006 u007 u008 u009 \
              u010 u011 u012 u013 u014 u015 u016 u017 u018 u019 \
  --lora-rank 64 --lora-alpha 128 --lora-steps 1500 --lora-lr 5e-4 \
  --engram-steps 1500 --engram-lr 0.5 \
  --eval-tokens 262144 --device-bs 8 --max-seq-len 1024 \
  --out $USER_AS_ENGRAM_ROOT/results/layered_reproduction.json
```

## Third-party data

We do not redistribute LOCOMO. To rerun those experiments, obtain the public
10-conversation `locomo10.json` release associated with Maharana et al. (2024),
place it at `data/locomo10.json`, and run `code/scripts/locomo_eval.py`. Our
committed LOCOMO-derived numerical outputs are included so the reported
aggregates can be inspected without redistributing the source benchmark.
