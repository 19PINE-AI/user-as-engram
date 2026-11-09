# Mechanism follow-up scripts (as run)

Results, report and the paper-number map: [`../../results/mechanism_followup/`](../../results/mechanism_followup/).

| Path | What it does |
|---|---|
| `summarize.py` | CPU only. Recomputes every number in the paper from the committed records. |
| `T2/b_mech.py` | Part B: Engram Joint OPT vs per-user LoRA on Mini-Engram-d20; per-fact residual change, attribution, patching. Imports `code/scripts/head_to_head_locality.py`. |
| `T2/train_lora.py`, `T2/analyze.py` | Part A: per-user LoRA on Qwen3-8B (Q/K/V and MLP), then layer windows, answer-writing neurons, ablations. |
| `T2/aggregate.py` | Builds `summary.json` / `aggregate.txt`. |
| `T2/run_full.sh`, `T2/run_analysis.sh` | The order the jobs were run in. |
| `T6/t6full.py` (+ `t6lib.py`, `t6blib.py`, `t6run.py`, `cells.py`) | Neuron / rank-1 LoRA / ROME / Q/K/V edits on Qwen3-8B, batched; `--phase lr`, `sweep`, `multi`. |
| `T6/t6agg.py`, `T6/t6tables.py` | Aggregation and report tables. `pipeline_full.sh` is the run order. |
| `T5/*` | Neuron audit: `collect.py` → `label_sim.py` → `substitute.py` → `midrange.py` → `analyze_t5.py`. |

Paths are hard-coded to the original workspace (`~/clean-tests/...`); edit them to re-run.
