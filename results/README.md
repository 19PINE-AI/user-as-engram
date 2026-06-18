# Results

Every JSON here is an experiment output that the paper figures are built from.
Rebuild all 38 figures from these with no GPU:

```bash
cd ../paper && pip install -r requirements.txt && bash make_all_figures.sh
```

The full **figure → data → experiment-script** map is in
[`../REPRODUCE.md`](../REPRODUCE.md).

## File shape

Most files are `{config, rows, summary}`:
- `config` — the run's arguments (checkpoint, corpus, flags), kept for provenance;
- `rows` — the per-item measurements;
- `summary` — the aggregate numbers the figures/tables use.

## Naming

Pretraining-grid results:
`engram_d{depth}[_w{width}|_v2|_{capacity}_{tokens}]__{eval}.json`, where
`{eval}` ∈ `scale` (insertion/recall), `locomo[_cat*][_judge]` (LOCOMO),
`factscale_n{N}` (per-fact recall at N facts).

Other families:
- `layered_*` — the six-condition F-vs-B result, plus cross-schema / SFT variants; `seeds/` holds the 3-seed variance.
- `joint_opt_*`, `multifact_lora_*`, `opt_strong_*` — density curves and LoRA baselines.
- `layered_rag_*`, `qwen_rag_*` — the RAG comparisons and KB-scaling.
- `mechanistic_*`, `glassbox_*`, `lora_vs_engram_*`, `depth_*` — the glass-box probes.
- `serving_eval_*`, `scalability_benchmark*` — multi-tenant serving.
- `*.csv` — small aggregate tables.

Per-run console logs (`*.console.log`, `*.log`) are gitignored.
