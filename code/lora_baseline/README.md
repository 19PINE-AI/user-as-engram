# Per-user-LoRA baseline + data generators

Standalone (Hugging Face Transformers + PEFT; no nanochat). This is the paper's
per-user-LoRA / POLAR-style baseline (the "per-user LoRA" condition) together
with the generators for the synthetic data. Run scripts from **this directory**.

## Data generators
- `python -m synth_users` → `../../data/users/` (the per-user fact sets).
- `python -m synth_users_medical` → `../../data/users_medical/` (medical schema).

## Per-user-LoRA pipeline
- `stage_a.py` — train a per-user LoRA adapter (NTP on paraphrases + QA);
  `stage_a_recite.py` is the recitation-default variant.
- `stage_b.py` — synthesize recite-then-reason traces.
- `stage_c.py` / `stage_c_pilot.py` — meta-train the base to "read" any user's adapter.
- `baseline_icl.py` (in-context ceiling), `leakage_test.py` (cross-user leakage),
  `aggregate_results.py`.
- `trace_v2/` — the agent-based teacher-trace pipeline (needs a served teacher LM;
  run `bash trace_v2/pipeline.sh`).

## Setup
Dependencies are in [`../requirements.txt`](../requirements.txt). Outputs default
to `$USER_AS_ENGRAM_ROOT/results/lora_baseline/` and `$USER_AS_ENGRAM_ROOT/data/`
— set `USER_AS_ENGRAM_ROOT` (see [`../README.md`](../README.md)).
