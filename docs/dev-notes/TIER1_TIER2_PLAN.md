# Tier 1 + Tier 2 Plan: Convert arXiv-strong → NeurIPS-strong

**Status:** in autonomous execution, started 2026-05-18
**Goal:** Eliminate the top reviewer concerns by adding cross-base layered
results, semantic LLM-judge evaluation, instruction-tuned base contamination
data, and cross-schema generalization.

## Sequenced execution

### Tier 1 — must-do before submission

| # | Item | Effort | GPU? | Status |
|---|---|---|---|---|
| 1a | Layered on Mini-Engram-d12@1280 (second size) | 1 h | ~10 GB | queued |
| 1b | Layered on instruction-tuned Engram (stretch) | 1-2 days | ~10 GB | optional |
| 2 | LLM-judge on layered indirect probes | 2-3 h | 30+ GB free needed (or use Qwen-7B judge) | queued |
| 3 | Regenerate fig_pareto.pdf with layered point | 30 min | no | queued |
| 4 | Fix remaining paper inconsistencies (n=20 −0.12, cross-schema scoping, future-work list) | 1 h | no | **in progress** |

### Tier 2 — strengthens significantly

| # | Item | Effort | GPU? | Status |
|---|---|---|---|---|
| 5 | Val_bpb contamination of per-user LoRA on Qwen-3B-Instruct | 6 GPU h | ~15 GB | queued |
| 6 | Cross-schema test of layered design (new schema corpus + experiment) | 1-2 days | ~10 GB | queued |
| 7 | Joint training of per-user Engram + shared LoRA (optional) | 1 day | ~10 GB | deferred to Tier 3 |

## Execution order

1. **Now (no-GPU):** Fix paper inconsistencies (Tier 1 #4). Look at fig_pareto.py source to understand what data it needs.
2. **First GPU batch (~2 h):** Tier 1 #1a — train shared LoRA on d12@1280, run layered experiment.
3. **Second GPU batch (~3 h):** Tier 1 #2 — LLM-judge eval on existing d20 + new d12@1280 layered runs. Use Qwen-7B-Instruct if GPU contention prevents Qwen-14B.
4. **No-GPU:** Tier 1 #3 — regenerate fig_pareto with new data points.
5. **Third GPU batch (~6 h):** Tier 2 #5 — Qwen-3B per-user LoRA + val_bpb measurement. Needs a new script (existing head_to_head_locality.py is Mini-Engram-only).
6. **Corpus + GPU (~1-2 days):** Tier 2 #6 — design new schema, regenerate users, train new shared LoRA, run cross-schema layered comparison.
7. **Final paper integration:** rewrite Section 8 with cross-base data, add LLM-judge column to layered table, add cross-schema subsection.

## Files

- New scripts:
  - `nanochat/scripts/judge_layered.py` — LLM-judge eval
  - `nanochat/scripts/qwen_lora_bpb.py` — Qwen-3B per-user LoRA + val_bpb
  - `nanochat/scripts/build_corpus_alt_schema.py` — new schema corpus
- New result JSONs:
  - `results/layered_d12_w1280_r16.json` (Tier 1 #1a)
  - `results/layered_judge.json` (Tier 1 #2)
  - `results/qwen3b_lora_bpb.json` (Tier 2 #5)
  - `results/layered_cross_schema.json` (Tier 2 #6)
- Paper edits to `paper/main.tex`:
  - Section 3 + abstract: add Qwen-3B val_bpb contamination row
  - Section 8: cross-base layered table, LLM-judge column, cross-schema subsection
  - Regenerated `figs/fig_pareto.pdf` with layered point

## Success criteria

- **Tier 1 #1a:** layered F still Pareto-dominates B at d12@1280 (smaller margin acceptable as long as F never gets worse than B)
- **Tier 1 #2:** F > B under LLM-judge, even if margin smaller than substring-match
- **Tier 2 #5:** per-user LoRA on Qwen-3B shows val_bpb $\Delta > 0$ (any size confirms architectural claim)
- **Tier 2 #6:** F still preserves locality ($\Delta$bpb $\approx 0$ vs shared-LoRA-alone) on cross-schema; indirect performance may drop but the *layering* claim still holds
