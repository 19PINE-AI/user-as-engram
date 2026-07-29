# Reproducibility Information Still Needed

This is the sole author-facing ledger for finishing Section 4 of the
reproducibility checklist. It records the current answer, what has already been
recovered, exactly what remains needed, and where any recovered information
belongs. Do not add statements about unavailable information to the paper merely
to improve a checklist answer.

Last reconciled with `ReproducibilityChecklist.tex`: 2026-07-28.

Current partial items are 4.2, 4.4, 4.7, 4.8, 4.12, and 4.13. Items 4.1, 4.3,
4.5, 4.6, and 4.9--4.11 are complete. The highest-value recovery work is the
missing cross-base outputs and aggregation/bootstrap code (4.4), followed by
the historical RNG mapping (4.7) and executed final configurations (4.13).
Computing-environment details (4.8) require access to the original execution
environment or its records. Do not change a partial answer to yes until all
information listed under that item's **Information needed** heading is
recovered and added at the stated location.

## 4.1 Computational experiments — complete

No additional information is needed.

## 4.2 Development-time search — partial

**Already recovered**

- Engram capacity candidates: 5K x 64, 20K x 128, 50K x 128,
  50K x 256, and 100K x 256, evaluated in a five-capacity x three-token-budget
  x two-base-model matrix.
- Pretraining-token candidates: d8 used 0.5B, 1B, and 2B; d12 used 0.5B,
  1.32B, and 2.5B.
- Shared-LoRA ranks 4, 16, and 64 were compared; rank 16 gave the best recorded
  balance of indirect accuracy and contamination.
- Per-user-LoRA ranks 8, 32, 64, and 128 were evaluated as a sensitivity study;
  the repository does not establish that rank 64 was selected by tuning.
- Joint OPT used convergence traces to support 2,000 steps at 100 facts and
  8,000 steps at 1,000 facts.
- FP8 and bf16 were tried during engineering; bf16 was retained after lower
  observed FP8 throughput. Treat this as an engineering trial unless the
  authors intended it as hyperparameter selection.

**Information needed**

- Confirm that the implemented capacity score
  `USER_OPT_top1 + 5 * LOCOMO_Joint_OPT_F1 + 0.5 * ORG_OPT_top1`
  was the criterion actually used.
- Explain why final larger models used 12 tokens per scaling parameter instead
  of the superseded 22.727-tokens-per-parameter extrapolation.
- Identify which settings were deliberately tuned rather than inherited
  defaults, engineering trials, or scientific ablations.
- For each deliberately tuned setting, identify the development data and final
  selection rule.

The first formula discrepancy is between `code/scripts/pick_optimal.py`'s
description and implementation. The token-budget discrepancy is between the
superseded `results/optimal_config.json` extrapolation and the final nanochat
12-tokens-per-parameter runs. Do not present scientific ablations, inherited
defaults, or engineering trials as tuning solely to obtain a `yes`.

**Where it belongs**

- “Experiments,” immediately after “Protocol and comparison controls,” using a
  compact search table or concise selection summary.

## 4.3 Preprocessing code — complete

No additional information is needed for the checklist answer. The archive
contains the synthetic-data generators, user-schema generators, runtime RAG
construction, LOCOMO parsing/filtering, and committed derived data. LOCOMO
itself remains third-party and is intentionally not redistributed.

## 4.4 Experiment and analysis source — partial

**Already recovered**

- Entry points and committed outputs exist for every other active experiment
  family, and the anonymous archive build and validator cover them.
- The active Qwen2.5-3B comparison used 30 users; Qwen2.5-7B, Llama-3.1-8B,
  Mistral-7B, and the Mini-Engram base used 20 users each.
- The committed `results/qwen3b_lora_bpb.json` is an older 10-user run and is
  not the final 30-user provenance.
- Searching committed history found no final Qwen-7B, Llama, Mistral, or
  30-user Qwen artifacts under recognizable names.

**Information needed**

- Final per-user output records for the active Qwen2.5-3B/7B,
  Llama-3.1-8B, and Mistral-7B comparison.
- The final executable experiment entry points or documented invocations for
  Qwen2.5-7B, Llama-3.1-8B, and Mistral-7B.
- The executable aggregation command or script that produced the active
  cross-base ranges.
- The paired-bootstrap implementation and saved output underlying the active
  confidence interval.

**Where it belongs**

- Files and commands: the code/data appendix, with the cross-base experiment
  and aggregation entry points added to its README.
- Any resulting correction to sample sizes or values: “Why Facts Should Not Be
  Per-User LoRA.”

## 4.5 Public release and licensing — complete

No additional information is needed. Preserve the anonymized Apache-2.0 notice,
the vendored harness's MIT license, and the LOCOMO acquisition notice.

## 4.6 Implementation comments — complete

No additional information is needed. Core new-method files use the format
`“Section Name” (sec:label)`.

## 4.7 Randomness and seeds — partial

**Already recovered**

- Corpus-generation seeds are 31337, 20260505, and 20260506.
- Shared-LoRA training uses seed 42; its code seeds Python and PyTorch.
- RAG distractor construction uses seed 0.
- The canonical S0 layered artifact evaluates 262,144 tokens; the retained S1
  artifact evaluates 524,288. Do not describe all seed runs as identical on
  this dimension unless S2 and the historical launch records establish that.

**Information needed**

- Integer RNG-state mapping for S0, S1, and S2.
- Whether each seed retrained the shared adapter, per-user LoRAs, Engram rows,
  or some subset.
- Per-user LoRA and Joint-OPT RNG states, if they were recorded externally.
- Serving request-construction seed.

**Where it belongs**

- Compact operative disclosure: “Experiments,” in “Protocol and comparison
  controls,” alongside the currently reported corpus, shared-LoRA, and RAG
  seeds.
- Any fuller mapping and determinism settings must also remain in the paper.

## 4.8 Computing infrastructure — partial

**Already recovered**

- Runs used one NVIDIA RTX PRO 6000 Blackwell GPU with bf16.
- The vendored harness declares Python >= 3.10, PyTorch 2.9.1, a CUDA 12.8
  wheel source, and Transformers >= 4.57.3. `code/requirements.txt` contains
  minimum rather than executed versions for several other packages.
- These manifests describe supported environments; they do not prove the exact
  historical environment for every reported run.

**Information needed**

- CPU model and system RAM.
- Whether GPU memory was reported as 96 GiB or approximately 102 decimal GB.
- OS release, NVIDIA driver, CUDA runtime, and Python version.
- Exact executed versions of unpinned libraries, especially Transformers,
  PEFT, and sentence-transformers.
- Whether every experiment campaign used the same host and environment.

**Where it belongs**

- Compact hardware/software disclosure: “Experiments,” in “Protocol and
  comparison controls.”
- Any fuller environment manifest must also remain in the paper.

## 4.9 Metrics and motivation — complete

No additional information is needed. The paper defines ranking,
answer-matching differences, LOCOMO normalization/aggregation, user
aggregation, memory-token accounting, throughput, and latency percentiles.

## 4.10 Runs per result — complete

No additional information is needed.

## 4.11 Variation and confidence — complete

No additional information is needed. The three-seed table reports individual
values and ranges.

## 4.12 Statistical tests — partial

**Already recovered**

- The layered-minus-LoRA comparison reports a paired-bootstrap 95% interval of
  `[+31,+37]` percentage points across the three seed outputs.
- Storage ratios, row counts, and parameter counts are deterministic. The
  remaining mechanistic, density, LOCOMO/RAG, cross-schema, and serving results
  are presently framed as descriptive results, so they should not receive
  post-hoc tests solely to strengthen the checklist response.

**Information needed**

- Decide which active claims, if any, are intended as population-level
  inferential claims rather than descriptive results.
- For each inferential claim, specify a test or interval, paired sampling unit,
  number of resamples or test repetitions, correction family if multiple
  hypotheses are tested, and RNG seed.
- Recover and include the implementation and saved outputs for the reported
  paired-bootstrap interval.

**Where it belongs**

- Test definition beside the corresponding claim, principally “Shared Skill,
  Local Content.”
- Operative statistical procedure and parameters: the same paper section. Any
  recovered implementation is included as experiment-analysis source under
  checklist item 4.4, not used as a substitute for the paper disclosure.

## 4.13 Final hyperparameters — partial

**Already recovered**

- Mini-Engram pretraining: nanochat RustBPE tokenizer (32,768-token
  vocabulary), bf16, sequence length 1024, total batch 131,072 tokens, table
  50K x 256, and the depths, widths, layers, token budgets, and validation bpb
  in Table 1. Launch scripts preserve many larger-model settings, but trainer
  defaults are not proof of an executed setting unless the historical version
  and absence of an override are confirmed.
- Headline Joint OPT density: Adam on row leaves, initialization scale 20,
  learning rate 0.5, 2,000 steps at 100 facts, and 8,000 at 1,000 facts. The
  active 35% point comes from `results/joint_opt_1000.json`, not the distinct
  d12@1280 result.
- Canonical per-user LoRA: rank 64, alpha 128, Q/K/V projections, Adam, 1,500
  steps, learning rate 5e-4, and first-token cross-entropy.
- Shared skill LoRA: rank 16, alpha 32 by default, Q/K/V projections, Adam,
  2,000 steps, learning rate 3e-4, maximum length 320, seed 42, and 10 training
  users. Confirm saved metadata and overrides for each seed variant.
- RAG scaling: all-MiniLM-L6-v2 normalized embeddings, cosine/dot-product
  ranking, top-k 1 and 3, knowledge-base sizes 34/100/200/300/500/1000, 20
  users, and distractor seed 0.
- Multi-hop: d20, 63 chains, two OPT-15 insertions per chain, learning rate 0.5,
  and initialization scale 20.
- Serving: d12@1280, 1,000 Joint OPT steps, learning rate 0.5, initialization
  scale 20, and one 600-request run per configuration.

**Information needed**

- Executed pretraining launch overrides for each Mini-Engram checkpoint rather
  than trainer defaults alone.
- Exact intermediate fact-count schedules across the distinct density/model
  campaigns.
- Final per-base cross-base LoRA configurations and seeds.
- Exact LOCOMO optimization settings and eligible counts for every active
  model/category cell.
- Serving warmup/cache procedure and request RNG seed.

**Where it belongs**

- Concise settings needed to interpret active claims: “Experiments,” in
  “Protocol and comparison controls.”
- Any complete family matrix must also remain in the paper.
