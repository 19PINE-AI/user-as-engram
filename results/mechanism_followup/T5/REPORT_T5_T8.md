# T5 / T8 — Neuron audits on nanochat d8 models (Engram vs dense; neuron development over training)

All files are on rtx-pro under `~/clean-tests/results/T5_T8/`. Code is in `code/`; raw outputs are in `T5/<model>/` and `T8/`.
Labeler, simulator and judge: MiMo Pro (`mimo-v2.6-pro-mopd-local`, box 2/3), temperature 0, blind to which model a neuron came from.

## Shared audit pipeline (identical protocol for every model and checkpoint)
- **Neuron:** one of the 2,048 hidden units of a block's MLP. These models use relu², so `post = relu(pre)²`. The "negative pole" is taken from the signed pre-activation.
- **Collection** (`collect.py`): 2.0M held-out ClimbMix tokens from the validation shard (shard_06542), packed into 256-token sequences. Per sampled neuron: the top-20 contexts, the 20 most negative pre-activation contexts, and the 10 tokens its output direction most promotes and most suppresses (projected through the unembedding).
- **Essay-style label** (`label_sim.py`): MiMo sees the top-20 contexts (activating token marked, activation scaled 0–10), the 10 most negative contexts, and the promoted/suppressed tokens. It returns a label, a negative-pole description, a verdict (`one_job` / `two_related` / `mixed_unrelated` / `unclear`), a category (`entity_fact` / `lexical_token` / `syntax_format` / `semantic_topic` / `other`) and a confidence.
- **Gao-style simulation:** given only the label, MiMo predicts a 0–10 activation for every token of 10 held-out 32-token windows. The windows come from a disjoint 300k-token split: 5 are each neuron's top windows there and 5 are random.
  - `sim_corr`: Pearson correlation of simulated vs real activations over the 320 tokens.
  - **Substitution** (`substitute.py`): replace the neuron's activation on each window with the simulated one, then measure next-token KL to the clean model from the window start to the end of the sequence. The baselines are zero-ablation and mean-ablation of the same positions. An identity substitution gives KL = 0.000, so there's no noise floor.
  - `damage_avoided = 1 − KL_sim / KL_zero`. "Raw" maps a simulated 10 to the neuron's max activation. "Calibrated" fits a per-neuron affine map from simulated to real on the same windows: an in-sample, optimistic upper bound for label-based simulation.
  - `pass50` is the share with damage_avoided ≥ 0.5, the threshold used in the public `neurons-kinda-clean` Qwen3-4B replication.
- **Mid-range coverage** (`midrange.py`; extra, tests the critics' "top activations mislead" point): per neuron, 10 contexts where it fires at 10–50% of its max plus 5 where it is 0, shuffled. MiMo judges whether each marked token fits the label. `mid_fit` is the fraction of mid-range contexts judged to fit; `zero_fit` is the false-positive rate on silent contexts.

## T5 — does Engram make MLP neurons more monosemantic?
**Models.** `base_d8` (dense) vs `engram_d8`, both from the AGIdock archive. Both are d8: n_embd 512, 8 layers, 2,048 MLP units per layer. Both trained with seed 42 for 4,000 steps × 131k tokens ≈ 524M tokens with identical hyperparameters. Engram is attached at layers 2 and 5 (20k slots per N-gram, 128-dim). Final val bpb is 0.9286 (dense) vs 0.9241 (Engram).
**No dense d12 baseline exists** in the archive (only `base_d6` and `base_d8`), so the comparison is d8 only.
**Sample.** 100 random live neurons per layer, so 200 per layer group, 800 per model. No neuron was dead. Every audited neuron got the simulation, substitution and mid-range judgment: 800/800 in each model.

### Results (mean or median with bootstrap 95% CI; diff = Engram − dense)
| layers | metric | dense | Engram | diff |
|---|---|---|---|---|
| all | essay `one_job` | 0.96 [0.95,0.98] | 0.96 [0.95,0.98] | 0.00 [−0.02,0.02] |
| all | `one_job` or `two_related` | 0.99 | 0.99 | 0.00 |
| all | `entity_fact` category | 0.01 | 0.01 | −0.00 [−0.01,0.01] |
| all | sim_corr (median) | 0.43 [0.39,0.47] | 0.46 [0.43,0.49] | 0.03 [−0.03,0.08] |
| all | damage avoided, raw (median) | 0.03 [0.01,0.05] | 0.04 [0.02,0.06] | 0.01 [−0.02,0.04] |
| all | damage avoided, calibrated (median) | 0.15 [0.13,0.17] | 0.14 [0.12,0.16] | −0.01 [−0.04,0.02] |
| all | pass50 raw / calibrated | 0.09 / 0.12 | 0.09 / 0.11 | −0.01 / −0.01 |
| all | mid-range fit vs zero-context fit | 0.48 vs 0.34 | 0.48 vs 0.33 | 0.00 |
| 0,1 | sim_corr / cal. damage avoided | 0.63 / 0.11 | 0.69 / 0.13 | +0.06 [−0.02,0.14] / +0.02 |
| 2,5 (after Engram injection) | sim_corr / cal. damage avoided | 0.44 / 0.13 | 0.39 / 0.09 | −0.04 / −0.04 [−0.10,0.02] |
| 3,4 | sim_corr / cal. damage avoided | 0.42 / 0.13 | 0.39 / 0.12 | −0.02 / −0.01 |
| 6,7 | sim_corr / cal. damage avoided | 0.32 / 0.20 | 0.32 / 0.19 | 0.01 / −0.01 |

The full tables are in `T5/t5_summary.json`.

Other numbers:
- **Simulations often do more harm than zeroing.** Raw simulation is worse than zero-ablation for 24.5% (dense) and 24.7% (Engram) of neurons. Calibrated: 15.5% / 15.6%.
- **Few neurons are fully explained.** Calibrated damage avoided ≥ 0.8 for only 1.6% / 1.0%.
- **Correlation is not explanation.** Of the 200 dense neurons whose simulation correlates > 0.7 with reality, 48 still avoid < 20% of the zero-ablation damage. Example: `0:903`, "adjectives/nouns with negative valence (bad, dangerous, hazardous…)", has corr 0.91, calibrated damage avoided 0.02, and mid-range fit 0.11.
- **The two standards disagree.** Among essay-style "one_job" neurons, only 9% (dense) and 9% (Engram) pass Gao's 50% bar. Spearman(sim_corr, damage avoided) ≈ 0.65. Spearman(mid_fit, damage avoided) ≈ −0.14.

### T5 conclusions
1. **The hypothesis is not supported.** Moving facts into Engram did not make d8 MLP neurons measurably more monosemantic on any measure. Every CI on the difference includes 0, and the layers right after Engram injection (2, 5) lean slightly worse on simulation. Entity/fact neurons are rare in both (~1%), so at d8 / 524M tokens there may be little factual content for Engram to absorb. The test has power for differences of roughly ±0.04–0.06.
2. **Essay standard vs Gao standard.** With the essay's protocol (label from top activations), ~96% of neurons have "one job". That reproduces the essay's ~90% headline even in a tiny model, but the measure is at ceiling and doesn't discriminate. With Gao's protocol, the same labels avoid a median of only 3–4% (raw) or 14–15% (calibrated) of the damage of zeroing the neuron, and only 9–12% pass the 50% bar. That is in line with the public Qwen3-4B replication (11/278 passing). On these models, "looks like one job" is far from "can be replaced by its description", which is what Gao's critique predicts.
3. **Why labels fail substitution.** They describe peaks, not the bulk. Labels fit only 48% of mid-range activating contexts vs 33–34% of silent contexts. Neurons fire on ~19–40% of tokens, and much of what zeroing destroys lives in those weak and mid activations. This supports Gao's "top activations mislead" point and cuts against "weak activations are expendable" for these models (T0 tests that directly).

## T8 — recursive generalization over training
### Cross-run matching on the archived token-budget series (weak, so not used)
`engram_d8_v2` (= `large_t05B`, 3,841 steps), `large_t1B` (7,629) and `large_t2B` (15,259) share seed 42 and data order: their step-0 and step-10 losses are identical to 4 decimals. They diverge only through the LR schedule. Correlating all 2,048 × 2,048 neurons per layer on 65k held-out tokens:
- **t05B vs t2B:** median best-match corr 0.38–0.51 per layer; same-index corr 0.09–0.27; best match is the same index for 11–40%.
- **t1B vs t2B:** best-match 0.42–0.51; same-index 0.07–0.25.
- **Calibration, `base_d8` vs `engram_d8`:** best-match 0.33–0.41; same-index 0.01–0.04.

Median best-match is below ~0.5 in most layers, so following the plan I trained a fresh run with saved checkpoints instead (`T8/budget/match_*.json`).

### Fresh run
- **Setup:** dense d8 trained with `base_d8`'s exact config and seed (4,000 steps; device batch 4 × 32 grad-accum steps = same 131k-token batch).
- **Checkpoints:** steps 50, 100, 200, 400, 800, 1600, 2400, 3200 and 4000.
- **Reproduction check:** final val bpb 0.9296 vs archived `base_d8` 0.9286, so the run reproduces the model audited in T5.
- **Sample:** 400 neurons tracked by identity (50 per layer, two disjoint random sets). Each checkpoint was audited with the same 1M-token collection and label protocol. MiMo then classified each neuron's label sequence.

**Activation stability:** median same-index correlation with the final model, across layers:

| step | 50 | 100 | 200 | 400 | 800 | 1600 | 2400 | 3200 |
|---|---|---|---|---|---|---|---|---|
| corr | 0.06–0.13 | 0.11–0.20 | 0.19–0.28 | 0.29–0.41 | 0.45–0.56 | 0.62–0.73 | 0.74–0.85 | 0.88–0.94 |

**Label-noise calibration:** MiMo judged whether a neuron's label at step X and at the final step describe the same job.
- **s3200 vs final:** same 95.8%, related 3.3%, different 1.0%. Label noise is low.
- **s800 vs final:** 52% / 26% / 22%.
- **s200 vs final:** 17% / 22% / 61%.

**Per checkpoint:**

| step | 50 | 100 | 200 | 400 | 800 | 1600 | 3200 | 4000 |
|---|---|---|---|---|---|---|---|---|
| `one_job` | 0.77 | 0.83 | 0.90 | 0.94 | 0.95 | 0.96 | 0.95 | 0.96 |
| mixed/unclear | 0.14 | 0.10 | 0.06 | 0.03 | 0.03 | 0.02 | 0.01 | 0.01 |
| median firing rate | 0.53 | 0.45 | 0.41 | 0.26 | 0.20 | 0.17 | 0.17 | 0.19 |
| distinct top-20 tokens | 4 | 5 | 7 | 9 | 9 | 9 | 9 | 9 |
| modal-token share | 0.62 | 0.55 | 0.45 | 0.35 | 0.35 | 0.35 | 0.30 | 0.35 |

**Trajectory classes** (bootstrap 95% CI):
| window | narrow → related offshoots | mixed → specialized | stable | drift to unrelated | final job contains earliest job |
|---|---|---|---|---|---|
| all 9 checkpoints (n=399) | 0.085 [0.06,0.11] | 0.23 [0.19,0.27] | 0.02 | 0.67 [0.62,0.71] | 0.12 [0.09,0.15] |
| step 400 → 4000 (n=400) | 0.27 [0.23,0.32] | 0.21 [0.17,0.25] | 0.31 [0.26,0.35] | 0.21 [0.17,0.25] | 0.56 [0.52,0.61] |

By layer (from step 400): offshoots 0.24 / 0.31 / 0.26 and stable 0.33 / 0.26 / 0.35 for layers 0–2 / 3–5 / 6–7.

**Example offshoot trajectory, `0:879`:** hyphenated modifiers such as "free-", "non-", "mega-" (step 400) → "non"/"eco" prefixes → "FREE", "Easy" in marketing → promotional adjectives (free, easy, quick, simple) → "simple, easy, quick, free… adjectives for low-effort offerings" (step 4000).

### T8 conclusions
1. **Early training is mostly rewriting, not growth.** Over the first ~800 steps (~100M tokens), most neurons' jobs are rewritten rather than extended. Over all checkpoints, 67% "drift to unrelated" and only 12% keep their earliest job. The calibration shows this is not labeler noise (96% "same" between near-identical checkpoints). Early neurons are token detectors (modal top token 62% at step 50) with high firing rates. They become sparser and semantically broader as training proceeds (distinct top tokens 4 → 9; firing rate 0.53 → 0.17).
2. **After the job settles (≥ step 400), recursive generalization is a real but minority pattern.** Narrow-then-related-offshoots covers 27% [23,32]. Stable is 31%, mixed-then-specialized 21%, and unrelated drift 21%. 56% of neurons keep their step-400 job in the final model. The essay's motif exists (e.g. `0:879`), but it is one of several comparably common patterns, not the main explanation of how neurons form.
3. **Essay-style "one job" rises early and then plateaus at ~95% from step 400 on.** Meanwhile the jobs themselves keep changing (activation correlation with the final model only reaches 0.62–0.73 at step 1600). A clean-looking label at any one checkpoint says little about stability.

## Caveats
- These are small models (d8, ~126M parameters including embeddings, 25M in transformer matrices) with dense relu² MLPs. The essay audits Qwen3-32B and vision models. T8 is a single seed.
- One labeler, simulator and judge (MiMo Pro), with no human spot-check. The trajectory and pair judgments are LLM judgments over labels, so their absolute values depend on wording. The pair calibration bounds noise for "same job", not for the finer trajectory classes.
- Simulation windows are 32 tokens without the left context the model saw. That handicaps the simulator on context-dependent neurons, which is equally true of Gao's protocol. Per-neuron substitution uses ~2.4k scored positions, so individual estimates are noisy; group medians and CIs are the robust quantities. The calibrated score is optimistic (2 parameters fitted in-sample).
- The fresh run used device batch 4 × 32 accumulation instead of 8 × 16. The total batch is the same, so it is not bit-identical to `base_d8`, but val bpb agrees within 0.001.
- MiMo concurrency: short label bursts from the second T8 neuron set overlapped the T5 simulations, putting me at about 3 calls per box at times vs the 2 allotted.
- One neuron (`s00050`) kept a malformed verdict after a retry, so it is excluded from the 9-checkpoint trajectories (n = 399).

## Files
- `T5/{base_d8,engram_d8}/`: `neurons.json`, `labels.json`, `sims.json`, `subst.json`, `midrange.json`, `midrange_judge.json`; summary in `T5/t5_summary.json`.
- `T8/budget/match_*.json`: cross-run matching.
- `T8/fresh/`: per-checkpoint audits `s*/` and `b_s*/`; `trajectories.json`, `per_checkpoint.json`, `t8_summary.json`; `from400/` (trajectories from step 400); `pairs.json` and `pairs_summary.json`; `match_s*__final.json`.
- Checkpoints: `~/clean-tests/nanochat_ckpts/engram_runs/{base_d8,engram_d8,engram_d8_v2,engram_d8_large_t1B,engram_d8_large_t2B,t8_dense_d8}`. ClimbMix shards are symlinked from T9's copy.
