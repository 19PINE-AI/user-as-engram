# New mechanistic experiments (trained Mini-Engram) — for §4 rewrite

Scripts: `nanochat/scripts/mech_glassbox.py`, `mech_lora_vs_engram.py`, `mech_depth.py`
Results: `results/glassbox_{d12_1280,d20}.json`, `lora_vs_engram_*`, `depth_*`
Figures: `paper/figs/fig_glassbox.pdf`, `fig_lora_vs_engram.pdf`

## The five-stage glass-box account (revised by evidence)

1. **Address** (deterministic). Hash → sparse R_f (~16 rows). T1 audit: 0.027% occupancy,
   4.5% key-token collision. (existing)

2. **Self-gating write** (NEW). Writing R_f raises the trigger-position gate from
   α≈0.015 (baseline) to **α≈0.99** (OPT, deployed) / 0.6 (UNEMBED), while every
   non-trigger position stays at α≈0.03–0.04. The row both supplies the value AND
   opens its own gate at exactly the trigger. d12@1280: 0.014→0.616→0.988. d20: 0.016→0.591→0.988.

3. **Injection = value path** (NEW; reproduces appendix 0.998 on TRAINED model).
   The residual change the deployed OPT row induces at the trigger is cosine
   **0.998 (d12@1280) / 0.999 (d20)** aligned with its analytic value-path projection
   W_V·e. Gate and short-conv scale but do not redirect. (UNEMBED's small-magnitude
   marker is dominated by the conv nonlinearity → noisy direction; OPT drives the row
   norm up 99→132 so the value path dominates.)

4. **Exact locality, population-level** (NEW; was single heatmap). Max non-trigger
   residual change = **0.000** across ALL 16 facts × both strategies × all layers, and
   0.000 before/at the Engram layer. Both models.

5. **What the row encodes** (NEW honest nuance). Closed-form UNEMBED_P points the value
   path at the gold token's unembedding (cos **0.59–0.65**); gradient OPT/Joint-OPT trade
   some direct alignment (→**0.16–0.24**) for higher recall by shaping the full
   next-token head. "Row stores the gold value" is literal for closed-form,
   same-hemisphere-approximate for deployed.

## Depth-of-insertion (causal) — E6

Same insertion, same OPT-15 budget, early vs late Engram layer:
- d20 (early=2, late=11): OPT top1 **0.25 → 1.00**; UNEMBED 0.12 → 0.19.
- d12@1280 (early=2, late=7): OPT top1 **0.31 → 1.00**; UNEMBED 0.12 → 0.25.
A late edit overrides an already-near-final prediction; an early one must survive the
stack and washes out. Converts LogitLens "effective deepening" from correlational to causal.

## LoRA vs Engram per-position effect — E5

- Engram surgical insertion: max non-trigger change **0.000**.
- A per-user LoRA learning the SAME fact: nonzero at EVERY position/layer
  (range [5, 430]) AND perturbs unrelated text ("The capital of France is") by
  mean **107 (d12@1280) / 146 (d20)**.
→ Side-by-side figure = "addressed write vs global function-bend."
