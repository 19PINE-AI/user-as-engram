# User as Engram: Internalizing Per-User Memory as Local Parametric Edits

**Working title (choose one)**:
1. *(recommended)* "User as Engram: Internalizing Per-User Memory as Local Parametric Edits"
2. "User as Engram: Internalizing User Memory as Parametric LLM Edits" *(original suggestion)*
3. "User as Engram: Per-User Memory Without LoRA's Invocation Tax"

**Status**: outline + section budgets. Body content drawn from
existing `~/user-as-lora/notes.md` (LoRA negative results) and
existing `PAPER.md` (Engram method, experiments, serving).

**Target venue/length**: ICLR or NeurIPS main (8–9 pages + appendix), or
NeurIPS Datasets & Benchmarks (longer). Outline below targets ~9 main
pages with appendix for ablations.

---

## One-line thesis

Per-user LoRA edits are *global* and *reasoning-negative*; Engram
override edits are *local* and *reasoning-neutral*; the latter scales
to millions of users with sub-millisecond per-request swap.

## Headline numbers (to advertise on page 1)

- **LoRA-as-memory is reasoning-negative**: per-user POLAR LoRA reaches
  1.000 direct recall but indirect recall is 0.150 — *below* the
  no-adapter base of 0.170 in 5/10 users (User-as-LoRA, Stage A).
- **Engram override is reasoning-neutral**: surgical row insertion at
  the trigger N-gram changes the model output by 62 L2-norm at the
  trigger position and by **0.000 at every other position**, through
  every subsequent attention/MLP layer (insertion-attribution test, d12).
- **Joint OPT achieves 68% top-1 / 96% top-5 at 100 facts/user** in
  88 KB of override storage (vs 13.5 MB POLAR-class LoRA at 99% / 100%).
  At 1 K users × 100 facts: **98 MB Engram vs 13.5 GB LoRA = 137× less storage**.
- **Live multi-tenant serving on Mini-Engram-d12 (30 users × 50 facts × 600 reqs):
  47.9 req/s, p99 latency 27.8 ms, override apply p99 2.3 ms,
  cross-user privacy = 0% by construction.**

---

## Outline (page budgets are approximate, ~9 main pages)

### Abstract (0.5 pg)

Per-user memory in LLMs is an open problem. The dominant approach —
per-user LoRA adapters — achieves perfect direct recall but, we
measure, is *reasoning-negative*: indirect-question accuracy drops
*below* the no-adapter base in 5/10 users. The cause is that LoRA
edits the model's weights globally, contaminating every forward pass
the user makes. We propose **User as Engram**: instead of training a
global per-user adapter, surgically insert per-user fact rows into the
hash-keyed embedding table of an Engram-pretrained base. The edit is
*local* (one row per fact, fired only at the trigger N-gram), so
non-trigger forwards are unaffected (we measure exactly 0.000 L2-norm
contamination at every other position). We present (i) a nano-scale
public reproduction of Engram at 137 M / 339 M params, (ii)
**Joint OPT**, a sparse-row joint training method that recovers
recall under high fact density, (iii) a working multi-tenant serving
system with sub-millisecond override swap and zero cross-user leak
by construction, and (iv) a comprehensive comparison vs ICL, RAG,
SFT-LoRA, and POLAR. **Joint OPT achieves 68% top-1 / 96% top-5 at
100 facts per user in 88 KB of storage**, 137× smaller than
POLAR-class LoRA at the same scale.

### 1. Introduction (1.0 pg)

- The personal-memory problem: millions of users sharing one base
  model; need per-user knowledge without retraining the base.
- The LoRA-as-memory family (POLAR, OPPU, HYDRA, MemLoRA, PRAG,
  T2L, ...): conceptually clean, achieves direct recall, but...
- **Punch line**: per-user LoRA *hurts* indirect reasoning. We
  measure 0.150 indirect with adapter vs 0.170 base alone (5/10 users).
  Mechanism: LoRA edits weights globally → contaminates every forward.
- **Our contribution**: User-as-Engram. The Engram architecture
  (Cheng et al. 2026) gives us hash-keyed embedding tables. Per-user
  facts become surgical row writes that fire only at trigger N-grams.
  Local edit ⇒ no contamination ⇒ no reasoning-negative.
- Contributions list (4 items: nano-scale reproduction, OPT/Joint OPT,
  multi-tenant serving, vs-LoRA/SFT/ICL/RAG comparison).

### 2. Background (0.75 pg)

#### 2.1 Engram (Cheng et al. 2026)
Hashed N-gram embeddings, context-aware gate, deterministic O(1)
lookup. Reference Figures 1, 4 from the original.

#### 2.2 LoRA-as-memory
Brief survey of POLAR, OPPU, MemLoRA, etc. Common pattern: train a
per-user/per-document LoRA, freeze the base, hot-swap at inference.

### 3. The LoRA-as-memory invocation gap *(LARGELY FROM USER-AS-LORA NOTES)* (1.75 pg)

This is the *measurement-driven motivation*. Use User-as-LoRA's
empirical findings to set up the problem Engram solves.

#### 3.1 Setup
- Qwen2.5-3B base, POLAR LoRA per user (rank 64, NTP on
  observation/fact/QA mixture, 12 epochs × 200 samples).
- 10 synthetic users × 50 facts; direct + indirect probes.

#### 3.2 Direct recall is solved; indirect is not
| metric | mean |
|---|---|
| direct recall (w/ adapter) | **1.000** |
| indirect recall (w/ adapter) | **0.150** |
| indirect recall (base only, no adapter) | 0.170 |

In 5 of 10 users, the adapter is *worse* than the base alone on
indirect questions. **The adapter is reasoning-negative.**

#### 3.3 Recite-then-reason traces only partially help
- Within-schema indirect held: 0.41 (with trace-mixed Stage A)
- Cross-schema indirect held: **0.046** — collapse below base alone
- Stage C meta-train (full-param base meta-trained over LoRA distribution):
  held-user lift = +0.000 (fails to transfer at minimum scale)

#### 3.4 Diagnosis
LoRA writes a low-rank delta into Q/K/V across all layers. Every
forward — even ones where the user's stored fact is irrelevant — sees
the delta. Empirically this disrupts the model's general indirect
reasoning, especially on facts whose surface form weakly matches the
trained QA pattern.

The *adapter contains the facts* (CoT prompt + adapter ≈ ICL ceiling
at 0.30); the *adapter doesn't help indirect reasoning spontaneously*;
and the *adapter's weight perturbation actively interferes* with the
model's normal reasoning.

This sets up the design requirement: **per-user edits must be local,
firing only when the relevant trigger appears**.

### 4. Method: User as Engram (1.5 pg)

#### 4.1 Surgical row insertion
Tokenize the trigger; compute the suffix-N-gram hash addresses at the
last Engram layer; write a marker vector into those rows.

#### 4.2 Three insertion strategies (with cost/quality)
- **UNEMBED_P** (closed-form): $W_V^\dagger U_y$. ~1 ms per fact.
- **OPT independent** (15-step gradient on the row): ~1 s per fact.
- **Joint OPT** (sparse-row jointly trained for N facts together):
  ~K/N s per fact for K total steps. *This is the recommended default
  for ≥30 facts per user.*

#### 4.3 Per-user override tables and additive composition
Override map = `{global_row_idx → row_vector}`. Apply per request,
restore after. Disjoint addresses commute, so corporate + user +
project Engrams stack without retraining a combiner (analogous to
Stable Diffusion LoRA stacking, but additive at the row level).

### 5. Nano-scale reproduction of Engram (0.75 pg)

We graft Engram into nanochat and train two Mini-Engrams (d8: 137 M,
d12: 339 M params) on 524–786 M ClimbMix tokens in ~50 min – 3 h on a
single Blackwell. **First public reproduction outside DeepSeek-AI.**

| | base d8 | engram d8 | engram d12 |
|---|---|---|---|
| total params | 125.8 M | 136.7 M | 339.2 M |
| final val_bpb | 0.929 | 0.924 | 0.849 |

§6.3 (sensitivity asymmetry) reproduced at d8: factual top-5 retains
93.3% under suppression; reading top-5 retains 100%; Δ = +0.067 in the
paper's direction (Cheng et al. report Δ ≈ +0.6 at 27 B / 262 B).

§6.1 (effective deepening) reproduced: layer-3 LogitLens KL is 3.66
lower for engram-d8 vs base-d8 — engram converges faster in early
layers, matching Figure 4(a) shape.

### 6. User-as-Engram experiments (3.5 pg)

#### 6.1 Single-fact recall (1000 facts × USER + 1000 × ORG, d12)
| method | USER top-1 | ORG top-1 | per-fact wall |
|---|---|---|---|
| ICL (= optimal RAG) | 95.8% | 99.9% | per query |
| OPT independent | 87.6% | 90.8% | ~1 s |

#### 6.2 Reasoning-neutrality (THE central claim, contrasting §3)
**Insertion attribution**: writing a UNEMBED_P marker at the last
Engram layer changes the residual stream by L2-norm 62.75 at the
trigger position, layer 6 of d8. At every other position, in every
layer (including layers AFTER the insertion), the L2-norm change is
**exactly 0.000**. Compare with §3's LoRA result: every forward is
contaminated by the LoRA delta. Engram doesn't pay the reasoning tax.

#### 6.3 Within-user fact-density and Joint OPT
| n facts simultaneous | OPT independent | Joint OPT |
|---|---|---|
| 100 | 36% / 54% | **68% / 96%** |
| 1000 | 13% / 31% | **35% / 72%** |

Joint OPT roughly doubles top-1 and triples top-5 at every density
without changing storage. Recommended default at ≥ 30 facts/user.

#### 6.4 Multi-domain additive composition
Corp (10 facts) + User (10 facts) stacked at inference: **lossless on
both domains** (corp 80% / 90%, user 90% / 100% — same as alone).
Address overlap 2.7% (disjoint templates).

When templates collide (4 user-style domains), composition degrades:
D=4 mean top-1 = 29%. Architectural conclusion: additive composition
for *disjoint-template* domains, per-user override for *same-template*
tenants. Complementary, not competing.

#### 6.5 Paraphrase generalization
Single-trigger insertion: 50% top-1 across 5 paraphrases (free, from
suffix-N-gram overlap). Multi-trigger insertion (insert at all
paraphrases): 100% top-1.

#### 6.6 Comparison vs SFT-LoRA, POLAR-class, and ICL
At 100 simultaneous facts:
- ICL (perfect retriever): 100 facts × ~10 tokens = ~1000 context
  tokens — impractical at chat scale.
- SFT-LoRA per-fact (rank 8, 30 steps): 100% top-1 but 1700× more
  storage per fact than one Engram row.
- **Multi-fact LoRA rank 64 (POLAR-class)**: 99% top-1, 13.5 MB.
- **Joint OPT**: 68% top-1, **88 KB** (153× smaller).

At 1000 simultaneous facts, the gap closes: LoRA 43.8% vs Joint OPT
35.1% — Joint OPT is **45× smaller and 2.7× faster to train**.

#### 6.7 Multi-tenant serving (live system)
EngramServer (50 lines of Python wrapping the model). Override apply
in 0.4 ms (d8) / 2.2 ms (d12). 30 users × 50 facts × 600 reqs:
47.9 req/s, p99 latency 27.8 ms, own-fact recall 75.5% top-1 / 98.3%
top-5, cross-user leak = 0 architecturally (the 3.9% probe rate is
gold-value coincidence from a 20-item answer pool).

Compare to LoRA serving (S-LoRA / Punica): no router, no fused
kernel; just `embedding.weight[rows] = ...` per request.

### 7. Discussion and limitations (0.75 pg)

- **Shared multi-hop limitation**: both LoRA and Engram are
  surface-trigger keyed; neither composes facts across triggers
  ("if my doctor is Patel and Patel works at Globex, ..."). Engram
  inherits the User-as-LoRA gap on this. Mitigation: train Engram
  with recite-then-reason traces (User-as-LoRA's Stage A recipe).
- **Engram pretraining required**: User-as-Engram needs an Engram
  base. We trained ours; production needs DeepSeek to release weights
  or pretraining your own (~3 h on a single Blackwell at our scale).
- **Density ceiling**: Joint OPT at 1000 facts gets 35% top-1.
  Hash-table sparsity is not the constraint (slot occupancy < 1%);
  the constraint is forward-pass interference among many live rows.
  Per-user scoped tables (only load relevant rows for the current
  query) is a candidate future fix.
- **No retrieval grounding**: Engram is opaque parametric memory. For
  citation/provenance, complement with RAG.

### 8. Conclusion (0.25 pg)

Per-user LoRA edits are *global* and *reasoning-negative*; Engram
override edits are *local* and *reasoning-neutral*. The latter
scales to millions of users with sub-millisecond per-request swap,
~150× less storage than LoRA at the same recall regime, and
additive composition across domains. The bottleneck of personal
memory moves from gradient training to substrate selection — and the
right substrate is hash-keyed embedding tables.

### Appendix (no page limit)

- A. Full Engram architecture and our nano-scale port (code listing).
- B. POLAR / User-as-LoRA detailed tables (Stages A/B/C, follow-ups
  1/2/3) — full negative-result data behind §3.
- C. Joint OPT algorithm + hyperparameter sweep.
- D. EngramServer code listing and benchmark methodology.
- E. Mechanistic figures (LogitLens KL, insertion-attribution heatmap).
- F. Reproducibility checklist and full reproduce script.

---

## What gets dropped from the standalone papers

**From User-as-LoRA**:
- Stage B trace-synthesis details (programmatic vs teacher-distilled)
  → moved to appendix or cited briefly.
- Cross-base generalization, attention-based memory plugins
  literature survey → cite once in §2.

**From the current standalone User-as-Engram paper**:
- Long architecture explanation of Engram → compressed to §2.1
  (Cheng et al. is now a citation, not a tutorial).
- Salt-table vs override-table comparison → moved to appendix C
  (per-user override is the headline; salt is alternative).
- LogitLens d8 vs base d8 KL table → keep in §5; insertion
  attribution stays in §6.2 (it's load-bearing for the central claim).

## Open decisions to make before writing prose

1. **Length**: 8 pages (NeurIPS main) requires aggressive trimming;
   12 pages (ICLR or workshop) gives breathing room; 20 pages
   (NeurIPS D&B) lets all of it live in the main text.
2. **Framing**: "we measure LoRA's failure → propose Engram fix"
   (current outline) vs. "we propose Engram and show it beats LoRA"
   (more conventional). I recommend the former — it's distinctive.
3. **Negative-result framing**: how aggressively to push the
   "reasoning-negative" finding. It's true and striking but might
   antagonize the LoRA-as-memory community.
4. **Anonymisation**: this run is solo-author (boj@); double-blind
   submissions are simpler.
