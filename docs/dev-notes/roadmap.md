# User as Engram: roadmap and feasibility pilot

**Draft v0.1 (2026-05-04)**
Author: boj@19pine.ai
Companions: **POLAR** (`~/polar-research`), **UserAsCode** (`~/UserAsCode`),
**User-as-LoRA** (`~/user-as-lora`)

---

## 0. Premise, in one paragraph

`User-as-LoRA` established empirically that the bottleneck in adapter-based
personal memory is not *storage* but *invocation*: POLAR adapters reach 100%
direct recall yet collapse on indirect questions until either an explicit CoT
prompt or a trace-mix recipe forces recite-then-reason. Even with the recipe,
generalisation is schema-conditioned (within-schema 0.41, cross-schema 0.046).
DeepSeek's **Engram** (Cheng et al., 2026; arXiv:2601.07372) introduces a
conditional-memory primitive whose context-aware gate fires *deterministically*
on N-gram patterns and, per the paper's sensitivity ablation (§6.3), serves
as the **primary parametric fact store** of the trained model — suppressing it
collapses TriviaQA to 29% retained while preserving reading comprehension at
93%. **User as Engram** asks whether per-user memory can be implemented as
hash-addressed user-fact rows written into an Engram-pretrained base, with
spontaneous invocation handled by the architecture rather than learned.

---

## 1. Substrate constraint we hit immediately

DeepSeek released:
- The paper PDF.
- `engram_demo_v1.py` (422 lines) — an architectural reference that
  *mocks Attention/MoE/mHC*, exercises only the Engram module, and is
  randomly initialised.
- No trained model weights.

**Implication.** The pilot in the User-as-LoRA plan was framed as
"experiments using Engram-27B." That is not currently possible; there is no
Engram-pretrained checkpoint on Hugging Face or anywhere else. Anything we
do has to either (i) build a small Engram-pretrained model ourselves, or
(ii) probe the *architectural* feasibility on the demo code (random weights)
where the questions are structural rather than empirical.

This constraint is load-bearing. The roadmap below is structured around it.

---

## 2. Three tiers of pilot

### Tier 1 — architectural feasibility on demo code (this week)

Cheap, runs on the existing GPU, answers structural questions that do *not*
depend on having trained weights:

- **T1.1** Demo smoke test. Confirm `engram_demo_v1.py` runs end-to-end on
  the local hardware (RTX PRO 6000, 96 GB).
- **T1.2** Hash-slot reachability. Given a synthetic user fact set, hash
  each fact's surface form, list the (n, head) → slot indices that
  retrieval would touch. Confirm the slots are well-defined and
  reproducible from `(layer_id, seed, multipliers)`.
- **T1.3** Slot read/write surgery. Write a known vector into a target
  slot row of `MultiHeadEmbedding.embedding`, run the Engram forward with
  random key/value projections, verify the gate produces a non-trivial
  scalar and the value projection produces an output derivable from the
  written vector. This is the minimum existence proof that *the
  architecture supports surgical insertion at all*.
- **T1.4** Cross-user collision audit. Generate N = 100 synthetic users
  with ~30 facts each, hash everyone's keys, count collisions per
  (n, head, slot). Answers: do we need per-user salt in the hash, or are
  natural collisions rare enough at scale?

**Tier 1 go/no-go.** If T1.3 fails (the architecture cannot surface a
written row through the gate), the project halts and we reframe. If T1.4
shows >10% slot collision across users at modest scale, we know per-user
salt is mandatory — still doable, but a real design constraint.

### Tier 2 — mini-Engram pretraining (2–4 weeks)

If Tier 1 is green, we need a *trained* Engram base small enough to iterate
on. Two routes:

- **Option A — train from scratch.** Pretrain a 200–400M Engram model on
  a clean corpus (FineWeb-Edu sample, 10–20B tokens). Reproduces the
  paper recipe (Engram modules at layers 2 and ~mid, max-N = 3, 8 hash
  heads, conv init zero, embeddings with 5× LR no weight decay). On a
  single 96 GB GPU this is roughly a week of training time; bf16, batch
  ~256, sequence ~2048.
- **Option B — graft Engram onto Qwen2.5-3B.** Start from a frozen Qwen
  base (already in the POLAR toolchain, cache-resident), insert Engram
  modules at layers 2 and 14, train *only the Engram parameters* and a
  small co-adapt of layers 0–2 on a few-billion-token mix. Cheaper,
  faster, but it is no longer a clean Engram architecture — the rest of
  the base was never trained to consult the gate, so Q1 ("does the gate
  consult an inserted user row") is partly confounded.

We start with Option A unless wall-clock pressure forces Option B. Option A
gives us a clean Engram base to study; Option B is a fallback.

**Tier 2 deliverable.** A loadable Mini-Engram checkpoint with a working
hash mapping, used for all Tier 3 experiments.

### Tier 3 — User-as-Engram experiments (the actual paper)

Mirrors the User-as-LoRA evaluation skeleton, with substrate swapped:

- **T3.1 (Q1).** Subspace alignment. Write a user-fact vector into a free
  hash slot. Three insertion strategies — random, nearest-neighbour to
  existing trained rows, or LM-head-direction (write the unembed of the
  answer token). Measure: does the gate fire, does unembed surface the
  right token at the right N-gram?
- **T3.2 (Q2).** Surface vs. semantic retrieval. Insert "doctor → Dr. Patel".
  Query with paraphrases ("GP", "physician", "tooth cleaning"). Measure
  recall vs. paraphrase distance. Compare against POLAR's NTP-based
  paraphrase robustness.
- **T3.3 (Q3).** Indirect / cross-schema reasoning. Run the User-as-LoRA
  follow-up-1/2/3 batteries on User-as-Engram. Headline claim alive iff
  cross-schema held > 0.05 (POLAR baseline) without trace-mix.
- **T3.4 (Q4).** Insertion cost. Quantify wall-clock and GPU-mem to
  insert one user (target: < 1 s, no gradient) vs. POLAR Stage A (~8
  GPU-min/user). This is the production case for the paper.
- **T3.5** Multi-user composition. Two users in one conversation. Show
  no cross-user leakage (the gate disambiguates by N-gram).
- **T3.6** Privacy / unlearning. Demonstrate exact deletion by zeroing
  the affected rows.

Baselines reused from User-as-LoRA: ICL upper bound, RAG, POLAR adapter,
POLAR + CoT prompt, trace-mix Stage A.

---

## 3. Open structural questions before Tier 3

These determine whether the paper exists; flagging them now so they get
answered first.

1. **Subspace problem.** W_K, W_V in the Engram module are trained against
   the *learned* row distribution. An arbitrary written vector may be
   gated to zero. Tier 1 (T1.3) gives a partial answer (random weights,
   structural sanity). Tier 3 (T3.1) gives the real answer.
2. **Surface-only retrieval.** N-gram hashing keys on tokens, not
   semantics. POLAR's NTP gives paraphrase robustness for free; we may
   need a paraphrase-expansion step at insert time.
3. **Per-user salt.** If Tier 1 (T1.4) shows non-trivial collision, we
   add `user_id` to the hash seed. Engineering, not science.
4. **Compositional invocation.** When the gate's training distribution is
   global text statistics, will it invoke a sparse user row when the user
   is talking *about* that fact? Engram's Zipfian assumption favours
   high-frequency rows; user facts are by definition rare.

---

## 4. Decision gates

- **G1 (end of Tier 1):** demo runs, slots are addressable, collision
  rate < 10% at N = 100 users. If yes → Tier 2.
- **G2 (end of Tier 2):** Mini-Engram trained, factual benchmarks
  reproduce qualitative findings of the 27B paper at smaller scale
  (Engram suppression collapses TriviaQA-style probes more than reading
  comprehension). If yes → Tier 3.
- **G3 (mid Tier 3):** T3.1 shows non-zero gate activation and >0.5
  unembed accuracy on a written user fact for at least one of the three
  insertion strategies. If yes → continue. If no → reframe to a learned
  insertion encoder (a real shift in scope).
- **G4 (end of Tier 3):** Cross-schema indirect ≥ 0.20 on Mini-Engram.
  If yes → write paper. If no → likely still publishable as a negative
  result on parametric per-user memory, but a different paper.

---

## 5. What is in this directory

```
user-as-engram/
├── roadmap.md         ← this file
├── notes.md           ← running pilot notes (Tier 1 starts here)
├── refs/
│   ├── engram_demo_v1.py   ← cloned from deepseek-ai/Engram
│   └── engram_README.md
├── src/               ← pilot scripts (probes, micro-tests, eventually Mini-Engram)
├── data/              ← synthetic user fact sets (will mirror user-as-lora's)
└── results/           ← outputs of pilot runs
```

---

## 6. Risks and how the roadmap absorbs them

| Risk | Mitigation in roadmap |
|---|---|
| No Engram weights ever released | Tier 2 trains our own Mini-Engram |
| Subspace problem kills T3.1 | G3 reframe to learned insertion encoder |
| Mini-Engram is too weak to show useful indirect reasoning | Compare to POLAR on the same Qwen-class base; relative gain is the claim |
| Compute runs out before main paper | G2 gate ensures we don't enter expensive Tier 3 without a working Mini-Engram |
| DeepSeek releases weights mid-project | We immediately switch substrate; the science is the same |

---

*End of roadmap v0.1. Next action: Tier 1 pilot — start with T1.1.*
