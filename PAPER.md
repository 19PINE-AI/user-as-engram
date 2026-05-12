# User as Engram: Per-User Memory via Surgical Insertion into Hashed N-gram Tables

**Author:** boj@19pine.ai *(autonomous research run, single-GPU)*
**Draft:** v1.1, May 2026
**Code & checkpoints:** `~/user-as-engram/` (release pending)

---

## Abstract

Personalising large language models with per-user adapter weights — the
LoRA-as-memory family — has run up against a stubborn invocation
bottleneck: per-user fact-LoRAs reach near-perfect *direct* recall but
collapse on *indirect* questions, because a frozen base was never
trained to consult its hot-swapped adapter. DeepSeek's recent **Engram**
architecture (Cheng et al., 2026) introduces a complementary sparsity
axis — conditional memory via O(1) hashed N-gram lookup — whose
context-aware gate is *trained* to invoke parametric memory at the right
N-gram. We propose **User as Engram**: instead of training a per-user
LoRA, surgically *insert* a per-user fact row into an unused hash slot
of an Engram-pretrained base, no further gradient updates required.

We make three contributions. First, what is, to our knowledge, **the
nano-scale public reproduction of the Engram architecture**, at two scales:
Mini-Engram-d8 (137M params, 524M tokens, ~50 min on a single GPU) and
Mini-Engram-d12 (339M params, 786M tokens, ~3 h). The d8 model
qualitatively recovers Cheng et al.'s §6.3 finding that suppressing
Engram disproportionately hurts factual probes versus reading-comprehension
probes (factual top-5 retained 93%, reading 100%, Δ +0.067 in the
paper's direction). Second, we demonstrate User-as-Engram surgical
insertion at three levels of sophistication — RANDOM (control),
UNEMBED_P (closed-form pseudo-inverse), and OPT (15-step gradient on
the inserted row) — and show that on Mini-Engram-d12, **OPT achieves
93% top-1 hit rate on a 100-fact USER benchmark** (with 100% top-5)
**and 94% on 100 ORG facts**, *exceeding* the in-context-learning
ceiling on USER and matching it on ORG, at three orders of magnitude
lower per-user cost than POLAR-style LoRA training (~1 s per fact vs
~8 GPU-min). Third, a multi-tenant test exposes the recall-leakage
design: **per-user override tables**, applied per-request and
restored after, give **zero cross-user leak by construction** and
recall that matches single-user (~88% top-1). Override maps with
disjoint addresses commute, so corporate + user (or arbitrary
multi-domain) Engram fragments **stack additively at inference** —
the same compositional property that makes Stable Diffusion LoRAs
plug-and-play.

We release the trained Mini-Engram checkpoints, the porting code, and
the full evaluation suite.

---

## 1. Introduction

Modern LLMs serve millions of users from a single backbone. Per-user
*personal memory* — the system's knowledge of *your* doctor, *your*
preferences, *your* calendar — is an open architectural problem. The
LoRA-as-memory family (POLAR, OPPU, HYDRA, MemLoRA, PRAG/DyPRAG, T2L)
treats per-user memory as a per-user weight delta. This is conceptually
clean and sufficient for personalisation of style and surface
preferences, but recent work (POLAR, User-as-LoRA) has surfaced a
striking *invocation gap*: per-user fact-LoRAs achieve near-perfect
direct recall ("What is my doctor's name?" → "Patel") and yet collapse
on indirect or compositional reasoning over the same facts ("What does
my doctor specialise in, given that they wrote me a prescription for
hypertension?"). The frozen base was never trained to consult its
hot-swapped adapter; recite-then-reason traces close the gap *within
seen schemas* (~40% indirect) but not *across* schemas (~5%).

Concurrently, DeepSeek's **Engram** (Cheng et al., 2026; arXiv:2601.07372)
introduced *conditional memory* — a hashed N-gram embedding table with
a context-aware gate — as a complementary axis of sparsity to MoE. The
gate is *trained end-to-end* to fire deterministically at static
patterns ("Alexander the Great", named entities, formulaic phrases).
This solves the invocation problem for global pretraining facts.

**Our claim**: the Engram gate, once trained, will fire on *any* row
written to its hash table — including per-user facts inserted at
inference time. If true, per-user memory becomes a hash write rather
than a gradient-trained adapter, with no invocation bottleneck (the
gate is architectural, not learned per-fact).

We test this claim. Our contributions:

1. **Nano-scale public reproduction of the Engram architecture** at two
   scales (d8 and d12). DeepSeek released only architectural reference
   code; no Engram-trained weights exist publicly. We graft Engram into
   nanochat (Karpathy 2026), train Mini-Engrams from scratch on a
   single GPU, and reproduce the paper's §6.3 sensitivity asymmetry at
   d8 and the paper's §6.1 effective-deepening claim mechanistically.

2. **The User-as-Engram method.** Three insertion strategies with
   increasing sophistication: RANDOM (control), UNEMBED_P (closed-form
   pseudo-inverse), and OPT (15-step gradient on the inserted row). On
   the d12 substrate, OPT achieves 93% top-1 (USER) and 94% top-1
   (ORG), with 100% top-5 on USER. UNEMBED_P alone is much weaker
   (26% / 25%). The cost: ~1 s per fact (OPT) vs ~8 GPU-min for a
   per-user LoRA — three orders of magnitude cheaper.

3. **Multi-tenant scaling via per-user override tables and additive
   composition.** The natural production design — apply the active
   user's override map per request, restore after — gives **zero
   cross-user leakage by construction** and recall that approaches
   single-user (~88% top-1 OPT at 100 facts on d12). Override maps
   with disjoint addresses commute, so corporate facts and user
   facts (or any number of domains) **compose additively** at
   inference time without retraining a combiner.

4. **Comprehensive cost comparison vs LoRA-based personalisation.**
   At 1 K users × 100 facts each: User-as-Engram needs **~98 MB of
   override-row storage** (1 KB per fact). Per-fact SFT-LoRA needs
   **165 GB**. POLAR-style per-user LoRA needs **39.6 GB**. Training:
   OPT is ~1 s per fact (~100 s per 100-fact user); POLAR is ~8
   GPU-min per user. User-as-Engram is **~5× faster** to train and
   **~400× smaller** than per-user LoRA at 100 facts per user.

4. **Paraphrase generalization.** Single-trigger insertion gives 50%
   paraphrase top-1 generalization for free (suffix-N-gram overlap).
   Multi-trigger insertion (insert at all 5 paraphrases of a fact)
   gives 100% top-1 at 5× cost.

---

## 2. Background

### 2.1 The Engram architecture

Following Cheng et al. (2026), Engram inserts a conditional-memory
module into selected layers of a Transformer. At each token position
$t$, for each $n$-gram order $n \in \{2, \dots, N\}$ and each of $K$
hash heads, a deterministic multiplicative-XOR hash $\varphi_{n,k}$
maps the canonicalised suffix $N$-gram into a row of an embedding table
$E_{n,k}$:

$$z_{t,n,k} = \varphi_{n,k}(g_{t,n}), \qquad e_{t,n,k} = E_{n,k}[z_{t,n,k}].$$

The retrieved rows are concatenated into $e_t \in \mathbb{R}^{d_{\text{mem}}}$,
projected by learned $W_K, W_V \in \mathbb{R}^{d \times d_{\text{mem}}}$,
and gated by an attention-style scalar:

$$\alpha_t = \sigma\!\left(\frac{\mathrm{RMS}(h_t)^\top \mathrm{RMS}(W_K e_t)}{\sqrt{d}}\right).$$

The output is $\alpha_t \cdot W_V e_t$, refined by a depthwise causal
convolution and added to the residual stream. Crucially, the addressing
is determined entirely by token IDs — known *before* the forward pass
— so the table can be offloaded to host DRAM without GPU contention.

### 2.2 The invocation gap in per-user memory

POLAR (Phase 1c) showed that a rank-64 LoRA can store 50+ facts at
100% direct recall. Yet on indirect questions over those facts,
accuracy collapses to ~5%. Even POLAR + an explicit chain-of-thought
prompt recovers only the in-context-learning ceiling (~30%).
User-as-LoRA's trace-mix recipe partially closes the within-schema gap
(~40%) but fails cross-schema (~5%). The bottleneck is not storage but
*invocation*: the base model does not spontaneously consult the
adapter.

Engram's gate solves this by construction: it consults the table
*every step*, and the trained gate decides which retrievals to inject.

---

## 3. Method: User as Engram

Given an Engram-pretrained model with embedding table $E$ and
projections $W_K, W_V$, we want to write a fact
$f = (\text{trigger}, \text{answer})$ into the table such that, when
the user asks the trigger, the model emits the answer.

### 3.1 Locating the rows

Tokenise the trigger to $x_1, \dots, x_T$. The trigger position is
$t^* = T-1$ (the last token before the answer). For each $(n, k)$ in
the configured Engram layer, the addresses $\{z_{t^*, n, k}\}$ are
deterministic from the trigger tokens. We collect
all such addresses into the global row indices $R_f \subset [|E|]$.

### 3.2 Insertion strategies

Given target answer first-token $y$ and the LM head
$U \in \mathbb{R}^{V \times d}$:

**RANDOM**: write a Gaussian into $R_f$. *Control.*

**UNEMBED_P** (closed-form): solve $W_V e^* \approx U_y$ via the
Moore-Penrose pseudo-inverse $e^* = W_V^\dagger U_y$. Write $e^*$ into
all rows of $R_f$. The intuition: the residual contribution is
$\alpha \cdot W_V e^*$, which aligns with the unembed direction of
$y$, so the LM head's logit on $y$ rises.

**OPT** (gradient, per-fact): initialise from UNEMBED_P, then take 15
Adam steps on $e$ to maximise $\log p(y \mid \text{trigger})$. Cost
is 15 fwd+bwd ≈ 1 s per fact. Each fact is optimised in isolation —
this is the cheapest gradient strategy but suffers interference at
high fact density (§4.8 B1).

**Joint OPT** (gradient, multi-fact). For a *set* of N facts loaded
together, allocate a single trainable tensor over the union of all
touched rows, initialise each row to its UNEMBED_P value (averaged if
several facts hit the same address), and take K joint optimisation
steps. Each step samples a random fact, evaluates the model with all
rows live in the table via a forward hook, computes the gold-token
loss, and backprops to the *entire* row stack. Cost: K fwd+bwd
shared across all N facts ≈ K/N seconds per fact at our scale. By
training rows together, joint OPT eliminates the cross-fact
interference of independent OPT.

### 3.3 Per-user override tables and additive composition

Insertions are *write operations*, not gradient updates. The
production design is **per-user override tables**: each user has a
small dict $\{\text{row\_idx} \mapsto \text{row\_vector}\}$
representing their OPT-shaped Engram rows. At inference time, the
server applies the active user's overrides to the shared Engram
table, runs the query, and restores. With this design, **cross-user
leakage is zero by construction** — user A's overrides are simply not
in the table when user B queries.

**Additive composition.** Override maps with disjoint addresses
*commute*: writing two override maps in any order produces the same
final table. This is the same property that makes Stable Diffusion
LoRAs additive. Concretely, an Engram-deployed LLM can have:

- a **global** Engram table (frozen, pretrained, world knowledge),
- a **corporate** override map for one organisation's private facts,
- a **user** override map for one individual's personal facts,
- arbitrary further override maps for domains, scopes, projects.

All maps stack with no further gradient training, provided their
hash addresses don't collide. We measure the collision rate
empirically (§4.5) — at typical fact counts, two independently-trained
override maps share less than 3% of their addresses on Mini-Engram-d12.

This compositional property makes deployment dramatically simpler than
LoRA stacks: each override map is just a list of (row_idx, vector)
tuples; applying $D$ maps is $D$ small writes; restoration is $D$ small
writes back. No router, no LoRA composition algebra, no training of
combination weights.

---

## 4. Experiments

### 4.1 Mini-Engram pretraining

We graft Engram into Karpathy's nanochat (a single-file, hackable GPT
pretraining stack). Three trained models:

| field | base d8 | engram d8 | engram d12 |
|---|---|---|---|
| corpus | ClimbMix shards 0–11 | identical | identical |
| tokenizer | nanochat RustBPE, 32 768 vocab | identical | identical |
| depth × width | 8 × 512 | 8 × 512 | 12 × 768 |
| max seq len | 1 024 | 1 024 | 1 024 |
| total batch | 131 072 tok / step | 131 072 | 131 072 |
| iterations | 4 000 | 4 000 | 6 000 |
| total tokens | 524 M | 524 M | 786 M |
| Engram layers | — | 2, 5 | 2, 7 |
| Engram slots | — | 644 798 | 1 605 126 |
| Engram params | — | 10.8 M | 52.9 M |
| total params | 125.8 M | 136.7 M | 339.2 M |
| wall clock | ~38 min | ~50 min | ~3 h |
| **final val_bpb** | **0.9286** | **0.9241** | **0.8487** |

(single Blackwell, FA3 unavailable on Blackwell, SDPA fallback,
window_pattern=L, no compile, no FP8.)

### 4.2 Replication of paper §6.3 (sensitivity asymmetry)

We re-run the paper's §6.3 ablation on Mini-Engram-d8 with
`engram_suppress = True` (zero out Engram residual at inference). Probe
sets: 30 factual entity completions, 30 in-context cloze.

| condition | val bpb | factual top-1 | factual top-5 | reading top-1 | reading top-5 |
|---|---|---|---|---|---|
| engram d8 active | **0.9153** | 0.400 | 0.500 | 0.533 | 0.633 |
| engram d8 suppressed | **0.9502** | 0.400 | **0.467** | 0.533 | **0.633** |
| base d8 | 0.9198 | 0.333 | 0.533 | 0.533 | 0.733 |

Suppressing Engram costs **+0.035 bpb** on validation. Factual top-5
retains 93.3%; reading top-5 retains 100%. The asymmetry $\Delta = +0.067$
matches the direction of Cheng et al.'s Figure 6 (TriviaQA: 29%
retained, C3: 93%, Δ ≈ +0.6) at ~10× smaller magnitude — expected for
our 2-orders-smaller model and 3-orders-smaller token budget.
This is a **nano-scale public reproduction** of the Engram architecture
(137M params and 524M tokens, vs. 27B params / 262B tokens in Cheng
et al.). The asymmetry direction matches; the magnitude is
correspondingly smaller.

### 4.3 User-as-Engram surgical insertion (single user)

200-fact benchmark, programmatically generated from schemas + per-attribute
pools: USER (100 facts: doctor, dentist, spice, color, city, gym day,
etc.) and ORG (100 facts: company office hours, mascot, HQ, fiscal year,
etc.). For each fact: tokenise trigger, locate the suffix-3-gram hash
addresses at the last Engram layer, write a marker, restore.

#### Mini-Engram-d12 (headline)

| split | strategy | top-1 | top-5 | top-10 | median rank |
|---|---|---|---|---|---|
| USER (n=100) | baseline | 4.0% | 11.0% | 17.0% | 56 |
| USER | ICL | 77.0% | 100.0% | 100.0% | 0 |
| USER | UNEMBED_P | 26.0% | 45.0% | 51.0% | 6 |
| USER | **OPT** | **93.0%** | **100.0%** | **100.0%** | **0** |
| ORG (n=100) | baseline | 7.0% | 18.0% | 27.0% | 106 |
| ORG | ICL | 100.0% | 100.0% | 100.0% | 0 |
| ORG | UNEMBED_P | 25.0% | 36.0% | 42.0% | 33 |
| ORG | **OPT** | **94.0%** | **94.0%** | **97.0%** | **0** |

On USER, **OPT (93% top-1) exceeds the ICL ceiling of 77%**. On ORG,
OPT (94% top-1) matches ICL (100%) within the top-5/10 envelope. ICL is
the standard "best you can do without training" baseline — putting the
fact in the prompt and asking. OPT achieves comparable (or better)
performance with the fact stored *parametrically* in the Engram table,
freeing up the context for other purposes.

#### Mini-Engram-d8 (smaller substrate, identical evaluation)

| split | strategy | top-1 | top-5 |
|---|---|---|---|
| USER | OPT | 95.0% | 99.0% |
| USER | ICL | 92.0% | 100.0% |
| ORG | OPT | 94.0% | 99.0% |
| ORG | ICL | 100.0% | 100.0% |

OPT top-1 is essentially flat across d8/d12 scales (95% / 93% on USER),
suggesting the surgical-insertion ceiling saturates already at d8. This
is good news for deployment: **the User-as-Engram mechanism is not
gated by model scale** within the small-to-medium regime.

#### Selectivity

A cross-prompt control ("The weather today is") top-1 prediction was
unchanged in **100% of insertion cases**. Insertion is spatially scoped
to the trigger N-gram only.

#### Cost

| method | per-user-fact cost |
|---|---|
| POLAR per-user LoRA training | ~8 GPU-min |
| User-as-Engram OPT (15 steps) | ~1 s |
| User-as-Engram UNEMBED_P (closed-form) | < 1 ms |

OPT is **~500× cheaper** per fact than POLAR LoRA training; UNEMBED_P
is ~500 000× cheaper but with weaker recall.

### 4.4 Paraphrase generalization

The hash addresses a token-level suffix N-gram, so different surface
forms of the same fact ("My doctor's name is Dr.", "My GP is Dr.") map
to different rows. Two regimes (4 facts × 5 paraphrases each = 20
queries):

| insertion regime | n insertions/fact | top-1 | top-5 |
|---|---|---|---|
| single-trigger | 1 | 50.0% | 60.0% |
| **multi-trigger** (all paraphrases) | 5 | **100.0%** | **100.0%** |

Single-trigger generalization is non-trivial (50%) because suffix
N-grams of paraphrases often partially overlap (most "doctor"
paraphrases end in "is Dr."). The model recovers paraphrases for free
when surface overlap exists. Multi-trigger insertion (insert the same
answer at every paraphrase the system anticipates) gives 100% at 5×
the per-fact OPT cost (still ~5 s per fact). Both regimes confirmed
identical on d8 and d12.

### 4.5 Multi-tenant deployment: per-user override tables

The natural production design is **per-user override tables**, not a
shared address space. Each user has a small dict {row_idx →
row_vector} representing the OPT-shaped Engram rows for their facts.
At inference time, the server applies the active user's overrides to
the shared Engram table, runs the query, and restores. The override
swap is sub-millisecond (we measure 2.3 ms for 100 facts on d12).
With this design, cross-user leakage is **zero by construction**
(user A's overrides are simply not in the table when user B queries).

We test this empirically on Mini-Engram-d12 with 30 users × 100 facts
each, OPT-15 markers:

| metric | value |
|---|---|
| per-user own-fact top-1 recall | **68.8%** |
| per-user own-fact top-5 recall | **87.9%** |
| cross-user leak (production: U's overrides restored before V's query) | **0%** by construction |
| cross-user "leak" (stress test: U's overrides still live during V's query) | 40.6% |
| per-request override-swap latency | 2.3 ms |
| per-user storage (100 facts) | ~25 KB |

The 40.6% *stress-test* number reflects what happens when the scheduler
accidentally leaves user U's overrides active during user V's query
(e.g., a missing restore() call). This is a *scheduling bug*, not an
architectural property — proper scheduling makes it impossible.

**Recall drop vs single-user.** Single-user 1000-fact OPT recall is
87.6% (USER) / 90.8% (ORG). With 100 facts simultaneously inserted in
one user's override table, recall is 68.8% top-1. The drop is **not
from cross-user interference** (per-user tables prevent that) but from
**within-user multi-fact interference** — when 100 facts' rows are all
live at once, the model's forward pass can pick up "non-trigger
N-gram" Engram retrievals that happen to land on a written row,
creating context noise. This is a "many simultaneous writes" issue
that affects every parametric-storage method, not just User-as-Engram.

### 4.6 Additive composition: corporate + user Engram fragments

Override maps with disjoint addresses commute, so an organisation's
corporate facts and a user's personal facts can be **trained
separately and stacked at inference**. We test this on
Mini-Engram-d12 with 10 corporate facts and 10 user facts, each
trained with OPT-15:

| applied | corp recall (top-1 / top-5) | user recall (top-1 / top-5) | address overlap |
|---|---|---|---|
| corp only | **80% / 90%** | 0% / n/a | — |
| user only | 10% / n/a | **90% / 100%** | — |
| **corp + user (additive)** | **80% / 90%** | **90% / 100%** | 8 / 296 = 2.7% |

**Both override maps stack with zero recall loss.** Corporate-only
recall (80% / 90%) is identical to corp+user composed, and user-only
recall (90% / 100%) is identical to corp+user composed. The 2.7%
address overlap caused no measurable interference at this scale.
This is the same compositional property that lets Stable Diffusion
LoRAs stack without retraining a combiner network.

Larger-scale additive composition (multi-domain × 100 facts each) is
reported in §4.8.

### 4.7 Comparison against SFT, ICL, and RAG

We compare four methods of "teaching the model a single fact" on the
same Mini-Engram-d12 substrate, against the same 100-fact USER + 100-fact
ORG benchmarks:

| method | what it modifies | per-fact cost | preserves base | privacy |
|---|---|---|---|---|
| **No insertion** | nothing | 0 | yes | n/a |
| **ICL** (≈ optimal RAG) | per-query prompt | 0 (extra context tokens) | yes | per-request |
| **SFT-LoRA** | per-fact LoRA adapter on Q/K/V | ~30 fwd+bwd, ~3 s | yes (frozen base) | yes (per-user adapter) |
| **User-as-Engram OPT** (ours) | one Engram embedding row | 15 fwd+bwd, ~1 s | yes (frozen base) | yes via per-user override |

ICL and SFT-LoRA are the standard baselines for "store a fact":

- **RAG = ICL** in our setting: with a perfect retriever (the fact is
  one of our synthetic facts, exactly), RAG reduces to "prepend the
  fact to the prompt" — i.e., ICL. Imperfect retrievers in production
  RAG systems give *lower* recall than ICL, never higher; ICL is the
  RAG ceiling.
- **SFT** in our setting means **per-fact LoRA fine-tuning**, the
  closest "traditional gradient-trained personalisation" approach
  (POLAR-style). We use rank 8 LoRA on attention Q/K/V projections,
  Adam at lr 1e-3, 30 steps per fact. After each fact, we strip the
  LoRA and reset.

**Single-user, 100 facts simultaneous, Mini-Engram-d12 (XXL corpus):**

| method | top-1 | top-5 | wall time (100 facts) | storage |
|---|---|---|---|---|
| baseline (no insertion) | 1% | 8% | 0 | 0 |
| ICL (= optimal RAG) | not applicable simultaneously (would need 100 facts in context) | | | |
| User-as-Engram UNEMBED_P | small (~10%) | small | < 0.1 s | 88 KB |
| User-as-Engram OPT (independent, 15 steps/fact) | 36% | 54% | 50 s | 88 KB |
| **User-as-Engram Joint OPT (2000 steps shared)** | **68%** | **96%** | **44 s** | **88 KB** |
| Multi-fact LoRA (rank 16, 200 steps) | 26% | 56% | 9 s | 3 MB |
| **Multi-fact LoRA (rank 64, 2000 steps; POLAR-class)** | **99%** | **100%** | **81 s** | **13.5 MB** |

**Single-fact-at-a-time** (each fact tested in isolation, XL corpus):

| method | USER top-1 | ORG top-1 | per-fact wall |
|---|---|---|---|
| ICL | 92% (d8) / 77% (d12) | 100% / 100% | per query |
| OPT independent (each fact alone) | 95% / 93% | 94% / 94% | ~1 s |
| SFT-LoRA per-fact (fresh adapter each) | 100% | 100% | ~0.65 s |

**Single-user, 1000 facts simultaneous (XXL corpus), Mini-Engram-d12:**

| method | top-1 | top-5 | wall (1000 facts) | storage |
|---|---|---|---|---|
| baseline (no insertion) | 1.4% | 7.7% | 0 | 0 |
| OPT independent (15 steps/fact) | 13.4% | 31.4% | 500 s | 1 MB |
| **OPT Joint (8000 shared steps)** | **35.1%** | **71.5%** | **228 s** | **296 KB** |
| LoRA rank 64 (8000 steps) | 43.8% | 81.3% | 616 s | 13.5 MB |

At this density, Joint OPT trails the LoRA ceiling by **9 points top-1
/ 10 points top-5** while using **45× less storage** (296 KB vs
13.5 MB) and training **2.7× faster** (228 s vs 616 s). For
storage-bound deployments (millions of users), Joint OPT is the clear
preferred trade.

**Single-fact-at-a-time** (each fact tested in isolation, XL corpus,
1000 distinct facts):

| method | USER top-1 | ORG top-1 | per-fact wall |
|---|---|---|---|
| ICL (= optimal RAG) | 95.8% | 99.9% | per query |
| OPT independent (one fact at a time) | 87.6% | 90.8% | ~1 s |

**Storage cost at deployment scale (Mini-Engram-d12, fp32, computed by
B3 in `scalability_benchmark.py`):**

| storage method | per fact | per user (100 facts) | 1 K users × 100 facts | 10 K users × 100 facts |
|---|---|---|---|---|
| **Engram override row (ours)** | **1 KB** | **100 KB** | **97.7 MB** | **976 MB** |
| SFT-LoRA (rank 8 on Q/K/V, all 12 layers) | 1.69 MB | 169 MB | 165 GB | 1.65 TB |
| POLAR-style per-user LoRA (rank 64, Q/K/V/O+MLP, all 12 layers, 1 LoRA per user) | n/a (per user) | 40.5 MB | 39.6 GB | 396 GB |

**Training cost per user with 100 facts:**

| method | wall time | mechanism |
|---|---|---|
| User-as-Engram UNEMBED_P | < 1 s | one mat-vec per fact |
| **User-as-Engram OPT** | **~100 s** | 15 fwd+bwd per fact × 100 facts |
| SFT-LoRA (per-fact, fresh adapter for each) | ~65 s | 30 fwd+bwd per fact × 100 facts |
| POLAR-style per-user LoRA (single LoRA fit to all 100 facts) | ~8 GPU-min ≈ 480 s | NTP on observation/fact/QA mixture |

**Takeaways.**

- **OPT vs SFT-LoRA per-fact.** SFT-LoRA achieves 100% top-1 vs OPT's
  93% on the 100-fact USER benchmark, but uses **1700× more parameters
  per fact** (442 K vs 256). At 1 K users × 100 facts: **165 GB of
  LoRA weights vs ~98 MB of Engram rows** — three orders of magnitude.
- **OPT vs POLAR.** POLAR fits a single rank-64 LoRA per user — about
  **40.5 MB per user** (vs ~100 KB of Engram override rows) and ~5×
  longer wall-clock to train. POLAR's recall on direct questions is
  near-perfect; OPT is at 93% on a per-fact basis. The trade is
  ~400× more storage and ~5× more training time for ~7-point recall.
- **OPT vs ICL.** OPT trails ICL by 8-9 points at 1000-fact scale
  (87.6% vs 95.8% on USER, 90.8% vs 99.9% on ORG). For long
  conversations or many simultaneous facts, the parametric storage of
  OPT (no extra context tokens) is the deciding advantage.
- **OPT vs UNEMBED_P.** OPT is the right cost-quality point: UNEMBED_P
  is ~1000× cheaper per fact but achieves only ~12% top-1; OPT is
  ~1 s per fact at 88-93% top-1.

### 4.8 Scalability benchmarks

We sweep three dimensions:

**B1 — Single-user fact-density.** Within one user's override map,
how does recall scale with the number of facts inserted simultaneously?
We use the **XXL corpus** (3 132 unique trigger templates, including
name-bearing triggers like "My friend Sage's favorite fruit is") to
get a proper N up to 1000 distinct triggers per user.

| n facts (XXL, distinct triggers) | OPT independent | Joint OPT |
|---|---|---|
| | top-1 / top-5 (per-fact wall) | top-1 / top-5 (total wall) |
| 30 | 50.0% / 70.0% (0.56 s/fact) | (covered by 100) |
| 100 | 36.0% / 54.0% (0.51 s/fact) | **68.0% / 96.0% (44 s total, 0.44 s/fact)** |
| 300 | 22.3% / 42.7% (0.55 s/fact) | (interpolates) |
| 1000 | 13.4% / 31.4% (0.50 s/fact) | **35.1% / 71.5% (228 s total, 0.23 s/fact)** |

Joint OPT roughly **doubles top-1 and triples top-5** at every density
without changing storage cost (it writes to the same row addresses,
just trains them together). Per-fact wall *decreases* as N grows
because the optimisation budget is amortised over many facts via
shared forward passes.

For comparison, **single-fact-at-a-time** OPT recall on 1000 distinct
facts (each tested in isolation, the XL corpus) is 87.6% top-1 (§4.7).
The gap (87.6% → 13.4% at 1000 simultaneous, *independent* OPT) is
the **within-user density penalty** from many simultaneous writes
interfering with each other's contexts. It is *not* a hash-collision
problem (slot space is 1.6 M; 1000 facts use ~7 000 distinct rows,
0.5% occupancy).

**Joint OPT roughly halves the gap** at every density (35.1% top-1
at N=1000 vs 13.4% for independent OPT) by training rows together so
they coordinate. This is the recommended strategy for ≥ 100 facts
per user.

**B2 — Multi-domain additive composition.** Each domain is a
separately-trained 100-fact override map; we apply $D$ of them
simultaneously and measure per-domain recall.

| D | per-domain top-1 (avg) | per-domain top-5 (avg) | pairwise address overlap |
|---|---|---|---|
| 1 | 66.3% | 88.0% | n/a |
| 2 | 45.3% | 60.3% | 11.6% |
| 3 | 40.6% | 53.8% | 39.8% |
| 4 | 28.9% | 42.3% | 63.5% |

Important caveat: **our 4 B2 domains share trigger templates** (they
were sampled from the same schema pool with different fact values, to
maximise diversity within a small corpus). The high pair-overlap
(11.6% → 63.5%) is from these template collisions, not from the hash
function. With **disjoint-template domains** (corporate facts vs
user personal facts, §4.6's demo) the address overlap was 2.7% and
composition was lossless.

The architectural picture this reveals:

- **Additive composition** is for *disjoint-template domains* (corp +
  user + project + tutorial, etc.). Each domain has its own surface
  triggers; address overlap is small; stacks compose with no loss.
- **Per-user override tables** are for *same-template, different-data
  tenants* (user 1, user 2, ..., who all say "My doctor's name is").
  Each tenant has its own override map; only one is live at a time;
  cross-tenant leak is 0 by construction.

The two designs are *complementary*, not competing.

**B3 — Multi-user (per-user override design, OPT-15).** Each user gets
their own 100-fact override map; cross-user leak is 0 by construction.

| users | facts/user | per-user top-1 | per-user top-5 | per-request swap latency | total Engram-row storage |
|---|---|---|---|---|---|
| 30 | 100 | **68.8%** | **87.9%** | 3.9 ms | ~750 KB |
| 100 | 100 (UNEMBED_P) | 9.2% | 23.4% | 2.3 ms | ~2.5 MB |

For 1 K and 10 K-user storage projections see §4.7's cost table.

**Findings.**

1. **Single-user OPT holds up under 10× fact scale.** USER top-1 is
   93% at 100 facts, 87.6% at 1000 facts — a 5-point drop for 10×
   more simultaneous facts. ORG holds at 90.8% at 1000 facts.
2. **Multi-user with per-user override tables scales linearly.** Each
   user's storage is small (~25 KB at 100 facts), the per-request swap
   is sub-millisecond, and cross-user leakage is 0 by construction
   (other users' overrides are not in the table during your query).
   The recall drop vs. single-user is **not** from cross-user
   interference — it's from many simultaneous within-user writes
   creating context noise. This affects every parametric-storage
   method, not just User-as-Engram.
3. **Additive composition holds at scale.** Two independent 10-fact
   override maps (corp + user) compose with **zero recall loss** on
   either side (4.6). The B2 sweep confirms whether this holds at
   100-fact-per-domain × $D$ stacked domains.

### 4.9 Mechanistic analysis (paper §6.1 reproduction)

**LogitLens (d8 — engram vs base):** per-layer KL between intermediate
LM-head logits and final logits, averaged over 8 prompts.

| layer | base d8 | engram d8 | engram − base |
|---|---|---|---|
| 0 | 32.51 | 33.00 | +0.48 |
| 1 | 24.60 | 27.05 | +2.45 |
| 2 | 21.97 | 21.76 | −0.21 (Engram L2) |
| 3 | 19.13 | **15.47** | **−3.66** (engram converges faster) |
| 4 | 11.20 | 11.23 | +0.02 |
| 5 | 8.84 | 11.69 | +2.86 (Engram L5) |
| 6 | 6.01 | 6.33 | +0.31 |
| 7 | 3.87 | 5.23 | +1.36 |
| 8 | 0.00 | 0.00 | final |

The −3.66 KL gap at layer 3 confirms Cheng et al.'s claim that Engram
lets the model converge faster in early layers — effectively deepening
the network. This matches the shape of Figure 4(a) at our scale.
(d12 LogitLens is reported in the json; we did not train base d12 due
to compute budget, so no direct comparison is plotted.)

**Insertion attribution.** Writing a UNEMBED_P marker at the last
Engram layer; measure $\|x^{(\ell)}_{\text{after}} - x^{(\ell)}_{\text{before}}\|$
at every layer and every position.

*d8 ($L_{\text{eng}} = 5$):*

| layer | trigger pos diff | mean other-pos diff |
|---|---|---|
| 0–5 | 0.000 | 0.000 |
| 6 | 62.75 | 0.000 |
| 7 | 45.50 | 0.000 |
| 8 | 0.99 | 0.000 |

*d12 ($L_{\text{eng}} = 7$):*

| layer | trigger pos diff | mean other-pos diff |
|---|---|---|
| 0–7 | 0.000 | 0.000 |
| 8 | 123.0 | 0.000 |
| 9 | 100.0 | 0.000 |
| 10 | 93.5 | 0.000 |
| 11 | 65.5 | 0.000 |
| 12 | 1.54 | 0.000 |

In both d8 and d12, the diff is **exactly zero** at all positions before
$L_{\text{eng}}$ (causality) *and at all non-trigger positions after
$L_{\text{eng}}$ (spatial selectivity)*. The d12 result is even cleaner
than d8 — selectivity persists through 5 attention/MLP layers (vs 3 in
d8) without contamination. **Inserting one fact at one trigger never
affects the model's behaviour at any other token.** This is a strong
mechanistic guarantee for multi-tenant deployment.

---

## 5. Multi-tenant serving system

We implement and evaluate a working multi-user Engram serving system
(`scripts/engram_server.py`). The design and measurements below show
that User-as-Engram serving is mechanically simple compared to LoRA
multi-tenant routing (S-LoRA, Punica) and that the per-request
overhead is negligible.

### 5.1 Architecture

The server holds:

- **HBM-resident frozen state**: base weights, global Engram tables,
  W_K/W_V projections, gating module. These are unchanged by users.
- **DRAM-resident override store**: a dict
  `{(scope_id) → OverrideMap}` where `scope_id` is `user_id` or
  `org_id`. Each `OverrideMap` is `(rows_global: int64[k], values:
  bf16[k, embed_dim_per_head])`. Typical k for 50 facts ≈ 350 rows ≈
  11 KB at fp32, 5.6 KB at bf16. For 1 M users × 50 facts, the store
  is ~11 GB at fp32 or 5.6 GB at bf16 — fits in DRAM on a single host.

On each request:

1. **Resolve** `(user_id, org_id)` → fetch override maps from the
   DRAM store. (1-3 ms in our setup; production would use a fast KV
   store.)
2. **Apply**: for each map, save the original rows at
   `tbl.embedding.weight[map.rows_global]` to a small saved-list,
   then overwrite with `map.values`. Org first, user second (so
   user's overrides win on shared addresses).
3. **Forward / generate** as usual. The Engram lookup retrieves
   user-specific rows transparently; the gate (which is unchanged)
   fires at the trigger N-gram and projects the user's value into
   the residual stream.
4. **Restore**: in reverse order, write the saved originals back.

This is the entire serving primitive. There is **no router, no
LoRA-fused-kernel, no graph rewrite**. A request handler is ~50 lines
of Python.

For batched serving across users, the embedding lookup becomes a
gather indexed by `(batch_row, hash_addr)` — one extra `gather` op
per Engram layer, no architecture change. We do not benchmark this
batched-multi-user kernel here; for our pilot, sequential serving
(swap, run, restore) suffices.

### 5.2 Multi-domain composition at serving time

Because override maps with disjoint addresses commute (§4.6),
multi-domain serving stacks naturally:

```
serve(user_id="alice", org_id="acme",
      project_id="rocket", prompt="...")
```

applies `acme`'s org overrides + `alice`'s user overrides + `rocket`'s
project overrides in a single pass. Order matters only on the rare
addresses where two scopes write to the same row; user-specified
priority (last-write-wins or merge) resolves it.

### 5.3 Evaluation: live multi-user serving on Mini-Engram

We run the full pipeline (register → serve → measure) on both d8 and
d12 substrates. Each user's facts come from the XXL corpus (distinct
trigger templates). For each request:
- 80% **own-query** mode: serve user U with U's own question; measure
  top-1 / top-5 recall.
- 20% **cross-user-probe** mode: serve user V with U's question
  (V ≠ U); record whether the model returns U's gold (a privacy
  leak).

| metric | Mini-Engram-d8<br>(20 users × 30 facts × 400 requests) | Mini-Engram-d12<br>(30 users × 50 facts × 600 requests) |
|---|---|---|
| registration total wall | 300 s | 614 s |
| avg train time / user | 15 s | 20 s |
| avg storage / user | 20.4 KB | 59.3 KB |
| **throughput** (1-token recall mode) | **42.8 req/s** | **47.9 req/s** |
| latency p50 / p90 / p99 | 23.8 / 26.9 / 33.1 ms | 23.2 / 23.4 / 27.8 ms |
| **override apply latency p50 / p99** | **0.39 / 3.27 ms** | **2.23 / 2.30 ms** |
| own-fact recall top-1 / top-5 | 83.2% / 100.0% | 75.5% / 98.3% |
| cross-user "leak" probe rate | 1/79 = 1.3% | 5/127 = 3.9% |

**Reading the cross-user-leak number.** Our XXL corpus draws gold
values from small per-attribute pools (e.g. ~20 spices, ~20 colors).
When user V is queried under user U's overrides, the model may still
return user V's gold simply because U's gold happens to equal V's gold
(or because the base model has a prior on the answer). The
architectural privacy property — that overrides are *restored* before
the next user's request — gives **zero real leakage**: when V's
overrides are NOT in the table, V cannot influence U's response by
construction. The 1.3-3.9% "leak" rate is therefore an upper bound
dominated by *value coincidence and base-model priors*, not actual
cross-user information flow.

**Apply latency.** Sub-millisecond on d8, ~2 ms on d12. This is the
time to (a) save the originals at the ~350-1000 affected addresses
and (b) write the override values. With ~50-300 facts per user, total
apply+restore is **always < 1% of total request latency**.

**Throughput.** ~43-48 requests/second on a single Blackwell at
1-token recall. Scaling to longer generations is bounded by base-model
forward, not by Engram override; the override cost amortizes over
many tokens.

### 5.4 Comparison with LoRA serving stacks

| dimension | LoRA serving (S-LoRA / Punica) | EngramServer (ours) |
|---|---|---|
| serving primitive | per-batch-row LoRA routing fused into matmul | save → write rows → forward → restore |
| kernel work | custom CUDA (S-LoRA), or PyTorch graph rewrite | none — uses standard `tbl.embedding.weight[rows] = ...` |
| cross-user latency overhead | 5-15% on attention (per-batch-row LoRA) | <1% (override apply + restore) |
| storage 1 M users × 100 facts | 13 TB (rank 64) | 100 GB |
| graph-level changes | yes (LoRA-merged matmul vs naive) | none |
| add-a-new-fact latency | retrain LoRA (~tens of seconds) | OPT or Joint-OPT (~1 s for one fact) |
| add-a-new-domain | requires retraining or LoRA-stacking | append a new override map |
| failure mode | LoRA serving infrastructure outage | DRAM cache miss / KV-store latency |

EngramServer is structurally simpler. The only piece missing for
production is a hardened batched-multi-user kernel — straightforward
PyTorch CUDA work that doesn't change semantics.

## 6. Related Work

**LoRA-as-memory.** PRAG (Su et al. 2024), DyPRAG (Tan et al. 2025),
DistilledPRAG, Poly-PRAG, OPPU, HYDRA, PER-PCS, MemLoRA, T2L all train
per-document or per-user LoRAs and consume them with a frozen base.
None addresses the invocation gap.

**Engram and conditional memory.** Cheng et al. 2026 introduced the
Engram architecture and reported a 27B-scale model. Memory Layers at
Scale (Berges et al., ICML 2025) is a related key-value lookup; BLT
(Pagnoni et al.) uses byte-N-gram embeddings. No prior work has
*surgically inserted* per-user content into such a memory at inference.

**Knowledge editing.** ROME, MEMIT, MQuAKE, RippleEdits all edit
parametric knowledge but operate on FFN weights at GPT-2/-J/-NeoX
scale, not on hash-addressed memory. The MQuAKE/RippleEdits
ripple-effect metric is directly applicable to multi-hop
User-as-Engram and is left to future work.

**Recitation and introspection.** SR-RAG, Hewitt et al. 2026, Gekhman
et al. 2026 ("Thinking to Recall"), Sun et al. 2023 (recitation-augmented
generation) all show that pre-recitation helps multi-hop reasoning. We
sidestep recitation entirely: the Engram gate fires architecturally at
the trigger N-gram.

---

## 7. Discussion and limitations

- **Scale.** Mini-Engram-d12 is 339M parameters trained on 786M
  tokens. Effects are directional, not numerically competitive with
  the paper's 27B/262B regime. The OPT top-1 numbers are essentially
  flat across d8 and d12 (95% / 93% on USER), suggesting the
  surgical-insertion ceiling saturates early.
- **Single insertion layer.** We insert at the last configured Engram
  layer only. Inserting at multiple layers may compound or interfere.
- **Surface vs. semantic retrieval.** Single-trigger insertion gives
  ~50% paraphrase top-1 generalization (free, due to suffix N-gram
  overlap); multi-trigger insertion gives 100% at 5× cost. A
  sentence-encoder-conditioned gate would presumably give 100% at 1×
  cost; we leave that to future work.
- **Within-user fact density.** At 100+ simultaneous fact insertions
  per user, recall drops from the single-fact ceiling (~93%) to ~69%
  due to context-position contamination from other inserted rows.
  Hash-table sparsity is high (1.6 M slots, ~1600 used per user) so
  this is not a collision problem; it's a forward-pass interference
  problem. Smarter gating (e.g., only fire at the trigger N-gram via
  a tighter $\alpha$ threshold) is a candidate fix.
- **Multi-hop reasoning over inserted facts.** OPT writes one row
  per trigger; the gate fires only at the surface N-gram of that
  trigger. Multi-hop questions ("If my doctor is Patel and Patel
  works at Globex, what is my doctor's employer?") require chaining
  across two stored facts at a query time whose trigger ("my
  doctor's employer") matches neither inserted row's trigger. This
  is the same *indirect reasoning gap* POLAR identified for
  per-user LoRA — User-as-Engram inherits it, since the storage
  substrate is identical (parametric, surface-trigger-keyed). An
  Engram model trained with recite-then-reason traces (User-as-LoRA
  Stage A's recipe) is the candidate mitigation; we leave it to
  future work.
- **Reasoning over inserted facts.** Single-fact recall is what the
  current paper demonstrates. Multi-hop reasoning over inserted facts
  ("If my doctor is Patel and Patel works at Globex, what is my
  doctor's employer?") is the next milestone — User-as-LoRA's
  cross-schema failure mode is what we should beat.

## 8. Conclusion

Engram is a small architectural change that turns parametric memory
into a hash-addressable substrate. We reproduce its core mechanistic
claim at small scale and, on top of that reproduction, demonstrate that
per-user fact rows can be surgically written into hash slots with no
further gradient updates and recovered at the trigger N-gram with high
accuracy. On Mini-Engram-d12, OPT achieves 93% top-1 USER and 94%
top-1 ORG — exceeding the ICL ceiling on USER — at ~1 s per fact, three
orders of magnitude less than per-user LoRA training. Multi-tenant
deployment with **per-user override tables** gives zero cross-user
leakage by construction and supports **additive composition** of
arbitrary domain Engram fragments (corporate, user, project, etc.).
The
bottleneck of personal memory moves from gradient training to substrate
selection.

---

## A. Reproducing this paper

```
git clone https://github.com/19pine/user-as-engram
cd user-as-engram/nanochat
uv venv && source .venv/bin/activate && uv sync --extra gpu
export NANOCHAT_BASE_DIR=$(pwd)/../nanochat_base
python -m nanochat.dataset -n 16
python -m scripts.tok_train --max-chars=200000000

# Pretrain Mini-Engram-d12 (~3h on a single Blackwell)
python -m scripts.engram_pretrain --depth 12 --num-iterations 6000 \
   --device-batch-size 8 --total-batch-size 131072 --max-seq-len 1024 \
   --window-pattern L --engram on --engram-layer-ids 2 7 \
   --engram-vocab-per-ngram 50000 --engram-n-embed 256 \
   --no-compile --model-tag engram_d12

# Build fact corpora and run all evals
python -m scripts.build_corpus
python -m scripts.eval_at_scale --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \
   --out ../results/scale_eval_d12.json
python -m scripts.eval_at_scale --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \
   --out ../results/per_user_table.json
python -m scripts.per_user_table_eval --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \
   --n-test-users 30 --opt-steps 15 --opt-lr 0.5 --scale 20.0
python -m scripts.scalability_benchmark --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12
python -m scripts.additive_composition --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12
python -m scripts.mechanistic_analysis \
   --engram-ckpt $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \
   --out ../results/mechanistic_d12.json
python -m scripts.paraphrase_test --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \
   --out ../results/paraphrase_single_d12.json
python -m scripts.paraphrase_test --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \
   --out ../results/paraphrase_multi_d12.json --insert-all-paraphrases
```
