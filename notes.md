# User-as-Engram — Tier 1 pilot notes

Day 1, 2026-05-04. Plan in `roadmap.md`.

---

## Substrate situation

- DeepSeek's `engram_demo_v1.py` (mocked Attention/MoE/mHC, randomly initialised
  Engram module) is the only public release. **No Engram-pretrained
  weights exist anywhere** — DeepSeek hasn't published any, and no third-party
  reproduction has surfaced as of May 2026.
- Adjacent public-weights options surveyed: `facebookresearch/memory`
  (training code only, no checkpoints), `facebook/blt-1b` and
  `facebook/blt-7b` (released, byte-N-gram-style architecture but byte-level
  rather than subword).
- Tier 1 work below uses the demo. It answers structural questions that
  do not depend on having trained weights.

## Environment

- RTX PRO 6000 Blackwell, 96 GB HBM, ~80 GB free at session start.
- `polar-env` already has torch 2.10 + cu128, transformers 4.57.6, sympy,
  tokenizers — no new install needed.
- Demo's default `BackBoneConfig`: hidden_size 1024, hc_mult 4,
  vocab_size 129280, num_layers 30. `EngramConfig`: layer_ids [1, 15],
  max_ngram_size 3, n_head_per_ngram 8, engram_vocab_size [646400, 646400].
- Bug fixed in `refs/engram_demo_v1.py` line 363 (`hash_input_ids` device
  was CPU; coerced to embedding's device). Two-character change, recorded
  for Tier 2.

---

## T1.1 — demo smoke test

`engram_demo_v1.py` runs end-to-end on the local GPU. Output shape sanity:
input `[1, 14]` → logits `[1, 14, 129280]`. **PASS.**

## T1.2 — hash mechanism on user facts (`src/probe_hash.py`)

30-fact synthetic user (matches user-as-lora schema), DeepSeek-V3
tokenizer.

| Quantity | Value |
|---|---|
| Total addressable slots (2 layers × 2 ngram orders × 8 heads × prime size) | 20,692,406 |
| Compressed vocab (NFKC + lowercase + space normalise) | 98,627 (vs 129,280 raw) |
| Distinct addresses per fact | 224 – 320 |
| Union of addresses for the 30-fact user | 5,616 |
| **Slot occupancy fraction** | **0.027 %** |
| Free slots in (layer 1, 2-gram, head 0) prime table | 646,232 / 646,403 = 99.97 % |

Within-user, 640 addresses are shared by ≥2 facts of the same user. The
top intra-user collisions are slots touched by all 30 facts — these are
the suffix N-grams ending at common scaffolding tokens (`<bos>`, `My`,
`I`, `is`). **Capacity is not a constraint** for per-user insertion.

## T1.3 — surgical slot read/write through the gate (`src/probe_readwrite.py`)

Existence proof: write a known marker into the embedding rows that a
target N-gram hashes to, run the Engram forward, measure how the output
changes at the trigger position vs everywhere else. Random init for
W_K, W_V, conv (no training).

Trigger position: ` Patel` in the sentence `"Today my doctor is Dr Patel
and that is final."`. Control position: ` and`.

| Metric | Value |
|---|---|
| ‖O_B[t*] − O_A[t*]‖ at trigger | 116.96 |
| ‖O_B[t_c] − O_A[t_c]‖ at control | 0.00 |
| Mean ‖O_B − O_A‖ at non-trigger positions | 0.65 |
| Max ‖O_B − O_A‖ at non-trigger positions | 7.55 |
| **Trigger / non-trigger mean ratio** | **179.7 ×** |
| cos(ΔO[t*], W_V(e_marker)) | **0.9978** |

Sanity on the non-trigger spread: the conv has kernel 4 with dilation =
max_ngram = 3, so the trigger marker leaks up to 12 positions
downstream — that explains the 7.55 max, which is still 15× smaller than
the trigger delta.

**Both checks PASS.** Spatial selectivity is essentially perfect; the
delta at t* is a near-pure projection of our written marker through
W_V. Conclusion: the Engram architecture mechanically supports surgical
per-user-row insertion. Whether the *gate fires usefully* on an inserted
row in a *trained* model is Tier 3's Q1 — but the architectural
existence question is settled in the affirmative.

## T1.4 — cross-user hash collision audit (`src/probe_collisions.py`)

100 synthetic users, ~30 facts each, hashed via the same demo config.
Two regimes:

- **R1 — all addresses** (includes scaffolding tokens like `My`, `is`):
  - 45,298 distinct addresses across the population.
  - 62.4 % shared by ≥2 users, 1,440 universal (every user).
  - Mean pairwise |A∩B|/|A| = **0.50**.
- **R2 — distinguishing-token addresses only** (suffix N-gram ending at a
  user-unique value, e.g. `Patel`, `Portland`, `1991`):
  - 20,437 distinct, 49.5 % shared by ≥2, **0 universal**.
  - Mean pairwise overlap = **0.0454** (≈ 4.5 %).

R2 overlap is dominated by *semantic* coincidence (same surname picked
from the same pool, e.g. two users both have `doctor=Patel`), not
hash-bucket collision. Pure hash-bucket collision is a small fraction of
the 4.5 %.

**Implication.** Per-user salt in the hash function is the recommended
design choice. The current demo hash is keyed only on `(layer_id,
ngram_order)`; adding a `user_id` term (e.g. XOR with `seed + user_id *
PRIME`) gives every user disjoint hash multipliers and reduces
cross-user collision to ~1/P (random). This is a one-line change to
`NgramHashMapping`. Engineering, not science. Without per-user salt,
two users with the same `Patel` doctor share rows and one will
overwrite the other.

## Tier 1 verdict: GREEN

| Question | Result |
|---|---|
| Demo runs? | Yes |
| Hashes are deterministic and addressable? | Yes |
| Free-slot capacity for user insertion? | Trivially yes (0.027 % occupancy / 99.97 % free) |
| Surgical write at a target N-gram changes only the trigger position? | Yes (179× ratio) |
| The change at the trigger carries the written information? | Yes (cos 0.998 with W_V projection) |
| Cross-user collisions manageable? | Yes with per-user salt; without salt ~4.5 % pairwise |

**Path-1 (true User-as-Engram on a Mini-Engram we pretrain ourselves) is
mechanically possible at the architectural level.** The remaining
unknowns are training-dependent (does the gate, post-pretraining,
*choose* to consult an inserted row) and are Tier 2/3 questions.

## Files produced

- `src/probe_hash.py`            (T1.2 — hash mechanism)
- `src/probe_readwrite.py`       (T1.3 — slot read/write through gate)
- `src/probe_collisions.py`      (T1.4 — cross-user collision audit)
- `results/t1_2_probe_hash.json`
- `results/t1_3_readwrite.json`
- `results/t1_4_collisions.json`
- `refs/engram_demo_v1.py`       (cloned from deepseek-ai/Engram, with
                                    one-line device fix at line 363)
- `refs/engram_README.md`

## Next decision

Tier 1 is green. The ranked options for Tier 2 substrate, in light of
the questions above:

1. **Pretrain a Mini-Engram on top of Karpathy's nanochat scaffolding.**
   nanochat already has data loading, tokenizer training, training loop,
   eval — saves 1–2 weeks of plumbing. Cost to graft Engram modules in:
   ~1 engineering day. Cost to pretrain a 200–400M dense + 200–500M
   Engram-table model on FineWebEDU: 3–7 days single-GPU. **Recommended
   default.**
2. Pretrain Mini-Engram from scratch without nanochat. Same compute
   cost, more plumbing.
3. Pivot to BLT 7B as substrate (its byte encoder uses hashed N-gram
   embeddings; weights public). Skip Tier 2. Trade-off: paper has to be
   reframed as User-as-conditional-memory rather than strictly
   User-as-Engram.

User-decision pending: which of the above for Tier 2.
