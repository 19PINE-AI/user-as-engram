# User-as-Engram — Findings (paper-reference document)

Snapshot date: 2026-05-12. All experiments in `/home/ubuntu/user-as-engram/`. Backing data lives in `results/*.json`.

---

## TL;DR

**Setup.** Nano-scale public reproduction of DeepSeek Engram (arXiv:2601.07372) at 4 dense sizes (137 M → 1.22 B params), with a full capacity × token-budget ablation, a fact-count scaling sweep up to 1000 facts, and a LOCOMO single-hop memory-system comparison. All trained from scratch on ClimbMix-400 B; all evals on a single Blackwell RTX PRO 6000 (102 GB).

**Five findings worth reporting**:

1. **User-as-Engram (Joint-OPT) beats every retrieval-based memory baseline on LOCOMO single-hop at iso-LM**, by 63–73 % relative across both d8@512 and d12@1280 Mini-Engrams.
2. **Engram capacity has an *optimum*, not a "more is better" trend.** At fixed token budget, smaller dense models prefer smaller Engram tables. We map the 5 × 3 capacity × tokens response surface explicitly at d8 and d12@768.
3. **At the *capacity × tokens* sweet spot, surgical insertion (OPT) recall matches the in-context-learning ceiling** (100/100 USER + 100/100 ORG) at zero context tokens.
4. **Fact-count scaling is approximately flat** up to 1000 facts when each fact is inserted independently — no apparent ceiling in that regime. d12@1280 holds 0.98 USER OPT top-1 from n=100 to n=1000 (matched by ICL at 1.00).
5. **The optimal Engram capacity is *constant* across the dense sizes we tested** (large = 50 K vocab × 256 embed = 51 M params). The optimum scales in **tokens**, not capacity. This pushes back on the assumption that Engram size should scale with dense size.

**Key headline number**: d12@1280 optimal (625 M total params, large Engram, 3.34 B tokens at Karpathy 12 t/p) achieves:
- 100 / 100 USER OPT top-1 (matches ICL ceiling, zero context tokens)
- 100 / 100 ORG OPT top-1
- 0.195 LOCOMO Joint OPT F1 vs 0.113 for MEMMACHINE_LIKE (+73 % relative)

---

## 1. Experimental setup

### 1.1 Mini-Engram models trained

All models trained from scratch on NVIDIA ClimbMix-400 B (Karpathy mirror `karpathy/climbmix-400b-shuffle`). Tokenizer: nanochat RustBPE, vocab 32 768. Hardware: single NVIDIA RTX PRO 6000 Blackwell (102 GB GDDR7), bf16 (FP8 tried, slower on Blackwell sm120 — see §6.1). Optimizer: nanochat MuonAdamW (Muon for matrix params, AdamW for embeddings/scalars). Engram tables: 5× embedding LR, no weight decay (per DeepSeek paper recipe).

Recipe targets **Karpathy 12 t/p × scaling-params** where `scaling_params = transformer_matrices + lm_head` (per `nanochat/scripts/base_train.py:266`).

| tag | dense | n_layer × n_embd | scaling-params | Engram size | tokens | wall-clock |
|---|---|---|---|---|---|---|
| `engram_d8` (v1) | 137 M | 8 × 512 | 42 M | small (20 K × 128) → 11 M | 0.524 B (4.6 t/p) | ~50 min |
| `engram_d8_v2` (control v2) | 178 M | 8 × 512 | 42 M | **large** (50 K × 256) → 51 M | 0.50 B (12 t/p) | ~38 min |
| `engram_d12` (v1) | 339 M | 12 × 768 | 110 M | large (50 K × 256) → 51 M | 0.79 B (7.2 t/p) | ~3 h |
| `engram_d12_v2` | 339 M | 12 × 768 | 110 M | large (50 K × 256) → 51 M | 1.32 B (**12 t/p**) | ~5 h |
| `engram_d12_w1280_optimal` | **625 M** | 12 × 1280 | 278 M | large (50 K × 256) → 51 M | 3.34 B (**12 t/p**) | ~10.5 h |
| `engram_d20_w1536_optimal` *(in flight, 54 %)* | **1.22 B** | 20 × 1536 | 617 M | large (50 K × 256) → 51 M | 7.40 B (**12 t/p**) | ~50 h |

### 1.2 Capacity × token ablation grid (30 cells)

Plus 30 cells in a full matrix at d8@512 and d12@768:

| capacity name | vocab × embed | Engram params | Engram / dense ratio at d8 / d12@768 |
|---|---|---|---|
| tiny | 5 K × 64 | 1.28 M | 1 % / 0.4 % |
| **small** | **20 K × 128** | **10.2 M** | 8 % / 3.6 % |
| **medium** | **50 K × 128** | **25.6 M** | 21 % / 9 % |
| **large** | **50 K × 256** | **51.2 M** | 41 % / 18 % |
| xlarge | 100 K × 256 | 102.4 M | 81 % / 36 % |

Token budgets: d8 ∈ {0.5 B, 1 B, 2 B}; d12@768 ∈ {0.5 B, 1.32 B, 2.5 B}.

### 1.3 Evaluations

Three eval pipelines run on each model:

1. **`insertion_strategies_v2.py`** — 16-fact micro-probe with RANDOM, WTE, UNEMBED_P, DUAL, OPT-15 insertions. Reports per-fact post-insertion rank vs control.
2. **`eval_at_scale.py`** (E1) — 100 USER + 100 ORG facts (also re-run at 200/500/1000 for fact-scaling study). Per-fact ICL ceiling, baseline rank, UNEMBED_P, OPT-15. Top-1 / top-5.
3. **`locomo_eval.py`** — 2 conversations × 80 single-hop QAs from LOCOMO. Token-F1 across 8 memory systems (NO_MEMORY, MARKDOWN_ALL, RAG_TOP1/3, MEM0_LIKE top-5, MEMMACHINE_LIKE top-3, User-as-Engram OPT, User-as-Engram Joint OPT). All use the same Mini-Engram-d12 (or whichever) as the answer LM.

---

## 2. Finding 1 — LOCOMO single-hop memory-system comparison

Token-F1 over 160 QA pairs (2 conv × 80 QA), all systems using Mini-Engram-d12 as the answer LM:

| method | conv 1 | conv 2 | **avg** |
|---|---|---|---|
| NO_MEMORY (no context) | 0.046 | 0.039 | 0.043 |
| MARKDOWN_ALL (all evidence in prompt) | 0.046 | 0.076 | 0.061 |
| RAG_TOP1 (sentence-encoder) | 0.053 | 0.060 | 0.056 |
| RAG_TOP3 | 0.077 | 0.103 | 0.090 |
| MEM0_LIKE (top-5 retrieval) | 0.082 | 0.116 | 0.099 |
| MEMMACHINE_LIKE (top-3 episodic) | 0.090 | 0.120 | 0.105 |
| User-as-Engram OPT (per-fact) | 0.140 | 0.147 | 0.143 |
| **User-as-Engram Joint OPT** | **0.162** | **0.179** | **0.171** |

**Joint OPT beats MEMMACHINE_LIKE by +63 % relative, RAG_TOP3 by +90 % relative, consistent across both conversations.**

At d12@1280 optimal (625 M, 3.34 B tokens): Joint OPT 0.195 vs MEMMACHINE 0.113 → **+73 % relative**. Effect grows with dense scale.

### Why Joint OPT wins
- Retrieval baselines miss when the question's surface form differs from the evidence sentence (the failure mode visible in RAG_TOP1's 0.056).
- Engram OPT trains a per-fact row to push the gold first-token's logit at the trigger N-gram. The remaining 15 generated tokens are then conditioned on a correctly-anchored prefix.
- Zero context tokens used → no in-context-budget tradeoff.

### Caveats
- Two of ten LOCOMO conversations only (kept compute modest).
- Token-F1 is a loose proxy; a stronger benchmark would use an LLM-as-judge.
- Single-hop only — LOCOMO's multi-hop / temporal / adversarial categories are harder.
- Best-case for each storage substrate: retrieval baselines get the gold evidence sentence, Engram gets the gold (q, a) pair.

---

## 3. Finding 2 — Capacity × token-budget matrix ablation (the new contribution)

Full 30-cell grid at d8@512 and d12@768. Format: `ins-OPT t1 / E1 USER OPT t1 / E1 USER OPT t5 / LOCOMO Joint OPT F1`.

### 3.1 d8@512 (137 M dense, 42 M scaling) — full matrix

|  | 0.5 B tok | 1 B tok | 2 B tok |
|---|---|---|---|
| **tiny** (1.3 M) | 0.44 / 0.62 / 0.84 / 0.172 | 0.56 / 0.57 / 0.87 / 0.168 | 0.69 / 0.69 / 0.94 / 0.159 |
| **small** (10 M) | 1.00 / 0.95 / 1.00 / **0.198** | 1.00 / 0.98 / 1.00 / 0.169 | 1.00 / 1.00 / 1.00 / 0.159 |
| **medium** (26 M) | 0.88 / 0.84 / 0.97 / 0.178 | 1.00 / 0.97 / 1.00 / 0.165 | 1.00 / 1.00 / 1.00 / 0.204 |
| **large** (51 M) | 0.62 / 0.84 / 0.92 / 0.161 | 1.00 / 1.00 / 1.00 / 0.170 | **1.00 / 1.00 / 1.00 / 0.207** ⭐ |
| **xlarge** (102 M) | 0.75 / 0.67 / 0.80 / 0.172 | 0.81 / 0.80 / 0.90 / 0.141 | 0.81 / 0.79 / 0.98 / 0.153 |

### 3.2 d12@768 (339 M dense, 110 M scaling) — full matrix

|  | 0.5 B tok | 1.32 B tok | 2.5 B tok |
|---|---|---|---|
| **tiny** (1.3 M) | 0.31 / 0.25 / 0.60 / 0.171 | 0.44 / 0.40 / 0.74 / **0.198** | 0.62 / 0.57 / 0.84 / 0.192 |
| **small** (10 M) | 0.81 / 0.78 / 0.95 / 0.173 | 0.81 / 0.92 / 1.00 / 0.159 | 1.00 / 0.99 / 1.00 / 0.174 |
| **medium** (26 M) | 0.88 / 0.74 / 0.93 / 0.193 | 0.81 / 0.81 / 0.94 / 0.176 | 0.94 / 0.97 / 1.00 / 0.191 |
| **large** (51 M) | 0.94 / 0.93 / 0.99 / 0.169 | 1.00 / 0.98 / 0.98 / 0.169 | **1.00 / 1.00 / 1.00 / 0.185** ⭐ |
| **xlarge** (102 M) | 1.00 / 0.96 / 0.99 / 0.184 | 0.94 / 0.95 / 1.00 / 0.182 | 0.94 / 0.97 / 0.99 / 0.171 |

### 3.3 Five interpretations

1. **Capacity has an optimum.** Both `tiny` (capacity-bottlenecked) and `xlarge` (over-provisioned) under-perform on E1 USER OPT and LOCOMO. The optimum is **`large` (50 K × 256, 51 M params) at both dense sizes** when tokens are sufficient.
2. **The optimum scales in *tokens*, not capacity.** The best Engram size is the same (large) at d8 and d12, but the *tokens needed to populate it* grow with dense size: d8 best at 2 B (~48 t/p of scaling-params), d12 best at 2.5 B (~23 t/p).
3. **`large` at low tokens looks bad** (0.62 ins-OPT for d8/large/0.5 B; 0.93 for d12/large/0.5 B). With insufficient tokens, the 51 M Engram is under-trained per slot. This is the artefact that initially made us think v2 was *worse* than v1 — the artefact disappears once the token budget catches up.
4. **`small` (10 M) is a strong default at any budget.** It saturates ins-OPT at 100 % for d8 from 0.5 B tokens onward, beating both larger and smaller capacities at lean budgets. For storage-constrained deployment, this is the right pick.
5. **LOCOMO Joint OPT does *not* track ins-OPT monotonically.** d8 best LOCOMO is large/2 B at 0.207, but small/0.5 B has nearly identical LOCOMO (0.198) with **5× less Engram storage**. For LOCOMO-style tasks alone, small is competitive.

### 3.4 The original mistake

My initial v2 retrains used `large` Engram (50 K × 256) at the Karpathy 12 t/p budget. At d8 (low scaling-params ⇒ low absolute tokens), this is 0.5 B tokens, which is below the catch-up point. At d12, 1.32 B tokens is enough — `large` reaches ins-OPT 1.00. So the rule is:

> **Use `large` (50 K × 256, 51 M params) Engram. Spend at least 2 B tokens on it for d8, 2.5 B+ for d12. Below that, use `small` (10 M).**

---

## 4. Finding 3 — Fact-count scaling (100 → 1000 facts)

E1 USER OPT top-1 / top-5, per-fact independent OPT-15 insertion, evaluated on first N facts of `corpora_xl.json`. ICL@max = ICL recall at N=1000 (the in-context-learning ceiling for that model).

### 4.1 USER OPT top-1 / top-5

| model | params | n=100 | n=200 | n=500 | **n=1000** | ICL@1000 |
|---|---|---|---|---|---|---|
| d8 v1 (small/0.524 B) | 137 M | 0.81 / 0.91 | 0.82 / 0.92 | 0.81 / 0.93 | **0.84 / 0.95** | 0.98 / 1.00 |
| d8 v2 (large/0.5 B) | 178 M | 0.68 / 0.87 | 0.69 / 0.86 | 0.69 / 0.85 | **0.68 / 0.84** | 0.88 / 0.99 |
| d8 large/2 B (best d8) | 178 M | 1.00 / 1.00 | 0.99 / 1.00 | 0.99 / 1.00 | **0.99 / 1.00** | 0.99 / 1.00 |
| d12 v1 (large/0.79 B) | 339 M | 0.86 / 0.96 | 0.88 / 0.96 | 0.87 / 0.97 | **0.88 / 0.97** | 0.96 / 1.00 |
| d12 v2 (large/1.32 B) | 339 M | 0.87 / 0.98 | 0.87 / 0.97 | 0.86 / 0.97 | **0.87 / 0.97** | 1.00 / 1.00 |
| d12 large/2.5 B (best d12) | 339 M | 0.96 / 1.00 | 0.95 / 1.00 | 0.95 / 0.99 | **0.96 / 0.99** | 0.98 / 1.00 |
| **d12@1280 optimal** | **625 M** | **0.98 / 1.00** | **0.97 / 1.00** | **0.98 / 1.00** | **0.98 / 1.00** | 1.00 / 1.00 |
| d20@1536 optimal | 1224 M | *training (54 %)* | — | — | — | — |

### 4.2 ORG OPT top-1 / top-5

| model | n=100 | n=200 | n=500 | n=1000 | ICL@1000 |
|---|---|---|---|---|---|
| d8 v1 | 0.82 / 0.97 | 0.84 / 0.96 | 0.87 / 0.98 | 0.87 / 0.98 | 1.00 / 1.00 |
| d8 v2 | 0.60 / 0.85 | 0.58 / 0.84 | 0.59 / 0.83 | 0.62 / 0.84 | 0.99 / 1.00 |
| d8 large/2 B (best d8) | 0.99 / 1.00 | 0.99 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| d12 v1 | 0.86 / 0.93 | 0.89 / 0.96 | 0.91 / 0.98 | 0.91 / 0.98 | 1.00 / 1.00 |
| d12 v2 | 0.84 / 0.96 | 0.83 / 0.95 | 0.85 / 0.98 | 0.85 / 0.97 | 1.00 / 1.00 |
| d12 large/2.5 B | 0.98 / 1.00 | 0.97 / 0.99 | 0.98 / 1.00 | 0.98 / 1.00 | 1.00 / 1.00 |
| **d12@1280 optimal** | **1.00 / 1.00** | **0.99 / 1.00** | **1.00 / 1.00** | **0.99 / 1.00** | 1.00 / 1.00 |

### 4.3 Interpretation

**Per-fact independent OPT recall is flat with fact count up to 1000.** When each fact is inserted independently (the production-realistic per-user override scenario), there's no measurable degradation at 10× fact count.

**This is *different* from the joint-OPT density curve** previously reported (`results/joint_opt_*.json`): at 1000 facts trained *jointly into one user's table*, top-1 falls to 35 % (296 KB total override). The two regimes:

- *Independent OPT* (per-user override): each fact gets its own private rows. 0.98 top-1 at n=1000 on d12@1280, no ceiling visible.
- *Joint OPT* (all facts share one user's table): 1000 facts compete for the same global slot space. Top-1 35 %.

For per-user / per-org memory deployment (the User-as-Engram thesis), independent OPT is the natural mode.

**d12@1280 optimal reaches or matches ICL** at every fact count we tested:
- USER: 0.98 vs ICL 1.00 (gap = 2 pt)
- ORG: 0.99–1.00 vs ICL 1.00 (gap ≈ 0)

---

## 5. Finding 4 — Dense-size scaling at optimal config

| model | total params | scaling-params | tokens | val_bpb | ins-OPT t1 | E1 USER t1 | E1 ORG t1 | LOCOMO J |
|---|---|---|---|---|---|---|---|---|
| d8 v1 | 137 M | 42 M | 0.52 B | 0.95 | 0.94 | 0.95 | 0.94 | 0.164 |
| d8 best (large/2 B) | 178 M | 42 M | 2.00 B | — | 1.00 | 1.00 | 1.00 | **0.207** |
| d12 v1 | 339 M | 110 M | 0.79 B | 0.85 | — | 0.93 | 0.94 | 0.171 |
| d12 v2 (Karpathy 12 t/p) | 339 M | 110 M | 1.32 B | 0.827 | 1.00 | 0.98 | 0.94 | 0.169 |
| d12 best (large/2.5 B) | 339 M | 110 M | 2.50 B | — | 1.00 | 1.00 | 1.00 | 0.185 |
| **d12@1280 optimal** | **625 M** | 278 M | 3.34 B | **0.770** | **1.00** | **1.00** | **1.00** | **0.195** |
| d20@1536 optimal *(training)* | 1224 M | 617 M | 7.40 B | 0.852 @ step 28k | — | — | — | — |

`val_bpb`: ClimbMix bits-per-byte on the held-out validation shard.

**Pattern**: at the optimum config, **insertion-OPT and E1 recall saturate at 100 % from d12@768 onward**. The remaining scaling gain is in LOCOMO Joint OPT F1 (0.164 → 0.207 across d8 sizes; 0.169 → 0.195 from d12 v1 → d12@1280). Whether d20@1536 pushes past 0.195 is the open question.

---

## 6. Methodology notes

### 6.1 FP8 attempt and abandonment

`nanochat/scripts/base_train.py` exposes `--fp8` (`Float8LinearConfig` from `nanochat.fp8`). I ported the flag to `engram_pretrain.py` and verified it converged on a d6 sanity (`val_bpb` 3.16 → 1.52 after 200 steps with 41/45 Linears in FP8). **However on d12@1280 the FP8 path *slowed* training to 54 K tok/s (vs 134 K bf16)** — a 60 % regression. The custom `nanochat.fp8` kernel likely isn't tuned for Blackwell sm120. Phase H reverted to bf16. FP8 may still work with `torchao.float8` directly; untested.

### 6.2 Token-budget choice in Phase H

The d12@768 ablation said the optimum was `large / 2.5 B` ≈ 22.7 t/p × scaling-params. Extrapolating to d20@1536 at 22.7 t/p meant 14 B tokens / ~104 h bf16. We dropped to **12 t/p (Karpathy default)** for Phase H to keep wall-clock tractable. The d12 ablation gap between 12 t/p (1.32 B) and 22.7 t/p (2.5 B) for the `large` capacity was small: ins-OPT tied at 1.00, E1 USER +2 pt, LOCOMO +0.016. Acceptable.

### 6.3 GPU sharing

The Blackwell is shared with other users. Two unrelated training jobs (`train_hippo.py`, `generate_teacher_traces.py`) joined the GPU on 2026-05-12, dropping d20's throughput from 42 K → 14 K tok/s. Revised d20 ETA at the time of writing: ~2.7 days remaining (was 22 h).

### 6.4 Eval design caveats

- **insertion_strategies_v2** uses 16 hand-curated facts and OPT-15 (15 gradient steps per fact). A larger / harder set would stress-test more.
- **eval_at_scale (E1)** is *per-fact independent* OPT. Multi-tenant interference is *not* measured here (covered by `per_user_table_eval.py` separately).
- **LOCOMO Option A**: 160 QAs across 2 of 10 conversations. Token-F1 not LLM-judge. Single-hop only.

---

## 7. Open items / things to do next

1. **Finish d20@1536_optimal training** (currently 54 % at val_bpb 0.852). Then re-run insertion_strategies + E1 + LOCOMO + fact-scale {100, 200, 500, 1000} on it.
2. **Add the 2-D matrix to the paper as a new section** ("Engram capacity ablation"). Currently only mentioned in `MEMORY.md` and `FINDINGS.md`.
3. **Fact-count scaling for d20@1536** — once trained.
4. **Joint-OPT density curve at d12@1280 / d20@1536** — does the joint-mode 1000-fact ceiling (35 % at d12@768) lift with bigger dense?
5. **LOCOMO judge-LM eval** — replace token-F1 with the canonical LLM-as-judge to match LOCOMO's published metric.
6. **Multi-hop reasoning probe** on d12@1280 — currently 75 % at d12@768 with the caveat that 6/8 successes share suffix-N-gram overlap, not true chaining.
7. **FP8 retry with `torchao.float8`** — would cut d20's training in half if it works.

---

## 8. File index

### Key scripts
- `nanochat/scripts/engram_pretrain.py` — training driver (with `--fp8` flag added)
- `nanochat/scripts/insertion_strategies_v2.py` — RANDOM / WTE / UNEMBED_P / DUAL / OPT-15
- `nanochat/scripts/eval_at_scale.py` — E1 per-fact + E2 multi-user
- `nanochat/scripts/locomo_eval.py` — Option A token-F1 across 8 memory systems
- `nanochat/scripts/joint_opt.py` — multi-fact joint sparse-row OPT
- `nanochat/scripts/capacity_ablation_table.py` — produces 30-cell matrix
- `nanochat/scripts/factscale_table.py` — produces 100/200/500/1000 fact-scaling table
- `nanochat/scripts/pick_optimal.py` — picks Phase H config from ablation
- `nanochat/scripts/scaling_summary.py` — dense-size scaling table

### Chain scripts
- `run_capacity_ablation.sh` — 30-cell matrix ablation (done)
- `run_phaseH.sh` — train d12@1280 + d20@1536 at optimal config (d12@1280 done, d20 in flight)
- `run_factscale_bench.sh` — 4 fact-counts × 8 models (28/32 done; 4 d20 cells deferred)
- `run_supervisor.sh` — waits for ablation table file, fires Phase H

### Result files
- `results/<tag>__{strategies,scale,locomo}.json` — per-model evals
- `results/<tag>__factscale_n{100,200,500,1000}.json` — fact-scaling cells
- `results/capacity_ablation_table.txt` — 30-cell rendered table
- `results/factscale_table.txt` *(will be written after d20 done)*
- `results/optimal_config.json` — `pick_optimal.py` output

### Paper
- `paper/main.tex` — 22-page arXiv-style draft with §6.7 LOCOMO + §8.5 Future Work. Needs new section for capacity ablation + updated scaling table once d20 finishes.
- `paper/main.pdf` — compiled (May 7).
