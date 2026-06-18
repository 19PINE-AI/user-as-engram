# Tier 2 — Mini-Engram pretraining on nanochat (autonomous run)

**Started 2026-05-04. Single Blackwell GPU (96 GB). Single user, single
session, autonomous.**

---

## Goals, in order of importance

1. **Replicate the qualitative Engram finding at small scale.**
   Specifically: produce a Mini-Engram that, when its Engram modules are
   suppressed at inference, *loses more on factual-recall probes than on
   reading-comprehension probes*. That asymmetry is the load-bearing
   mechanistic claim of the Engram paper (their §6.3, Fig 6:
   factual 29–44% retained, reading 81–93% retained). If we reproduce
   this directionally — e.g. a 20-point delta between the two — we have
   the first public reproduction of the Engram architecture. That is a
   standalone contribution given no third-party reproduction exists.
2. **Demonstrate per-user surgical insertion on the trained Mini-Engram.**
   Write user-fact directions into free hash slots, show that the
   trained gate consults them at the trigger N-gram, and that the LM
   recovers the inserted fact at decode time. This is the User-as-Engram
   experiment.
3. **Demonstrate per-organisation insertion** (same mechanism, different
   namespace). Trivial extension if (2) works.

## Compute envelope

- One Blackwell ≈ one H100 ± in raw FP8 throughput; nanochat's speedrun
  is 8×H100 for 1.65 h (~$48). Single GPU equivalent ≈ 13 GPU-h.
- Engram tables roughly double per-step cost vs dense baseline.
- Realistic budget for this autonomous run:
    - Base GPT (control) and Engram-GPT (treatment), matched compute
    - ~3 GPU-h base + ~6 GPU-h Engram
    - Plus shakedown smoke runs (~30 min total)
    - Plus probes (minutes)
  - **Total ~10 GPU-h compute, ~12–24 h wall clock with monitoring.**
- This is enough to demonstrate directional effects, not to compete
  with Engram-27B's headline numbers. That's appropriate for a
  reproduction at small scale.

## Phase plan

### Phase 0 — Setup and inventory

- P0.1 Clone `karpathy/nanochat` into `user-as-engram/nanochat/`.
- P0.2 Read `nanochat/gpt.py` end-to-end. Identify the Block class, the
  optimiser, the data path.
- P0.3 Check tokenizer pipeline. Decide: train nanochat's BPE from
  scratch (slow), or hot-swap in the DeepSeek-V3 tokenizer the Engram
  demo already supports.
- P0.4 Verify nanochat's data-prep script runs and at what scale we
  can fit a corpus on disk.

### Phase 1 — Port Engram module

- P1.1 Write `nanochat/engram_module.py` based on the debugged demo.
  Keep `NgramHashMapping`, `MultiHeadEmbedding`, `ShortConv`, `Engram`.
  Strip the `BackBoneConfig`/`engram_cfg` global state — accept config
  as constructor args.
- P1.2 Rebuild `CompressedTokenizer` to map whichever tokenizer
  nanochat uses (or DeepSeek-V3 if we hot-swap).
- P1.3 Modify nanochat's Block to call Engram before attention at
  configured layer indices. Residual add: `h = h + Engram(h, input_ids)`.
- P1.4 Wire Engram embedding-table params into the optimiser with the
  paper's recipe: 5× LR, no weight decay, conv init zero so Engram
  starts as identity.
- P1.5 Sanity test: forward pass parity (Engram demo output reproduced
  for a fixed input within numerical tolerance), gradient flows, no
  NaN.

### Phase 2 — Pretrain control + treatment

- P2.1 Pick smallest meaningful config:
    - depth ~12, hidden ~768, ~100 M dense params
    - Engram: layers 2 and 7, max-N = 3, 8 heads, table size ~100 K
      slots × 8 heads × 2 ngrams × 2 layers ≈ ~200 M extra params
    - sequence length 1024
    - batch tokens per step ~256 K (depending on memory)
    - 1–3 B tokens (fits FineWebEDU sample, ~2–6 GB on disk)
- P2.2 Shakedown smoke run: 50 steps, both configs, just verify loss
  decreases.
- P2.3 Launch full base run in background. Save to `runs/base/`.
- P2.4 Launch full Engram run in background once base is done (or
  concurrently if memory allows).
- P2.5 Monitor: log loss curves, gradient norms, GPU util. If loss
  plateaus or diverges, debug.

### Phase 3 — Engram replication validation

- P3.1 Eval val loss on held-out FineWebEDU sample. Engram should be
  lower than base (paper claims significant validation-loss
  improvement at iso-FLOPs). With matched compute, look for ≥0.02
  delta.
- P3.2 Build a tiny factual-recall probe set (entity completion: "The
  capital of France is …", "The first US president was …" — 50–100
  items). Build a tiny reading-comprehension probe set (paragraph +
  question, 50–100 items, e.g. from TruthfulQA's MC2 or homemade).
- P3.3 Suppression ablation: at inference, set Engram output to zero.
  Measure factual-recall delta vs reading-comprehension delta.
  *Replication signal:* factual delta > reading delta by ≥10 points.
- P3.4 LogitLens-style sanity: layer-wise KL to final-layer
  distribution. Engram should show steeper descent in early layers.
- P3.5 Decision gate G2:
    - PASS → Phase 4.
    - FAIL → write up negative result honestly, halt User-as-Engram.

### Phase 4 — User-as-Engram surgical insertion

- P4.1 Reuse the 100 synthetic users from T1.4. Pick 10 for primary
  evaluation.
- P4.2 For each user, for each fact, identify:
    - The "trigger" N-gram positions (suffix N-gram ending at a
      key-token like `Patel`).
    - The hash-slot rows those positions touch.
    - Verify those rows are not in the "universal" set (the 1,440
      always-touched rows — never write into those).
- P4.3 Insertion strategies to test (Q1 from the original analysis):
    - **(a) Direction-as-LM-head.** Write the unembed-direction of the
      answer-token into the trigger rows. Predicted: gate fires, LM
      decodes the answer.
    - **(b) Random direction.** Control. Should not produce a
      meaningful answer.
    - **(c) Trained micro-adapter.** A tiny encoder f(facts) → row,
      trained for ~100 steps to maximise answer log-prob. Real but
      expensive.
- P4.4 Metrics:
    - Direct recall: prompt with fact-context, measure if the LM emits
      the inserted answer.
    - Cross-fact specificity: with user A's adapter active, prompt with
      user B's question — do not leak.
    - Spontaneous invocation: prompt with a question that *implies*
      the fact rather than naming it; does the gate fire?
- P4.5 Same protocol with organisational facts (company-internal
  policies, hours, contacts). Mechanically identical to user facts.

### Phase 5 — Write-up

- P5.1 Notes file with all numbers, plots, decisions.
- P5.2 If Phase 3 passes: short replication report for `notes.md`.
- P5.3 If Phase 4 passes: summary of findings for the User-as-Engram
  paper outline.

---

## Risks and how I will handle them autonomously

| Risk | Triggered by | Action |
|---|---|---|
| Pretraining diverges or NaNs | loss > 1e3 or NaN | reduce LR by 4×, restart |
| Loss plateaus above corpus-floor | val loss > 5.0 after 1 G tokens | inspect data, scale model up modestly, try again |
| Replication fails (P3.3 ratio < 1) | direct measurement | write up honestly as negative result; do not proceed to Phase 4 |
| User-fact insertion fails in (a) | direct measurement | try (b) and (c) before declaring failure |
| Wall-clock blows past 24 h | obvious | halt, summarise progress, leave training resumable |
| Out of disk for FineWebEDU | df check | use smaller subset (1 B tokens, ~3 GB) |
| Out of GPU memory | OOM error | reduce batch size or model dim |

## Decision authority

Per autonomous-research mode: I run experiments and analyse results
without stopping for confirmation. I will NOT:
- push to remote, change git config, or edit files outside
  `~/user-as-engram/` and the cloned `nanochat/`
- delete or destructively modify other projects
- spend on external services

I will produce a final report when training and probes are done, or
when I hit a halt condition above.
