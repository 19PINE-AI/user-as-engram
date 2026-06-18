# Layered Architecture: Meta-Skill (LoRA) + Content (Engram)

**Status:** in autonomous execution
**Owner:** Claude (autonomous), reviewed by boj
**Started:** 2026-05-17

## Thesis

Personal memory in LLMs is **two problems, not one**:
- **Content** — the user's specific facts (names, dates, preferences)
- **Meta-skill** — the reasoning patterns that *use* facts to answer questions

The current paper conflates them by storing both in the same substrate
(either LoRA or Engram, alternatively). This experiment tests whether
matching each problem to its right substrate is strictly better:

- **Content → per-user Engram-row insertion** (local edit, low storage)
- **Meta-skill → one shared LoRA** (global edit, amortised across users)

If the layered design works cleanly, the paper's central contribution
upgrades from *"Engram replaces LoRA for per-user memory"* (currently
contested by the val_bpb_delta finding being the only robust result) to
*"personal memory is a layered systems problem; here is the right
decomposition."*

## Hypotheses, with falsifiable predictions

### H1 (combination A: per-user LoRA + per-user Engram together)

**Prediction:** No improvement over per-user LoRA alone, because LoRA's
global contamination (Δbpb +0.34) is already present and Engram-row
insertion can't subtract it. Storage doubles. Recall ceiling is already
saturated by LoRA at rank ≥8.

**Falsification:** if combination A shows Δbpb < per-user LoRA alone (e.g.,
+0.10 vs +0.34), then the Engram override is somehow *correcting* the
LoRA-perturbed gate, which would be a separate interesting finding.

### H2 (combination B: shared LoRA + per-user Engram, the layered design)

**Prediction:** Three sub-hypotheses, ranked by how surprising they'd be:

| Sub-H | Prediction | Implication |
|---|---|---|
| H2a | Δbpb scales monotonically with shared-LoRA rank | trade-off knob exists |
| H2b | Direct recall ≈ per-user Engram alone (Engram does the lookup) | content/skill cleanly separable |
| H2c | Indirect-style recall > per-user Engram alone (shared LoRA enables completion) | meta-skill is the real upgrade |

**Falsification:** if Δbpb stays at +0.34 regardless of rank, the shared LoRA
isn't actually narrower than a per-user LoRA → layering buys nothing.

### H3 (ablation over shared LoRA rank)

Sweep shared-LoRA rank ∈ {4, 16, 64}. Predict a Pareto curve in
(Δbpb, indirect-recall) space. The interesting region is rank-4: small
enough to plausibly keep Δbpb low, large enough to plausibly enable the
meta-skill.

## Why measurement is hard, and the workaround

Mini-Engram-d20 is a **base LM** (no instruction tuning). The
user-as-lora indirect probes were designed for Qwen-3B-Instruct in a
"Question: ... Answer: ..." format that base LMs don't follow reliably.

Two workarounds:

1. **Completion-style indirect probes.** Reformat each indirect QA as a
   completion: `"Facts: born 1993, year 2026. Age: "` → expect `"33"`.
   The base LM doesn't naturally do this arithmetic; the **shared LoRA**
   trained on cross-user (facts, indirect-Q, gold) tuples is supposed to
   learn it. This becomes the test of the "meta-skill" claim: does the
   shared LoRA enable indirect reasoning that the base alone can't do?

2. **Direct-recall + val_bpb as the robust subset.** Even if (1) is
   noisy, the direct-recall (does the model surface the user's facts?)
   and val_bpb (does the edit contaminate unrelated text?) measurements
   are unambiguous on a base LM. These remain the load-bearing tests.

## Experimental design

### Base model
Mini-Engram-d20@1536 (1.22 B total, 617 M scaling, 51.2 M Engram table).
Reason: largest Engram we have; smoke test of head-to-head already showed
clean LoRA Δbpb +0.34 vs Engram +0.00003 here.

### Users
- **Test users:** u000–u019 (20 users), each with ~34 facts and ~33 indirect QAs
- **Held-out training users:** u020–u029 (10 users) — used only to train the
  shared LoRA; their facts never appear in any test-user evaluation

### Conditions (per test user)

| Code | Edit | Storage / user |
|---|---|---|
| A | NO_EDIT | 0 |
| B | per-user LoRA (rank-64) | 14.2 MB |
| C | per-user Engram Joint OPT | ~88 KB |
| D | per-user LoRA + per-user Engram | 14.2 MB + 88 KB |
| E | shared LoRA only | 0 per user (amortised) |
| F | shared LoRA + per-user Engram | 88 KB per user (LoRA amortised) |

### Metrics

1. **direct top-1, top-5** on user's facts (completion-format prompts: e.g.,
   `"My doctor is Dr. → Krause"`)
2. **indirect top-1, top-5** on user's indirect QAs in completion format
3. **val_bpb** on a held-out 524 K-token ClimbMix shard (locality test)
4. **train_seconds** per method (training cost)

### Shared-LoRA training procedure (one-time per rank)

Per training user u ∈ {u020..u029}:
1. Render each fact as a completion sample: `"<prompt prefix> → <answer>"`.
2. Render each indirect QA as: `"Facts: <fact1>. <fact2>. ... Q: <indirect_q> A: <gold>"`,
   keeping only the facts in `required_fact_keys`. This is "in-context
   reasoning": the LoRA learns the *pattern* "given facts, do arithmetic /
   schema-following / day-set logic," without memorising which facts go
   with which user.
3. Train shared LoRA on this mixed corpus for K steps at rank r,
   alpha=2r, attaching to all Q/K/V projections.
4. Save adapter to `nanochat_base/shared_lora_d20/r{r}/`.

### Ablation over shared-LoRA rank

Train shared LoRA at **r ∈ {4, 16, 64}**. For each rank, evaluate
conditions E and F (the ones that use shared LoRA) on the full 20 test
users.

### Run plan and time budget

| Phase | Step | Time |
|---|---|---|
| 0 | Build cross-user training corpus from u020–u029 | <5 min |
| 1 | Train 3 shared LoRAs (r ∈ {4, 16, 64}) | ~45 min |
| 2 | Conditions A–F on 20 test users (r=16 as default) | ~3 h |
| 3 | Conditions E,F at r=4 and r=64 (ablation subset, 5 users each) | ~1 h |

**Total wall-clock:** ~4.5–5 h GPU on Mini-Engram-d20.

## Success criteria

The layered architecture is a *real* finding if **all three** hold:

1. **Locality survives layering**: Δbpb(F) significantly < Δbpb(B)
   (shared LoRA + Engram has less contamination than per-user LoRA).
2. **Direct recall maintained**: top-1(F) ≈ top-1(C) (Engram is doing the
   lookup; shared LoRA isn't fighting it).
3. **Meta-skill enables indirect reasoning**: indirect(F) > indirect(C)
   *and* indirect(F) > indirect(E) (the synergy is real, not just shared
   LoRA doing all the work alone).

If only (1) and (2) hold but (3) doesn't, the finding is "you can
combine them without harm, but no synergy" → still useful, smaller
paper-update.

If (1) fails (Δbpb of F is still ~0.34), the layering hypothesis is
falsified → report as a clean negative result; the architectural claim
falls back to "Engram-row insertion alone is local; LoRA is not."

## Decision tree

```
Phase 1 done (shared LoRAs trained)
       │
       ▼
Phase 2 done (full 20-user eval at r=16)
       │
       ▼
   Does Δbpb(F=r16) < Δbpb(B)?
       ├── YES → layering preserves locality at r=16
       │         continue to Phase 3 ablation
       │         ┌─────────┐
       │         ▼         ▼
       │  Δbpb(F=r4) ≈ 0?  Δbpb(F=r64) ≈ Δbpb(B)?
       │         │              │
       │         ▼              ▼
       │   Sub-H confirmed: rank knob works
       │   ▼
       │   Check indirect(F) > indirect(C)?
       │         ├── YES → strong synergy result. Headline paper update.
       │         └── NO  → "no harm, no help"; minor paper update.
       │
       └── NO → layering fails to preserve locality.
                 Combination A and B both inherit LoRA contamination.
                 Honest negative result; paper stays with current framing
                 but adds a "we tried layering, it doesn't help" section.
```

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Mini-Engram base can't do completion-style indirect reasoning even with shared LoRA | Train shared LoRA with longer schedule; if still bad, fall back to direct-only eval and report negative-on-indirect |
| Forward-hook conflicts between LoRA forward-override and Engram embedding-hook | Sanity-check apply/restore order; verify both edits are simultaneously active via a smoke test |
| 5-h GPU budget infeasible | Run subset first (1 rank, 5 users) as a smoke test |
| Shared-LoRA training overfits to held-out user attributes | Ensure test users (u000–u019) have no overlap with training users (u020–u029); inspect a few samples |

## Files in this experiment

- `nanochat/scripts/train_shared_lora.py` — Phase 1 trainer
- `nanochat/scripts/layered_architecture.py` — Phases 2 + 3 evaluator
- `run_layered.sh` — orchestrates Phases 0–3 (smoke, train 3 shared LoRAs,
  full 20-user at r=16, ablation at r=4 and r=64)
- `chain_layered.sh` — running in background (PID see logs/chain.log); polls
  for `run_scaling_and_h2h.sh` to exit, then launches `run_layered.sh`.
  Decoupled from the running queue script so bash never re-reads a script
  it's actively executing.
- Output paths:
  - `nanochat_base/shared_lora_d20/r{4,16,64}/{lora_state.pt,meta.json}` — Phase 1
  - `results/layered_d20_smoke.json` — Phase 0
  - `results/layered_d20_r16_full.json` — Phase 2 (headline)
  - `results/layered_d20_r{4,64}_abl.json` — Phase 3 (ablation)
- Log paths:
  - `logs/chain.log` — chain script status
  - `logs/shared_lora_r{R}.log` — Phase 1 per-rank
  - `logs/layered_smoke.log` — Phase 0
  - `logs/layered_r{16,4,64}_*.log` — Phases 2+3
  - `logs/layered.state` — single rc on completion
