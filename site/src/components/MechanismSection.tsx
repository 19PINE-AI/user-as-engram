import type { ReactNode } from "react";

interface Stage {
  n: number;
  title: string;
  kicker: string;
  body: ReactNode;
  evidence: string;
  source: string;
}

const STAGES: Stage[] = [
  {
    n: 1,
    title: "Addressing",
    kicker: "a fact's trigger N-gram is a hash key",
    body: (
      <>
        A fact is written at the embedding-table row(s) its trigger suffix
        N-gram hashes to, via Engram's deterministic multiplicative-XOR hash.
        The address is a pure function of surface tokens — no learning, no
        per-user state — so two users' edits only interfere if their{" "}
        <em>distinguishing</em> tokens collide.
      </>
    ),
    evidence: "4.5% key-token collision · 0.027% table occupancy / 30-fact user",
    source: "T1 collision audit",
  },
  {
    n: 2,
    title: "Read",
    kicker: "the row stores the value the model needed to emit the gold token",
    body: (
      <>
        On the reference architecture, the change a written row induces at the
        trigger position has cosine similarity <strong>0.998</strong> with the
        analytically predicted <code>W_V · marker</code> projection — the row{" "}
        <em>is</em> that value vector, not an inscrutable embedding.
        Behaviourally, ablating the Engram pathway degrades <em>factual</em>{" "}
        recall (93.3% retained) while <em>reading</em> comprehension is untouched
        (100%). The magnitude is modest at this scale, but the direction
        localises factual content to the Engram pathway, not the dense backbone.
      </>
    ),
    evidence: "cosine 0.998 to W_V·marker · 6.7 pp factual-vs-reading retention gap",
    source: "T2 read/write probe + sensitivity asymmetry",
  },
  {
    n: 3,
    title: "Write / locality",
    kicker: "the gate fires only on the trigger, inert everywhere else",
    body: (
      <>
        The gated lookup is an attention-style scalar that is non-zero only when
        the suffix N-gram is present. Inserting a row perturbs the residual
        stream by <strong>exactly 0.000</strong> at every non-trigger position
        and at every position before the Engram layer (causality), propagated
        through five later attention/MLP layers, while the trigger position moves
        123.0 → 1.54. Because the edit is <em>conditional</em>, a per-user write
        contaminates no unrelated forward pass, and cross-user leakage is zero by
        construction.
      </>
    ),
    evidence: "0.000 perturbation off-trigger · Δbpb +0.00005 vs LoRA's +1.78",
    source: "locality verification",
  },
  {
    n: 4,
    title: "Depth",
    kicker: "the edit lands where the model has already 'deepened'",
    body: (
      <>
        Rows are read at a late Engram layer, after the network has effectively
        committed to a next-token distribution. A LogitLens trace shows
        Mini-Engram converging to its output distribution <em>faster</em> than
        the base — layer-3 KL is 3.66 lower — so a row overrides a near-final
        prediction at the position where it is decoded, rather than steering an
        early, still-malleable representation (where a global LoRA does its
        damage).
      </>
    ),
    evidence: "−3.66 KL gap at layer 3 (effective deepening)",
    source: "LogitLens",
  },
  {
    n: 5,
    title: "The limit",
    kicker: "a surface-N-gram gate cannot compose across triggers",
    body: (
      <>
        The same mechanism that makes the edit local bounds what it can do. On a
        balanced 63-pair corpus, chained queries succeed <strong>91%</strong> of
        the time when the query's suffix overlaps the second fact's trigger, but
        only <strong>13%</strong> (≲10% after filtering coincidences) when true
        composition is required — though per-fact direct recall is 99.2% in both.
        The near-bimodal split is exactly what a surface-keyed lookup predicts:
        there is no chaining circuit to find because, by construction, there is
        none.
      </>
    ),
    evidence: "91% surface-overlap vs 13% true-chaining (per-fact recall 99.2%)",
    source: "multi-hop decomposition",
  },
];

export function MechanismSection() {
  return (
    <div>
      <h2>How it works: the mechanism</h2>
      <p className="prose-paper text-ink-600">
        User-as-Engram is not an opaque learned circuit but a{" "}
        <strong>content-addressable memory</strong> whose every step is either
        deterministic (hashing, gating) or directly measurable (the value a row
        writes, the locality of its effect). We trace one inserted fact through
        five observable stages — the same evidence the paper consolidates in its
        mechanistic-analysis appendix.
      </p>

      <ol className="mt-6 space-y-3 list-none pl-0">
        {STAGES.map((s) => (
          <li key={s.n} className="card p-5 flex gap-4">
            <div className="shrink-0 w-9 h-9 rounded-full bg-accent-600 text-white flex items-center justify-center font-semibold tabular-nums">
              {s.n}
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span className="font-semibold text-ink-900">{s.title}</span>
                <span className="text-ink-500 text-sm">— {s.kicker}</span>
              </div>
              <p className="text-sm text-ink-700 mt-1.5">{s.body}</p>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <span className="badge badge-green tabular-nums">{s.evidence}</span>
                <span className="text-[11px] uppercase tracking-wider text-ink-400">
                  {s.source}
                </span>
              </div>
            </div>
          </li>
        ))}
      </ol>

      <div className="card mt-4 p-5 bg-accent-50/50 border-accent-200">
        <div className="text-[11px] font-mono uppercase tracking-widest text-accent-700 mb-1">
          Synthesis
        </div>
        <p className="text-sm text-ink-800">
          User-as-Engram hashes a fact's surface trigger to a sparse row, stores
          the gold value there (cosine 0.998 to the predicted projection), reads
          it through a gate that fires only on that trigger (0.000 perturbation
          elsewhere) at a late, already-deepened layer — and therefore retrieves
          single facts with zero contamination but cannot compose across facts.
          This is why the substrate cleanly separates <em>content</em> (the rows)
          from <em>meta-skill</em> (composition, supplied by the shared LoRA in
          the layered design): the content/skill split is a direct consequence of
          the mechanism, not an engineering convenience.
        </p>
      </div>
    </div>
  );
}
