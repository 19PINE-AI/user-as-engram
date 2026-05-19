export function Abstract() {
  return (
    <div className="prose-paper">
      <h2>Abstract</h2>
      <p>
        Personal memory in LLMs is two problems, not one: <em>content</em>
        (per-user facts) and <em>meta-skill</em> (reasoning patterns that use
        facts to answer questions). Per-user LoRA conflates them in one
        substrate, producing <strong>15,000× more contamination</strong> than
        per-user Engram-row insertion on the same Mini-Engram-d20 base
        (Δbpb +1.56 vs +0.0001).
      </p>
      <p>
        This motivates a <strong>layered architecture</strong>: one shared LoRA
        holds cross-user meta-skill (amortised over the population), and per-user
        Engram-row overrides hold per-user content (local, 88 KB/user). At n=20
        test users on Mini-Engram-d20, the layered design <strong>Pareto-dominates</strong> every
        per-user single-substrate baseline: 100% direct top-1, 6.8× better
        indirect reasoning than per-user LoRA (44% vs 7%), 4.0× less
        contamination, and never hurts indirect reasoning relative to the no-adapter
        base (0/20 users worse vs 17/20 for per-user LoRA).
      </p>
      <p>
        Head-to-head with RAG reveals a <strong>cross-over driven by KB size</strong>.
        At a toy KB (34 facts/user), Qwen-3B + RAG (52–57%) and J (RAG + shared
        LoRA, 54%) match or slightly beat F (44%, 0 context tokens). At realistic
        KB sizes (N=1000 facts/user, 1008-fact distractor pool),
        all-MiniLM-L6-v2 top-3 recall collapses from 62% to 9%, and
        Qwen-3B + RAG-top-3 drops to <strong>30% indirect_any — 14 pp below F</strong>,
        which is invariant in this axis because per-user Engram tables don't grow
        with population size. At N ≥ 100, F (zero context, no retrieval) beats both
        naive RAG and a ~2.5× larger instruction-tuned model with RAG. Substrate
        ranking is decided by deployment scale, not by which substrate alone.
      </p>
      <div className="mt-6 grid sm:grid-cols-2 gap-3 text-sm">
        <a className="card p-4 hover:border-accent-500 transition-colors no-underline"
           href="https://github.com/bojieli/user-as-engram">
          <div className="font-medium text-ink-900">📦 Code & data</div>
          <div className="text-ink-600 mt-1">5 new scripts, 7 result JSONs, 4 Mini-Engram weights</div>
        </a>
        <a className="card p-4 hover:border-accent-500 transition-colors no-underline" href="#data">
          <div className="font-medium text-ink-900">🔍 Browse the results</div>
          <div className="text-ink-600 mt-1">Per-user numbers, conditions, KB sizes</div>
        </a>
      </div>
    </div>
  );
}
