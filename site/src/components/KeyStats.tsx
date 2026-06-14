export function KeyStats() {
  return (
    <div>
      <h2>Key findings</h2>
      <p className="prose-paper text-ink-600">
        Headline numbers from our experiments on Mini-Engram-d20 (1.22 B base LM),
        n = 20 test users × 20 indirect-reasoning probes each.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
        <FindingCard
          tag="Layered F vs per-user LoRA (B)"
          color="badge-blue"
          stat="7.4×"
          headline="higher indirect-reasoning"
          subline="44% vs 6% indirect_any · 4.6× less contamination · 0/20 vs 17/20 users worse"
        />
        <FindingCard
          tag="F vs Qwen-3B + RAG @ KB=1000"
          color="badge-green"
          stat="+14 pp"
          headline="F beats a 2.5× larger LM + RAG"
          subline="F invariant at 44% indirect_any at 0 context tokens; Qwen-3B+RAG drops to 30% as retrieval recall@3 collapses 62% → 9%"
        />
        <FindingCard
          tag="Storage per user"
          color="badge-purple"
          stat="88 KB"
          headline="≈ 161× smaller than POLAR-class LoRA"
          subline="Per-user Engram-row override is 1 KB/fact; matches LoRA's 100% direct top-1 at 14.2 MB/user"
        />
        <FindingCard
          tag="Architectural contamination"
          color="badge-rose"
          stat="34 000×"
          headline="less than per-user LoRA"
          subline="Engram-row insertion's Δbpb = +0.00005 on held-out text vs LoRA's +1.78 — which more than triples val_bpb (0.74 → 2.52) on the same Mini-Engram-d20 base"
        />
      </div>
    </div>
  );
}

function FindingCard({ tag, color, stat, headline, subline }: {
  tag: string; color: string; stat: string; headline: string; subline: string;
}) {
  return (
    <div className="card p-5">
      <span className={`badge ${color}`}>{tag}</span>
      <div className="mt-3 flex items-baseline gap-3">
        <div className="text-4xl font-semibold tabular-nums text-ink-900">{stat}</div>
        <div className="text-ink-700">{headline}</div>
      </div>
      <p className="text-sm text-ink-600 mt-2">{subline}</p>
    </div>
  );
}
