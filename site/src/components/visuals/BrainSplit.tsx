const pairs = [
  {
    bio: "Hippocampus",
    bioSub: "fast · sparse · local trace (the engram)",
    arch: "Per-user Engram row",
    archSub: "88 KB · fires only on its trigger N-gram",
  },
  {
    bio: "Neocortex",
    bioSub: "slow · distributed · shared skills",
    arch: "Shared LoRA + frozen backbone",
    archSub: "one model everyone uses · learns to reason",
  },
];

export function BrainSplit() {
  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-7">
      <div className="grid md:grid-cols-2 gap-px bg-rule rounded-lg overflow-hidden">
        {pairs.map((p) => (
          <div key={p.bio} className="bg-paper-50 p-5">
            <div className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-ink-50">the brain</div>
            <div className="display text-xl md:text-2xl text-ink mt-1">{p.bio}</div>
            <div className="font-body text-[0.82rem] text-ink-50 mt-1">{p.bioSub}</div>

            <div className="my-3.5 flex items-center gap-2">
              <span className="h-px flex-1 bg-rule" />
              <span className="font-mono text-[0.56rem] uppercase tracking-[0.16em] text-engram">becomes</span>
              <span className="h-px flex-1 bg-rule" />
            </div>

            <div className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-engram">user as engram</div>
            <div className="display text-xl md:text-2xl text-engram-deep mt-1">{p.arch}</div>
            <div className="font-body text-[0.82rem] text-ink-50 mt-1">{p.archSub}</div>
          </div>
        ))}
      </div>
      <p className="font-body text-[0.82rem] text-ink-50 leading-snug mt-4 max-w-prose">
        Complementary learning systems, made architectural: a sparse local trace for content, a slow
        shared system for skill — so a new fact is written without overwriting how the model thinks.
      </p>
    </div>
  );
}
