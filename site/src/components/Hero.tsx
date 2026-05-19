export function Hero() {
  return (
    <header className="border-b border-ink-200 bg-gradient-to-b from-accent-50/60 to-white">
      <div className="container-page py-16 md:py-20">
        <div className="text-xs font-mono uppercase tracking-widest text-accent-700 mb-3">
          arXiv preprint · 2026
        </div>
        <h1 className="leading-tight">
          User as Engram
        </h1>
        <p className="text-xl md:text-2xl text-ink-600 mt-3 max-w-3xl font-light">
          Internalizing per-user memory as <em>local</em> parametric edits,
          beating retrieval at production-scale KBs.
        </p>
        <div className="mt-6 flex flex-wrap items-center gap-3 text-sm">
          <span className="font-medium text-ink-700">Bojie Li</span>
          <span className="text-ink-400">·</span>
          <span className="text-ink-600">Pine AI</span>
        </div>
        <div className="mt-8 flex flex-wrap gap-3">
          <a className="inline-flex items-center gap-2 bg-ink-900 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-ink-800 no-underline"
             href="#abstract">Read paper ↓</a>
          <a className="inline-flex items-center gap-2 bg-white border border-ink-300 text-ink-800 px-4 py-2 rounded-md text-sm font-medium hover:bg-ink-50 no-underline"
             href="https://github.com/bojieli/user-as-engram" target="_blank" rel="noreferrer">
            GitHub ↗
          </a>
          <a className="inline-flex items-center gap-2 bg-white border border-ink-300 text-ink-800 px-4 py-2 rounded-md text-sm font-medium hover:bg-ink-50 no-underline"
             href="#data">Browse data →</a>
        </div>
        <div className="mt-10 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl">
          <BadgeStat number="44%" label="F indirect_any (n=20, 0 ctx)" />
          <BadgeStat number="14 pp" label="F vs Qwen+RAG at KB=1000" />
          <BadgeStat number="88 KB" label="Storage per user" />
          <BadgeStat number="15 000×" label="Less contamination than LoRA" />
        </div>
      </div>
    </header>
  );
}

function BadgeStat({ number, label }: { number: string; label: string }) {
  return (
    <div className="bg-white border border-ink-200 rounded-lg p-3">
      <div className="text-2xl font-semibold text-ink-900 tabular-nums">{number}</div>
      <div className="text-[11px] uppercase tracking-wider text-ink-500 mt-1 leading-tight">
        {label}
      </div>
    </div>
  );
}
