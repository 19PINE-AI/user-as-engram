import { useState } from "react";
import { headline, PAPER_URL, CODE_URL } from "../../data/headline";

const BIBTEX = `@article{li2026userasengram,
  title         = {User as Engram: Internalizing Per-User Memory
                   as Local Parametric Edits},
  author        = {Li, Bojie},
  year          = {2026},
  eprint        = {2606.19172},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2606.19172}
}`;

export function Resources() {
  const [copied, setCopied] = useState(false);
  return (
    <section id="resources" className="relative px-6 md:px-12 lg:px-20 py-28 border-t hairline">
      <div className="mx-auto max-w-5xl">
        <div className="kicker mb-6">Take it further</div>
        <h2 className="display text-[clamp(2rem,5vw,3.4rem)] text-ink max-w-[16ch]">
          Stop asking one set of weights to be both the memory and the mind.
        </h2>

        <div className="grid sm:grid-cols-3 gap-4 mt-12">
          <a href={PAPER_URL} className="group rounded-xl border hairline bg-paper-50 p-5 hover:border-engram transition-colors">
            <div className="font-mono text-[0.66rem] uppercase tracking-[0.16em] text-engram">Paper</div>
            <div className="font-body text-lg text-ink mt-1 group-hover:text-engram-deep">Full results & proofs ↗</div>
          </a>
          <a href={CODE_URL} className="group rounded-xl border hairline bg-paper-50 p-5 hover:border-engram transition-colors">
            <div className="font-mono text-[0.66rem] uppercase tracking-[0.16em] text-engram">Code</div>
            <div className="font-body text-lg text-ink mt-1 group-hover:text-engram-deep">All scripts & data ↗</div>
          </a>
          <div className="rounded-xl border hairline bg-paper-50 p-5">
            <div className="font-mono text-[0.66rem] uppercase tracking-[0.16em] text-engram">Checkpoints</div>
            <div className="font-mono text-sm text-ink mt-1.5 flex flex-wrap gap-1.5">
              {headline.checkpoints.map((c) => (
                <span key={c} className="px-2 py-0.5 rounded bg-paper-200 text-ink-100">{c}</span>
              ))}
            </div>
          </div>
        </div>

        {/* bibtex */}
        <div className="mt-6 rounded-xl border hairline bg-ink overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3 border-b border-white/10">
            <span className="font-mono text-[0.66rem] uppercase tracking-[0.16em] text-paper-200">BibTeX</span>
            <button
              onClick={() => {
                navigator.clipboard?.writeText(BIBTEX);
                setCopied(true);
                setTimeout(() => setCopied(false), 1600);
              }}
              className="font-mono text-[0.66rem] uppercase tracking-[0.16em] text-engram-lighter hover:text-paper-50 transition-colors"
            >
              {copied ? "copied ✓" : "copy"}
            </button>
          </div>
          <pre className="font-mono text-[0.74rem] text-paper-200 p-5 overflow-x-auto leading-relaxed">{BIBTEX}</pre>
        </div>

        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 mt-12 font-mono text-[0.68rem] uppercase tracking-[0.16em] text-ink-50">
          <span>© 2026 Bojie Li · Pine AI</span>
          <span>A century after Semon named the engram.</span>
        </div>
      </div>
    </section>
  );
}
