import { useState } from "react";

const BIB = `@article{li2026userasengram,
  title  = {User as Engram: Internalizing Per-User Memory as Local Parametric Edits},
  author = {Li, Bojie},
  year   = {2026},
  journal = {arXiv preprint},
  note   = {Pine AI},
  url    = {https://github.com/bojieli/user-as-engram}
}`;

export function CitationCard() {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(BIB).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div>
      <h2>Cite this work</h2>
      <p className="prose-paper text-ink-600">
        Code, data, and Mini-Engram checkpoints are released at the GitHub repo.
      </p>
      <div className="card mt-4 p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="text-xs uppercase tracking-wider font-mono text-ink-500">BibTeX</div>
          <button onClick={copy}
                  className="text-xs px-2.5 py-1 rounded border border-ink-300 bg-white hover:bg-ink-50">
            {copied ? "✓ Copied" : "Copy"}
          </button>
        </div>
        <pre className="!bg-ink-50 !text-ink-800 !text-xs !p-3 overflow-x-auto">{BIB}</pre>
      </div>
    </div>
  );
}
