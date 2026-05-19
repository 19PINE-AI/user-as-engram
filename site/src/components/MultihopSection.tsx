import { useEffect, useState } from "react";
import { loadMultihopRows, type MultihopRow } from "../lib/data";

export function MultihopSection() {
  const [rows, setRows] = useState<MultihopRow[] | null>(null);
  useEffect(() => { loadMultihopRows().then(setRows); }, []);

  if (!rows) return <div className="text-ink-500">Loading multi-hop data…</div>;

  return (
    <div>
      <h2>Multi-hop: chained fact reasoning</h2>
      <p className="prose-paper text-ink-600">
        8 chained-fact pairs (e.g., <code>doctor = Patel · Patel works at Globex</code>{" "}
        → "where does my doctor work?"). Engram alone gets 75% via surface-trigger
        N-gram overlap (see §6.11 in the paper). With a flat KB of all 16 facts,
        RAG-top-2 retrieves both gold facts on every item (100% ret_both) and
        the Mini-Engram-d20 backbone reads the chain off cleanly. RAG-top-1
        only retrieves one of the two facts and collapses to 50%.
      </p>
      <p className="text-xs text-ink-500">
        Note: n=8 is small; Wilson 95% CI on a 6/8 point estimate is [40.9%, 93.0%]
        and overlaps every row. Treat this as qualitative.
      </p>

      <div className="card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-ink-200 bg-ink-50/60">
              <th className="text-left px-3 py-2 text-xs font-semibold uppercase tracking-wider text-ink-600">method</th>
              <th className="text-right px-3 py-2 text-xs font-semibold uppercase tracking-wider text-ink-600">multi-hop top-1</th>
              <th className="text-right px-3 py-2 text-xs font-semibold uppercase tracking-wider text-ink-600">multi-hop top-5</th>
              <th className="text-right px-3 py-2 text-xs font-semibold uppercase tracking-wider text-ink-600">ret_both</th>
              <th className="text-right px-3 py-2 text-xs font-semibold uppercase tracking-wider text-ink-600">ctx tokens</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b border-ink-100 last:border-0">
                <td className="px-3 py-2">
                  <span className={`badge ${r.family === "engram" ? "badge-green" : "badge-orange"} mr-2`}>
                    {r.family === "engram" ? "Engram" : "RAG"}
                  </span>
                  {r.label}
                </td>
                <td className={`px-3 py-2 text-right tabular-nums ${r.top1 >= 0.99 ? "text-accent-700 font-semibold" : ""}`}>
                  {(r.top1 * 100).toFixed(0)}%
                </td>
                <td className="px-3 py-2 text-right tabular-nums">{(r.top5 * 100).toFixed(0)}%</td>
                <td className="px-3 py-2 text-right tabular-nums">{(r.ret_both * 100).toFixed(0)}%</td>
                <td className="px-3 py-2 text-right tabular-nums">{Math.round(r.ctx_tokens)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
