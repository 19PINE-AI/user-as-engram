import { useEffect, useMemo, useState } from "react";
import { loadComparisonTable } from "../lib/data";
import type { ComparisonRow } from "../types";

const FAMILY_LABELS: Record<string, { label: string; color: string }> = {
  baseline: { label: "baseline", color: "badge-gray" },
  engram: { label: "Engram", color: "badge-green" },
  lora: { label: "per-user LoRA", color: "badge-rose" },
  layered: { label: "Layered", color: "badge-blue" },
  rag: { label: "RAG", color: "badge-orange" },
  qwen_rag: { label: "Qwen-3B + RAG", color: "badge-purple" },
};

type SortKey = "method" | "direct_top1" | "indirect_top1" | "indirect_any" |
               "retrieval_acc" | "ctx_tokens" | "storage_per_user_kb" | "val_bpb_delta";

export function ComparisonTableSection({ focus }: { focus: "rag" | "all" }) {
  const [rows, setRows] = useState<ComparisonRow[] | null>(null);
  const [families, setFamilies] = useState<Set<string>>(
    () => new Set(Object.keys(FAMILY_LABELS)));
  const [sort, setSort] = useState<SortKey>("indirect_any");
  const [dir, setDir] = useState<"asc" | "desc">("desc");

  useEffect(() => { loadComparisonTable().then(setRows); }, []);

  const visible = useMemo(() => {
    if (!rows) return null;
    const r = rows.filter(r => families.has(r.family));
    const sortVal = (x: ComparisonRow): number | string => {
      const v = (x as any)[sort];
      if (v == null) return -Infinity;
      return v;
    };
    return r.sort((a, b) => {
      const av = sortVal(a), bv = sortVal(b);
      if (av === bv) return 0;
      if (typeof av === "string" || typeof bv === "string")
        return dir === "asc" ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
      return dir === "asc" ? (av as number) - (bv as number) : (bv as number) - (av as number);
    });
  }, [rows, families, sort, dir]);

  const onSort = (k: SortKey) => {
    if (k === sort) setDir(d => d === "asc" ? "desc" : "asc");
    else { setSort(k); setDir(k === "method" ? "asc" : "desc"); }
  };

  const toggleFamily = (f: string) => {
    const next = new Set(families);
    if (next.has(f)) next.delete(f); else next.add(f);
    setFamilies(next);
  };

  if (!rows) return <div className="text-ink-500">Loading comparison table…</div>;

  const title = focus === "rag" ? "RAG vs the layered design (head-to-head)" : "All methods compared";
  const subtitle = focus === "rag"
    ? "At KB = 34 facts/user. Same 20-user × 20-probe split. Naive RAG (G/H/I) doesn't reach F on indirect; J (RAG + shared LoRA) does but at 44 context tokens. Qwen-3B + RAG slightly beats F at this small KB — the picture inverts at scale."
    : "Every method we ran, on the same 20-user × 20-probe split (KB = 34 facts/user). Filter by family with the chips below, sort by clicking any column header.";

  return (
    <div>
      <h2>{title}</h2>
      <p className="prose-paper text-ink-600">{subtitle}</p>

      <div className="card mt-6">
        <div className="border-b border-ink-200 px-4 py-3 flex flex-wrap items-center gap-2 text-xs">
          {Object.entries(FAMILY_LABELS).map(([k, v]) => (
            <button key={k} onClick={() => toggleFamily(k)}
                    className={`px-2.5 py-1 rounded-full border transition-colors ${
                      families.has(k)
                        ? "border-ink-300 bg-white text-ink-800"
                        : "border-ink-200 bg-ink-50 text-ink-400"
                    }`}>
              <span className={`badge ${v.color} mr-1.5`} style={{ padding: "2px 6px" }}>
                {v.label}
              </span>
            </button>
          ))}
          <span className="text-ink-400 ml-auto">{visible?.length ?? 0} methods</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-ink-200 bg-ink-50/60">
                <Th onClick={() => onSort("method")} active={sort === "method"} dir={dir}>method</Th>
                <th className="text-left px-3 py-2 text-xs font-semibold uppercase tracking-wider text-ink-600">family</th>
                <th className="text-left px-3 py-2 text-xs font-semibold uppercase tracking-wider text-ink-600">backbone</th>
                <Th onClick={() => onSort("direct_top1")} active={sort === "direct_top1"} dir={dir} align="right">dir@1</Th>
                <Th onClick={() => onSort("indirect_top1")} active={sort === "indirect_top1"} dir={dir} align="right">ind@1</Th>
                <Th onClick={() => onSort("indirect_any")} active={sort === "indirect_any"} dir={dir} align="right">ind_any</Th>
                <Th onClick={() => onSort("retrieval_acc")} active={sort === "retrieval_acc"} dir={dir} align="right">ret_acc</Th>
                <Th onClick={() => onSort("ctx_tokens")} active={sort === "ctx_tokens"} dir={dir} align="right">ctx</Th>
                <Th onClick={() => onSort("storage_per_user_kb")} active={sort === "storage_per_user_kb"} dir={dir} align="right">KB/user</Th>
                <Th onClick={() => onSort("val_bpb_delta")} active={sort === "val_bpb_delta"} dir={dir} align="right">Δbpb</Th>
              </tr>
            </thead>
            <tbody>
              {visible?.map((r) => (
                <tr key={r.id}
                    className={`border-b border-ink-100 last:border-0 hover:bg-ink-50/50 ${
                      r.code === "F" || r.code === "J" ? "font-medium" : ""
                    }`}>
                  <td className="px-3 py-2">
                    {r.code && <span className="font-mono text-ink-500 mr-1.5">{r.code}</span>}
                    {r.method}
                  </td>
                  <td className="px-3 py-2">
                    <span className={`badge ${FAMILY_LABELS[r.family].color}`}>
                      {FAMILY_LABELS[r.family].label}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs text-ink-600">{r.backbone}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{pct(r.direct_top1)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{pct(r.indirect_top1)}</td>
                  <td className={`px-3 py-2 text-right tabular-nums ${r.code === "F" || r.code === "J" ? "text-accent-700 font-semibold" : ""}`}>
                    {pct(r.indirect_any)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">{pct(r.retrieval_acc)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{r.ctx_tokens != null ? Math.round(r.ctx_tokens) : "—"}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{r.storage_per_user_kb ? formatKB(r.storage_per_user_kb) : "—"}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{r.val_bpb_delta != null ? `${r.val_bpb_delta >= 0 ? "+" : ""}${r.val_bpb_delta.toFixed(3)}` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Th({ children, onClick, active, dir, align }: {
  children: React.ReactNode; onClick: () => void; active: boolean; dir: "asc" | "desc"; align?: "right";
}) {
  return (
    <th onClick={onClick}
        className={`px-3 py-2 text-xs font-semibold uppercase tracking-wider text-ink-600 cursor-pointer hover:text-ink-900 select-none ${align === "right" ? "text-right" : "text-left"}`}>
      {children}
      {active && <span className="text-accent-600 ml-1">{dir === "asc" ? "↑" : "↓"}</span>}
    </th>
  );
}

function pct(n?: number) {
  if (n == null) return "—";
  return `${(n * 100).toFixed(0)}%`;
}
function formatKB(kb: number) {
  if (kb >= 1024) return `${(kb / 1024).toFixed(1)} MB`;
  return `${kb.toFixed(0)} KB`;
}
