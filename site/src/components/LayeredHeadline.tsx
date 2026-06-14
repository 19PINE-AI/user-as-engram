import { useEffect, useState } from "react";
import { loadLayered, loadRAG } from "../lib/data";
import type { LayeredFull, RAGFull } from "../types";

interface Row {
  code: string;
  bold?: boolean;
  group: "layered" | "rag";
  method: string;
  direct1?: number;
  direct5?: number;
  ind1?: number;
  indAny?: number;
  bpb?: number;
  bpbWorse?: number; // 0..1
  ctx: number;
}

const COLS = ["code", "method", "direct1", "direct5", "ind1", "indAny", "bpb", "ctx"] as const;
type Col = typeof COLS[number];
const COL_LABEL: Record<Col, string> = {
  code: "code", method: "method",
  direct1: "direct@1", direct5: "direct@5",
  ind1: "ind@1", indAny: "ind_any",
  bpb: "Δbpb (worse/20)", ctx: "ctx",
};

function pct(n?: number) {
  if (n == null) return "—";
  return `${(n * 100).toFixed(0)}%`;
}
function fmtBpb(d?: number, w?: number) {
  if (d == null) return "—";
  const sign = d >= 0 ? "+" : "";
  const wf = w == null ? "" : ` (${Math.round(w * 20)}/20)`;
  return `${sign}${d.toFixed(3)}${wf}`;
}

export function LayeredHeadline() {
  const [data, setData] = useState<{ layered: LayeredFull; rag: RAGFull } | null>(null);
  const [sort, setSort] = useState<Col>("indAny");
  const [dir, setDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    Promise.all([loadLayered(), loadRAG()]).then(([layered, rag]) => setData({ layered, rag }));
  }, []);

  if (!data) return <div className="text-ink-500">Loading layered table…</div>;
  const { layered, rag } = data;
  const a = layered.agg, ra = rag.agg;

  const rows: Row[] = [
    { code: "A", group: "layered", method: "no edit", direct1: a.A_no_edit_direct_top1, direct5: a.A_no_edit_direct_top5, ind1: a.A_no_edit_indirect_top1, indAny: a.A_no_edit_indirect_any, bpb: 0, bpbWorse: 0, ctx: 0 },
    { code: "B", group: "layered", method: "per-user LoRA r=64", direct1: a.B_per_user_lora_direct_top1, direct5: a.B_per_user_lora_direct_top5, ind1: a.B_per_user_lora_indirect_top1, indAny: a.B_per_user_lora_indirect_any, bpb: a.B_per_user_lora_val_bpb_delta, bpbWorse: a.B_per_user_lora_val_bpb_delta_gt0, ctx: 0 },
    { code: "C", group: "layered", method: "per-user Engram J-OPT", direct1: a.C_per_user_engram_direct_top1, direct5: a.C_per_user_engram_direct_top5, ind1: a.C_per_user_engram_indirect_top1, indAny: a.C_per_user_engram_indirect_any, bpb: a.C_per_user_engram_val_bpb_delta, bpbWorse: a.C_per_user_engram_val_bpb_delta_gt0, ctx: 0 },
    { code: "D", group: "layered", method: "per-user LoRA + Engram", direct1: a.D_per_user_lora_plus_engram_direct_top1, direct5: a.D_per_user_lora_plus_engram_direct_top5, ind1: a.D_per_user_lora_plus_engram_indirect_top1, indAny: a.D_per_user_lora_plus_engram_indirect_any, bpb: a.D_per_user_lora_plus_engram_val_bpb_delta, bpbWorse: a.D_per_user_lora_plus_engram_val_bpb_delta_gt0, ctx: 0 },
    { code: "E", group: "layered", method: "shared LoRA r=16 only", direct1: a.E_shared_lora_only_direct_top1, direct5: a.E_shared_lora_only_direct_top5, ind1: a.E_shared_lora_only_indirect_top1, indAny: a.E_shared_lora_only_indirect_any, bpb: a.E_shared_lora_only_val_bpb_delta, bpbWorse: a.E_shared_lora_only_val_bpb_delta_gt0, ctx: 0 },
    { code: "F", bold: true, group: "layered", method: "Engram + shared LoRA", direct1: a.F_layered_direct_top1, direct5: a.F_layered_direct_top5, ind1: a.F_layered_indirect_top1, indAny: a.F_layered_indirect_any, bpb: a.F_layered_val_bpb_delta, bpbWorse: a.F_layered_val_bpb_delta_gt0, ctx: 0 },
    { code: "G", group: "rag", method: "RAG top-1", direct1: ra.G_rag1_direct_top1, direct5: ra.G_rag1_direct_top5, ind1: ra.G_rag1_indirect_top1, indAny: ra.G_rag1_indirect_any, ctx: ra.G_rag1_indirect_ctx_tokens_avg },
    { code: "G'", group: "rag", method: "oracle top-1", direct1: ra.G_oracle1_direct_top1, direct5: ra.G_oracle1_direct_top5, ind1: ra.G_oracle1_indirect_top1, indAny: ra.G_oracle1_indirect_any, ctx: ra.G_oracle1_indirect_ctx_tokens_avg },
    { code: "H", group: "rag", method: "RAG top-3", direct1: ra.H_rag3_direct_top1, direct5: ra.H_rag3_direct_top5, ind1: ra.H_rag3_indirect_top1, indAny: ra.H_rag3_indirect_any, ctx: ra.H_rag3_indirect_ctx_tokens_avg },
    { code: "I", group: "rag", method: "RAG all (markdown)", direct1: ra.I_ragall_direct_top1, direct5: ra.I_ragall_direct_top5, ind1: ra.I_ragall_indirect_top1, indAny: ra.I_ragall_indirect_any, ctx: ra.I_ragall_indirect_ctx_tokens_avg },
    { code: "J", bold: true, group: "rag", method: "RAG top-3 + shared LoRA", direct1: ra.J_rag3_sharedLoRA_direct_top1, direct5: ra.J_rag3_sharedLoRA_direct_top5, ind1: ra.J_rag3_sharedLoRA_indirect_top1, indAny: ra.J_rag3_sharedLoRA_indirect_any, ctx: ra.J_rag3_sharedLoRA_indirect_ctx_tokens_avg },
  ];

  const sortKey = (r: Row): number | string => {
    if (sort === "code") return r.code;
    if (sort === "method") return r.method;
    if (sort === "ctx") return r.ctx;
    if (sort === "bpb") return r.bpb ?? -1;
    return (r as any)[sort] ?? -1;
  };
  const sorted = [...rows].sort((a, b) => {
    const av = sortKey(a), bv = sortKey(b);
    if (av === bv) return 0;
    if (typeof av === "string" || typeof bv === "string")
      return dir === "asc" ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    return dir === "asc" ? (av as number) - (bv as number) : (bv as number) - (av as number);
  });

  const onSort = (c: Col) => {
    if (c === sort) setDir(d => d === "asc" ? "desc" : "asc");
    else { setSort(c); setDir(c === "code" || c === "method" ? "asc" : "desc"); }
  };

  return (
    <div>
      <h2>Layered architecture: 10 conditions × 20 users</h2>
      <p className="prose-paper text-ink-600">
        All conditions on the same Mini-Engram-d20 base, same 20 test users and
        20 indirect-reasoning probes. Click column headers to sort. The
        head-to-head story: <strong>F (0 ctx)</strong> matches per-user LoRA's
        direct recall while delivering 7.4× higher indirect_any;{" "}
        <strong>J (44 ctx)</strong> adds RAG on top to push indirect_any to 54%.
        Naive RAG without the shared LoRA (G/H/I) stays below F.
      </p>
      <div className="card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-ink-200 bg-ink-50/60">
              {COLS.map(c => (
                <th key={c}
                    className="text-left px-3 py-2 text-xs font-semibold uppercase tracking-wider text-ink-600 cursor-pointer hover:text-ink-900"
                    onClick={() => onSort(c)}>
                  {COL_LABEL[c]}
                  {sort === c && <span className="text-accent-600 ml-1">{dir === "asc" ? "↑" : "↓"}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => {
              const isF = r.code === "F";
              const isJ = r.code === "J";
              const rowBg = r.group === "rag" ? "bg-orange-50/30" : "";
              return (
                <tr key={i}
                    className={`border-b border-ink-100 last:border-0 ${rowBg} ${(isF || isJ) ? "font-medium" : ""}`}>
                  <td className="px-3 py-2 font-mono">
                    <span className={`badge ${r.group === "rag" ? "badge-orange" : "badge-blue"}`}>
                      {r.code}
                    </span>
                  </td>
                  <td className="px-3 py-2">{r.method}</td>
                  <td className="px-3 py-2 tabular-nums">{pct(r.direct1)}</td>
                  <td className="px-3 py-2 tabular-nums">{pct(r.direct5)}</td>
                  <td className="px-3 py-2 tabular-nums">{pct(r.ind1)}</td>
                  <td className={`px-3 py-2 tabular-nums ${isF || isJ ? "text-accent-700 font-semibold" : ""}`}>
                    {pct(r.indAny)}
                  </td>
                  <td className="px-3 py-2 tabular-nums">{fmtBpb(r.bpb, r.bpbWorse)}</td>
                  <td className="px-3 py-2 tabular-nums">{r.ctx ? Math.round(r.ctx) : 0}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Legend />
    </div>
  );
}

function Legend() {
  return (
    <div className="mt-4 flex flex-wrap gap-2 text-xs text-ink-600">
      <span><span className="badge badge-blue mr-1">A–F</span>parametric substrates (no retrieval)</span>
      <span><span className="badge badge-orange mr-1">G–J</span>retrieval-augmented (this work)</span>
    </div>
  );
}
