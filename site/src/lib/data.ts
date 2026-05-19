// Data loaders. All result JSONs live in /public/data/ and are fetched on demand.

import type {
  LayeredFull, RAGFull, ScaleFile, QwenScaleFile, MultihopFile,
  ComparisonRow,
} from "../types";

const BASE = import.meta.env.BASE_URL || "/";
const dpath = (name: string) => `${BASE}data/${name}`;

async function loadJSON<T>(file: string): Promise<T> {
  const r = await fetch(dpath(file));
  if (!r.ok) throw new Error(`Failed to load ${file}: ${r.status}`);
  return (await r.json()) as T;
}

export const loadLayered = () => loadJSON<LayeredFull>("layered_d20_r16_full.json");
export const loadRAG = () => loadJSON<RAGFull>("layered_rag_full.json");
export const loadRAGScale = () => loadJSON<ScaleFile>("layered_rag_scale_v2.json");
export const loadQwenScale = () => loadJSON<QwenScaleFile>("qwen_rag_scale_v2.json");
export const loadQwenSingleKB = () => loadJSON<any>("qwen_rag_full.json");
export const loadMultihop = () => loadJSON<MultihopFile>("multihop_rag.json");

export async function loadCSV(file: string): Promise<Record<string, string>[]> {
  const r = await fetch(dpath(file));
  const t = await r.text();
  const lines = t.trim().split("\n");
  const headers = lines[0].split(",");
  return lines.slice(1).map(line => {
    const cells = line.split(",");
    return Object.fromEntries(headers.map((h, i) => [h.trim(), cells[i]?.trim() ?? ""]));
  });
}

// Build the unified comparison table from all sources.
export async function loadComparisonTable(): Promise<ComparisonRow[]> {
  const [layered, rag, qwenSingle] = await Promise.all([
    loadLayered(),
    loadRAG(),
    loadQwenSingleKB(),
  ]);

  const rows: ComparisonRow[] = [];

  // A-F from layered_d20_r16_full.json
  const layeredCodes = [
    { code: "A", id: "A_no_edit", method: "No edit (baseline)", family: "baseline" as const },
    { code: "B", id: "B_per_user_lora", method: "Per-user LoRA r=64", family: "lora" as const },
    { code: "C", id: "C_per_user_engram", method: "Per-user Engram J-OPT", family: "engram" as const },
    { code: "D", id: "D_per_user_lora_plus_engram", method: "Per-user LoRA + Engram", family: "lora" as const },
    { code: "E", id: "E_shared_lora_only", method: "Shared LoRA r=16 only", family: "layered" as const },
    { code: "F", id: "F_layered", method: "Layered: Engram + shared LoRA", family: "layered" as const },
  ];
  for (const c of layeredCodes) {
    const a = layered.agg;
    rows.push({
      id: c.id,
      code: c.code,
      family: c.family,
      method: c.method,
      backbone: "Mini-Engram-d20 (1.22B)",
      direct_top1: a[`${c.id}_direct_top1`],
      direct_top5: a[`${c.id}_direct_top5`],
      indirect_top1: a[`${c.id}_indirect_top1`],
      indirect_any: a[`${c.id}_indirect_any`],
      ctx_tokens: 0,
      val_bpb_delta: a[`${c.id}_val_bpb_delta`],
      storage_per_user_kb: c.id === "B_per_user_lora" || c.id === "D_per_user_lora_plus_engram"
        ? 14200 : (c.id === "C_per_user_engram" || c.id === "F_layered" ? 88 : 0),
    });
  }

  // G-J from layered_rag_full.json
  const ragCodes = [
    { code: "G", id: "G_rag1", method: "RAG top-1", family: "rag" as const, ctx_label: 27 },
    { code: "G'", id: "G_oracle1", method: "Oracle top-1", family: "rag" as const, ctx_label: 28 },
    { code: "H", id: "H_rag3", method: "RAG top-3", family: "rag" as const, ctx_label: 44 },
    { code: "I", id: "I_ragall", method: "RAG all (markdown)", family: "rag" as const, ctx_label: 302 },
    { code: "J", id: "J_rag3_sharedLoRA", method: "RAG top-3 + shared LoRA", family: "rag" as const, ctx_label: 44 },
  ];
  for (const c of ragCodes) {
    const a = rag.agg;
    rows.push({
      id: c.id,
      code: c.code,
      family: c.family,
      method: c.method,
      backbone: "Mini-Engram-d20 (1.22B)",
      direct_top1: a[`${c.id}_direct_top1`],
      direct_top5: a[`${c.id}_direct_top5`],
      indirect_top1: a[`${c.id}_indirect_top1`],
      indirect_any: a[`${c.id}_indirect_any`],
      retrieval_acc: a[`${c.id}_retrieval_acc`],
      ctx_tokens: a[`${c.id}_indirect_ctx_tokens_avg`] ?? c.ctx_label,
      ms_per_query: a[`${c.id}_ms_per_indirect`],
      storage_per_user_kb: c.id === "J_rag3_sharedLoRA" ? 88 : 30,
    });
  }

  // Qwen-3B + RAG single-KB
  const qwenCodes = [
    { id: "NO_CONTEXT", method: "Qwen-3B, no context", ctx: 67 },
    { id: "RAG_TOP1", method: "Qwen-3B + RAG top-1", ctx: 82 },
    { id: "RAG_TOP3", method: "Qwen-3B + RAG top-3", ctx: 102 },
    { id: "RAG_ALL", method: "Qwen-3B + RAG all", ctx: 390 },
    { id: "ORACLE_TOP1", method: "Qwen-3B + oracle top-1", ctx: 83 },
  ];
  const qa = qwenSingle.agg ?? {};
  for (const c of qwenCodes) {
    rows.push({
      id: `qwen_${c.id}`,
      family: "qwen_rag",
      method: c.method,
      backbone: "Qwen2.5-3B-Instruct",
      indirect_top1: qa[`${c.id}_indirect_top1`],
      indirect_any: qa[`${c.id}_indirect_any`],
      retrieval_acc: qa[`${c.id}_retrieval_acc`],
      ctx_tokens: qa[`${c.id}_ctx_tokens_avg`] ?? c.ctx,
      ms_per_query: qa[`${c.id}_ms_per_query`],
      storage_per_user_kb: 30,
    });
  }

  return rows;
}

// Series for the Pareto chart: (x = context tokens, y = indirect_any).
export interface ParetoPoint {
  label: string;
  code?: string;
  family: string;
  x: number; // log-friendly ctx tokens (clamp small values)
  y: number; // 0..100 indirect_any %
  ms?: number;
}

export async function loadParetoPoints(): Promise<ParetoPoint[]> {
  const rows = await loadComparisonTable();
  const out: ParetoPoint[] = [];
  for (const r of rows) {
    const ind = r.indirect_any;
    if (ind == null) continue;
    const ctx = r.ctx_tokens ?? 0;
    out.push({
      label: r.code ? `${r.code}: ${r.method}` : r.method,
      code: r.code,
      family: r.family,
      x: Math.max(ctx, 0.5),
      y: ind * 100,
      ms: r.ms_per_query,
    });
  }
  return out;
}

// KB-scale data: combine layered_rag_scale_v2 + qwen_rag_scale_v2.
export interface KBPoint {
  series: string;
  family: string;
  kb_size: number;
  indirect_any: number;     // 0..100
  retrieval_acc: number;    // 0..100
  ctx_tokens: number;
}

export async function loadKBScale(): Promise<KBPoint[]> {
  const [mini, qwen, layered] = await Promise.all([
    loadRAGScale(),
    loadQwenScale(),
    loadLayered(),
  ]);
  const out: KBPoint[] = [];

  const miniConds = [
    { code: "G_rag1", label: "Mini-Engram + RAG top-1", family: "rag_mini" },
    { code: "H_rag3", label: "Mini-Engram + RAG top-3", family: "rag_mini" },
    { code: "J_rag3_sharedLoRA", label: "RAG-3 + shared LoRA (J)", family: "j" },
  ];
  for (const cond of miniConds) {
    for (const kb of mini.config.kb_sizes) {
      const key = `${cond.code}_kb${kb}`;
      const a = mini.agg[key];
      if (!a) continue;
      out.push({
        series: cond.label,
        family: cond.family,
        kb_size: kb,
        indirect_any: a.indirect_any * 100,
        retrieval_acc: a.retrieval_acc * 100,
        ctx_tokens: a.indirect_ctx_tokens_avg,
      });
    }
  }

  for (const k of qwen.config.top_ks) {
    for (const kb of qwen.config.kb_sizes) {
      const key = `qwen_rag${k}_kb${kb}`;
      const a = qwen.agg[key];
      if (!a) continue;
      out.push({
        series: `Qwen-3B + RAG top-${k}`,
        family: "qwen",
        kb_size: kb,
        indirect_any: a.indirect_any * 100,
        retrieval_acc: a.retrieval_acc * 100,
        ctx_tokens: a.indirect_ctx_tokens_avg,
      });
    }
  }

  // F (layered) baseline horizontal — emit one point per KB so the line is straight
  const F = layered.agg["F_layered_indirect_any"] * 100;
  for (const kb of mini.config.kb_sizes) {
    out.push({
      series: "F: layered (Engram + shared LoRA)",
      family: "f",
      kb_size: kb,
      indirect_any: F,
      retrieval_acc: 100, // not retrieval-bound
      ctx_tokens: 0,
    });
  }
  return out;
}

// Multi-hop summary as four rows.
export interface MultihopRow {
  label: string;
  family: string;
  top1: number; top5: number; ret_both: number; ctx_tokens: number;
}

export async function loadMultihopRows(): Promise<MultihopRow[]> {
  const m = await loadMultihop();
  const out: MultihopRow[] = [];
  // Engram references (constant 75% from the paper, all 3 sizes)
  out.push({ label: "Mini-Engram-d20 (Engram only)", family: "engram",
             top1: 0.75, top5: 0.75, ret_both: 1.0, ctx_tokens: 0 });
  for (const [k, v] of Object.entries(m.summary)) {
    out.push({
      label: k.replace(/_/g, " "),
      family: "rag",
      top1: v.top1,
      top5: v.top5,
      ret_both: v.ret_both,
      ctx_tokens: v.ctx_tokens_avg,
    });
  }
  return out;
}
