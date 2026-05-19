import { useEffect, useMemo, useState } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend, Cell,
} from "recharts";
import { loadParetoPoints, type ParetoPoint } from "../lib/data";

const FAMILY_COLORS: Record<string, string> = {
  baseline: "#94a3b8",
  engram: "#10b981",
  lora: "#ef4444",
  layered: "#3b82f6",
  rag: "#f59e0b",
  qwen_rag: "#8b5cf6",
};

const FAMILY_LABELS: Record<string, string> = {
  baseline: "No edit",
  engram: "Engram (no shared LoRA)",
  lora: "Per-user LoRA",
  layered: "Layered (Engram + shared LoRA)",
  rag: "Mini-Engram + RAG",
  qwen_rag: "Qwen-3B + RAG",
};

export function ParetoSection() {
  const [points, setPoints] = useState<ParetoPoint[] | null>(null);
  const [enabled, setEnabled] = useState<Record<string, boolean>>(
    Object.fromEntries(Object.keys(FAMILY_COLORS).map(k => [k, true])));

  useEffect(() => { loadParetoPoints().then(setPoints); }, []);

  const filtered = useMemo(() => {
    if (!points) return null;
    const map: Record<string, ParetoPoint[]> = {};
    for (const p of points) {
      if (!enabled[p.family]) continue;
      (map[p.family] ||= []).push(p);
    }
    return map;
  }, [points, enabled]);

  if (!points) return <div className="text-ink-500">Loading Pareto data…</div>;

  const toggle = (k: string) => setEnabled(e => ({ ...e, [k]: !e[k] }));

  return (
    <div>
      <h2>Pareto: indirect_any vs context tokens</h2>
      <p className="prose-paper text-ink-600">
        Each point is one method on the same 20-user × 20-probe split. The x-axis
        is the per-query context cost (log scale); the y-axis is
        indirect-reasoning accuracy (gold substring in 16-token greedy
        continuation). Toggle method families on/off. The interesting region
        is the upper-left (high accuracy, low context cost): F and E anchor it.
      </p>

      <div className="card mt-6 p-4">
        <div className="flex flex-wrap gap-2 mb-3">
          {Object.entries(FAMILY_LABELS).map(([k, label]) => (
            <button key={k} onClick={() => toggle(k)}
                    className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                      enabled[k] ? "border-ink-300 bg-white" : "border-ink-200 bg-ink-50 text-ink-400"
                    }`}>
              <span className="inline-block w-2.5 h-2.5 rounded-full mr-1.5 align-middle"
                    style={{ background: FAMILY_COLORS[k] }} />
              {label}
            </button>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={420}>
          <ScatterChart margin={{ top: 10, right: 30, bottom: 36, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              type="number" dataKey="x" scale="log" domain={[0.3, 600]}
              ticks={[0.5, 1, 10, 100, 500]}
              tickFormatter={(v) => v < 1 ? "0" : String(v)}
              label={{ value: "Avg context tokens per query (log)", position: "insideBottom", offset: -20, style: { fontSize: 12, fill: "#475569" } }}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              type="number" dataKey="y" domain={[0, 65]} tickCount={7}
              label={{ value: "indirect_any (%)", angle: -90, position: "insideLeft", style: { fontSize: 12, fill: "#475569" } }}
              tick={{ fontSize: 11 }}
            />
            <Tooltip content={<ParetoTip />} />
            <Legend verticalAlign="top" height={0} />
            <ReferenceLine y={44.5} stroke="#3b82f6" strokeDasharray="6 4"
                           label={{ value: "F = 44.5%", position: "insideTopRight", fill: "#3b82f6", fontSize: 11 }} />
            {filtered && Object.entries(filtered).map(([family, pts]) => (
              <Scatter key={family} name={FAMILY_LABELS[family]} data={pts}
                       fill={FAMILY_COLORS[family]}>
                {pts.map((_, i) => (
                  <Cell key={i} />
                ))}
              </Scatter>
            ))}
          </ScatterChart>
        </ResponsiveContainer>

        <p className="text-xs text-ink-500 mt-2">
          F (layered) sits on the 0-context Pareto frontier. J (RAG-top3 + shared
          LoRA, 44 ctx) and Qwen-3B + RAG (82–390 ctx) buy a few accuracy points
          at substantial context cost. These numbers are at KB=34 facts/user;
          the comparison sharpens at scale — see the <a href="#scale">KB-Scale</a> section.
        </p>
      </div>
    </div>
  );
}

function ParetoTip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d: ParetoPoint = payload[0].payload;
  return (
    <div className="bg-white border border-ink-200 rounded-md px-3 py-2 text-xs shadow-sm">
      <div className="font-medium text-ink-900">{d.label}</div>
      <div className="text-ink-600 mt-1">
        ctx ≈ {Math.round(d.x)} tokens · indirect_any = {d.y.toFixed(1)}%
      </div>
      {d.ms != null && (
        <div className="text-ink-500 text-[10px] mt-0.5">{Math.round(d.ms)} ms/query</div>
      )}
    </div>
  );
}
