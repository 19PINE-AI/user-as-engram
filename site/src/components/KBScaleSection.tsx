import { useEffect, useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine,
} from "recharts";
import { loadKBScale, type KBPoint } from "../lib/data";

const SERIES_COLORS: Record<string, string> = {
  "Mini-Engram + RAG top-1": "#fb923c",
  "Mini-Engram + RAG top-3": "#ef4444",
  "RAG-3 + shared LoRA (J)": "#a855f7",
  "Qwen-3B + RAG top-1": "#10b981",
  "Qwen-3B + RAG top-3": "#84cc16",
  "F: layered (Engram + shared LoRA)": "#3b82f6",
};

const KB_SIZES = [34, 100, 200, 300, 500, 1000];

export function KBScaleSection() {
  const [data, setData] = useState<KBPoint[] | null>(null);
  const [kbIndex, setKbIndex] = useState(0); // 0..5
  const [mode, setMode] = useState<"chart" | "bars">("chart");

  useEffect(() => { loadKBScale().then(setData); }, []);

  const seriesData = useMemo(() => {
    if (!data) return [];
    const seriesMap: Record<string, { kb: number; ind: number; ret: number }[]> = {};
    for (const p of data) {
      (seriesMap[p.series] ||= []).push({ kb: p.kb_size, ind: p.indirect_any, ret: p.retrieval_acc });
    }
    return Object.entries(seriesMap).map(([series, pts]) => ({
      series,
      pts: pts.sort((a, b) => a.kb - b.kb),
    }));
  }, [data]);

  // Long-form for recharts line chart
  const lineData = useMemo(() => {
    return KB_SIZES.map(kb => {
      const row: any = { kb };
      for (const s of seriesData) {
        const pt = s.pts.find(p => p.kb === kb);
        if (pt) row[s.series] = pt.ind;
      }
      return row;
    });
  }, [seriesData]);

  // Snapshot at selected KB for the bar chart
  const currentKB = KB_SIZES[kbIndex];
  const barData = useMemo(() => {
    return seriesData.map(s => {
      const pt = s.pts.find(p => p.kb === currentKB);
      return {
        series: s.series,
        indirect_any: pt?.ind ?? null,
        retrieval_acc: pt?.ret ?? null,
        color: SERIES_COLORS[s.series] ?? "#475569",
      };
    }).filter(d => d.indirect_any != null);
  }, [seriesData, currentKB]);

  if (!data) return <div className="text-ink-500">Loading KB-scale data…</div>;

  return (
    <div>
      <h2>RAG vs Engram across KB sizes</h2>
      <p className="prose-paper text-ink-600">
        We augment each test user's 34 facts with sampled distractors from a
        pool of 1008 facts (30 schema-family users) to reach KB sizes
        N ∈ {`{`}34, 100, 200, 300, 500, 1000{`}`}. F (layered) is invariant in this
        axis because per-user Engram tables don't grow with population size.
        Retrieval-based methods degrade monotonically; <strong>Qwen-3B + RAG-top-3
        falls from 52% at KB=34 to 30% at KB=1000 — 14 pp below F</strong>.
      </p>

      <div className="card mt-6">
        <div className="border-b border-ink-200 px-4 py-2 flex flex-wrap items-center gap-3 text-sm">
          <div className="flex rounded-md overflow-hidden border border-ink-200">
            <button onClick={() => setMode("chart")}
                    className={`px-3 py-1.5 ${mode === "chart" ? "bg-ink-900 text-white" : "bg-white text-ink-700"}`}>
              Trend
            </button>
            <button onClick={() => setMode("bars")}
                    className={`px-3 py-1.5 ${mode === "bars" ? "bg-ink-900 text-white" : "bg-white text-ink-700"}`}>
              Snapshot @ KB = {currentKB}
            </button>
          </div>
          {mode === "bars" && (
            <div className="flex-1 min-w-[260px]">
              <label className="text-xs text-ink-500 font-mono uppercase tracking-wider">KB size</label>
              <div className="flex items-center gap-2">
                <input type="range" min={0} max={KB_SIZES.length - 1} step={1}
                       value={kbIndex} onChange={e => setKbIndex(parseInt(e.target.value))}
                       className="w-full" />
                <span className="font-mono text-xs w-16 text-right">{currentKB}</span>
              </div>
            </div>
          )}
        </div>

        <div className="p-4">
          {mode === "chart" ? (
            <ResponsiveContainer width="100%" height={420}>
              <LineChart data={lineData} margin={{ top: 10, right: 24, bottom: 24, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="kb" type="number" scale="log" domain={[30, 1100]}
                       ticks={KB_SIZES} tick={{ fontSize: 11 }}
                       label={{ value: "KB size (per-user facts + distractors, log)", position: "insideBottom", offset: -10, style: { fontSize: 12, fill: "#475569" } }} />
                <YAxis domain={[0, 65]} tickCount={7} tick={{ fontSize: 11 }}
                       label={{ value: "indirect_any (%)", angle: -90, position: "insideLeft", style: { fontSize: 12, fill: "#475569" } }} />
                <Tooltip content={<LineTip />} />
                {seriesData.map(s => (
                  <Line key={s.series}
                        type="monotone"
                        dataKey={s.series}
                        stroke={SERIES_COLORS[s.series] ?? "#475569"}
                        strokeWidth={s.series.startsWith("F:") ? 3 : 2}
                        dot={{ r: 4 }}
                        connectNulls
                        isAnimationActive
                  />
                ))}
                <ReferenceLine x={100} stroke="#94a3b8" strokeDasharray="3 3"
                               label={{ value: "Cross-over ≈ N=100", position: "top", fill: "#64748b", fontSize: 10 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={barData} layout="vertical"
                        margin={{ top: 10, right: 30, bottom: 24, left: 200 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" domain={[0, 65]} tick={{ fontSize: 11 }}
                       label={{ value: "indirect_any (%)", position: "insideBottom", offset: -8, style: { fontSize: 12, fill: "#475569" } }} />
                <YAxis type="category" dataKey="series" width={200}
                       tick={{ fontSize: 11 }} />
                <Tooltip content={<BarTip />} />
                <Bar dataKey="indirect_any" radius={[0, 4, 4, 0]} isAnimationActive>
                  {barData.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
                <ReferenceLine x={44.5} stroke="#3b82f6" strokeDasharray="5 3"
                               label={{ value: "F = 44.5%", position: "top", fill: "#3b82f6", fontSize: 11 }} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
        <div className="border-t border-ink-200 px-4 py-3 bg-ink-50/50">
          <div className="flex flex-wrap gap-3 text-xs">
            {Object.entries(SERIES_COLORS).map(([series, color]) => (
              <span key={series} className="inline-flex items-center gap-1.5 text-ink-700">
                <span className="inline-block w-3 h-3 rounded" style={{ background: color }} />
                {series}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function LineTip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-ink-200 rounded-md px-3 py-2 text-xs shadow-sm">
      <div className="font-mono text-ink-500">KB = {label}</div>
      {payload
        .slice()
        .sort((a: any, b: any) => (b.value ?? 0) - (a.value ?? 0))
        .map((p: any) => (
          <div key={p.dataKey} className="flex items-center gap-2 mt-0.5">
            <span className="inline-block w-2 h-2 rounded-sm" style={{ background: p.color }} />
            <span className="text-ink-700 flex-1">{p.dataKey}</span>
            <span className="tabular-nums font-medium text-ink-900">{p.value?.toFixed(1)}%</span>
          </div>
        ))}
    </div>
  );
}
function BarTip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-ink-200 rounded-md px-3 py-2 text-xs shadow-sm">
      <div className="font-medium text-ink-900">{d.series}</div>
      <div className="text-ink-700 mt-1">indirect_any = {d.indirect_any.toFixed(1)}%</div>
      <div className="text-ink-500 mt-0.5">retrieval@k = {d.retrieval_acc?.toFixed(1)}%</div>
    </div>
  );
}
