import { useEffect, useState } from "react";
import {
  loadLayered, loadRAG, loadRAGScale, loadQwenScale, loadMultihop,
} from "../lib/data";

interface Dataset {
  id: string;
  title: string;
  description: string;
  file: string;
  loader: () => Promise<any>;
}

const DATASETS: Dataset[] = [
  { id: "layered", title: "Layered architecture (A–F)", description: "20 users × 6 conditions on Mini-Engram-d20. Direct + indirect + Δbpb.", file: "layered_d20_r16_full.json", loader: loadLayered },
  { id: "rag", title: "RAG conditions (G–J)", description: "20 users × 5 RAG conditions. Direct + indirect + retrieval accuracy.", file: "layered_rag_full.json", loader: loadRAG },
  { id: "scale", title: "KB-scale sweep, Mini-Engram", description: "20 users × {34,100,200,300,500,1000} KB × 3 conditions = 360 runs.", file: "layered_rag_scale_v2.json", loader: loadRAGScale },
  { id: "qwen-scale", title: "KB-scale sweep, Qwen-3B + RAG", description: "20 users × {34..1000} KB × top-1/top-3 = 240 runs.", file: "qwen_rag_scale_v2.json", loader: loadQwenScale },
  { id: "multihop", title: "Multi-hop RAG (8 chained pairs)", description: "Each pair: 2 facts in a 16-fact KB. RAG top-1/2/all and RAG-2 + shared LoRA.", file: "multihop_rag.json", loader: loadMultihop },
];

export function DataBrowser() {
  const [active, setActive] = useState<string>(DATASETS[0].id);
  return (
    <div>
      <h2>Data browser</h2>
      <p className="prose-paper text-ink-600">
        Every result JSON used in the paper. Pick a dataset on the left to
        explore per-user numbers, conditions, and KB sizes. Each file can also
        be downloaded via the link in the dataset header.
      </p>
      <div className="card mt-6 grid grid-cols-1 md:grid-cols-[260px_1fr] divide-y md:divide-y-0 md:divide-x divide-ink-200">
        <aside className="p-3">
          <ul className="space-y-1">
            {DATASETS.map(d => (
              <li key={d.id}>
                <button
                  onClick={() => setActive(d.id)}
                  className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
                    active === d.id
                      ? "bg-accent-50 text-accent-900 font-medium"
                      : "text-ink-700 hover:bg-ink-50"
                  }`}>
                  {d.title}
                </button>
              </li>
            ))}
          </ul>
        </aside>
        <div className="p-5 min-h-[400px]">
          <DatasetPanel dataset={DATASETS.find(d => d.id === active)!} />
        </div>
      </div>
    </div>
  );
}

function DatasetPanel({ dataset }: { dataset: Dataset }) {
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    setLoading(true);
    dataset.loader().then(d => { setData(d); setLoading(false); });
  }, [dataset.id]);

  if (loading || !data) return <div className="text-ink-500">Loading {dataset.file}…</div>;

  return (
    <div>
      <div className="flex items-baseline justify-between flex-wrap gap-2">
        <h3 className="!mt-0 !mb-1">{dataset.title}</h3>
        <a className="text-xs no-underline hover:underline" href={`${import.meta.env.BASE_URL || "/"}data/${dataset.file}`} download>
          ⬇ {dataset.file}
        </a>
      </div>
      <p className="text-sm text-ink-600">{dataset.description}</p>

      {dataset.id === "layered" && <LayeredView data={data} />}
      {dataset.id === "rag" && <RAGView data={data} />}
      {dataset.id === "scale" && <ScaleView data={data} />}
      {dataset.id === "qwen-scale" && <QwenScaleView data={data} />}
      {dataset.id === "multihop" && <MultihopView data={data} />}
    </div>
  );
}

function LayeredView({ data }: { data: any }) {
  const codes = ["A_no_edit", "B_per_user_lora", "C_per_user_engram", "D_per_user_lora_plus_engram", "E_shared_lora_only", "F_layered"];
  const labels: Record<string, string> = {
    A_no_edit: "A no edit", B_per_user_lora: "B LoRA", C_per_user_engram: "C Engram",
    D_per_user_lora_plus_engram: "D LoRA+Engram", E_shared_lora_only: "E shared LoRA", F_layered: "F layered",
  };
  return (
    <div className="mt-4">
      <h4 className="text-sm font-semibold text-ink-900 mb-2">Per-user indirect_any</h4>
      <ScrollTable
        headers={["uid", ...codes.map(c => labels[c])]}
        rows={data.per_user.map((u: any) => [
          u.uid,
          ...codes.map(c => `${u[c].indirect_any}/${u[c].indirect_total}`),
        ])}
      />
    </div>
  );
}

function RAGView({ data }: { data: any }) {
  const codes = ["G_rag1", "H_rag3", "I_ragall", "G_oracle1", "J_rag3_sharedLoRA"];
  return (
    <div className="mt-4">
      <h4 className="text-sm font-semibold text-ink-900 mb-2">Per-user indirect_any + retrieval@k</h4>
      <ScrollTable
        headers={["uid", ...codes.flatMap(c => [`${c} ind`, `${c} ret`])]}
        rows={data.per_user.map((u: any) => [
          u.uid,
          ...codes.flatMap(c => [
            `${u[c].indirect_any}/${u[c].indirect_total}`,
            `${(u[c].retrieval_acc * 100).toFixed(0)}%`,
          ]),
        ])}
      />
    </div>
  );
}

function ScaleView({ data }: { data: any }) {
  const [kb, setKB] = useState<number>(34);
  const kbs: number[] = data.config.kb_sizes;
  const runs = data.per_run.filter((r: any) => r.kb_size === kb);
  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 mb-3">
        <label className="text-xs text-ink-500 uppercase tracking-wider font-mono">KB</label>
        <select value={kb} onChange={e => setKB(parseInt(e.target.value))}
                className="border border-ink-300 rounded px-2 py-1 text-sm">
          {kbs.map(k => <option key={k} value={k}>{k}</option>)}
        </select>
        <span className="text-xs text-ink-500">{runs.length} runs</span>
      </div>
      <ScrollTable
        headers={["uid", "condition", "direct@1", "ind@1", "ind_any", "ret_acc", "ctx", "wall(s)"]}
        rows={runs.map((r: any) => [
          r.uid, r.condition,
          `${r.direct_top1}/${r.direct_total}`,
          `${r.indirect_top1}/${r.indirect_total}`,
          `${r.indirect_any}/${r.indirect_total}`,
          `${(r.retrieval_acc * 100).toFixed(0)}%`,
          Math.round(r.indirect_ctx_tokens_avg),
          r.wall_s.toFixed(1),
        ])}
      />
    </div>
  );
}

function QwenScaleView({ data }: { data: any }) {
  const [kb, setKB] = useState<number>(34);
  const kbs: number[] = data.config.kb_sizes;
  const runs = data.per_run.filter((r: any) => r.kb_size === kb);
  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 mb-3">
        <label className="text-xs text-ink-500 uppercase tracking-wider font-mono">KB</label>
        <select value={kb} onChange={e => setKB(parseInt(e.target.value))}
                className="border border-ink-300 rounded px-2 py-1 text-sm">
          {kbs.map(k => <option key={k} value={k}>{k}</option>)}
        </select>
      </div>
      <ScrollTable
        headers={["uid", "k", "ind@1", "ind_any", "ret_acc", "ctx", "wall(s)"]}
        rows={runs.map((r: any) => [
          r.uid, r.k,
          `${r.indirect_top1}/${r.indirect_total}`,
          `${r.indirect_any}/${r.indirect_total}`,
          `${(r.retrieval_acc * 100).toFixed(0)}%`,
          Math.round(r.indirect_ctx_tokens_avg),
          r.wall_s.toFixed(1),
        ])}
      />
    </div>
  );
}

function MultihopView({ data }: { data: any }) {
  return (
    <div className="mt-4">
      <h4 className="text-sm font-semibold text-ink-900 mb-2">Per-item outputs</h4>
      <ScrollTable
        headers={["#", "query", "expected", "RAG-1", "RAG-2", "RAG-all", "RAG-2 + sLoRA"]}
        rows={data.items.map((it: any, i: number) => [
          i + 1,
          it.query,
          it.expected.trim(),
          it.RAG_TOP1?.top1_text ?? "—",
          it.RAG_TOP2?.top1_text ?? "—",
          it.RAG_ALL?.top1_text ?? "—",
          it.RAG_TOP2_sharedLoRA?.top1_text ?? "—",
        ])}
      />
    </div>
  );
}

function ScrollTable({ headers, rows }: { headers: string[]; rows: any[][] }) {
  return (
    <div className="border border-ink-200 rounded-md overflow-auto max-h-[460px]">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-ink-50">
          <tr>
            {headers.map((h, i) => (
              <th key={i} className="text-left px-3 py-1.5 font-semibold text-ink-700 border-b border-ink-200 whitespace-nowrap">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-ink-100 last:border-0 odd:bg-white even:bg-ink-50/40">
              {row.map((c, j) => (
                <td key={j} className="px-3 py-1.5 tabular-nums whitespace-nowrap">{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
