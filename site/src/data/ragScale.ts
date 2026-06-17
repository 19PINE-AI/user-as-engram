// KB-size sweep, mirroring fig_rag_scale (indirect accuracy + retrieval recall)
// across KB = test user's 34 facts + distractors. Values track the paper figure.
export const kbSizes = [34, 100, 200, 300, 500, 1000] as const;

export type Series = {
  key: string;
  label: string;
  color: "engram" | "engramLight" | "rust" | "rustLight" | "ochre";
  dashed?: boolean;
  flat?: boolean;
  vals: number[]; // indirect accuracy (%) at each KB size
};

// indirect-reasoning accuracy (%)
export const accuracySeries: Series[] = [
  { key: "layered", label: "Layered (no retrieval)", color: "engram", flat: true, vals: [44, 44, 44, 44, 44, 44] },
  { key: "j", label: "RAG + shared LoRA", color: "engramLight", vals: [54, 52, 51, 51, 48, 47] },
  { key: "qwen3", label: "Qwen-3B + RAG", color: "ochre", dashed: true, vals: [52, 44.5, 38, 36.5, 32, 31.5] },
  { key: "rag3", label: "naive RAG", color: "rust", dashed: true, vals: [38.5, 34.5, 31, 30, 29.5, 29.5] },
];

// fraction of probes where retrieval covers every fact the question needs (%)
export const retrievalRecall = {
  top3: [62, 39, 27, 19, 11, 9],
  top1: [22, 13, 9, 8, 7, 6],
};
