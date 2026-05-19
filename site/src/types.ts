// Shared types for paper-data shape.

export interface LayeredAgg {
  n_users: number;
  // Conditions A-F each have direct_top1, direct_top5, indirect_top1, indirect_any, val_bpb_delta, val_bpb_delta_gt0
  [k: string]: number;
}

export interface LayeredFull {
  config: { ckpt_dir: string; test_uids: string[] };
  baseline_bpb: number;
  shared_only_bpb: number;
  per_user: PerUserLayered[];
  agg: LayeredAgg;
}

export interface PerUserLayered {
  uid: string;
  n_facts: number;
  n_probes: number;
  A_no_edit: ConditionRecord;
  B_per_user_lora: ConditionRecord;
  C_per_user_engram: ConditionRecord;
  D_per_user_lora_plus_engram: ConditionRecord;
  E_shared_lora_only: ConditionRecord;
  F_layered: ConditionRecord;
}

export interface ConditionRecord {
  direct_top1: number;
  direct_top5: number;
  direct_total: number;
  indirect_top1: number;
  indirect_any: number;
  indirect_total: number;
  val_bpb: number;
  val_bpb_delta: number;
  wall_s?: number;
}

// layered_rag_full.json
export interface RAGRunRecord {
  label: string;
  direct_top1: number;
  direct_top5: number;
  direct_total: number;
  indirect_top1: number;
  indirect_any: number;
  indirect_total: number;
  retrieval_acc: number;
  indirect_ctx_tokens_avg: number;
  direct_ctx_tokens_avg: number;
  indirect_wall_per_query_ms: number;
}

export interface PerUserRAG {
  uid: string;
  n_facts: number;
  n_probes: number;
  G_rag1: RAGRunRecord;
  H_rag3: RAGRunRecord;
  I_ragall: RAGRunRecord;
  G_oracle1: RAGRunRecord;
  J_rag3_sharedLoRA: RAGRunRecord;
}

export interface RAGFull {
  config: any;
  per_user: PerUserRAG[];
  agg: Record<string, number>;
}

// layered_rag_scale_v2.json - per_run array
export interface ScaleRun {
  uid: string;
  kb_size: number;
  condition: string;
  label: string;
  direct_top1: number;
  direct_top5: number;
  direct_total: number;
  indirect_top1: number;
  indirect_any: number;
  indirect_total: number;
  retrieval_acc: number;
  indirect_ctx_tokens_avg: number;
  wall_s: number;
}

export interface ScaleFile {
  config: any;
  per_run: ScaleRun[];
  agg: Record<string, ScaleAggEntry>;
}

export interface ScaleAggEntry {
  kb_size: number;
  condition: string;
  n_users: number;
  direct_top1: number;
  direct_top5: number;
  indirect_top1: number;
  indirect_any: number;
  retrieval_acc: number;
  indirect_ctx_tokens_avg: number;
}

// qwen_rag_full.json
export interface QwenRAGAgg {
  n_users: number;
  [k: string]: number;
}

// qwen_rag_scale_v2.json
export interface QwenScaleRun {
  uid: string;
  kb_size: number;
  k: number;
  indirect_top1: number;
  indirect_any: number;
  indirect_total: number;
  retrieval_acc: number;
  indirect_ctx_tokens_avg: number;
  wall_s: number;
}

export interface QwenScaleAggEntry {
  kb_size: number;
  k: number;
  n_users: number;
  indirect_top1: number;
  indirect_any: number;
  retrieval_acc: number;
  indirect_ctx_tokens_avg: number;
}

export interface QwenScaleFile {
  config: any;
  per_run: QwenScaleRun[];
  agg: Record<string, QwenScaleAggEntry>;
}

// multihop_rag.json
export interface MultihopRAGSummary {
  top1: number;
  top5: number;
  ret_both: number;
  ctx_tokens_avg: number;
  ms_per_query: number;
}

export interface MultihopFile {
  config: any;
  items: any[];
  summary: Record<string, MultihopRAGSummary>;
}

// One row in the unified comparison table.
export interface ComparisonRow {
  id: string;
  code?: string;
  family: 'baseline' | 'engram' | 'lora' | 'rag' | 'layered' | 'qwen_rag';
  method: string;
  backbone: string;
  direct_top1?: number;
  direct_top5?: number;
  indirect_top1?: number;
  indirect_any?: number;
  retrieval_acc?: number;
  ctx_tokens?: number;
  ms_per_query?: number;
  storage_per_user_kb?: number;
  val_bpb_delta?: number;
  notes?: string;
}
