// Six parametric conditions — descriptive names (no single-letter tags),
// values from fig_layered_conditions (Mini-Engram-d20, seed S0).
export type Condition = {
  name: string;
  short: string;
  direct: number; // direct top-1 recall (%)
  indirect: number; // indirect_any (%)
  bpb: number; // Δ val bits/byte on unrelated text
  ours?: boolean; // the proposed layered design
  cost?: "high" | "none" | "shared"; // contamination character
};

export const conditions: Condition[] = [
  { name: "Untouched base", short: "base", direct: 29, indirect: 19, bpb: 0.0, cost: "none" },
  { name: "Per-user LoRA", short: "per-user LoRA", direct: 99, indirect: 6, bpb: 1.784, cost: "high" },
  { name: "Per-user Engram", short: "per-user Engram", direct: 100, indirect: 23, bpb: 0.00005, cost: "none" },
  { name: "LoRA + Engram stack", short: "LoRA+Engram", direct: 100, indirect: 8, bpb: 1.819, cost: "high" },
  { name: "Shared LoRA alone", short: "shared LoRA", direct: 54, indirect: 44, bpb: 0.386, cost: "shared" },
  {
    name: "Layered design",
    short: "Engram + shared LoRA",
    direct: 100,
    indirect: 44,
    bpb: 0.386,
    ours: true,
    cost: "shared",
  },
];
