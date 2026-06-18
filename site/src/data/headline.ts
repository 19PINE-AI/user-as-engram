// SINGLE SOURCE OF TRUTH for the site's numbers.
// Transcribed from the current paper (post-2026-06-17 revisions) so the site
// cannot drift from the paper. Canonical-seed (S0) and 3-seed-mean both kept.

export const ARXIV_ID = "2606.19172";
export const PAPER_URL = `https://arxiv.org/abs/${ARXIV_ID}`;
export const CODE_URL = "https://github.com/bojieli/user-as-engram";

export const headline = {
  // Layered design vs per-user LoRA, indirect reasoning.
  indirectMeanRatio: 5.6, // 3-seed mean
  indirectBestRatio: 7.4, // canonical seed S0
  layeredIndirectPct: 44, // canonical seed
  loraIndirectPct: 6, // canonical seed
  layeredIndirectMeanPct: 41,
  loraIndirectMeanPct: 7,

  // Direct recall.
  layeredDirectPct: 100,
  loraDirectPct: 99,

  // Contamination on unrelated text (Δ val bits/byte).
  contamRatio: 33_000, // 3-seed mean (~34,000x on S0)
  loraBpb: 1.78,
  engramBpb: 0.00005,
  layeredBpb: 0.386,

  // Per-user reasoning regressions vs the untouched base.
  usersWorseLayered: 0, // out of 60 (3 seeds x 20)
  usersWorseLora: 49,
  usersTotal: 60,

  // Storage.
  engramKB: 88,
  loraMB: 14.2,
  storageRatio: 161, // at 100 facts/user

  // Deployment vs retrieval.
  ragCrossoverFacts: 100,
  largerBackbone: 2.5, // Qwen2.5-3B vs Mini-Engram-d20

  // Mechanism (glass box, trained Mini-Engram-d20).
  gateBefore: 0.02,
  gateAfter: 0.99,
  gateNonTrigger: 0.04,
  valuePathCosine: 0.999,
  offTriggerDelta: 0, // exact
  recallLateLayer: 1.0,
  recallEarlyLayer: 0.25,

  // Serving.
  reqPerSec: 226,
  crossUserLeak: 0, // by construction

  // Checkpoints (total params).
  checkpoints: ["178M", "339M", "625M", "1.22B"] as const,
} as const;
