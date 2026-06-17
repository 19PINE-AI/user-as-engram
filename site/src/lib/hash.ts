// A toy multiplicative-XOR hash, used purely to *animate* the addressing idea
// (suffix N-gram -> a sparse, deterministic set of row addresses). Not the real
// Engram hash; just stable and well-spread for the visualization.
function fnvxor(s: string, salt: number): number {
  let h = 0x811c9dc5 ^ salt;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
    h ^= h >>> 15;
  }
  return h >>> 0;
}

/** Deterministic set of `k` row indices in [0, rows) for a trigger string. */
export function triggerRows(trigger: string, rows: number, k = 16): number[] {
  // Use the last few tokens as the "suffix N-gram".
  const toks = trigger.trim().split(/\s+/);
  const suffix = toks.slice(-3).join(" ").toLowerCase();
  const out = new Set<number>();
  let salt = 1;
  while (out.size < k && salt < 200) {
    out.add(fnvxor(suffix, salt) % rows);
    salt++;
  }
  return [...out];
}
