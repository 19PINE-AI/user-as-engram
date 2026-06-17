import { C } from "../../theme";

/**
 * The recurring motif: a content-addressed memory table. Most cells stay a faint
 * rule; a sparse, deterministic set is "written" in slate-blue. Fully static.
 */
export function MemoryGrid({
  cols = 26,
  rows = 16,
  lit = [],
  gap = 3,
  className = "",
}: {
  cols?: number;
  rows?: number;
  lit?: number[];
  gap?: number;
  className?: string;
}) {
  const total = cols * rows;
  const litSet = new Set(lit);
  const cellW = 100 / cols;
  const cellH = 100 / rows;
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className={className} aria-hidden>
      {Array.from({ length: total }).map((_, idx) => {
        const r = Math.floor(idx / cols);
        const c = idx % cols;
        const on = litSet.has(idx);
        return (
          <rect
            key={idx}
            x={c * cellW + gap / 20}
            y={r * cellH + gap / 20}
            width={cellW - gap / 10}
            height={cellH - gap / 10}
            rx={0.4}
            fill={on ? C.engram : C.rule}
            opacity={on ? 0.9 : 0.22}
          />
        );
      })}
    </svg>
  );
}
