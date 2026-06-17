import { motion } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import { C } from "../../theme";
import { triggerRows } from "../../lib/hash";

/**
 * The recurring motif: a content-addressed memory table. Most cells stay dark
 * (a faint rule); writing a fact lights a sparse, deterministic set at one
 * address in slate-blue. In `ambient` mode it writes random facts on a loop
 * (used as the hero backdrop).
 */
export function MemoryGrid({
  cols = 26,
  rows = 16,
  lit = [],
  ambient = false,
  gap = 3,
  className = "",
}: {
  cols?: number;
  rows?: number;
  lit?: number[]; // flat indices (r*cols+c) to light
  ambient?: boolean;
  gap?: number;
  className?: string;
}) {
  const total = cols * rows;
  const [ambientLit, setAmbientLit] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (!ambient) return;
    const seeds = [
      "my cardiologist is dr",
      "favorite spice is",
      "office hours start at",
      "headquarters is in",
      "i was born in",
      "allergic to penicillin",
    ];
    let i = 0;
    const tick = () => {
      const s = seeds[i % seeds.length];
      i++;
      // map ~14 hashed addresses into a contiguous-ish band of the grid
      const addrs = triggerRows(s, total, 14);
      setAmbientLit(new Set(addrs));
    };
    tick();
    const iv = setInterval(tick, 2600);
    return () => clearInterval(iv);
  }, [ambient, total]);

  const litSet = useMemo(
    () => (ambient ? ambientLit : new Set(lit)),
    [ambient, ambientLit, lit],
  );

  const cellW = 100 / cols;
  const cellH = 100 / rows;

  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      className={className}
      aria-hidden
    >
      {Array.from({ length: total }).map((_, idx) => {
        const r = Math.floor(idx / cols);
        const c = idx % cols;
        const on = litSet.has(idx);
        return (
          <motion.rect
            key={idx}
            x={c * cellW + gap / 20}
            y={r * cellH + gap / 20}
            width={cellW - gap / 10}
            height={cellH - gap / 10}
            rx={0.4}
            initial={false}
            animate={{
              fill: on ? C.engram : C.rule,
              opacity: on ? 0.95 : 0.22,
            }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          />
        );
      })}
    </svg>
  );
}
