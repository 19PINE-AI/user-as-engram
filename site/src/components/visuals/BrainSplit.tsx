import { useState } from "react";
import { motion } from "motion/react";

const pairs = [
  {
    bio: "Hippocampus",
    bioSub: "fast · sparse · local trace (the engram)",
    arch: "Per-user Engram row",
    archSub: "88 KB · fires only on its trigger N-gram",
    side: "left" as const,
  },
  {
    bio: "Neocortex",
    bioSub: "slow · distributed · shared skills",
    arch: "Shared LoRA + frozen backbone",
    archSub: "one model everyone uses · learns to reason",
    side: "right" as const,
  },
];

export function BrainSplit() {
  const [hover, setHover] = useState<number | null>(null);
  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-8">
      <div className="grid md:grid-cols-2 gap-px bg-rule rounded-lg overflow-hidden">
        {pairs.map((p, i) => (
          <div
            key={p.bio}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
            className="bg-paper-50 p-6 transition-colors"
            style={{ background: hover === i ? "#fff" : undefined }}
          >
            <div className="font-mono text-[0.64rem] uppercase tracking-[0.18em] text-ink-50">
              the brain
            </div>
            <div className="display text-2xl md:text-3xl text-ink mt-1">{p.bio}</div>
            <div className="font-body text-sm text-ink-50 mt-1">{p.bioSub}</div>

            {/* connector */}
            <div className="my-4 flex items-center gap-2">
              <span className="h-px flex-1 bg-rule" />
              <motion.span
                className="font-mono text-[0.6rem] text-engram"
                animate={{ opacity: hover === i ? 1 : 0.4 }}
              >
                becomes
              </motion.span>
              <span className="h-px flex-1 bg-rule" />
            </div>

            <div className="font-mono text-[0.64rem] uppercase tracking-[0.18em] text-engram">
              user as engram
            </div>
            <div className="display text-2xl md:text-3xl text-engram-deep mt-1">{p.arch}</div>
            <div className="font-body text-sm text-ink-50 mt-1">{p.archSub}</div>
          </div>
        ))}
      </div>
      <p className="font-body text-sm text-ink-50 leading-snug mt-5 max-w-prose">
        Complementary learning systems, made architectural: a sparse local trace for content,
        a slow shared system for skill — so a new fact is written without overwriting how the
        model thinks.
      </p>
    </div>
  );
}
