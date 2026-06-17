import { motion } from "motion/react";
import { useMemo } from "react";
import { MemoryGrid } from "../visuals/MemoryGrid";
import { headline, PAPER_URL, CODE_URL } from "../../data/headline";
import { triggerRows } from "../../lib/hash";
import { EASE } from "../../theme";

export function Hero() {
  // a fixed, sparse "written" pattern — static, no animation loop
  const lit = useMemo(() => {
    const cols = 30,
      rows = 20,
      total = cols * rows;
    const seeds = ["my cardiologist is dr", "allergic to penicillin", "favorite spice is"];
    const s = new Set<number>();
    seeds.forEach((p) => triggerRows(p, total, 13).forEach((a) => s.add(a)));
    return [...s];
  }, []);

  return (
    <section id="top" className="relative min-h-screen flex flex-col justify-center overflow-hidden px-6 md:px-12 lg:px-20">
      {/* static memory table backdrop */}
      <div className="absolute inset-0 -z-10">
        <MemoryGrid lit={lit} cols={30} rows={20} className="w-full h-full" />
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(120% 90% at 16% 42%, rgba(244,240,230,0.98) 0%, rgba(244,240,230,0.9) 40%, rgba(244,240,230,0.62) 100%)",
          }}
        />
      </div>

      <div className="max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: EASE, delay: 0.1 }}
          className="kicker mb-5"
        >
          Per-user memory · a local parametric edit
        </motion.div>

        <h1 className="display text-[clamp(2.2rem,6vw,4.5rem)] text-ink leading-[0.95]">
          <motion.span
            className="block"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: EASE, delay: 0.15 }}
          >
            User as
          </motion.span>
          <motion.span
            className="block display-italic text-engram"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: EASE, delay: 0.26 }}
          >
            Engram
          </motion.span>
        </h1>

        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: EASE, delay: 0.4 }}
          className="lede mt-6 max-w-[44ch]"
        >
          A user's facts become a few rows in a content-addressed memory table —
          <span className="text-ink"> not a rewrite of the model.</span>
        </motion.p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.9, delay: 0.6 }}
          className="flex flex-wrap gap-x-9 gap-y-4 mt-9"
        >
          <Stat big={`${headline.indirectMeanRatio}×`} label="more indirect reasoning than a per-user LoRA" />
          <Stat big={`${headline.contamRatio.toLocaleString()}×`} label="less disruption to unrelated text" />
          <Stat big={`${headline.engramKB} KB`} label="per user · backbone untouched" />
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.9, delay: 0.78 }}
          className="flex items-center gap-3 mt-9"
        >
          <a
            href={PAPER_URL}
            className="font-mono text-[0.68rem] uppercase tracking-[0.18em] px-5 py-2.5 rounded-full bg-ink text-paper-50 hover:bg-engram transition-colors"
          >
            Read the paper ↗
          </a>
          <a
            href={CODE_URL}
            className="font-mono text-[0.68rem] uppercase tracking-[0.18em] px-5 py-2.5 rounded-full border hairline text-ink-100 hover:border-engram hover:text-engram transition-colors"
          >
            Code & weights
          </a>
        </motion.div>
      </div>

      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 font-mono text-[0.58rem] uppercase tracking-[0.22em] text-ink-50">
        scroll
      </div>
    </section>
  );
}

function Stat({ big, label }: { big: string; label: string }) {
  return (
    <div className="max-w-[11rem]">
      <div className="display text-3xl md:text-4xl text-engram-deep">{big}</div>
      <div className="font-body text-[0.82rem] text-ink-50 leading-snug mt-1">{label}</div>
    </div>
  );
}
