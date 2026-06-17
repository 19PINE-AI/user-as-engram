import { motion } from "motion/react";
import { MemoryGrid } from "../visuals/MemoryGrid";
import { headline, PAPER_URL, CODE_URL } from "../../data/headline";
import { EASE } from "../../theme";

export function Hero() {
  return (
    <section id="top" className="relative min-h-screen flex flex-col justify-center overflow-hidden px-6 md:px-12 lg:px-20">
      {/* ambient memory table */}
      <div className="absolute inset-0 -z-10">
        <MemoryGrid ambient cols={30} rows={20} className="w-full h-full" />
        {/* parchment fade so text stays legible */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(120% 90% at 18% 40%, rgba(244,240,230,0.97) 0%, rgba(244,240,230,0.86) 38%, rgba(244,240,230,0.55) 100%)",
          }}
        />
      </div>

      <div className="max-w-5xl">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: EASE, delay: 0.1 }}
          className="kicker mb-6"
        >
          Per-user memory · a local parametric edit
        </motion.div>

        <h1 className="display text-[clamp(3rem,11vw,9rem)] text-ink leading-[0.9]">
          <motion.span
            className="block"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, ease: EASE, delay: 0.15 }}
          >
            User as
          </motion.span>
          <motion.span
            className="block display-italic text-engram"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, ease: EASE, delay: 0.28 }}
          >
            Engram
          </motion.span>
        </h1>

        <motion.p
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: EASE, delay: 0.45 }}
          className="lede mt-8 max-w-[40ch]"
        >
          A user's facts become a few rows in a content-addressed memory table —
          <span className="text-ink"> not a rewrite of the model.</span>
        </motion.p>

        {/* key stats */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.7 }}
          className="flex flex-wrap gap-x-10 gap-y-4 mt-10"
        >
          <Stat big={`${headline.indirectMeanRatio}×`} label="more indirect reasoning than a per-user LoRA" />
          <Stat big={`${headline.contamRatio.toLocaleString()}×`} label="less disruption to unrelated text" />
          <Stat big={`${headline.engramKB} KB`} label="per user · backbone untouched" />
        </motion.div>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.9 }}
          className="flex items-center gap-3 mt-10"
        >
          <a
            href={PAPER_URL}
            className="font-mono text-[0.74rem] uppercase tracking-[0.18em] px-6 py-3 rounded-full bg-ink text-paper-50 hover:bg-engram transition-colors"
          >
            Read the paper ↗
          </a>
          <a
            href={CODE_URL}
            className="font-mono text-[0.74rem] uppercase tracking-[0.18em] px-6 py-3 rounded-full border hairline text-ink-100 hover:border-engram hover:text-engram transition-colors"
          >
            Code & weights
          </a>
        </motion.div>
      </div>

      {/* scroll cue */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2 font-mono text-[0.62rem] uppercase tracking-[0.22em] text-ink-50"
        animate={{ opacity: [0.3, 1, 0.3], y: [0, 4, 0] }}
        transition={{ duration: 2.4, repeat: Infinity }}
      >
        scroll
      </motion.div>
    </section>
  );
}

function Stat({ big, label }: { big: string; label: string }) {
  return (
    <div className="max-w-[12rem]">
      <div className="display text-4xl md:text-5xl text-engram-deep">{big}</div>
      <div className="font-body text-sm text-ink-50 leading-tight mt-1">{label}</div>
    </div>
  );
}
