import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { mayaFacts, questions } from "../../data/facts";

export function TwoJobs() {
  const [mode, setMode] = useState<"recall" | "reason">("recall");
  const q = questions[mode];
  // which facts "light up" for each job
  const active = mode === "recall" ? ["cardiologist"] : ["cardiologist", "clinic", "diet"];

  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-7">
      {/* toggle */}
      <div className="inline-flex rounded-full border hairline p-1 bg-paper mb-6">
        {(["recall", "reason"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`font-mono text-[0.72rem] uppercase tracking-[0.16em] px-5 py-2 rounded-full transition-colors ${
              mode === m ? "bg-engram text-paper-50" : "text-ink-50 hover:text-ink"
            }`}
          >
            {m}
          </button>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-7 items-center">
        {/* facts */}
        <div className="space-y-2">
          {mayaFacts.map((f) => {
            const on = active.includes(f.id);
            return (
              <motion.div
                key={f.id}
                animate={{ opacity: on ? 1 : 0.4, x: on ? 0 : -2 }}
                transition={{ duration: 0.4 }}
                className={`rounded-lg border px-4 py-2.5 ${on ? "border-engram/50 bg-engram-wash" : "border-rule bg-paper"}`}
              >
                <span className="font-mono text-[0.7rem] text-engram-deep">{f.trigger} </span>
                <span className="font-body text-ink">{f.answer}</span>
              </motion.div>
            );
          })}
        </div>

        {/* question + job */}
        <div>
          <div className="font-mono text-[0.64rem] uppercase tracking-[0.16em] text-ink-50 mb-2">
            Maya asks
          </div>
          <AnimatePresence mode="wait">
            <motion.p
              key={mode}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.35 }}
              className="font-body text-xl text-ink leading-snug"
            >
              “{q.q}”
            </motion.p>
          </AnimatePresence>
          <div className="mt-5 flex items-center gap-3">
            <span className="font-mono text-[0.64rem] uppercase tracking-[0.16em] text-ink-50">
              the job
            </span>
            <AnimatePresence mode="wait">
              <motion.span
                key={mode}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className={`font-body text-lg ${mode === "recall" ? "text-engram-deep" : "text-rust-deep"}`}
              >
                {mode === "recall" ? "Recall — " : "Reason — "}
                <span className="text-ink-100">{q.needs}</span>
              </motion.span>
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
