import { AnimatePresence, motion } from "motion/react";
import { useMemo, useState } from "react";
import { C } from "../../theme";
import { triggerRows } from "../../lib/hash";
import { mayaFacts } from "../../data/facts";

const COLS = 30;
const ROWS = 18;
const TOTAL = COLS * ROWS;
const K = 14; // rows lit per fact

export function WriteAFact() {
  const [written, setWritten] = useState<string[]>([]);
  const [active, setActive] = useState<string | null>(null);

  // precompute each fact's address set
  const addrs = useMemo(() => {
    const m: Record<string, number[]> = {};
    for (const f of mayaFacts) m[f.id] = triggerRows(f.trigger, TOTAL, K);
    return m;
  }, []);

  const litSet = useMemo(() => {
    const s = new Set<number>();
    written.forEach((id) => addrs[id].forEach((a) => s.add(a)));
    return s;
  }, [written, addrs]);

  const activeFact = mayaFacts.find((f) => f.id === active) ?? null;
  const activeAddrs = active ? new Set(addrs[active]) : new Set<number>();

  function write(id: string) {
    setActive(id);
    setWritten((w) => (w.includes(id) ? w : [...w, id]));
  }
  function reset() {
    setWritten([]);
    setActive(null);
  }

  const occupancy = ((litSet.size / TOTAL) * 100).toFixed(1);
  const kb = written.length === 0 ? 0 : Math.round((written.length / mayaFacts.length) * 88);

  const cellW = 100 / COLS;
  const cellH = 100 / ROWS;

  return (
    <div className="grid lg:grid-cols-[20rem_1fr] gap-8 items-start">
      {/* controls */}
      <div className="space-y-3">
        <div className="kicker !text-ink-50 mb-3">Write one of Maya's facts</div>
        {mayaFacts.map((f) => {
          const done = written.includes(f.id);
          return (
            <button
              key={f.id}
              onClick={() => write(f.id)}
              className={`group w-full text-left rounded-lg border px-4 py-3 transition-all duration-300 ${
                active === f.id
                  ? "border-engram bg-engram-wash"
                  : done
                    ? "border-engram/40 bg-paper-50"
                    : "border-rule bg-paper-50 hover:border-engram/60"
              }`}
            >
              <div className="font-mono text-[0.72rem] text-engram-deep flex items-center gap-2">
                <span
                  className={`inline-block w-1.5 h-1.5 rounded-full ${done ? "bg-engram" : "bg-rule"}`}
                />
                {f.trigger}
                <span className="text-ink-50">…</span>
              </div>
              <div className="font-body text-lg text-ink mt-0.5">{f.answer}</div>
            </button>
          );
        })}
        <button
          onClick={reset}
          className="font-mono text-[0.7rem] uppercase tracking-[0.18em] text-ink-50 hover:text-rust transition-colors pt-1"
        >
          ↺ reset table
        </button>
      </div>

      {/* visual */}
      <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-7">
        {/* pipeline labels */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 font-mono text-[0.7rem] mb-5 min-h-[2.4rem]">
          <AnimatePresence mode="wait">
            {activeFact ? (
              <motion.div
                key={activeFact.id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.4 }}
                className="flex flex-wrap items-center gap-x-2.5 gap-y-2"
              >
                <Step>“…{activeFact.trigger.split(" ").slice(-3).join(" ")}”</Step>
                <Arrow />
                <Step>suffix N-gram</Step>
                <Arrow />
                <Step accent>hash → {K} addresses</Step>
                <Arrow />
                <Step ours>write “{activeFact.answer}”</Step>
              </motion.div>
            ) : (
              <span className="text-ink-50">
                Pick a fact. Its trigger hashes to a sparse set of rows; only those rows change.
              </span>
            )}
          </AnimatePresence>
        </div>

        {/* the table */}
        <svg viewBox="0 0 100 62" className="w-full rounded-md overflow-hidden">
          {Array.from({ length: TOTAL }).map((_, idx) => {
            const r = Math.floor(idx / COLS);
            const c = idx % COLS;
            const on = litSet.has(idx);
            const isActive = activeAddrs.has(idx);
            return (
              <motion.rect
                key={idx}
                x={c * cellW + 0.12}
                y={(r * cellH + 0.12) * 0.62}
                width={cellW - 0.24}
                height={cellH * 0.62 - 0.24}
                rx={0.3}
                initial={false}
                animate={{
                  fill: on ? C.engram : C.rule,
                  opacity: on ? 1 : 0.2,
                  scale: isActive ? [1, 1.6, 1] : 1,
                }}
                transition={{
                  duration: isActive ? 0.7 : 0.4,
                  ease: [0.16, 1, 0.3, 1],
                }}
                style={{ transformOrigin: "center", transformBox: "fill-box" }}
              />
            );
          })}
        </svg>

        {/* readout */}
        <div className="flex flex-wrap gap-x-8 gap-y-2 mt-5 font-mono text-[0.74rem]">
          <Readout label="facts written" value={`${written.length} / ${mayaFacts.length}`} />
          <Readout label="rows occupied" value={`${litSet.size} / ${TOTAL}`} />
          <Readout label="table used" value={`${occupancy}%`} />
          <Readout label="override map" value={`${kb} KB`} ours />
          <Readout label="backbone" value="bit-identical" />
        </div>
      </div>
    </div>
  );
}

function Step({
  children,
  accent,
  ours,
}: {
  children: React.ReactNode;
  accent?: boolean;
  ours?: boolean;
}) {
  return (
    <span
      className={`px-2 py-1 rounded ${
        ours
          ? "bg-engram text-paper-50"
          : accent
            ? "bg-engram-wash text-engram-deep"
            : "bg-paper-200 text-ink-100"
      }`}
    >
      {children}
    </span>
  );
}
function Arrow() {
  return <span className="text-rule">→</span>;
}
function Readout({ label, value, ours }: { label: string; value: string; ours?: boolean }) {
  return (
    <div>
      <div className="text-ink-50 text-[0.62rem] uppercase tracking-[0.16em]">{label}</div>
      <div className={ours ? "text-engram-deep" : "text-ink"}>{value}</div>
    </div>
  );
}
