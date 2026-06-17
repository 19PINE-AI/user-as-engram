import { useMemo } from "react";
import { C } from "../../theme";
import { triggerRows } from "../../lib/hash";
import { mayaFacts } from "../../data/facts";

const COLS = 30;
const ROWS = 18;
const TOTAL = COLS * ROWS;
const K = 14;

export function WriteAFact() {
  const { litSet, exampleAddrs } = useMemo(() => {
    const s = new Set<number>();
    let example: number[] = [];
    mayaFacts.forEach((f, i) => {
      const a = triggerRows(f.trigger, TOTAL, K);
      if (i === 0) example = a;
      a.forEach((x) => s.add(x));
    });
    return { litSet: s, exampleAddrs: new Set(example) };
  }, []);

  const occupancy = ((litSet.size / TOTAL) * 100).toFixed(1);
  const cellW = 100 / COLS;
  const cellH = 100 / ROWS;
  const ex = mayaFacts[0];

  return (
    <div className="grid lg:grid-cols-[19rem_1fr] gap-7 items-start">
      {/* the facts (static legend) */}
      <div className="space-y-2">
        <div className="font-mono text-[0.62rem] uppercase tracking-[0.16em] text-ink-50 mb-2">
          Maya's facts, written in
        </div>
        {mayaFacts.map((f) => (
          <div key={f.id} className="rounded-lg border border-engram/30 bg-paper-50 px-3.5 py-2.5">
            <div className="font-mono text-[0.66rem] text-engram-deep flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-engram" />
              {f.trigger}…
            </div>
            <div className="font-body text-[0.95rem] text-ink mt-0.5">{f.answer}</div>
          </div>
        ))}
      </div>

      {/* visual */}
      <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-6">
        {/* one worked pipeline */}
        <div className="flex flex-wrap items-center gap-x-2 gap-y-2 font-mono text-[0.64rem] mb-5">
          <Step>“…{ex.trigger.split(" ").slice(-3).join(" ")}”</Step>
          <Arrow />
          <Step>suffix N-gram</Step>
          <Arrow />
          <Step accent>hash → {K} addresses</Step>
          <Arrow />
          <Step ours>write “{ex.answer}”</Step>
        </div>

        {/* the table */}
        <svg viewBox="0 0 100 62" className="w-full rounded-md overflow-hidden">
          {Array.from({ length: TOTAL }).map((_, idx) => {
            const r = Math.floor(idx / COLS);
            const c = idx % COLS;
            const on = litSet.has(idx);
            const isEx = exampleAddrs.has(idx);
            return (
              <rect
                key={idx}
                x={c * cellW + 0.12}
                y={(r * cellH + 0.12) * 0.62}
                width={cellW - 0.24}
                height={cellH * 0.62 - 0.24}
                rx={0.3}
                fill={on ? (isEx ? C.engramDeep : C.engram) : C.rule}
                opacity={on ? 1 : 0.2}
              />
            );
          })}
        </svg>

        <div className="flex flex-wrap gap-x-7 gap-y-2 mt-5 font-mono text-[0.7rem]">
          <Readout label="facts written" value={`${mayaFacts.length}`} />
          <Readout label="rows occupied" value={`${litSet.size} / ${TOTAL}`} />
          <Readout label="table used" value={`${occupancy}%`} />
          <Readout label="override map" value="88 KB" ours />
          <Readout label="backbone" value="bit-identical" />
        </div>
      </div>
    </div>
  );
}

function Step({ children, accent, ours }: { children: React.ReactNode; accent?: boolean; ours?: boolean }) {
  return (
    <span
      className={`px-2 py-1 rounded ${
        ours ? "bg-engram text-paper-50" : accent ? "bg-engram-wash text-engram-deep" : "bg-paper-200 text-ink-100"
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
      <div className="text-ink-50 text-[0.58rem] uppercase tracking-[0.16em]">{label}</div>
      <div className={ours ? "text-engram-deep" : "text-ink"}>{value}</div>
    </div>
  );
}
