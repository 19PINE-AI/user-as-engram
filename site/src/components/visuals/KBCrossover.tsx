import { useState } from "react";
import { motion } from "motion/react";
import { C } from "../../theme";
import { kbSizes, accuracySeries, retrievalRecall } from "../../data/ragScale";

const W = 100;
const H = 58;
const PAD_L = 2;
const PAD_R = 2;
const Y_MAX = 60;

const colorMap = {
  engram: C.engram,
  engramLight: C.engramLight,
  rust: C.rust,
  rustLight: C.rustLight,
  ochre: C.ochre,
};

function xAt(i: number) {
  // log spacing across the kb sizes
  const ls = kbSizes.map((k) => Math.log10(k));
  const min = ls[0];
  const max = ls[ls.length - 1];
  return PAD_L + ((ls[i] - min) / (max - min)) * (W - PAD_L - PAD_R);
}
function yAt(v: number) {
  return H - (v / Y_MAX) * H;
}

export function KBCrossover() {
  const [s, setS] = useState(kbSizes.length - 1); // selected KB index

  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-7">
      <div className="grid lg:grid-cols-[1fr_15rem] gap-7 items-center">
        {/* chart */}
        <div>
          <svg viewBox={`0 0 ${W} ${H + 8}`} className="w-full overflow-visible">
            {/* gridlines */}
            {[20, 40, 60].map((g) => (
              <g key={g}>
                <line x1={PAD_L} y1={yAt(g)} x2={W - PAD_R} y2={yAt(g)} stroke={C.rule} strokeWidth="0.2" strokeDasharray="0.6 0.8" />
                <text x={PAD_L} y={yAt(g) - 0.8} fontSize="2.4" fill={C.ink50} fontFamily="monospace">{g}%</text>
              </g>
            ))}

            {/* crossover marker at ~100 facts */}
            <line x1={xAt(1)} y1={0} x2={xAt(1)} y2={H} stroke={C.ochre} strokeWidth="0.25" strokeDasharray="0.8 0.8" opacity={0.7} />
            <text x={xAt(1) + 0.6} y={3} fontSize="2.3" fill={C.ochre} fontFamily="monospace">~100 facts</text>

            {/* series */}
            {accuracySeries.map((ser) => {
              const pts = ser.vals.map((v, i) => `${xAt(i)},${yAt(v)}`).join(" ");
              return (
                <motion.polyline
                  key={ser.key}
                  points={pts}
                  fill="none"
                  stroke={colorMap[ser.color]}
                  strokeWidth={ser.key === "layered" ? 1.1 : 0.8}
                  strokeDasharray={ser.dashed ? "1.4 1" : undefined}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  initial={{ pathLength: 0 }}
                  whileInView={{ pathLength: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 1.3, ease: [0.16, 1, 0.3, 1] }}
                />
              );
            })}

            {/* scrubber */}
            <line x1={xAt(s)} y1={0} x2={xAt(s)} y2={H} stroke={C.ink} strokeWidth="0.3" opacity={0.5} />
            {accuracySeries.map((ser) => (
              <circle key={ser.key} cx={xAt(s)} cy={yAt(ser.vals[s])} r="0.9" fill={colorMap[ser.color]} stroke={C.paper50} strokeWidth="0.3" />
            ))}

            {/* x labels */}
            {kbSizes.map((k, i) => (
              <text key={k} x={xAt(i)} y={H + 5} fontSize="2.3" fill={i === s ? C.ink : C.ink50} fontFamily="monospace" textAnchor="middle">{k}</text>
            ))}
          </svg>

          {/* slider */}
          <input
            type="range" min={0} max={kbSizes.length - 1} value={s}
            onChange={(e) => setS(+e.target.value)}
            className="w-full accent-engram mt-1"
            aria-label="knowledge-base size"
          />
          <div className="font-mono text-[0.62rem] text-ink-50 text-center mt-0.5">
            knowledge-base size — facts per user (log scale)
          </div>
        </div>

        {/* readout */}
        <div className="space-y-3">
          <div className="font-mono text-[0.64rem] uppercase tracking-[0.16em] text-ink-50">
            at {kbSizes[s].toLocaleString()} facts
          </div>
          {accuracySeries.map((ser) => (
            <div key={ser.key} className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-2 font-body text-sm text-ink-100">
                <span className="w-3 h-[3px] rounded-full" style={{ background: colorMap[ser.color] }} />
                {ser.label}
              </span>
              <span className="font-mono text-sm tabular-nums" style={{ color: ser.key === "layered" ? C.engramDeep : C.ink }}>
                {ser.vals[s].toFixed(0)}%
              </span>
            </div>
          ))}
          <div className="pt-3 border-t hairline">
            <div className="flex items-center justify-between gap-3">
              <span className="font-body text-sm text-ink-50">retrieval recall (top-3)</span>
              <span className="font-mono text-sm tabular-nums text-rust-deep">{retrievalRecall.top3[s]}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-paper-200 overflow-hidden mt-1">
              <div className="h-full rounded-full bg-rust transition-all duration-300" style={{ width: `${retrievalRecall.top3[s]}%` }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
