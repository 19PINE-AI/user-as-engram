import { C } from "../../theme";
import { kbSizes, accuracySeries, retrievalRecall } from "../../data/ragScale";

const W = 100;
const H = 56;
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
  const ls = kbSizes.map((k) => Math.log10(k));
  const min = ls[0];
  const max = ls[ls.length - 1];
  return PAD_L + ((ls[i] - min) / (max - min)) * (W - PAD_L - PAD_R);
}
const yAt = (v: number) => H - (v / Y_MAX) * H;

export function KBCrossover() {
  const last = kbSizes.length - 1;
  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-7">
      <div className="grid lg:grid-cols-[1fr_14rem] gap-7 items-center">
        {/* static chart */}
        <div>
          <svg viewBox={`0 0 ${W} ${H + 7}`} className="w-full overflow-visible">
            {[20, 40, 60].map((g) => (
              <g key={g}>
                <line x1={PAD_L} y1={yAt(g)} x2={W - PAD_R} y2={yAt(g)} stroke={C.rule} strokeWidth="0.2" strokeDasharray="0.6 0.8" />
                <text x={PAD_L} y={yAt(g) - 0.8} fontSize="2.2" fill={C.ink50} fontFamily="monospace">{g}%</text>
              </g>
            ))}

            {/* crossover marker */}
            <line x1={xAt(1)} y1={0} x2={xAt(1)} y2={H} stroke={C.ochre} strokeWidth="0.25" strokeDasharray="0.8 0.8" opacity={0.7} />
            <text x={xAt(1) + 0.6} y={2.8} fontSize="2.2" fill={C.ochre} fontFamily="monospace">~100 facts</text>

            {accuracySeries.map((ser) => {
              const pts = ser.vals.map((v, i) => `${xAt(i)},${yAt(v)}`).join(" ");
              return (
                <g key={ser.key}>
                  <polyline
                    points={pts}
                    fill="none"
                    stroke={colorMap[ser.color]}
                    strokeWidth={ser.key === "layered" ? 1.1 : 0.8}
                    strokeDasharray={ser.dashed ? "1.4 1" : undefined}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  {/* endpoint dot + value label at the right */}
                  <circle cx={xAt(last)} cy={yAt(ser.vals[last])} r="0.9" fill={colorMap[ser.color]} />
                  <text
                    x={xAt(last) + 1.4}
                    y={yAt(ser.vals[last]) + 0.8}
                    fontSize="2.4"
                    fill={ser.key === "layered" ? C.engramDeep : C.ink50}
                    fontFamily="monospace"
                    fontWeight={ser.key === "layered" ? 600 : 400}
                  >
                    {ser.vals[last]}%
                  </text>
                </g>
              );
            })}

            {kbSizes.map((k, i) => (
              <text key={k} x={xAt(i)} y={H + 5} fontSize="2.2" fill={C.ink50} fontFamily="monospace" textAnchor="middle">{k}</text>
            ))}
          </svg>
          <div className="font-mono text-[0.58rem] text-ink-50 text-center mt-1">
            knowledge-base size — facts per user (log scale)
          </div>
        </div>

        {/* static legend + endpoints */}
        <div className="space-y-2.5">
          {accuracySeries.map((ser) => (
            <div key={ser.key} className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-2 font-body text-[0.85rem] text-ink-100">
                <span className="w-3 h-[3px] rounded-full" style={{ background: colorMap[ser.color] }} />
                {ser.label}
              </span>
            </div>
          ))}
          <div className="pt-3 border-t hairline font-body text-[0.8rem] text-ink-50 leading-snug">
            Past ~100 facts, the layered design (flat at {accuracySeries[0].vals[0]}%) overtakes naive
            RAG and Qwen-3B + RAG — because retrieval recall collapses from{" "}
            <b className="text-rust-deep">{retrievalRecall.top3[0]}%</b> to{" "}
            <b className="text-rust-deep">{retrievalRecall.top3[last]}%</b> as the pool grows.
          </div>
        </div>
      </div>
    </div>
  );
}
