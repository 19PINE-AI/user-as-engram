import { C } from "../../theme";

// Static: three methods, each showing WHERE the cost lands and the axis it
// grows on — as a small fixed cost curve. No sliders.
const methods = [
  {
    method: "Per-user LoRA",
    where: "in the shared weights",
    grows: "paid on every query · stacks with users",
    axis: "users →",
    curve: [0.34, 0.52, 0.72, 0.88, 1.0],
    color: C.rust,
  },
  {
    method: "Engram row",
    where: "at the one address",
    grows: "grows only with a user's own facts",
    axis: "facts / user →",
    curve: [0.1, 0.16, 0.24, 0.32, 0.4],
    color: C.engram,
    ours: true,
  },
  {
    method: "Retrieval",
    where: "in the search index",
    grows: "grows with the candidate pool",
    axis: "pool size →",
    curve: [0.1, 0.32, 0.56, 0.8, 0.95],
    color: C.ochre,
  },
];

function Spark({ curve, color }: { curve: number[]; color: string }) {
  const W = 100,
    H = 44;
  const pts = curve.map((v, i) => [(i / (curve.length - 1)) * W, H - v * H]);
  const line = pts.map((p) => p.join(",")).join(" ");
  const area = `0,${H} ${line} ${W},${H}`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none">
      <polygon points={area} fill={color} opacity={0.12} />
      <polyline points={line} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={W} cy={H - curve[curve.length - 1] * H} r={2.4} fill={color} />
    </svg>
  );
}

export function PriceLocator() {
  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-7">
      <div className="grid sm:grid-cols-3 gap-4">
        {methods.map((m) => (
          <div
            key={m.method}
            className={`rounded-lg border p-4 ${m.ours ? "border-engram/40 bg-engram-wash" : "border-rule bg-paper"}`}
          >
            <div className="font-body text-base text-ink leading-tight">{m.method}</div>
            <div className="font-mono text-[0.62rem] uppercase tracking-[0.12em] mt-0.5" style={{ color: m.color }}>
              {m.where}
            </div>
            <div className="mt-4">
              <Spark curve={m.curve} color={m.color} />
              <div className="flex justify-between font-mono text-[0.56rem] text-ink-50 mt-1">
                <span>cost ↑</span>
                <span>{m.axis}</span>
              </div>
            </div>
            <div className="font-body text-[0.8rem] text-ink-50 leading-snug mt-3">{m.grows}</div>
          </div>
        ))}
      </div>
      <p className="font-body text-[0.82rem] text-ink-50 leading-snug mt-5 max-w-prose">
        No method avoids the price. They differ only in <i>where</i> it lands — and on which axis it
        grows. The Engram row keeps it lowest, and tied to a single user's own facts.
      </p>
    </div>
  );
}
