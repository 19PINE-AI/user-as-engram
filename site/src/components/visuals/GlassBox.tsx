import { C } from "../../theme";
import { headline } from "../../data/headline";

export function GlassBox() {
  return (
    <div className="grid md:grid-cols-3 gap-4">
      <GatePanel />
      <ValuePathPanel />
      <DepthPanel />
    </div>
  );
}

function Card({ n, title, children, foot }: { n: string; title: string; children: React.ReactNode; foot: React.ReactNode }) {
  return (
    <div className="rounded-xl border hairline bg-paper-50 p-4 flex flex-col">
      <div className="flex items-baseline gap-2 mb-3">
        <span className="font-mono text-[0.64rem] text-engram">{n}</span>
        <span className="font-mono text-[0.66rem] uppercase tracking-[0.14em] text-ink-100">{title}</span>
      </div>
      <div className="flex-1 flex items-center justify-center">{children}</div>
      <p className="font-body text-[0.8rem] text-ink-50 leading-snug mt-3">{foot}</p>
    </div>
  );
}

/** (1) the write opens its own gate — static end state */
function GatePanel() {
  return (
    <Card
      n="01"
      title="Opens its own gate"
      foot={
        <>
          Writing the fact turns the switch α at its trigger from <b>{headline.gateBefore}</b> to{" "}
          <b className="text-engram-deep">{headline.gateAfter}</b>; everywhere else it stays shut.
        </>
      }
    >
      <div className="w-full space-y-4">
        <GateBar label="trigger position" value={headline.gateAfter} ours />
        <GateBar label="every other position" value={headline.gateNonTrigger} />
      </div>
    </Card>
  );
}
function GateBar({ label, value, ours }: { label: string; value: number; ours?: boolean }) {
  return (
    <div>
      <div className="flex justify-between font-mono text-[0.6rem] text-ink-50 mb-1">
        <span>{label}</span>
        <span className={ours ? "text-engram-deep" : ""}>α {value.toFixed(2)}</span>
      </div>
      <div className="h-2.5 rounded-full bg-paper-200 overflow-hidden">
        <div className={`h-full rounded-full ${ours ? "bg-engram" : "bg-rule"}`} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

/** (2) injects just its value path — static vectors */
function ValuePathPanel() {
  return (
    <Card
      n="02"
      title="Injects just its value"
      foot={
        <>
          The change points almost exactly along the value the row carries — cosine{" "}
          <b className="text-engram-deep">{headline.valuePathCosine}</b>.
        </>
      }
    >
      <svg viewBox="0 0 120 90" className="w-[80%]">
        <circle cx="20" cy="70" r="2" fill={C.ink} />
        <line x1="20" y1="70" x2="104" y2="16" stroke={C.rule} strokeWidth="2.5" strokeLinecap="round" />
        <line x1="20" y1="70" x2="101" y2="19" stroke={C.engram} strokeWidth="3.2" strokeLinecap="round" />
        <text x="106" y="14" fontSize="6" fill={C.ink50} fontFamily="monospace">Wᵥe</text>
        <text x="38" y="44" fontSize="6.5" fill={C.engramDeep} fontFamily="monospace" fontWeight="600">
          cos {headline.valuePathCosine}
        </text>
      </svg>
    </Card>
  );
}

/** (3) only late layers work — static early-vs-late comparison */
function DepthPanel() {
  return (
    <Card
      n="03"
      title="Only late layers work"
      foot={
        <>
          Written <b>early</b>, the stack above reworks it; written <b className="text-engram-deep">late</b>,
          where the model has nearly decided, the row sticks.
        </>
      }
    >
      <div className="w-full flex gap-6 justify-center">
        <DepthCol label="written early" lit={2} recall={headline.recallEarlyLayer} ours={false} />
        <DepthCol label="written late" lit={9} recall={headline.recallLateLayer} ours />
      </div>
    </Card>
  );
}
function DepthCol({ label, lit, recall, ours }: { label: string; lit: number; recall: number; ours: boolean }) {
  const STACK = 11;
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="flex flex-col-reverse gap-[2px]">
        {Array.from({ length: STACK }).map((_, i) => (
          <div
            key={i}
            className="w-7 h-2 rounded-sm"
            style={{ background: i === lit ? (ours ? C.engram : C.rust) : C.rule, opacity: i === lit ? 1 : 0.35 }}
          />
        ))}
      </div>
      <div className="font-mono text-[0.56rem] uppercase tracking-[0.12em] text-ink-50">{label}</div>
      <div className={`display text-2xl ${ours ? "text-engram-deep" : "text-rust-deep"}`}>{recall.toFixed(2)}</div>
      <div className="font-mono text-[0.54rem] text-ink-50">recall</div>
    </div>
  );
}
