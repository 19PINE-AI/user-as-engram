import { motion } from "motion/react";
import { useState } from "react";
import { C } from "../../theme";
import { headline } from "../../data/headline";

export function GlassBox() {
  return (
    <div className="grid md:grid-cols-3 gap-5">
      <GatePanel />
      <ValuePathPanel />
      <DepthPanel />
    </div>
  );
}

function Card({
  n,
  title,
  children,
  foot,
}: {
  n: string;
  title: string;
  children: React.ReactNode;
  foot: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 flex flex-col">
      <div className="flex items-baseline gap-2 mb-4">
        <span className="font-mono text-[0.7rem] text-engram">{n}</span>
        <span className="font-mono text-[0.72rem] uppercase tracking-[0.14em] text-ink-100">
          {title}
        </span>
      </div>
      <div className="flex-1 flex items-center justify-center">{children}</div>
      <p className="font-body text-sm text-ink-50 leading-snug mt-4">{foot}</p>
    </div>
  );
}

/** (a) the write opens its own gate */
function GatePanel() {
  return (
    <Card
      n="01"
      title="It opens its own gate"
      foot={
        <>
          The lookup is gated by a switch α. Writing the fact turns the switch at its
          trigger from <b>off</b> to <b className="text-engram-deep">on</b>; everywhere else
          stays shut.
        </>
      }
    >
      <div className="w-full space-y-4">
        <GateBar label="trigger position" from={headline.gateBefore} to={headline.gateAfter} ours />
        <GateBar label="every other position" from={headline.gateNonTrigger} to={headline.gateNonTrigger} />
      </div>
    </Card>
  );
}

function GateBar({ label, from, to, ours }: { label: string; from: number; to: number; ours?: boolean }) {
  return (
    <div>
      <div className="flex justify-between font-mono text-[0.64rem] text-ink-50 mb-1">
        <span>{label}</span>
        <span className={ours ? "text-engram-deep" : ""}>
          α {from.toFixed(2)} → {to.toFixed(2)}
        </span>
      </div>
      <div className="h-3 rounded-full bg-paper-200 overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${ours ? "bg-engram" : "bg-rule"}`}
          initial={{ width: `${from * 100}%` }}
          whileInView={{ width: `${to * 100}%` }}
          viewport={{ once: true }}
          transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
        />
      </div>
    </div>
  );
}

/** (b) the injected change points along the value path */
function ValuePathPanel() {
  return (
    <Card
      n="02"
      title="It injects just its value"
      foot={
        <>
          The change the row makes points almost exactly along the value it carries —
          cosine <b className="text-engram-deep">{headline.valuePathCosine}</b>. It scales the
          value; it does not become something else.
        </>
      }
    >
      <svg viewBox="0 0 120 90" className="w-[80%]">
        {/* origin */}
        <circle cx="20" cy="70" r="2" fill={C.ink} />
        {/* value path (reference) */}
        <motion.line
          x1="20" y1="70" x2="104" y2="16"
          stroke={C.rule} strokeWidth="2.5" strokeLinecap="round"
          initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
          viewport={{ once: true }} transition={{ duration: 0.9 }}
        />
        {/* actual residual change (nearly aligned) */}
        <motion.line
          x1="20" y1="70" x2="101" y2="19"
          stroke={C.engram} strokeWidth="3.2" strokeLinecap="round"
          initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
          viewport={{ once: true }} transition={{ duration: 1.1, delay: 0.3 }}
        />
        <text x="106" y="14" fontSize="6" fill={C.ink50} fontFamily="monospace">Wᵥe</text>
        <text x="40" y="40" fontSize="6.5" fill={C.engramDeep} fontFamily="monospace" fontWeight="600">
          cos {headline.valuePathCosine}
        </text>
      </svg>
    </Card>
  );
}

/** (c) it only works near the end of the network — interactive depth slider */
function DepthPanel() {
  const STACK = 12;
  const ENGRAM_LATE = 10;
  const [layer, setLayer] = useState(ENGRAM_LATE); // 0 = early, 11 = late
  // recall: collapses for early insertion, perfect when late (paper: 0.25 -> 1.00)
  const t = layer / (STACK - 1);
  const recall =
    headline.recallEarlyLayer +
    (headline.recallLateLayer - headline.recallEarlyLayer) * Math.max(0, (t - 0.45) / 0.55);
  const recallClamped = Math.min(1, Math.max(headline.recallEarlyLayer, recall));

  return (
    <Card
      n="03"
      title="Only late layers work"
      foot={
        <>
          Drag where the row is written. Write it <b>early</b> and the stack above reworks it
          (recall ≈ {headline.recallEarlyLayer.toFixed(2)}); write it <b className="text-engram-deep">late</b>,
          where the model has nearly decided, and recall is {headline.recallLateLayer.toFixed(2)}.
        </>
      }
    >
      <div className="w-full flex items-center gap-4">
        {/* transformer stack */}
        <div className="flex flex-col-reverse gap-[3px]">
          {Array.from({ length: STACK }).map((_, i) => (
            <div
              key={i}
              className="w-10 h-2.5 rounded-sm transition-colors"
              style={{
                background: i === layer ? C.engram : C.rule,
                opacity: i === layer ? 1 : 0.4,
              }}
            />
          ))}
        </div>
        <div className="flex-1">
          <input
            type="range"
            min={0}
            max={STACK - 1}
            value={layer}
            onChange={(e) => setLayer(+e.target.value)}
            className="w-full accent-engram"
            aria-label="insertion layer"
          />
          <div className="flex justify-between font-mono text-[0.6rem] text-ink-50 mt-1">
            <span>early</span>
            <span>late</span>
          </div>
          <div className="mt-4">
            <div className="font-mono text-[0.6rem] uppercase tracking-[0.14em] text-ink-50">
              top-1 recall
            </div>
            <div className="flex items-center gap-2">
              <div className="display text-3xl text-ink tabular-nums" style={{ fontVariantNumeric: "tabular-nums" }}>
                {recallClamped.toFixed(2)}
              </div>
              <div className="flex-1 h-2 rounded-full bg-paper-200 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${recallClamped * 100}%`,
                    background: recallClamped > 0.7 ? C.engram : C.rust,
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
