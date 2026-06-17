import { motion } from "motion/react";
import { headline } from "../../data/headline";
import { Counter } from "../ui/Counter";

export function ServingFlow() {
  const users = [0, 1, 2];
  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-8">
      <div className="grid md:grid-cols-[1fr_auto] gap-8 items-center">
        {/* flow diagram */}
        <div className="flex items-center justify-between gap-2">
          {/* requests */}
          <div className="space-y-2">
            {users.map((u) => (
              <div key={u} className="flex items-center gap-2">
                <span className="w-7 h-7 rounded-full bg-paper-200 grid place-items-center font-mono text-[0.6rem] text-ink-50">
                  u{u}
                </span>
                <div className="flex gap-[3px]">
                  {Array.from({ length: 4 }).map((_, k) => (
                    <span key={k} className="w-1 h-3 rounded-full bg-engram-lighter" style={{ opacity: 0.5 + k * 0.12 }} />
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* animated request dot stream */}
          <div className="relative flex-1 h-24 mx-1">
            {[0, 1, 2].map((i) => (
              <motion.span
                key={i}
                className="absolute top-1/2 w-2 h-2 rounded-full bg-engram"
                initial={{ left: "0%", opacity: 0 }}
                animate={{ left: ["0%", "100%"], opacity: [0, 1, 1, 0] }}
                transition={{ duration: 1.8, delay: i * 0.6, repeat: Infinity, ease: "linear" }}
              />
            ))}
            <div className="absolute inset-x-0 top-1/2 h-px bg-rule" />
          </div>

          {/* server */}
          <div className="rounded-lg border border-engram/40 bg-engram-wash px-4 py-5 text-center">
            <div className="font-mono text-[0.6rem] uppercase tracking-[0.16em] text-engram-deep">
              EngramServer
            </div>
            <div className="font-body text-xs text-ink-50 mt-1 leading-tight">
              swap user's<br />override map<br />→ forward → restore
            </div>
          </div>
        </div>

        {/* stats */}
        <div className="space-y-5">
          <Stat
            value={<Counter value={headline.reqPerSec} suffix="" />}
            unit="req/s"
            note="flat as tenants triple"
          />
          <Stat value="< 1" unit="ms" note="override apply" />
          <Stat value="0" unit="leaks" note="cross-user, by construction" ours />
        </div>
      </div>
    </div>
  );
}

function Stat({
  value,
  unit,
  note,
  ours,
}: {
  value: React.ReactNode;
  unit: string;
  note: string;
  ours?: boolean;
}) {
  return (
    <div className="flex items-baseline gap-2">
      <span className={`display text-3xl ${ours ? "text-engram-deep" : "text-ink"}`}>{value}</span>
      <span className="font-mono text-[0.7rem] text-ink-50">{unit}</span>
      <span className="font-body text-sm text-ink-50">· {note}</span>
    </div>
  );
}
