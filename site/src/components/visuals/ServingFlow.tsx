import { headline } from "../../data/headline";

export function ServingFlow() {
  const users = [0, 1, 2];
  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-7">
      <div className="grid md:grid-cols-[1fr_auto] gap-8 items-center">
        {/* static flow diagram */}
        <div className="flex items-center justify-between gap-3">
          <div className="space-y-2.5">
            {users.map((u) => (
              <div key={u} className="flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-paper-200 grid place-items-center font-mono text-[0.56rem] text-ink-50">
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

          <div className="flex-1 mx-1 flex items-center">
            <span className="h-px flex-1 bg-rule" />
            <span className="font-mono text-[0.56rem] text-ink-50 px-2">→</span>
            <span className="h-px flex-1 bg-rule" />
          </div>

          <div className="rounded-lg border border-engram/40 bg-engram-wash px-4 py-4 text-center">
            <div className="font-mono text-[0.56rem] uppercase tracking-[0.16em] text-engram-deep">EngramServer</div>
            <div className="font-body text-[0.78rem] text-ink-50 mt-1 leading-tight">
              swap user's<br />override map<br />→ forward → restore
            </div>
          </div>
        </div>

        {/* static stats */}
        <div className="space-y-4">
          <Stat value={`${headline.reqPerSec}`} unit="req/s" note="flat as tenants triple" />
          <Stat value="< 1" unit="ms" note="override apply" />
          <Stat value="0" unit="leaks" note="cross-user, by construction" ours />
        </div>
      </div>
    </div>
  );
}

function Stat({ value, unit, note, ours }: { value: string; unit: string; note: string; ours?: boolean }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className={`display text-2xl ${ours ? "text-engram-deep" : "text-ink"}`}>{value}</span>
      <span className="font-mono text-[0.66rem] text-ink-50">{unit}</span>
      <span className="font-body text-[0.82rem] text-ink-50">· {note}</span>
    </div>
  );
}
