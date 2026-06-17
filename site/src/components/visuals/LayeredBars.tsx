import { conditions } from "../../data/conditions";

export function LayeredBars() {
  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-7">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 mb-5 font-mono text-[0.62rem] uppercase tracking-[0.14em]">
        <Key className="bg-rule" label="direct recall" />
        <Key className="bg-engram" label="indirect reasoning" />
        <span className="ml-auto text-ink-50 normal-case tracking-normal font-body text-[0.82rem]">
          contamination →
        </span>
      </div>

      <div className="space-y-2">
        {conditions.map((c) => (
          <div
            key={c.name}
            className={`grid grid-cols-[9.5rem_1fr_4rem] items-center gap-3 rounded-lg px-3 py-2 ${
              c.ours ? "bg-engram-wash ring-1 ring-engram/40" : ""
            }`}
          >
            <div className="flex items-center gap-1.5 min-w-0">
              <span className={`font-body text-[0.85rem] truncate ${c.ours ? "text-engram-deep font-medium" : "text-ink-100"}`}>
                {c.name}
              </span>
              {c.ours && (
                <span className="font-mono text-[0.5rem] uppercase tracking-[0.12em] text-engram bg-paper-50 border border-engram/40 rounded-full px-1.5 py-0.5">
                  ours
                </span>
              )}
            </div>

            <div className="space-y-1.5">
              <Bar value={c.direct} kind="direct" />
              <Bar value={c.indirect} kind="indirect" ours={c.ours} />
            </div>

            <div className="text-right">
              <span
                className={`font-mono text-[0.68rem] px-1.5 py-1 rounded ${
                  c.cost === "high"
                    ? "bg-rust-wash text-rust-deep"
                    : c.cost === "shared"
                      ? "bg-engram-wash text-engram-deep"
                      : "text-ink-50"
                }`}
              >
                {c.bpb < 0.001 ? "≈0" : `+${c.bpb.toFixed(2)}`}
              </span>
            </div>
          </div>
        ))}
      </div>

      <p className="font-body text-[0.82rem] text-ink-50 leading-snug mt-5 max-w-prose">
        The layered design matches per-user LoRA's <b className="text-ink-100">direct recall</b> while
        answering indirect questions far more often — at the{" "}
        <b className="text-engram-deep">same contamination as the shared skill alone</b>, and none
        added by the per-user rows.
      </p>
    </div>
  );
}

function Bar({ value, kind, ours }: { value: number; kind: "direct" | "indirect"; ours?: boolean }) {
  const color = kind === "direct" ? "bg-rule" : ours ? "bg-engram" : "bg-engram-lighter";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 rounded-full bg-paper-200 overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${value}%` }} />
      </div>
      <span className="font-mono text-[0.58rem] text-ink-50 w-7 text-right tabular-nums">{value}%</span>
    </div>
  );
}

function Key({ className, label }: { className: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-ink-50">
      <span className={`w-3 h-2 rounded-full ${className}`} />
      {label}
    </span>
  );
}
