import { mayaFacts, questions } from "../../data/facts";

// Static: both jobs shown at once — recall (look one fact up) and reason
// (chain several). No toggle.
const rows = [
  {
    mode: "recall" as const,
    q: questions.recall.q,
    needs: questions.recall.needs,
    active: ["cardiologist"],
    accent: "text-engram-deep",
    dot: "bg-engram",
  },
  {
    mode: "reason" as const,
    q: questions.reason.q,
    needs: questions.reason.needs,
    active: ["cardiologist", "clinic", "diet"],
    accent: "text-rust-deep",
    dot: "bg-rust",
  },
];

export function TwoJobs() {
  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-7 space-y-5">
      {rows.map((row) => (
        <div key={row.mode} className="grid md:grid-cols-[auto_1fr] gap-x-6 gap-y-3 items-start">
          {/* job label */}
          <div className="flex md:flex-col items-center md:items-start gap-2 md:w-28">
            <span className={`font-mono text-[0.66rem] uppercase tracking-[0.16em] ${row.accent}`}>
              {row.mode}
            </span>
            <span className="font-body text-[0.8rem] text-ink-50 leading-snug">{row.needs}</span>
          </div>

          {/* question + the facts it needs */}
          <div>
            <p className="font-body text-base md:text-lg text-ink leading-snug">“{row.q}”</p>
            <div className="flex flex-wrap gap-1.5 mt-2.5">
              {mayaFacts.map((f) => {
                const on = row.active.includes(f.id);
                return (
                  <span
                    key={f.id}
                    className={`font-mono text-[0.66rem] px-2 py-1 rounded border ${
                      on
                        ? "border-engram/40 bg-engram-wash text-engram-deep"
                        : "border-rule text-ink-50 opacity-50"
                    }`}
                  >
                    {f.answer}
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      ))}
      <p className="font-body text-[0.82rem] text-ink-50 leading-snug border-t hairline pt-4">
        The same store must do both — and the recall job and the reasoning job pull on the model in
        different ways.
      </p>
    </div>
  );
}
