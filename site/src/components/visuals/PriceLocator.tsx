import { useState } from "react";
import { motion } from "motion/react";
import { C } from "../../theme";

// Schematic: where each method's accuracy cost lands, and what axis it grows on.
// - per-user LoRA: in the shared weights — paid on every query, stacks with users.
// - Engram row:    at the one address — grows only with a user's own facts.
// - retrieval:     in the search index — grows with the candidate pool (users × facts).

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

export function PriceLocator() {
  const [facts, setFacts] = useState(0.45); // 0..1 (log facts/user)
  const [users, setUsers] = useState(0.4); // 0..1 (log users)

  const factCount = Math.round(lerp(1, 1000, facts ** 1.6));
  const userCount = Math.round(10 ** lerp(0, 6, users));

  const loraCost = Math.min(100, 34 + 66 * users);
  const engramCost = Math.min(100, 12 + 80 * facts);
  const ragCost = Math.min(100, 8 + 92 * (0.5 * facts + 0.5 * users));

  const cards = [
    {
      key: "lora",
      method: "Per-user LoRA",
      where: "in the shared weights",
      grows: "every query · stacks with users",
      cost: loraCost,
      color: C.rust,
    },
    {
      key: "engram",
      method: "Engram row",
      where: "at the one address",
      grows: "grows only with a user's own facts",
      cost: engramCost,
      color: C.engram,
      ours: true,
    },
    {
      key: "rag",
      method: "Retrieval",
      where: "in the search index",
      grows: "grows with the candidate pool",
      cost: ragCost,
      color: C.ochre,
    },
  ];

  return (
    <div className="rounded-xl border hairline bg-paper-50 p-5 md:p-7">
      <div className="grid sm:grid-cols-3 gap-5">
        {cards.map((c) => (
          <div
            key={c.key}
            className={`rounded-lg border p-4 ${c.ours ? "border-engram/40 bg-engram-wash" : "border-rule bg-paper"}`}
          >
            <div className="font-body text-lg text-ink leading-tight">{c.method}</div>
            <div className="font-mono text-[0.66rem] uppercase tracking-[0.12em] mt-0.5" style={{ color: c.color }}>
              {c.where}
            </div>
            {/* cost vessel */}
            <div className="mt-4 flex items-end gap-3">
              <div className="relative w-full h-28 rounded-md bg-paper-200 overflow-hidden">
                <motion.div
                  className="absolute bottom-0 left-0 right-0"
                  style={{ background: c.color }}
                  animate={{ height: `${c.cost}%` }}
                  transition={{ type: "spring", stiffness: 120, damping: 20 }}
                />
                <div className="absolute inset-0 grid grid-rows-4">
                  {[0, 1, 2, 3].map((i) => (
                    <div key={i} className="border-t border-paper-50/40" />
                  ))}
                </div>
              </div>
            </div>
            <div className="font-body text-xs text-ink-50 leading-snug mt-3 h-8">{c.grows}</div>
          </div>
        ))}
      </div>

      {/* controls */}
      <div className="grid sm:grid-cols-2 gap-6 mt-7">
        <Slider
          label="facts per user"
          value={facts}
          onChange={setFacts}
          readout={factCount.toLocaleString()}
        />
        <Slider
          label="number of users"
          value={users}
          onChange={setUsers}
          readout={userCount >= 1000 ? `${(userCount / 1000).toFixed(userCount >= 1e6 ? 1 : 0)}${userCount >= 1e6 ? "M" : "K"}` : `${userCount}`}
        />
      </div>
      <p className="font-body text-sm text-ink-50 leading-snug mt-5 max-w-prose">
        No method avoids the price. Drag the dimensions and watch <i>where</i> it lands —
        and how fast it grows on each method's own axis. That is the whole question.
      </p>
    </div>
  );
}

function Slider({
  label,
  value,
  onChange,
  readout,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  readout: string;
}) {
  return (
    <div>
      <div className="flex justify-between font-mono text-[0.68rem] uppercase tracking-[0.14em] text-ink-50 mb-1">
        <span>{label}</span>
        <span className="text-ink">{readout}</span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        className="w-full accent-engram"
        aria-label={label}
      />
    </div>
  );
}
