import { useState } from "react";

type Mode = "lora" | "engram" | "layered";

export function Architecture() {
  const [mode, setMode] = useState<Mode>("layered");

  return (
    <div>
      <h2>Architecture: substrate selector</h2>
      <p className="prose-paper text-ink-600">
        Click a substrate to see how it modifies the base LM forward pass.
        The fundamental invariant: <strong>per-user edits must be local</strong> —
        they should fire only on forwards whose token sequence matches the fact's
        trigger, and be mathematically inert everywhere else. POLAR-class LoRA
        fails this; Engram-row insertion passes by construction.
      </p>
      <div className="mt-4 flex flex-wrap gap-2">
        <ModeBtn label="Per-user LoRA (POLAR)" subtitle="global edit, +1.56 Δbpb" active={mode === "lora"} onClick={() => setMode("lora")} />
        <ModeBtn label="Per-user Engram override" subtitle="local edit, +0.0001 Δbpb" active={mode === "engram"} onClick={() => setMode("engram")} />
        <ModeBtn label="Layered: Engram + shared LoRA" subtitle="content + meta-skill" active={mode === "layered"} onClick={() => setMode("layered")} />
      </div>

      <div className="card mt-6 p-5">
        <ArchitectureSVG mode={mode} />
        <div className="mt-4 text-sm text-ink-700 leading-relaxed">
          <ArchitectureExplain mode={mode} />
        </div>
      </div>
    </div>
  );
}

function ModeBtn({ label, subtitle, active, onClick }: {
  label: string; subtitle: string; active: boolean; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`text-left px-3 py-2 rounded-md border transition-colors ${
        active
          ? "bg-accent-50 border-accent-500 text-accent-900"
          : "bg-white border-ink-200 text-ink-700 hover:border-ink-400"
      }`}>
      <div className="text-sm font-medium">{label}</div>
      <div className="text-[11px] text-ink-500 mt-0.5">{subtitle}</div>
    </button>
  );
}

function ArchitectureSVG({ mode }: { mode: Mode }) {
  // Highlights based on mode
  const loraActive = mode === "lora";
  const engActive = mode === "engram" || mode === "layered";
  const sharedActive = mode === "layered";

  return (
    <svg viewBox="0 0 760 320" className="w-full h-auto">
      {/* tokens */}
      <text x="60" y="30" fontSize="11" fill="#64748b" fontFamily="JetBrains Mono">
        Input tokens (q)
      </text>
      <rect x="50" y="40" width="120" height="36" rx="6"
            fill="#f1f5f9" stroke="#cbd5e1" />
      <text x="110" y="62" textAnchor="middle" fontSize="12" fill="#0f172a"
            fontFamily="JetBrains Mono">My doctor's name</text>

      {/* arrow into base */}
      <path d="M 170 58 L 220 58" stroke="#334155" strokeWidth="2" fill="none"
            markerEnd="url(#arr)" />

      {/* base LM core (transformer blocks) */}
      <rect x="220" y="20" width="200" height="280" rx="10"
            fill={"#f8fafc"} stroke={loraActive ? "#ef4444" : "#94a3b8"}
            strokeWidth={loraActive ? 2.5 : 1.5} strokeDasharray={loraActive ? "0" : "0"} />
      <text x="320" y="40" textAnchor="middle" fontSize="13" fill="#0f172a" fontWeight="600">
        Base LM (Mini-Engram-d20)
      </text>
      <text x="320" y="56" textAnchor="middle" fontSize="10" fill="#64748b">
        20 transformer blocks
      </text>

      {/* LoRA delta overlay */}
      {loraActive && (
        <g>
          {[80, 130, 180, 230].map((y, i) => (
            <g key={i}>
              <rect x="240" y={y} width="160" height="22" rx="4"
                    fill="#fee2e2" stroke="#ef4444" />
              <text x="320" y={y + 15} textAnchor="middle" fontSize="10"
                    fill="#7f1d1d" fontFamily="JetBrains Mono">
                Q/K/V/O + LoRA Δ
              </text>
            </g>
          ))}
          <text x="320" y="290" textAnchor="middle" fontSize="11" fill="#7f1d1d"
                fontFamily="JetBrains Mono">
            ⚠ every forward sees the delta
          </text>
        </g>
      )}

      {/* Shared LoRA (layered) - subtle, top of base */}
      {sharedActive && (
        <g>
          {[80, 130].map((y, i) => (
            <g key={i}>
              <rect x="240" y={y} width="160" height="22" rx="4"
                    fill="#dbeafe" stroke="#3b82f6" />
              <text x="320" y={y + 15} textAnchor="middle" fontSize="10"
                    fill="#1e3a8a" fontFamily="JetBrains Mono">
                shared LoRA (meta-skill)
              </text>
            </g>
          ))}
          <text x="320" y="290" textAnchor="middle" fontSize="11" fill="#1e3a8a"
                fontFamily="JetBrains Mono">
            cross-user reasoning pattern
          </text>
        </g>
      )}

      {/* Engram table - lives outside the base */}
      <rect x="470" y="60" width="240" height="200" rx="10"
            fill={"#f8fafc"} stroke={engActive ? "#10b981" : "#94a3b8"}
            strokeWidth={engActive ? 2.5 : 1.5} />
      <text x="590" y="80" textAnchor="middle" fontSize="13" fill="#0f172a" fontWeight="600">
        Engram override table
      </text>
      <text x="590" y="96" textAnchor="middle" fontSize="10" fill="#64748b">
        hash-keyed N-gram rows
      </text>
      {/* hash slots */}
      {[
        { y: 110, txt: 'h("doctor name")' },
        { y: 132, txt: 'h("favorite spice")' },
        { y: 154, txt: 'h("home city")' },
      ].map((r, i) => (
        <g key={i}>
          <rect x="490" y={r.y} width="200" height="20" rx="4"
                fill={engActive ? "#d1fae5" : "#f1f5f9"}
                stroke={engActive ? "#10b981" : "#cbd5e1"} />
          <text x="500" y={r.y + 14} fontSize="10"
                fill={engActive ? "#064e3b" : "#94a3b8"}
                fontFamily="JetBrains Mono">{r.txt}</text>
        </g>
      ))}
      {/* trigger arrow */}
      {engActive && (
        <g>
          <path d="M 420 145 L 470 145" stroke="#10b981" strokeWidth="2" fill="none"
                markerEnd="url(#arr-green)" />
          <text x="445" y="138" textAnchor="middle" fontSize="9" fill="#064e3b"
                fontFamily="JetBrains Mono">trigger</text>
          <text x="590" y="220" textAnchor="middle" fontSize="11" fill="#064e3b"
                fontFamily="JetBrains Mono">
            ✓ inert on non-trigger tokens
          </text>
          <text x="590" y="234" textAnchor="middle" fontSize="11" fill="#064e3b"
                fontFamily="JetBrains Mono">
            ✓ Δbpb = +0.0001 by construction
          </text>
        </g>
      )}

      {/* output */}
      <path d="M 420 220 L 470 250" stroke="#334155" strokeWidth="1.5" fill="none" />
      <rect x="50" y="240" width="120" height="36" rx="6"
            fill="#f1f5f9" stroke="#cbd5e1" />
      <text x="110" y="262" textAnchor="middle" fontSize="12" fill="#0f172a"
            fontFamily="JetBrains Mono">"Patel"</text>
      <path d="M 220 258 L 170 258" stroke="#334155" strokeWidth="2" fill="none"
            markerEnd="url(#arr)" />
      <text x="60" y="295" fontSize="11" fill="#64748b" fontFamily="JetBrains Mono">
        Predicted next token
      </text>

      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#334155" />
        </marker>
        <marker id="arr-green" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981" />
        </marker>
      </defs>
    </svg>
  );
}

function ArchitectureExplain({ mode }: { mode: Mode }) {
  if (mode === "lora") {
    return <p>
      <strong>Per-user LoRA (POLAR-class):</strong> a low-rank delta is added to
      every Q/K/V/O projection in every block. Every forward pass — including
      those that have nothing to do with the user's facts — sees the delta. We
      measure +1.56 val_bpb on held-out ClimbMix text (Mini-Engram-d20),
      a 2.1× degradation. 17/20 users worse than no-edit base on indirect
      reasoning. Direct recall is 100% on every base.
    </p>;
  }
  if (mode === "engram") {
    return <p>
      <strong>Per-user Engram-row insertion:</strong> the base is Engram-pretrained
      (DeepSeek's architecture); at each token position, suffix N-grams hash into
      embedding-table rows whose retrieval is gated by an attention-style scalar.
      We write per-user fact rows into an override map and apply it per request.
      Δbpb = +0.0001 (~15 000× less than LoRA) because the gate fires only on the
      fact's trigger N-gram and the lookup is the identity elsewhere.
    </p>;
  }
  return <p>
    <strong>Layered (F):</strong> one shared LoRA (rank 16, ~12 MB) is trained
    once on cross-user indirect-reasoning data and amortised across all users —
    this carries the meta-skill. Per-user Engram-row overrides (88 KB/user) carry
    the user's content. At inference both apply: Engram provides the fact via
    trigger lookup, shared LoRA provides the reasoning pattern. Direct recall
    100%, indirect-any 44%, 0/20 users worse than base, Δbpb +0.39 (only the
    shared-LoRA cost). Invariant to per-user fact count up to the Engram density
    ceiling.
  </p>;
}
