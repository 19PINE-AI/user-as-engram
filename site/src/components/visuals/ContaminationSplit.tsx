import { C } from "../../theme";
import { headline } from "../../data/headline";

const LAYERS = 9;
const POS = 16;
const TRIG = 11;
const ENGRAM_LAYER = 4;

function rnd(i: number) {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function Heatmap({ mode }: { mode: "engram" | "lora" }) {
  const cellW = 100 / POS;
  const cellH = 100 / LAYERS;
  const base = mode === "engram" ? C.engram : C.rust;
  return (
    <svg viewBox="0 0 100 56" className="w-full">
      {Array.from({ length: LAYERS * POS }).map((_, idx) => {
        const r = Math.floor(idx / POS);
        const c = idx % POS;
        let intensity = 0;
        if (mode === "engram") intensity = c === TRIG && r >= ENGRAM_LAYER ? 1 : 0;
        else intensity = 0.28 + 0.72 * rnd(idx);
        return (
          <rect
            key={idx}
            x={c * cellW + 0.15}
            y={(r * cellH + 0.15) * 0.56}
            width={cellW - 0.3}
            height={cellH * 0.56 - 0.3}
            rx={0.3}
            fill={intensity > 0 ? base : C.ink}
            opacity={intensity > 0 ? 0.25 + 0.75 * intensity : 0.08}
          />
        );
      })}
    </svg>
  );
}

function Strip({ mode }: { mode: "engram" | "lora" }) {
  const n = 22;
  return (
    <div className="mt-2">
      <div className="font-mono text-[0.58rem] text-ink-50 mb-1">
        unrelated text · “The capital of France is …”
      </div>
      <svg viewBox="0 0 100 6" className="w-full">
        {Array.from({ length: n }).map((_, i) => {
          const lit = mode === "lora" ? 0.3 + 0.7 * rnd(i + 99) : 0;
          return (
            <rect
              key={i}
              x={(100 / n) * i + 0.2}
              y={0.3}
              width={100 / n - 0.4}
              height={5.4}
              rx={0.4}
              fill={lit > 0 ? C.rust : C.ink}
              opacity={lit > 0 ? 0.25 + 0.7 * lit : 0.08}
            />
          );
        })}
      </svg>
    </div>
  );
}

export function ContaminationSplit() {
  return (
    <div className="grid md:grid-cols-[1fr_auto_1fr] gap-6 md:gap-4 items-center">
      <Panel
        tag="Engram row"
        color="engram"
        caption={
          <>
            one column moves at the trigger. every other position:{" "}
            <b className="text-engram-deep">exactly 0.000</b>.
          </>
        }
      >
        <Heatmap mode="engram" />
        <Strip mode="engram" />
      </Panel>

      <div className="flex md:flex-col items-center justify-center gap-1.5 py-2">
        <div className="display text-[clamp(1.6rem,3.4vw,2.6rem)] text-ink leading-none">
          {headline.contamRatio.toLocaleString()}×
        </div>
        <div className="font-mono text-[0.56rem] uppercase tracking-[0.14em] text-ink-50 text-center max-w-[8rem]">
          less disruption to unrelated text
        </div>
      </div>

      <Panel
        tag="Per-user LoRA"
        color="rust"
        caption={
          <>
            moves every position, every layer — and shifts{" "}
            <b className="text-rust-deep">unrelated</b> text too.
          </>
        }
      >
        <Heatmap mode="lora" />
        <Strip mode="lora" />
      </Panel>
    </div>
  );
}

function Panel({
  tag,
  color,
  caption,
  children,
}: {
  tag: string;
  color: "engram" | "rust";
  caption: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border hairline bg-paper-50 p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className={`w-1.5 h-1.5 rounded-full ${color === "engram" ? "bg-engram" : "bg-rust"}`} />
        <span className="font-mono text-[0.64rem] uppercase tracking-[0.16em] text-ink-100">{tag}</span>
      </div>
      {children}
      <p className="font-body text-[0.82rem] text-ink-50 leading-snug mt-3">{caption}</p>
    </div>
  );
}
