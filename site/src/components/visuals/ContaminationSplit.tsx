import { motion } from "motion/react";
import { useState } from "react";
import { C } from "../../theme";
import { Counter } from "../ui/Counter";
import { headline } from "../../data/headline";

const LAYERS = 9;
const POS = 16;
const TRIG = 11; // trigger position (column)
const ENGRAM_LAYER = 4; // engram read happens here; before it, exactly 0

// deterministic pseudo-random in [0,1]
function rnd(i: number) {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function Heatmap({
  mode,
  on,
}: {
  mode: "engram" | "lora";
  on: boolean;
}) {
  const cellW = 100 / POS;
  const cellH = 100 / LAYERS;
  const base = mode === "engram" ? C.engram : C.rust;
  return (
    <svg viewBox="0 0 100 56" className="w-full">
      {Array.from({ length: LAYERS * POS }).map((_, idx) => {
        const r = Math.floor(idx / POS);
        const c = idx % POS;
        let intensity = 0;
        if (on) {
          if (mode === "engram") {
            // exactly 0 except the trigger column at/after the engram layer
            intensity = c === TRIG && r >= ENGRAM_LAYER ? 1 : 0;
          } else {
            // nonzero at every position and every layer
            intensity = 0.28 + 0.72 * rnd(idx);
          }
        }
        return (
          <motion.rect
            key={idx}
            x={c * cellW + 0.15}
            y={(r * cellH + 0.15) * 0.56}
            width={cellW - 0.3}
            height={cellH * 0.56 - 0.3}
            rx={0.3}
            initial={false}
            animate={{
              fill: intensity > 0 ? base : C.ink,
              opacity: intensity > 0 ? 0.25 + 0.75 * intensity : 0.08,
            }}
            transition={{ duration: 0.6, delay: on ? (mode === "lora" ? rnd(idx) * 0.5 : 0.2) : 0 }}
          />
        );
      })}
    </svg>
  );
}

function Strip({ mode, on }: { mode: "engram" | "lora"; on: boolean }) {
  // "unrelated text" bar — Engram leaves it untouched; LoRA disrupts it.
  const n = 22;
  return (
    <div className="mt-2">
      <div className="font-mono text-[0.6rem] text-ink-50 mb-1">
        unrelated text · “The capital of France is …”
      </div>
      <svg viewBox="0 0 100 6" className="w-full">
        {Array.from({ length: n }).map((_, i) => {
          const lit = on && mode === "lora" ? 0.3 + 0.7 * rnd(i + 99) : 0;
          return (
            <motion.rect
              key={i}
              x={(100 / n) * i + 0.2}
              y={0.3}
              width={100 / n - 0.4}
              height={5.4}
              rx={0.4}
              initial={false}
              animate={{
                fill: lit > 0 ? C.rust : C.ink,
                opacity: lit > 0 ? 0.25 + 0.7 * lit : 0.08,
              }}
              transition={{ duration: 0.5, delay: on ? rnd(i) * 0.4 : 0 }}
            />
          );
        })}
      </svg>
    </div>
  );
}

export function ContaminationSplit() {
  const [on, setOn] = useState(false);
  return (
    <div>
      <div className="grid md:grid-cols-[1fr_auto_1fr] gap-6 md:gap-4 items-center">
        {/* Engram */}
        <Panel
          tag="Engram row"
          color="engram"
          caption={
            <>
              one column moves at the trigger.
              <br />
              every other position: <b className="text-engram-deep">exactly 0.000</b>
            </>
          }
        >
          <Heatmap mode="engram" on={on} />
          <Strip mode="engram" on={on} />
        </Panel>

        {/* counter */}
        <div className="flex md:flex-col items-center justify-center gap-2 py-2">
          <div className="font-display display text-[clamp(1.8rem,4vw,3rem)] text-ink leading-none">
            {on ? <Counter value={headline.contamRatio} suffix="×" /> : "—"}
          </div>
          <div className="font-mono text-[0.6rem] uppercase tracking-[0.16em] text-ink-50 text-center max-w-[8rem]">
            less disruption to unrelated text
          </div>
        </div>

        {/* LoRA */}
        <Panel
          tag="Per-user LoRA"
          color="rust"
          caption={
            <>
              moves every position, every layer —
              <br />
              and shifts <b className="text-rust-deep">unrelated</b> text too.
            </>
          }
        >
          <Heatmap mode="lora" on={on} />
          <Strip mode="lora" on={on} />
        </Panel>
      </div>

      <div className="flex justify-center mt-9">
        <button
          onClick={() => setOn((v) => !v)}
          className="font-mono text-[0.74rem] uppercase tracking-[0.2em] px-6 py-3 rounded-full bg-ink text-paper-50 hover:bg-engram transition-colors"
        >
          {on ? "↺ clear the write" : "Write Maya's penicillin allergy →"}
        </button>
      </div>
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
    <div className="rounded-xl border hairline bg-paper-50 p-5">
      <div className="flex items-center gap-2 mb-3">
        <span className={`w-2 h-2 rounded-full ${color === "engram" ? "bg-engram" : "bg-rust"}`} />
        <span className="font-mono text-[0.72rem] uppercase tracking-[0.16em] text-ink-100">
          {tag}
        </span>
      </div>
      {children}
      <p className="font-body text-sm text-ink-50 leading-snug mt-3">{caption}</p>
    </div>
  );
}
