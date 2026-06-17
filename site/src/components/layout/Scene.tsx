import type { ReactNode } from "react";
import { Reveal } from "./Reveal";

/**
 * A standard full-height scene: a kicker (mono, numbered), a serif display
 * headline, an optional one-line lede, and the visual. Minimal text by design.
 */
export function Scene({
  id,
  index,
  kicker,
  title,
  lede,
  children,
  wide = false,
}: {
  id: string;
  index: string;
  kicker: string;
  title: ReactNode;
  lede?: ReactNode;
  children?: ReactNode;
  wide?: boolean;
}) {
  return (
    <section
      id={id}
      className="relative min-h-screen flex flex-col justify-center px-6 md:px-12 lg:px-20 py-24"
    >
      <div className={wide ? "mx-auto w-full max-w-6xl" : "mx-auto w-full max-w-5xl"}>
        <Reveal>
          <div className="flex items-baseline gap-4 mb-5">
            <span className="kicker">{kicker}</span>
            <span className="flex-1 border-t hairline mt-2" />
            <span className="eyebrow-num">{index}</span>
          </div>
        </Reveal>
        <Reveal delay={0.05}>
          <h2 className="display text-[clamp(2.1rem,5.2vw,4rem)] text-ink max-w-[18ch]">
            {title}
          </h2>
        </Reveal>
        {lede && (
          <Reveal delay={0.12}>
            <p className="lede mt-6 max-w-prose">{lede}</p>
          </Reveal>
        )}
        {children && (
          <Reveal delay={0.18} className="mt-10 md:mt-14">
            {children}
          </Reveal>
        )}
      </div>
    </section>
  );
}
