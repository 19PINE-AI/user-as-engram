import type { ReactNode } from "react";
import { Reveal } from "./Reveal";

/**
 * A section in the continuous, centered single-column reading flow: a kicker
 * (mono, numbered), a serif display headline, an optional one-line lede, and
 * the visual. One shared max-width so every section lines up down the middle.
 */
export function Scene({
  id,
  index,
  kicker,
  title,
  lede,
  children,
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
    <section id={id} className="relative px-6 md:px-12 py-12 md:py-14 scroll-mt-16">
      <div className="mx-auto w-full max-w-4xl">
        <Reveal>
          <div className="flex items-baseline gap-4 mb-4">
            <span className="kicker">{kicker}</span>
            <span className="flex-1 border-t hairline mt-2" />
            <span className="eyebrow-num">{index}</span>
          </div>
        </Reveal>
        <Reveal delay={0.05}>
          <h2 className="display text-[clamp(1.5rem,3vw,2.3rem)] text-ink max-w-[20ch]">
            {title}
          </h2>
        </Reveal>
        {lede && (
          <Reveal delay={0.1}>
            <p className="lede mt-4 max-w-prose">{lede}</p>
          </Reveal>
        )}
        {children && (
          <Reveal delay={0.15} className="mt-7">
            {children}
          </Reveal>
        )}
      </div>
    </section>
  );
}
