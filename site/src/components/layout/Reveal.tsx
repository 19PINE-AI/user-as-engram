import type { ReactNode } from "react";

/**
 * Static wrapper. The site is meant to be read top-to-bottom, so content is
 * always visible — no scroll-triggered reveal that could leave a section blank.
 * (Kept as a component so callers don't change; `delay` is ignored.)
 */
export function Reveal({
  children,
  className = "",
}: {
  children: ReactNode;
  delay?: number;
  y?: number;
  className?: string;
}) {
  return <div className={className}>{children}</div>;
}
