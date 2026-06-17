import { animate, useInView, useMotionValue } from "motion/react";
import { useEffect, useRef, useState } from "react";

/** Counts up to `value` once when scrolled into view. */
export function Counter({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  duration = 1.6,
  className = "",
}: {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  duration?: number;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const mv = useMotionValue(0);
  const [display, setDisplay] = useState("0");

  useEffect(() => {
    if (!inView) return;
    const controls = animate(mv, value, {
      duration,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => {
        const n = decimals > 0 ? v.toFixed(decimals) : Math.round(v).toLocaleString();
        setDisplay(n);
      },
    });
    return () => controls.stop();
  }, [inView, value, decimals, duration, mv]);

  return (
    <span ref={ref} className={className}>
      {prefix}
      {display}
      {suffix}
    </span>
  );
}
