import { useEffect, useRef, useState } from "react";

interface AnimatedNumberProps {
  /** The real, already-known numeric value to display — never fabricated. */
  value: number;
  /** Formats both the animating and final value; defaults to a rounded, comma-grouped integer. */
  formatter?: (n: number) => string;
  durationMs?: number;
  className?: string;
  /** Escape hatch for non-HTML contexts (e.g. SVG <text>/<tspan>) where the default <span> wrapper isn't valid markup. */
  render?: (formatted: string) => React.ReactNode;
}

function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

// Eases the displayed number from whatever it last showed to a new real
// value — runs once on first mount (0 -> value) and again only when `value`
// actually changes (e.g. a re-fetched profile, a freshly trained model).
// Never loops, never invents intermediate data (it's just tweening between
// two real numbers), and always settles on the exact formatted source value.
export default function AnimatedNumber({ value, formatter, durationMs = 700, className, render }: AnimatedNumberProps) {
  const target = Number.isFinite(value) ? value : 0;
  const [display, setDisplay] = useState(target);
  const fromRef = useRef(target);
  const rafRef = useRef<number | null>(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    if (!mountedRef.current) {
      // First mount: animate up from 0 rather than snapping in place.
      mountedRef.current = true;
      fromRef.current = 0;
    }

    if (prefersReducedMotion()) {
      fromRef.current = target;
      setDisplay(target);
      return;
    }

    const from = fromRef.current;
    if (from === target) {
      setDisplay(target);
      return;
    }

    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    const start = performance.now();

    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      if (t >= 1) {
        setDisplay(target);
        fromRef.current = target;
        rafRef.current = null;
        return;
      }
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(from + (target - from) * eased);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, durationMs]);

  const format = formatter ?? ((n: number) => Math.round(n).toLocaleString());
  const formatted = format(display);
  return render ? <>{render(formatted)}</> : <span className={className}>{formatted}</span>;
}
