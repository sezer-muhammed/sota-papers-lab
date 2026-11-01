import { useEffect, useRef, useState } from "react";
import { cn } from "../lib/cn";

type GaugeProps = {
  className?: string;
  /** Arc colour. Defaults to the neutral foreground; pass a --ds-*-700 for a status gauge. */
  color?: string;
  label?: string;
  size?: number;
  strokeWidth?: number;
  /** Show the numeric value in the centre. */
  showValue?: boolean;
  unit?: string;
  /** 0–100 */
  value: number;
};

// rAF count-up from 0 to target (ease-out cubic); snaps instantly when the
// viewer prefers reduced motion.
function useCountUp(target: number, duration = 700): number {
  const [display, setDisplay] = useState(0);
  const raf = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (typeof window === "undefined") {
      setDisplay(target);
      return;
    }
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      setDisplay(target);
      return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(target * eased);
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [target, duration]);
  return display;
}

// A compact 270° speedometer-style gauge: neutral track + value arc. The value
// counts up on mount and the arc fills in step with it.
export function Gauge({
  className,
  color = "var(--ds-gray-1000)",
  label,
  size = 96,
  strokeWidth = 8,
  showValue = true,
  unit,
  value,
}: GaugeProps) {
  const target = Math.max(0, Math.min(100, value));
  const clamped = useCountUp(target);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const sweep = 0.75; // 270° of the circle
  const track = circumference * sweep;
  const filled = track * (clamped / 100);

  return (
    <div className={cn("inline-flex flex-col items-center", className)}>
      <div className="relative" style={{ height: size, width: size }}>
        <svg
          aria-hidden="true"
          className="-rotate-[135deg]"
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          width={size}
        >
          <circle
            cx={size / 2}
            cy={size / 2}
            fill="none"
            r={radius}
            stroke="var(--ds-gray-alpha-300)"
            strokeDasharray={`${track} ${circumference}`}
            strokeLinecap="round"
            strokeWidth={strokeWidth}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            fill="none"
            r={radius}
            stroke={color}
            strokeDasharray={`${filled} ${circumference}`}
            strokeLinecap="round"
            strokeWidth={strokeWidth}
          />
        </svg>
        {showValue ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-[20px] font-semibold leading-none text-[var(--ds-gray-1000)] [font-variant-numeric:tabular-nums]">
              {Math.round(clamped)}
              {unit ? (
                <span className="ml-0.5 text-[11px] font-medium text-[var(--ds-gray-700)]">
                  {unit}
                </span>
              ) : null}
            </span>
          </div>
        ) : null}
      </div>
      {label ? (
        <span className="mt-1 font-mono text-[11px] uppercase tracking-normal text-[var(--ds-gray-700)]">
          {label}
        </span>
      ) : null}
    </div>
  );
}
