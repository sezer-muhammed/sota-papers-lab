import type { CSSProperties, InputHTMLAttributes } from "react";
import { cn } from "../lib/cn";

/* A native range input in the system's look (styles: .ds-range in base.css).
   The track fills from `origin` to the value, so a bipolar slider (-100 to
   100, origin 0) reads from its centre the way Lightroom's do. `track`
   replaces the fill with a gradient, e.g. a white-balance scale. */
export function RangeInput({
  className,
  max,
  min,
  onValueChange,
  origin,
  step,
  style,
  track,
  value,
  ...props
}: Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "value" | "min" | "max" | "step" | "onChange"> & {
  max: number;
  min: number;
  onValueChange: (value: number) => void;
  /** Where the fill starts; defaults to `min`. */
  origin?: number;
  step?: number;
  /** A CSS background for the whole track instead of the fill. */
  track?: string;
  value: number;
}) {
  const span = max - min || 1;
  const position = (v: number) => ((Math.min(max, Math.max(min, v)) - min) / span) * 100;
  const a = position(origin ?? min);
  const b = position(value);
  const vars = {
    "--range-from": `${Math.min(a, b)}%`,
    "--range-to": `${Math.max(a, b)}%`,
    ...(track ? { "--range-track": track } : {}),
  } as CSSProperties;

  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(event) => onValueChange(Number(event.target.value))}
      className={cn("ds-range", className)}
      style={{ ...vars, ...style }}
      {...props}
    />
  );
}
