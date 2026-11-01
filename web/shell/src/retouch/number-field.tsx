import { useState } from "react";
import { cn } from "@spl/ui";

/* A number input that commits on Enter or blur, clamped to [min, max]. While
   you type, the text is yours: a half-typed "5" never becomes 2000 K. Escape
   puts the last committed value back. */
export function NumberField({
  className,
  disabled,
  id,
  label,
  max,
  min,
  onCommit,
  step = 1,
  value,
}: {
  className?: string;
  disabled?: boolean;
  id?: string;
  /** Accessible name when no <label> points at `id`. */
  label?: string;
  max: number;
  min: number;
  onCommit: (value: number) => void;
  step?: number;
  value: number;
}) {
  const [draft, setDraft] = useState<string>();

  const commit = () => {
    if (draft === undefined) return;
    const parsed = Number(draft);
    if (draft.trim() !== "" && Number.isFinite(parsed)) {
      const decimals = step < 1 ? Math.ceil(-Math.log10(step)) : 0;
      const factor = 10 ** decimals;
      onCommit(Math.round(Math.min(max, Math.max(min, parsed)) * factor) / factor);
    }
    setDraft(undefined);
  };

  return (
    <input
      id={id}
      type="number"
      inputMode={step < 1 ? "decimal" : "numeric"}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      aria-label={label}
      value={draft ?? String(value)}
      onFocus={(event) => event.currentTarget.select()}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") commit();
        if (event.key === "Escape") setDraft(undefined);
      }}
      className={cn(
        "h-8 w-24 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2 text-right font-mono text-[13px] tabular-nums text-[var(--ds-gray-1000)] outline-none focus-visible:shadow-[var(--ds-focus-ring)] disabled:opacity-50",
        className,
      )}
    />
  );
}
