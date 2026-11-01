import { useId, type ReactNode } from "react";
import { cn } from "../lib/cn";

export type SegmentedOption<T extends string> = { value: T; label: ReactNode };

/* A small set of exclusive choices (a model, a count). Native radios under
   the segments, so arrow keys, Tab and screen readers behave as a radio
   group with no extra code. */
export function SegmentedControl<T extends string>({
  className,
  disabled = false,
  label,
  onChange,
  options,
  value,
}: {
  className?: string;
  disabled?: boolean;
  /** Accessible name of the group. */
  label: string;
  onChange: (value: T) => void;
  options: SegmentedOption<T>[];
  value: T;
}) {
  const name = useId();

  return (
    <div
      role="radiogroup"
      aria-label={label}
      aria-disabled={disabled || undefined}
      className={cn(
        "inline-flex rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] p-0.5",
        disabled && "opacity-50",
        className,
      )}
    >
      {options.map((option) => (
        <label key={option.value} className="relative flex">
          <input
            type="radio"
            name={name}
            value={option.value}
            checked={value === option.value}
            disabled={disabled}
            onChange={() => onChange(option.value)}
            className="peer sr-only"
          />
          <span
            className={cn(
              "inline-flex h-7 min-w-9 cursor-pointer items-center justify-center rounded-[5px] px-2.5 text-[13px] font-medium text-[var(--ds-gray-900)] transition-colors",
              "hover:text-[var(--ds-gray-1000)] peer-checked:bg-[var(--ds-background-100)] peer-checked:text-[var(--ds-gray-1000)] peer-checked:shadow-[0_1px_2px_rgb(0_0_0_/_0.08)]",
              "peer-focus-visible:shadow-[var(--ds-focus-ring)] peer-disabled:cursor-default",
            )}
          >
            {option.label}
          </span>
        </label>
      ))}
    </div>
  );
}
