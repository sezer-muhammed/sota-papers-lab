import { useId, type ReactNode } from "react";
import { cn } from "../lib/cn";

/* An on/off setting: label and optional description on the left, the switch
   on the right. The label is a real <label> for the button, so clicking the
   text toggles it too. On is the action blue. */
export function Switch({
  checked,
  className,
  description,
  disabled = false,
  label,
  onCheckedChange,
}: {
  checked: boolean;
  className?: string;
  description?: ReactNode;
  disabled?: boolean;
  label: ReactNode;
  onCheckedChange: (checked: boolean) => void;
}) {
  const id = useId();
  const descriptionId = description ? `${id}-description` : undefined;

  return (
    <div className={cn("flex items-start justify-between gap-4", className)}>
      <div className={cn("min-w-0", disabled && "opacity-50")}>
        <label htmlFor={id} className="text-[14px] font-medium leading-6 text-[var(--ds-gray-1000)]">
          {label}
        </label>
        {description ? (
          <p id={descriptionId} className="text-[13px] leading-5 text-[var(--ds-gray-900)]">
            {description}
          </p>
        ) : null}
      </div>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        aria-describedby={descriptionId}
        disabled={disabled}
        onClick={() => onCheckedChange(!checked)}
        className={cn(
          "mt-0.5 inline-flex h-5 w-9 shrink-0 items-center rounded-full border outline-none transition-colors focus-visible:shadow-[var(--ds-focus-ring)] disabled:pointer-events-none disabled:opacity-50",
          checked
            ? "border-[var(--ds-blue-700)] bg-[var(--ds-blue-700)]"
            : "border-[var(--ds-gray-alpha-500)] bg-[var(--ds-gray-200)]",
        )}
      >
        <span
          aria-hidden="true"
          className={cn(
            "ml-0.5 h-3.5 w-3.5 rounded-full bg-white shadow-[0_1px_2px_rgb(0_0_0_/_0.25)] transition-transform motion-reduce:transition-none",
            checked && "translate-x-4",
          )}
        />
      </button>
    </div>
  );
}
