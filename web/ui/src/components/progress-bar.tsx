import { cn } from "../lib/cn";

/* A thin bar for long work: determinate with a `value` from 0 to 1,
   indeterminate (the shared sweep) without one. The fill is neutral like
   the sweep: progress is not a status, so it carries no colour. */
export function ProgressBar({
  className,
  label,
  value,
}: {
  className?: string;
  /** Accessible name, e.g. "Upload of DSC0001.ARW". */
  label: string;
  value?: number | null;
}) {
  const determinate = typeof value === "number" && Number.isFinite(value);
  const percent = determinate ? Math.round(Math.min(1, Math.max(0, value)) * 100) : undefined;

  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={determinate ? 0 : undefined}
      aria-valuemax={determinate ? 100 : undefined}
      aria-valuenow={percent}
      className={cn(
        "h-1 w-full overflow-hidden rounded-full bg-[var(--ds-gray-alpha-200)]",
        !determinate && "ds-indeterminate",
        className,
      )}
    >
      {determinate ? (
        <div
          className="h-full rounded-full bg-[var(--ds-gray-1000)] transition-[width] duration-300 motion-reduce:transition-none"
          style={{ width: `${percent}%` }}
        />
      ) : null}
    </div>
  );
}
