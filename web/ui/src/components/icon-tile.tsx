import type { LucideIcon } from "lucide-react";
import { cn } from "../lib/cn";

/* The ATSGrade 40px card tile: an icon, or a short mono label (e.g. app
   initials). Use tone="muted" on a muted card so the tile stays lighter. */
export function IconTile({
  className,
  icon: Icon,
  label,
  tone = "default",
}: {
  className?: string;
  icon?: LucideIcon;
  label?: string;
  tone?: "default" | "muted";
}) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "grid h-10 w-10 shrink-0 place-items-center rounded-[8px] border border-[var(--ds-gray-alpha-300)]",
        tone === "muted" ? "bg-[var(--ds-background-100)]" : "bg-[var(--ds-background-200)]",
        className,
      )}
    >
      {Icon ? (
        <Icon className="h-5 w-5 text-[var(--ds-gray-800)]" />
      ) : (
        <span className="font-mono text-[13px] font-semibold uppercase text-[var(--ds-gray-800)]">
          {label}
        </span>
      )}
    </span>
  );
}
