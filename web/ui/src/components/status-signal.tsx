import type { ReactNode } from "react";
import { cn } from "../lib/cn";

type StatusSignalVariant = "dot" | "inline" | "pill" | "cell" | "glass";

export function StatusSignal({
  children,
  className,
  color,
  pulse = false,
  variant = "inline",
}: {
  children?: ReactNode;
  className?: string;
  color: string;
  pulse?: boolean;
  variant?: StatusSignalVariant;
}) {
  const dot = (
    <span className="relative inline-flex h-2.5 w-2.5 shrink-0 items-center justify-center">
      {pulse ? (
        <span
          aria-hidden="true"
          className="ds-ping absolute inset-0 rounded-full"
          style={{ background: color }}
        />
      ) : null}
      <span
        className="relative inline-block h-2.5 w-2.5 rounded-full border border-[var(--ds-gray-alpha-500)]"
        style={{ background: color }}
      />
    </span>
  );

  if (variant === "dot") {
    // Wrap in a line-height-aware inline-flex box so the dot self-centers
    // against the first line of adjacent text (cap height) without callers
    // needing margin hacks like `mt-1.5`.
    return (
      <span
        className={cn(
          "inline-flex h-[1lh] items-center align-text-bottom",
          className,
        )}
      >
        {dot}
      </span>
    );
  }

  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 font-medium text-[var(--ds-gray-1000)]",
        variant === "inline" && "text-[13px]",
        variant === "pill" &&
          "h-6 rounded-[5px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-2 text-[12px]",
        variant === "cell" && "text-[13px]",
        variant === "glass" &&
          "h-7 rounded-[7px] border border-white/30 bg-white/[0.07] px-2.5 text-[12px] text-white shadow-[inset_0_1px_0_rgb(255_255_255_/_0.76),inset_0_-1px_0_rgb(255_255_255_/_0.12),0_10px_24px_-18px_rgb(0_0_0_/_0.72)] backdrop-blur-xl backdrop-brightness-110 backdrop-saturate-200",
        className,
      )}
    >
      {dot}
      {children}
    </span>
  );
}
