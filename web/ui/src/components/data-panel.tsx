import type { ReactNode } from "react";
import { cn } from "../lib/cn";
import { SectionHeader, Surface } from "./surface";

/* A titled panel: Surface + SectionHeader + body. With an `id`, the panel is a
   named region (aria-labelledby its own h2), so pages can scroll to it. */
export function DataPanel({
  action,
  busy,
  children,
  className,
  contentClassName,
  eyebrow,
  id,
  padded = false,
  summary,
  title,
  tone = "flat",
}: {
  action?: ReactNode;
  /** True while the body shows placeholders (sets aria-busy). */
  busy?: boolean;
  children: ReactNode;
  className?: string;
  contentClassName?: string;
  eyebrow?: string;
  id?: string;
  padded?: boolean;
  summary?: string;
  title: string;
  tone?: "flat" | "raised" | "sunken";
}) {
  const titleId = id ? `${id}-title` : undefined;

  return (
    <Surface
      id={id}
      tone={tone}
      aria-labelledby={titleId}
      aria-busy={busy || undefined}
      className={cn("overflow-hidden", className)}
    >
      <SectionHeader titleId={titleId} eyebrow={eyebrow} title={title} summary={summary} action={action} />
      <div className={cn(padded && "p-3", contentClassName)}>{children}</div>
    </Surface>
  );
}
