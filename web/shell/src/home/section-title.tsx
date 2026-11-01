import type { ReactNode } from "react";
import { cn, text } from "@spl/ui";

/* A Home section's heading: mono eyebrow, h2, one short line. No box. */
export function SectionTitle({
  id,
  eyebrow,
  title,
  children,
  className,
}: {
  id: string;
  eyebrow: string;
  title: string;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("max-w-2xl", className)}>
      <p className={text.eyebrow}>{eyebrow}</p>
      <h2 id={id} className="mt-2 text-[28px] font-semibold leading-9 text-balance text-[var(--ds-gray-1000)]">
        {title}
      </h2>
      {children ? <p className={cn(text.body, "mt-2")}>{children}</p> : null}
    </div>
  );
}
