import type { ReactNode } from "react";
import { ActionBar, cn, text } from "@spl/ui";

/*
 * Full-width call-to-action band (ATSGrade marketing/cta-band.tsx, actions
 * variant): text + buttons in two columns from md, plus an optional footer
 * row under a hairline (e.g. copyable commands).
 */
export function CtaBand({
  actions,
  className,
  description,
  eyebrow,
  footer,
  id,
  title,
}: {
  actions: ReactNode;
  className?: string;
  description?: string;
  eyebrow?: string;
  footer?: ReactNode;
  id: string;
  title: string;
}) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-title`}
      className={cn(
        "rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] p-5",
        className,
      )}
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_auto] md:items-center">
        <div>
          {eyebrow ? <p className={text.eyebrowSm}>{eyebrow}</p> : null}
          <h2 id={`${id}-title`} className={cn(text.bandTitle, "mt-2 text-[var(--ds-gray-1000)]")}>
            {title}
          </h2>
          {description ? <p className={cn(text.body, "mt-2 max-w-2xl")}>{description}</p> : null}
        </div>
        <ActionBar className="md:justify-end">{actions}</ActionBar>
      </div>
      {footer ? (
        <div className="mt-4 grid grid-cols-1 gap-3 border-t border-[var(--ds-gray-alpha-300)] pt-4 lg:grid-cols-2">
          {footer}
        </div>
      ) : null}
    </section>
  );
}
