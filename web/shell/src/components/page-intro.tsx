import type { ReactNode } from "react";
import { text } from "@spl/ui";

export function PageIntro({
  action,
  eyebrow,
  lead,
  title,
}: {
  action?: ReactNode;
  eyebrow: string;
  lead: string;
  title: string;
}) {
  return (
    <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="max-w-3xl">
        <p className={text.eyebrow}>{eyebrow}</p>
        <h1 className={`${text.pageTitle} mt-2 text-[var(--ds-gray-1000)]`}>{title}</h1>
        <p className={`${text.lead} mt-3`}>{lead}</p>
      </div>
      {action ? <div className="flex shrink-0 gap-2">{action}</div> : null}
    </div>
  );
}
