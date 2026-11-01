import type { ReactNode } from "react";
import { RefreshCw } from "lucide-react";
import { Button, Card, cn, text } from "@spl/ui";
import { useFocusHandoff } from "../lib/scroll";

/* A section's load error with a Retry button. `compact` when the section
   still shows its last result below it. A retry that works removes the
   notice, so focus on Retry moves to the section rather than to <body>. */
export function RetryNotice({
  children,
  className,
  compact = false,
  onRetry,
  title,
}: {
  children?: ReactNode;
  className?: string;
  compact?: boolean;
  onRetry: () => void;
  title: string;
}) {
  const retry = useFocusHandoff<HTMLDivElement>();

  return (
    <Card
      tone="warning"
      depth="flat"
      density={compact ? "compact" : "md"}
      role="status"
      className={cn("flex flex-wrap items-center justify-between gap-3", className)}
    >
      <div className="min-w-0">
        <p className={cn(text.small, "font-medium text-inherit")}>{title}</p>
        {children ? <p className={cn(text.caption, "mt-0.5 text-inherit")}>{children}</p> : null}
      </div>
      <div ref={retry} className="flex shrink-0">
        <Button size="sm" icon={RefreshCw} onClick={onRetry}>
          Retry
        </Button>
      </div>
    </Card>
  );
}

/* Stands in for a section with nothing to show while the host is unreachable;
   the header pill and the host strip already carry the error. */
export function WaitingNotice({ children }: { children: ReactNode }) {
  return (
    <Card tone="muted" depth="flat" role="status">
      <p className={text.small}>{children}</p>
    </Card>
  );
}
