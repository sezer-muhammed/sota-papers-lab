import { cn } from "../lib/cn";

/* Shimmering placeholder block; size it with className. Hidden from screen
   readers: the surrounding panel carries aria-busy. */
export function Skeleton({ className }: { className?: string }) {
  return <span aria-hidden="true" className={cn("ds-skeleton block rounded-[6px]", className)} />;
}
