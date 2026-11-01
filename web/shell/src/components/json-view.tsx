import { cn } from "@spl/ui";

export function JsonView({ value, className }: { value: unknown; className?: string }) {
  return (
    <pre
      className={cn(
        "max-h-[420px] overflow-auto rounded-[7px] border border-[var(--ds-gray-alpha-300)] bg-[var(--ds-background-200)] p-3 font-mono text-[12px] leading-5 text-[var(--ds-gray-1000)] shadow-[inset_0_1px_2px_#0000000a]",
        className,
      )}
    >
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}
