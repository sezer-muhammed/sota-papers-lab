import { cn, text } from "@spl/ui";
import { LOOK_SOURCE, formatChange, formatExposure, type Example } from "./examples";

const seconds = (s: number) => `${s.toFixed(1)} s`;

/* What one example is and what the app did to it: camera and exposure, the
   look that won, the five biggest slider moves, Claude's scene read, time. */
export function ExampleFacts({ example, className }: { example: Example; className?: string }) {
  const lens = example.lens ?? "Built-in lens";
  return (
    <div className={cn("min-w-0", className)}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <p className="text-[15px] font-medium text-[var(--ds-gray-1000)]">{example.camera}</p>
        <p className={text.meta}>
          {lens} · {formatExposure(example)}
        </p>
      </div>
      <dl className="mt-3 grid grid-cols-[72px_minmax(0,1fr)] gap-x-3 sm:grid-cols-[88px_minmax(0,1fr)] gap-y-2">
        <dt className={cn(text.eyebrowSm, "pt-px")}>Look</dt>
        <dd className={text.small}>
          <span className="font-medium text-[var(--ds-gray-1000)]">{example.look}</span>
          <span className="text-[var(--ds-gray-700)]"> · {LOOK_SOURCE[example.lookSource]}</span>
        </dd>
        <dt className={cn(text.eyebrowSm, "pt-px")}>Settings</dt>
        <dd>
          <ul className="flex flex-wrap gap-x-3 gap-y-0.5" aria-label="Five largest changes from as shot">
            {example.changes.map((change) => (
              <li key={change.key} className={text.mono}>
                {formatChange(change)}
              </li>
            ))}
          </ul>
        </dd>
        <dt className={cn(text.eyebrowSm, "pt-px")}>Claude</dt>
        <dd className={text.caption}>
          <q>{example.scene}</q>
          {example.refinement ? <span className="mt-1 block">Refined: {example.refinement}</span> : null}
        </dd>
        <dt className={cn(text.eyebrowSm, "pt-px")}>Time</dt>
        <dd className={text.mono}>
          {seconds(example.totalS)}
          <span className="text-[var(--ds-gray-700)]">
            {" "}
            · GPU search {seconds(example.searchS)} · {example.claudeCalls} Claude calls {seconds(example.claudeS)}
          </span>
        </dd>
      </dl>
    </div>
  );
}
