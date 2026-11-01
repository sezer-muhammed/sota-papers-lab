import { ArrowRight, Play } from "lucide-react";
import type { RunRecord } from "@spl/sdk";
import { ButtonLink, DataPanel, Skeleton, cn, text } from "@spl/ui";
import { describeError, type Loaded } from "../api";
import { RECENT_ROWS } from "../lib/catalog";
import { absoluteTime, relativeTime } from "../lib/format";
import { RetryNotice, WaitingNotice } from "./notice";
import { RunStatusSignal, runDuration } from "./run-status";

const columns = ["Status", "Op", "When", "Duration"];

/* The newest runs on Home, each row a link to that run on the Runs page. */
export function RecentRuns({
  hostDown,
  now,
  runs,
}: {
  hostDown: boolean;
  now: number;
  runs: Loaded<RunRecord[]>;
}) {
  const rows = runs.data?.slice(0, RECENT_ROWS);
  const loading = !runs.data && !runs.error;

  return (
    <DataPanel
      id="recent-runs"
      eyebrow="Activity"
      title="Recent runs"
      summary="Newest first. Each run keeps its input, output, metrics and the environment it ran in."
      busy={loading}
      className="reveal scroll-mt-4 outline-none md:scroll-mt-24"
      action={
        <ButtonLink href="#/runs" icon={ArrowRight} size="sm">
          All runs
        </ButtonLink>
      }
    >
      {runs.error && !hostDown ? (
        <div className="p-3">
          {rows ? (
            <RetryNotice
              compact
              title="Couldn't refresh runs. Showing the last result."
              onRetry={() => void runs.reload()}
            />
          ) : (
            <RetryNotice title="Couldn't load runs." onRetry={() => void runs.reload()}>
              {describeError(runs.error)}
            </RetryNotice>
          )}
        </div>
      ) : null}
      {!rows && runs.error && hostDown ? (
        <div className="p-3">
          <WaitingNotice>Waiting for the host. This list fills in when it answers.</WaitingNotice>
        </div>
      ) : null}
      {rows?.length === 0 ? (
        <div className="flex flex-col items-start gap-3 p-4">
          <p className={text.small}>
            No runs yet. Run an op and it appears here with its output, metrics and timings.
          </p>
          <ButtonLink href="#/ops" icon={Play} size="sm" variant="primary">
            Run an op
          </ButtonLink>
        </div>
      ) : null}
      {loading || (rows && rows.length > 0) ? (
        <div
          aria-hidden="true"
          className="hidden gap-x-4 border-b border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-4 py-2.5 md:grid md:grid-cols-[136px_minmax(0,1fr)_120px_96px]"
        >
          {columns.map((label, index) => (
            <span key={label} className={cn(text.eyebrowSm, index === columns.length - 1 && "text-right")}>
              {label}
            </span>
          ))}
        </div>
      ) : null}
      {loading ? (
        <div>
          {[0, 1, 2, 3].map((row) => (
            <div
              key={row}
              className="flex h-11 items-center gap-4 border-b border-[var(--ds-gray-alpha-300)] px-4 last:border-b-0"
            >
              <Skeleton className="h-3 w-24" />
              <Skeleton className="h-3 flex-1" />
              <Skeleton className="h-3 w-16" />
            </div>
          ))}
        </div>
      ) : null}
      {rows && rows.length > 0 ? (
        <ol>
          {rows.map((run) => (
            <RunRow key={run.id} run={run} now={now} />
          ))}
        </ol>
      ) : null}
    </DataPanel>
  );
}

/* Phones: two lines (op + duration / status + time). md+: four columns under
   the header row. The focus ring is inset: the panel clips an outer one. */
function RunRow({ now, run }: { now: number; run: RunRecord }) {
  return (
    <li className="border-b border-[var(--ds-gray-alpha-300)] last:border-b-0">
      <a
        href={`#/runs/${encodeURIComponent(run.id)}`}
        className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 px-4 py-3 outline-none transition-colors hover:bg-[var(--ds-gray-100)] focus-visible:shadow-[inset_0_0_0_2px_var(--ds-focus-color)] md:grid-cols-[136px_minmax(0,1fr)_120px_96px]"
      >
        <span className="col-start-1 row-start-2 md:col-start-auto md:row-start-auto">
          <RunStatusSignal status={run.status} />
        </span>
        <span className="col-start-1 row-start-1 min-w-0 md:col-start-auto md:row-start-auto">
          <span className={cn(text.mono, "block truncate")}>{run.op_id}</span>
          {run.status === "failed" && run.error_message ? (
            <span className={cn(text.caption, "block line-clamp-1")}>{run.error_message}</span>
          ) : null}
        </span>
        <time
          dateTime={new Date(run.created_at_ms).toISOString()}
          title={absoluteTime(run.created_at_ms)}
          className={cn(
            text.meta,
            "col-start-2 row-start-2 whitespace-nowrap text-right md:col-start-auto md:row-start-auto md:text-left",
          )}
        >
          {relativeTime(run.created_at_ms, now)}
        </time>
        <span
          className={cn(
            text.meta,
            "col-start-2 row-start-1 whitespace-nowrap text-right tabular-nums md:col-start-auto md:row-start-auto",
          )}
        >
          {runDuration(run)}
        </span>
      </a>
    </li>
  );
}
