import { useState, type ReactNode } from "react";
import { ArrowRight, Plus } from "lucide-react";
import type { OpDescriptor, RunRecord } from "@spl/sdk";
import {
  Badge,
  Button,
  ButtonLink,
  Card,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  DataPanel,
  IconTile,
  Skeleton,
  StatusSignal,
  cn,
  text,
} from "@spl/ui";
import { describeError, type Loaded } from "../api";
import { APP_CAP, APP_PAGES, OP_ROWS, initials, matchesApp, type AppSummary } from "../lib/catalog";
import { absoluteTime, plural, relativeTime } from "../lib/format";
import { CopyCommand } from "./copy-command";
import { RetryNotice, WaitingNotice } from "./notice";
import { statusColor } from "./run-status";
import { TokenPrompt } from "./token-prompt";

type RunsState = "loading" | "error" | "ready";

/* Home's app catalog: one card per app (the op id prefix), its ops as links
   into the Ops page, and its last run. Ends with the "next paper" slot. */
export function AppsPanel({
  apps,
  hostDown,
  locked,
  now,
  ops,
  runs,
  windowFull,
}: {
  apps: AppSummary[];
  hostDown: boolean;
  locked: boolean;
  now: number;
  ops: Loaded<OpDescriptor[]>;
  runs: Loaded<RunRecord[]>;
  /** The run window came back full, so counts are "recent", not all-time. */
  windowFull: boolean;
}) {
  const [query, setQuery] = useState("");
  const [showAll, setShowAll] = useState(false);

  const filterable = apps.length > APP_CAP;
  const q = filterable ? query.trim() : "";
  const filtering = q !== "";
  const matches = filtering ? apps.filter((app) => matchesApp(app, q)) : apps;
  const visible = !filtering && !showAll && filterable ? apps.slice(0, APP_CAP) : matches;
  const single = apps.length === 1 && !filtering;
  const runsState: RunsState = runs.data ? "ready" : runs.error ? "error" : "loading";
  // Several apps are ordered by their last run: keep placeholders until runs
  // settle, so the cards are sorted once and never move under the pointer.
  const ordering = apps.length > 1 && runsState === "loading";

  const summary = locked
    ? "This host needs a token to list apps and runs."
    : ops.data
      ? ops.data.length === 0
        ? "No apps are linked into this host yet."
        : `${plural(apps.length, "app")} · ${plural(ops.data.length, "op")}.` +
          (apps.length > 1 && runsState === "ready" ? " Most recently run first." : "") +
          " Select an op to open it on the Ops page."
      : ops.error
        ? undefined
        : "Loading apps…";

  const filterStatus = !filtering
    ? ""
    : matches.length === 0
      ? `No app or op matches “${q}”.`
      : `${plural(matches.length, "app")} ${matches.length === 1 ? "matches" : "match"}.`;

  const action = locked ? undefined : (
    <>
      {filterable ? (
        <>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter apps and ops"
            aria-label="Filter apps and ops"
            className={cn(
              text.meta,
              "h-8 min-w-0 flex-1 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 text-[var(--ds-gray-1000)] outline-none focus-visible:shadow-[var(--ds-focus-ring)] sm:w-56 sm:flex-none",
            )}
          />
          {/* Mounted with the input, so screen readers announce every change. */}
          <p role="status" className="sr-only">
            {filterStatus}
          </p>
        </>
      ) : null}
      <ButtonLink href="#/ops" icon={ArrowRight} size="sm">
        All ops
      </ButtonLink>
    </>
  );

  let body: ReactNode;
  if (locked) {
    body = (
      <div className="py-6">
        <TokenPrompt level={3} />
      </div>
    );
  } else if (!ops.data && ops.error) {
    body = hostDown ? (
      <WaitingNotice>Waiting for the host. This section fills in when it answers.</WaitingNotice>
    ) : (
      <RetryNotice title="Couldn't load the app list." onRetry={() => void ops.reload()}>
        {describeError(ops.error)}
      </RetryNotice>
    );
  } else if (!ops.data || ordering) {
    const placeholders = ordering ? Math.min(apps.length, APP_CAP) : 2;
    body = (
      <div className="stagger grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {Array.from({ length: placeholders }, (_, index) => (
          <AppCardSkeleton key={index} />
        ))}
        <NextPaperCard empty={false} />
      </div>
    );
  } else {
    body = (
      <>
        {ops.error && !hostDown ? (
          <RetryNotice
            compact
            className="mb-3"
            title="Couldn't refresh the app list. Showing the last result."
            onRetry={() => void ops.reload()}
          />
        ) : null}
        {filtering && matches.length === 0 ? (
          <Card tone="muted" depth="flat" className="flex flex-wrap items-center justify-between gap-3">
            <p className={text.small}>No app or op matches “{q}”.</p>
            <Button variant="ghost" size="sm" onClick={() => setQuery("")}>
              Clear filter
            </Button>
          </Card>
        ) : (
          <div className="stagger grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {visible.map((app) => (
              <AppCard
                key={app.id}
                app={app}
                now={now}
                runsState={runsState}
                single={single}
                windowFull={windowFull}
              />
            ))}
            {filtering ? null : <NextPaperCard empty={apps.length === 0} />}
            {!filtering && filterable ? (
              <div className="col-span-full flex justify-center">
                <Button
                  variant="ghost"
                  size="sm"
                  aria-expanded={showAll}
                  onClick={() => setShowAll((open) => !open)}
                >
                  {showAll ? "Show fewer" : `Show all ${apps.length} apps`}
                </Button>
              </div>
            ) : null}
          </div>
        )}
      </>
    );
  }

  return (
    <DataPanel
      id="apps"
      eyebrow="Apps"
      title="Apps on this host"
      summary={summary}
      padded
      busy={(!ops.data && !ops.error) || ordering}
      className="reveal scroll-mt-4 outline-none md:scroll-mt-24"
      action={action}
    >
      {body}
    </DataPanel>
  );
}

function AppCard({
  app,
  now,
  runsState,
  single,
  windowFull,
}: {
  app: AppSummary;
  now: number;
  runsState: RunsState;
  /** The only app on the host: spans two columns at xl, ops side by side. */
  single: boolean;
  windowFull: boolean;
}) {
  const shown = app.ops.slice(0, OP_ROWS);
  const more = app.ops.length - shown.length;
  const run = app.lastRun;
  const page = APP_PAGES[app.id];

  return (
    <Card depth="base" className={cn("flex flex-col", single && "xl:col-span-2")}>
      <div className="flex-1">
        <CardHeader action={<Badge>{plural(app.ops.length, "op")}</Badge>}>
          <IconTile label={initials(app.id)} />
          <CardTitle className="mt-4 truncate font-mono">{app.id}</CardTitle>
        </CardHeader>
        <ul className={cn("grid grid-cols-1 gap-1", single && "xl:grid-cols-2 xl:gap-x-3")}>
          {shown.map((op) => (
            <li key={op.id} className="min-w-0">
              <a
                href={`#/ops/${encodeURIComponent(op.id)}`}
                className="-mx-2 block rounded-[6px] px-2 py-1.5 outline-none transition-colors hover:bg-[var(--ds-gray-100)] focus-visible:shadow-[var(--ds-focus-ring)]"
              >
                <span className={cn(text.mono, "block truncate font-semibold")}>{op.id}</span>
                <span className={cn(text.caption, "block line-clamp-1")}>{op.summary}</span>
              </a>
            </li>
          ))}
        </ul>
        {more > 0 ? (
          <a
            href="#/ops"
            className={cn(
              text.meta,
              "mt-2 inline-block rounded-[4px] outline-none hover:text-[var(--ds-gray-1000)] focus-visible:shadow-[var(--ds-focus-ring)]",
            )}
          >
            +{plural(more, "more op")}
          </a>
        ) : null}
        {page ? (
          <ButtonLink href={page.href} icon={ArrowRight} size="sm" className="mt-3">
            {page.label}
          </ButtonLink>
        ) : null}
      </div>
      <CardFooter className="flex items-center justify-between gap-3">
        {runsState === "loading" ? (
          <Skeleton className="h-3 w-28" />
        ) : runsState === "error" ? (
          <span className={text.meta}>Run history unavailable</span>
        ) : run ? (
          <>
            <span className="flex min-w-0 items-center gap-2">
              <StatusSignal variant="inline" color={statusColor[run.status]} pulse={run.status === "running"}>
                {run.status}
              </StatusSignal>
              <time
                dateTime={new Date(run.created_at_ms).toISOString()}
                title={absoluteTime(run.created_at_ms)}
                className={cn(text.meta, "truncate")}
              >
                {relativeTime(run.created_at_ms, now)}
              </time>
            </span>
            <span className={cn(text.meta, "shrink-0")}>
              {plural(app.runCount, windowFull ? "recent run" : "run")}
            </span>
          </>
        ) : (
          <span className={text.meta}>{windowFull ? "No recent runs" : "No runs yet"}</span>
        )}
      </CardFooter>
    </Card>
  );
}

/* The slot a new paper fills: how to scaffold an app. Spans the row when the
   host has no apps at all. */
function NextPaperCard({ empty }: { empty: boolean }) {
  return (
    <Card tone="muted" depth="flat" className={cn("flex flex-col", empty && "sm:col-span-2 xl:col-span-3")}>
      <IconTile icon={Plus} tone="muted" />
      <CardTitle className="mt-4">{empty ? "Add the first paper" : "Add the next paper"}</CardTitle>
      <CardDescription>
        Scaffold an app from the template, write its ops and CUDA kernels, and record the paper's
        claimed numbers in paper.toml. Rebuild the host and it appears here.
      </CardDescription>
      <CopyCommand
        className="mt-3"
        command={'scripts/new-app.sh <name> "<one-line summary>"'}
        label="scaffold command"
      />
      <p className={cn(text.meta, "mt-auto pt-3")}>Contract: docs/APP_CONTRACT.md</p>
    </Card>
  );
}

function AppCardSkeleton() {
  return (
    <Card depth="base" className="flex flex-col">
      <div className="flex-1">
        <div className="mb-4">
          <Skeleton className="h-10 w-10 rounded-[8px]" />
          <Skeleton className="mt-4 h-5 w-32" />
        </div>
        <div className="grid grid-cols-1 gap-3">
          {[0, 1].map((row) => (
            <div key={row}>
              <Skeleton className="h-4 w-40 max-w-full" />
              <Skeleton className="mt-1 h-3 w-56 max-w-full" />
            </div>
          ))}
        </div>
      </div>
      <CardFooter>
        <Skeleton className="h-3 w-28" />
      </CardFooter>
    </Card>
  );
}
