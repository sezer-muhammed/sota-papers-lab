import type { ReactNode } from "react";
import { ArrowRight, RefreshCw, type LucideIcon } from "lucide-react";
import type { Health, MetricRecord, RunDetail, RunRecord } from "@spl/sdk";
import { Card, Skeleton, cn, text } from "@spl/ui";
import { describeError, type Loaded } from "../api";
import { readEnv } from "../lib/env";
import { formatMetric, plural, relativeTime, shortGpuName, splitDuration } from "../lib/format";
import { focusTokenInput, useFocusHandoff } from "../lib/scroll";
import { RunStatusSignal, statusColor } from "./run-status";

const nodes = [
  { label: "Paper", value: "paper.toml", note: "Claimed numbers" },
  { label: "App", value: "apps/<name>", note: "Rust ops, CUDA kernels" },
  { label: "Op", value: "<app>.<verb>", note: "Typed JSON in and out" },
];

const METRIC_ROWS = 3;
const NEUTRAL_RING = "var(--ds-gray-alpha-400)";

type LatestRunProps = {
  health: Loaded<Health>;
  hostDown: boolean;
  /** Detail (metrics) of the newest run. */
  latest: Loaded<RunDetail | undefined>;
  locked: boolean;
  now: number;
  onRetry: () => void;
  runs: Loaded<RunRecord[]>;
};

/* The hero's media slot (xl and up), as on ATSGrade: a hairline-grid
   schematic in place of the image, and the sample-report card showing the
   newest real run. */
export function LabSpecimen(props: LatestRunProps) {
  return (
    <div className="depth-surface hidden min-h-[480px] flex-col overflow-hidden rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] xl:flex">
      <figure className="hairline-grid flex flex-1 flex-col justify-center gap-4 p-5">
        <figcaption className={text.eyebrowSm}>How a paper enters the lab</figcaption>
        <ol className="grid grid-cols-3 gap-6">
          {nodes.map((node, index) => (
            <li
              key={node.label}
              className="relative min-w-0 rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] p-3"
            >
              <p className={text.eyebrowSm}>{node.label}</p>
              <p className={cn(text.mono, "mt-1 truncate font-semibold")}>{node.value}</p>
              <p className={cn(text.meta, "mt-1")}>{node.note}</p>
              {index < nodes.length - 1 ? (
                <ArrowRight
                  aria-hidden="true"
                  className="absolute -right-5 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--ds-gray-600)]"
                />
              ) : null}
            </li>
          ))}
        </ol>
      </figure>
      <LatestRunCard {...props} />
    </div>
  );
}

/* Every state renders the same Frame, so the card element (and focus on it)
   survives a change of state. */
function LatestRunCard({ health, hostDown, latest, locked, now, onRetry, runs }: LatestRunProps) {
  const run = runs.data?.[0];

  if (locked) {
    return (
      <Frame
        ring={<Ring color={NEUTRAL_RING} value="—" />}
        title="Runs need a token"
        meta={<Meta>This host sets SPL_AUTH_TOKEN.</Meta>}
        body={<p className={text.caption}>Enter it under Apps on this host to see runs and metrics.</p>}
        footer="Token required"
        action={<FooterAction onClick={focusTokenInput}>Enter token</FooterAction>}
      />
    );
  }

  if (run) return <Frame {...runFrame({ health, latest, now, run })} />;

  if (runs.data) {
    return (
      <Frame
        ring={<Ring color={NEUTRAL_RING} value="—" />}
        title="No runs yet"
        meta={<Meta>Run any op and it lands here with its metrics.</Meta>}
        footer="Nothing recorded"
        action={<FooterAction href="#/ops">Open Ops</FooterAction>}
      />
    );
  }

  if (runs.error) {
    return (
      <Frame
        ring={<Ring color={NEUTRAL_RING} value="—" />}
        title={hostDown ? "Host unreachable" : "Runs unavailable"}
        meta={<Meta>{hostDown ? "Live data fills in when it answers." : describeError(runs.error)}</Meta>}
        footer={hostDown ? "Host offline" : "No data"}
        action={<RetryAction onClick={onRetry} />}
      />
    );
  }

  return (
    <Frame
      busy
      ring={<Skeleton className="h-16 w-16 shrink-0 rounded-full" />}
      title={<Skeleton className="mt-1.5 h-4 w-44" />}
      meta={<Skeleton className="mt-2 h-3 w-56 max-w-full" />}
      body={<MetricRowsSkeleton />}
      footer={<Skeleton className="h-3 w-32" />}
      action={<Skeleton className="h-3 w-16" />}
    />
  );
}

/* The newest run as Frame props: duration ring, status, metrics, and the GPU
   it ran on. */
function runFrame({
  health,
  latest,
  now,
  run,
}: {
  health: Loaded<Health>;
  latest: Loaded<RunDetail | undefined>;
  now: number;
  run: RunRecord;
}): FrameProps {
  const duration = splitDuration(run);
  const pending = run.status === "running" || run.status === "queued";
  const gpu = readEnv(run.env).gpu ?? readEnv(health.data?.env).gpu;
  const device = gpu?.devices[0];
  const sha = run.git_sha && run.git_sha !== "unknown" ? run.git_sha.slice(0, 7) : undefined;
  // Only this run's detail, at this status: an older run's metrics never flash.
  const detail =
    latest.data?.run.id === run.id && latest.data.run.status === run.status ? latest.data : undefined;

  // A run that has started counts from its start. One still queued shows when
  // it was created; its status already says queued.
  const meta =
    pending && run.started_at_ms != null
      ? `· started ${relativeTime(run.started_at_ms, now)}`
      : `· ${relativeTime(run.created_at_ms, now)}${!pending && device ? ` · ${shortGpuName(device.name)}` : ""}`;

  const footer = [
    device
      ? `sm ${device.compute_capability}${gpu?.driver_version ? ` · CUDA ${gpu.driver_version}` : ""}`
      : "No GPU recorded",
    sha,
  ]
    .filter(Boolean)
    .join(" · ");

  let body: ReactNode;
  if (pending) {
    body = <p className={text.caption}>Metrics appear when the run finishes.</p>;
  } else if (run.status === "cancelled") {
    body = <p className={text.caption}>This run was cancelled.</p>;
  } else if (run.status === "failed") {
    body = (
      <p className={cn(text.caption, "line-clamp-3")}>
        {run.error_message ?? "No error message was recorded."}
      </p>
    );
  } else if (detail) {
    body = <MetricRows metrics={detail.metrics} />;
  } else if (latest.error && !latest.loading) {
    body = <p className={text.caption}>Metrics unavailable.</p>;
  } else {
    body = <MetricRowsSkeleton />;
  }

  return {
    // The ring matches the status dot: blue only while running.
    ring: <Ring color={statusColor[run.status]} value={duration?.value ?? "—"} unit={duration?.unit} />,
    title: run.op_id,
    meta: (
      <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
        <RunStatusSignal status={run.status} />
        <span className={text.meta}>{meta}</span>
      </div>
    ),
    body,
    footer,
    footerValues: true,
    action: <FooterAction href={`#/runs/${encodeURIComponent(run.id)}`}>Open run</FooterAction>,
  };
}

type FrameProps = {
  action: ReactNode;
  body?: ReactNode;
  busy?: boolean;
  footer: ReactNode;
  /** The footer holds machine values (GPU, git SHA): mono, not uppercased. */
  footerValues?: boolean;
  meta: ReactNode;
  ring: ReactNode;
  title: ReactNode;
};

/* The ATSGrade sample-report card: ring + title, body, footer row. */
function Frame({ action, body, busy = false, footer, footerValues = false, meta, ring, title }: FrameProps) {
  return (
    <Card
      depth="lifted"
      role="group"
      aria-label="Latest run"
      aria-busy={busy || undefined}
      className="m-5 mt-0 rounded-[10px] p-0 outline-none"
    >
      <div className="flex items-center gap-4 border-b border-[var(--ds-gray-alpha-300)] p-4">
        {ring}
        <div className="min-w-0 flex-1">
          <p className={text.eyebrowSm}>Latest run</p>
          {typeof title === "string" ? (
            <p className={cn(text.body, "mt-0.5 truncate font-mono font-semibold text-[var(--ds-gray-1000)]")}>
              {title}
            </p>
          ) : (
            title
          )}
          {meta}
        </div>
      </div>
      {body ? <div className="p-4">{body}</div> : null}
      {/* One hairline per gap: without a body, the top row's border already
          sits right above the footer. */}
      <div
        className={cn(
          "flex items-center justify-between gap-3 px-4 py-2.5",
          body ? "border-t border-[var(--ds-gray-alpha-300)]" : undefined,
        )}
      >
        <span className={cn(footerValues ? text.meta : text.eyebrowSm, "min-w-0 truncate")}>{footer}</span>
        {action}
      </div>
    </Card>
  );
}

/* The ATSGrade grade ring, holding the run's duration. */
function Ring({ color, unit, value }: { color: string; unit?: string; value: string }) {
  return (
    <div
      aria-hidden={unit ? undefined : true}
      className="grid h-16 w-16 shrink-0 place-items-center rounded-full border-4 bg-[var(--ds-background-200)]"
      style={{ borderColor: color }}
    >
      <div className="text-center">
        <span className="block text-[18px] font-semibold leading-5 tabular-nums text-[var(--ds-gray-1000)]">
          {value}
        </span>
        {unit ? <span className="block font-mono text-[11px] text-[var(--ds-gray-700)]">{unit}</span> : null}
      </div>
    </div>
  );
}

/* The line under the title: a sentence, so sans, as in ATSGrade's slot. */
function Meta({ children }: { children: ReactNode }) {
  return <p className={cn(text.caption, "mt-1 text-[var(--ds-gray-700)]")}>{children}</p>;
}

function MetricRows({ metrics }: { metrics: MetricRecord[] }) {
  if (metrics.length === 0) return <p className={text.caption}>No metrics recorded for this run.</p>;
  const shown = [...metrics].sort((a, b) => a.seq - b.seq).slice(0, METRIC_ROWS);
  const more = metrics.length - shown.length;

  return (
    <>
      <dl className="grid grid-cols-1 gap-2">
        {shown.map((metric) => (
          <div key={metric.seq} className="flex items-baseline justify-between gap-3">
            <dt className={cn(text.meta, "min-w-0 truncate")}>{metric.key}</dt>
            <dd className={cn(text.mono, "shrink-0 tabular-nums")}>{formatMetric(metric.value, metric.unit)}</dd>
          </div>
        ))}
      </dl>
      {more > 0 ? <p className={cn(text.meta, "mt-2")}>+{plural(more, "more metric")}</p> : null}
    </>
  );
}

function MetricRowsSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-2">
      {[0, 1, 2].map((row) => (
        <div key={row} className="flex h-5 items-center justify-between gap-3">
          <Skeleton className="h-3 w-24" />
          <Skeleton className="h-3 w-16" />
        </div>
      ))}
    </div>
  );
}

const footerActionClass = cn(
  text.link,
  "inline-flex shrink-0 items-center gap-1 rounded-[4px] outline-none focus-visible:shadow-[var(--ds-focus-ring)]",
);

/* The footer's blue text action: a link with href, otherwise a button. */
function FooterAction({
  children,
  href,
  icon: Icon = ArrowRight,
  onClick,
}: {
  children: ReactNode;
  href?: string;
  icon?: LucideIcon;
  onClick?: () => void;
}) {
  const content = (
    <>
      {children}
      <Icon aria-hidden="true" className="h-3.5 w-3.5" />
    </>
  );
  return href ? (
    <a href={href} className={footerActionClass}>
      {content}
    </a>
  ) : (
    <button type="button" onClick={onClick} className={footerActionClass}>
      {content}
    </button>
  );
}

/* The footer's Retry. Its own component, so it unmounts (and hands focus to
   the card) when a retry that works swaps it for another action. */
function RetryAction({ onClick }: { onClick: () => void }) {
  const ref = useFocusHandoff<HTMLButtonElement>('[role="group"]');

  return (
    <button ref={ref} type="button" onClick={onClick} className={footerActionClass}>
      Retry
      <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
    </button>
  );
}
