import type { ReactNode } from "react";
import { CircleStop, RefreshCw, RotateCcw, WandSparkles, X } from "lucide-react";
import { Button, ProgressBar, StatusSignal, cn, text } from "@spl/ui";
import { useNow } from "../api";
import { formatMs, formatPercent } from "./format";
import { canRetouch, isActive, photoError, photoPhase, type Phase, type Photo } from "./model";
import type { RetouchSession } from "./session";

/* Status lives in the dot: blue while something runs (pulsing), green done,
   red failed, amber cancelled, gray waiting. */
export const phaseColor: Record<Phase, string> = {
  uploading: "var(--ds-blue-700)",
  reading: "var(--ds-blue-700)",
  ready: "var(--ds-gray-700)",
  queued: "var(--ds-gray-700)",
  working: "var(--ds-blue-700)",
  done: "var(--ds-green-700)",
  failed: "var(--ds-red-700)",
  cancelled: "var(--ds-amber-700)",
};

const pulsing = new Set<Phase>(["uploading", "reading", "working"]);

/** A file still waiting for an upload slot reads "uploading" but stays gray
    and still: nothing is moving yet. */
export function PhaseSignal({ photo, variant = "inline" }: { photo: Photo; variant?: "inline" | "pill" }) {
  const phase = photoPhase(photo);
  const waiting = photo.upload.status === "waiting";
  return (
    <StatusSignal
      variant={variant}
      color={waiting ? phaseColor.queued : phaseColor[phase]}
      pulse={!waiting && pulsing.has(phase)}
      className="shrink-0"
    >
      {phase}
    </StatusSignal>
  );
}

const ordinal = (n: number) => {
  const tens = n % 100;
  const suffix = tens >= 11 && tens <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th";
  return `${n}${suffix}`;
};

/** What a photo is doing, as a sentence and (while it works) a bar. `detailed`
    adds elapsed time, the full error and the run's log. */
export function PhotoProgress({
  detailed = false,
  photo,
  position,
}: {
  detailed?: boolean;
  photo: Photo;
  /** Place in the retouch queue, 1 = next. */
  position?: number;
}) {
  const { auto, upload } = photo;
  const phase = photoPhase(photo);
  const name = photo.filename;

  if (upload.status === "waiting") return <Line label={`Upload of ${name}`} value={0} text="Waiting to upload" />;
  if (upload.status === "uploading") {
    return (
      <Line label={`Upload of ${name}`} value={upload.fraction} text={`Uploading · ${formatPercent(upload.fraction)}`} />
    );
  }
  if (phase === "failed") {
    const error = photoError(photo) ?? "Something went wrong.";
    return <p className={cn(text.caption, detailed ? "whitespace-pre-wrap break-words" : "line-clamp-2 break-words")}>{error}</p>;
  }
  if (phase === "reading") return <Line label={`Reading ${name}`} text="Reading the RAW file" />;
  if (auto.status === "queued") {
    const where = position ? ` · ${ordinal(position)} in line` : "";
    const waiting = photo.inspect.status === "running" ? " · after reading" : "";
    return <p className={text.caption}>Queued{where}{waiting}</p>;
  }
  if (auto.status === "restoring") return <Line label={`Retouch of ${name}`} text="Checking the run" />;
  if (auto.status === "running") {
    if (auto.cancelling) return <Line label={`Retouch of ${name}`} text="Cancelling…" />;
    if (auto.serverQueued) return <Line label={`Retouch of ${name}`} text="Waiting for the host to start it" />;
    return (
      <div className="space-y-1.5">
        <Line
          label={`Retouch of ${name}`}
          value={auto.progress?.fraction}
          text={auto.progress?.message || "Starting"}
          extra={detailed && auto.startedAtMs != null ? <Elapsed since={auto.startedAtMs} /> : undefined}
        />
        {auto.notice ? <p className={text.caption}>{auto.notice}</p> : null}
        {detailed && auto.logs.length > 0 ? <RunLog lines={auto.logs} /> : null}
      </div>
    );
  }
  if (phase === "cancelled") return <p className={text.caption}>Cancelled before it finished.</p>;
  if (phase === "done" && photo.result) {
    const output = photo.result.output;
    const chosen = output.candidates.find((c) => c.id === output.chosen);
    return (
      <p className={cn(text.caption, "truncate")}>
        {chosen ? `${chosen.label} · ` : ""}
        <span className="font-mono">{formatMs(output.timings?.total_ms)}</span>
      </p>
    );
  }
  if (phase === "done") return <p className={text.caption}>Loading the result…</p>;
  return <p className={text.caption}>Ready to retouch</p>;
}

function Line({
  extra,
  label,
  text: message,
  value,
}: {
  extra?: ReactNode;
  label: string;
  text: string;
  /** 0 to 1; undefined or null for an indeterminate bar. */
  value?: number | null;
}) {
  return (
    <div className="min-w-0 space-y-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <p className={cn(text.caption, "min-w-0 truncate")} title={message}>
          {message}
        </p>
        {extra ?? (typeof value === "number" ? <span className={text.meta}>{formatPercent(value)}</span> : null)}
      </div>
      <ProgressBar label={label} value={value} />
    </div>
  );
}

function Elapsed({ since }: { since: number }) {
  const now = useNow(1000);
  return <span className={cn(text.meta, "shrink-0 tabular-nums")}>{formatMs(Math.max(0, now - since))}</span>;
}

function RunLog({ lines }: { lines: string[] }) {
  return (
    <details className="group">
      <summary className="cursor-pointer rounded-[4px] text-[12px] text-[var(--ds-gray-900)] outline-none hover:text-[var(--ds-gray-1000)] focus-visible:shadow-[var(--ds-focus-ring)]">
        Log ({lines.length})
      </summary>
      <pre className="mt-2 max-h-48 overflow-auto rounded-[7px] border border-[var(--ds-gray-alpha-300)] bg-[var(--ds-background-200)] p-2 font-mono text-[11px] leading-4 text-[var(--ds-gray-1000)] whitespace-pre-wrap break-words">
        {lines.join("\n")}
      </pre>
    </details>
  );
}

/** The buttons that apply to a photo right now. */
export function PhotoActions({
  canRetryUpload,
  onRemoved,
  photo,
  primary = false,
  session,
}: {
  canRetryUpload: boolean;
  /** After Remove (the detail view goes back to the grid). */
  onRemoved?: () => void;
  photo: Photo;
  /** Make Retouch the blue action (the detail view's main button). */
  primary?: boolean;
  session: RetouchSession;
}) {
  const { auto, inspect, key, upload } = photo;
  const name = photo.filename;
  const phase = photoPhase(photo);
  const actions: ReactNode[] = [];

  if (upload.status === "waiting" || upload.status === "uploading") {
    actions.push(
      <Button key="cancel-upload" size="sm" icon={X} onClick={() => session.remove(key)}>
        Cancel upload<span className="sr-only"> of {name}</span>
      </Button>,
    );
  } else if (isActive(photo)) {
    const cancelling = auto.status === "running" && auto.cancelling;
    actions.push(
      <Button
        key="cancel"
        size="sm"
        icon={CircleStop}
        aria-disabled={cancelling || undefined}
        onClick={() => {
          if (!cancelling) session.cancel(key);
        }}
      >
        {cancelling ? "Cancelling" : "Cancel"}
        <span className="sr-only"> retouch of {name}</span>
      </Button>,
    );
  } else {
    if (upload.status === "failed" && canRetryUpload) {
      actions.push(
        <Button key="retry-upload" size="sm" icon={RefreshCw} onClick={() => session.retryUpload(key)}>
          Retry upload<span className="sr-only"> of {name}</span>
        </Button>,
      );
    }
    if (upload.status === "done" && auto.status === "idle" && inspect.status === "failed") {
      actions.push(
        <Button key="reread" size="sm" icon={RefreshCw} onClick={() => session.retryInspect(key)}>
          Read again<span className="sr-only"> {name}</span>
        </Button>,
      );
    }
    if (canRetouch(photo) && phase !== "reading") {
      const again = photo.result !== undefined || auto.status !== "idle";
      actions.push(
        <Button
          key="retouch"
          size="sm"
          variant={primary && !again ? "primary" : "secondary"}
          icon={again ? RotateCcw : WandSparkles}
          onClick={() => session.retouch(key)}
        >
          {again ? "Re-run" : "Retouch"}
          <span className="sr-only"> {name}</span>
        </Button>,
      );
    }
    if (phase !== "reading") {
      actions.push(
        <Button
          key="remove"
          size="sm"
          variant="ghost"
          icon={X}
          onClick={() => {
            session.remove(key);
            onRemoved?.();
          }}
        >
          Remove<span className="sr-only"> {name} from the session</span>
        </Button>,
      );
    }
  }

  return actions.length ? <>{actions}</> : null;
}
