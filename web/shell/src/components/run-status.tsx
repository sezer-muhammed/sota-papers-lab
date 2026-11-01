import type { RunStatus } from "@spl/sdk";
import { StatusSignal } from "@spl/ui";

const statusColor: Record<RunStatus, string> = {
  queued: "var(--ds-gray-700)",
  running: "var(--ds-blue-700)",
  succeeded: "var(--ds-green-700)",
  failed: "var(--ds-red-700)",
  cancelled: "var(--ds-amber-700)",
};

export function RunStatusSignal({ status }: { status: RunStatus }) {
  return (
    <StatusSignal variant="cell" color={statusColor[status]} pulse={status === "running"}>
      {status}
    </StatusSignal>
  );
}

export function runDuration(run: { started_at_ms: number | null; finished_at_ms: number | null }) {
  if (run.started_at_ms == null || run.finished_at_ms == null) return "—";
  const ms = run.finished_at_ms - run.started_at_ms;
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`;
}
