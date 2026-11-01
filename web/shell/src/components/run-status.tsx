import type { RunStatus } from "@spl/sdk";
import { StatusSignal } from "@spl/ui";
import { splitDuration } from "../lib/format";

export const statusColor: Record<RunStatus, string> = {
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

/** "1.25 s", or "—" while the run has not finished: splitDuration as one string,
    so Home, Ops and Runs read the same. */
export function runDuration(run: { started_at_ms: number | null; finished_at_ms: number | null }) {
  const duration = splitDuration(run);
  return duration ? `${duration.value} ${duration.unit}` : "—";
}
