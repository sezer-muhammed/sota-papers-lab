import { Fragment, useEffect, useRef, useState } from "react";
import { RefreshCw } from "lucide-react";
import type { RunDetail } from "@spl/sdk";
import { Button, Card, SectionHeader, Skeleton, Surface, cn, text } from "@spl/ui";
import { client, describeError, errorMessage, isUnauthorized, useLoad } from "../api";
import { JsonView } from "../components/json-view";
import { PageIntro } from "../components/page-intro";
import { RunStatusSignal, runDuration } from "../components/run-status";
import { TokenPrompt } from "../components/token-prompt";
import { formatMetric } from "../lib/format";
import { prefersReducedMotion } from "../lib/scroll";

export function RunsPage({ initialRunId }: { initialRunId?: string }) {
  const runs = useLoad(() => client.runs({ limit: 100 }));
  const [openId, setOpenId] = useState<string>();
  const [detail, setDetail] = useState<RunDetail>();
  const [detailError, setDetailError] = useState<unknown>();
  // The run asked for last: a slower answer for an earlier one is dropped.
  const requested = useRef<string | undefined>(undefined);
  const scrolledTo = useRef<string | undefined>(undefined);

  async function open(id: string) {
    requested.current = id;
    setOpenId(id);
    setDetail(undefined);
    setDetailError(undefined);
    try {
      const loaded = await client.runDetail(id);
      if (requested.current === id) setDetail(loaded);
    } catch (e) {
      if (requested.current === id) setDetailError(e);
    }
  }

  function toggle(id: string) {
    if (openId === id) {
      setOpenId(undefined);
      return;
    }
    void open(id);
  }

  // #/runs/<id> opens that run.
  useEffect(() => {
    if (initialRunId) void open(initialRunId);
  }, [initialRunId]);

  // ...and brings its row into view once the table has loaded (once per link),
  // or the panel above the table when the run is older than the newest 100.
  // Focus goes there too (the row's button, or the panel) unless the user has
  // already moved it.
  useEffect(() => {
    if (!initialRunId || !runs.data || scrolledTo.current === initialRunId) return;
    scrolledTo.current = initialRunId;
    const behavior = prefersReducedMotion() ? "instant" : "smooth";
    const inTable = runs.data.some((run) => run.id === initialRunId);
    const target = document.getElementById(inTable ? `run-${initialRunId}` : "pinned-run");
    target?.scrollIntoView({ block: inTable ? "center" : "start", behavior });
    const active = document.activeElement;
    if (!active || active === document.body || active.id === "main") {
      (inTable ? target?.querySelector("button") : target)?.focus({ preventScroll: true });
    }
  }, [initialRunId, runs.data]);

  // A linked run older than the newest 100 gets its own panel above the table.
  const pinned =
    initialRunId && runs.data && !runs.data.some((run) => run.id === initialRunId) ? initialRunId : undefined;
  const pinnedOpen = pinned !== undefined && openId === pinned;
  const pinnedDetail = pinnedOpen && detail?.run.id === pinned ? detail : undefined;

  const intro = (
    <PageIntro
      eyebrow="Runs"
      title="Run history"
      lead="Every run with its input, output, metrics and environment (GPU, driver, version), so results stay reproducible across machines."
      action={
        <Button icon={RefreshCw} onClick={() => void runs.reload()} disabled={runs.loading}>
          Refresh
        </Button>
      }
    />
  );

  // /v1/runs sits behind SPL_AUTH_TOKEN; /v1/health does not.
  if (isUnauthorized(runs.error)) {
    return (
      <div className="space-y-6">
        {intro}
        <TokenPrompt />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {intro}
      {pinned ? (
        <Surface id="pinned-run" tabIndex={-1} className="scroll-mt-4 outline-none md:scroll-mt-24">
          <SectionHeader
            eyebrow="Run"
            title={
              pinnedDetail?.run.op_id ??
              (!pinnedOpen ? "Older run" : detailError ? "Run unavailable" : "Loading run…")
            }
            summary="This run is older than the newest 100, so it is not in the table below."
          />
          <div className="p-4">
            {pinnedDetail ? (
              <RunDetailBody detail={pinnedDetail} />
            ) : !pinnedOpen ? (
              <Button
                size="sm"
                onClick={(event) => {
                  void open(pinned);
                  // The run replaces this button: keep focus on its panel.
                  event.currentTarget.closest<HTMLElement>("#pinned-run")?.focus({ preventScroll: true });
                }}
              >
                Show this run
              </Button>
            ) : detailError ? (
              <DetailError error={detailError} />
            ) : (
              <Skeleton className="h-24 rounded-[7px]" />
            )}
          </div>
        </Surface>
      ) : null}
      <Surface className="overflow-hidden">
        <SectionHeader title="Recent runs" summary="Newest first. Select a run to see its output and metrics." />
        {runs.error ? (
          <div className="p-4">
            <Card tone="warning" depth="flat">
              {errorMessage(runs.error)}
            </Card>
          </div>
        ) : null}
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-left">
            <thead>
              <tr className="border-b border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)]">
                {["Status", "Op", "Run", "Started", "Duration"].map((label) => (
                  <th key={label} className={`${text.eyebrowSm} px-4 py-2.5 font-normal`}>
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.data?.map((run) => (
                <Fragment key={run.id}>
                  <tr
                    id={`run-${run.id}`}
                    onClick={() => toggle(run.id)}
                    className={cn(
                      "cursor-pointer border-b border-[var(--ds-gray-alpha-300)] transition-colors hover:bg-[var(--ds-gray-100)]",
                      openId === run.id && "bg-[var(--ds-gray-100)]",
                    )}
                  >
                    <td className="px-4 py-3">
                      <RunStatusSignal status={run.status} />
                    </td>
                    <td className="px-4 py-3 font-mono text-[13px] text-[var(--ds-gray-1000)]">
                      {/* The whole row opens on click; this button opens it from the keyboard. */}
                      <button
                        type="button"
                        aria-expanded={openId === run.id}
                        onClick={(event) => {
                          event.stopPropagation(); // the row's onClick would toggle it straight back
                          toggle(run.id);
                        }}
                        className="rounded-[4px] text-left outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
                      >
                        {run.op_id}
                      </button>
                    </td>
                    <td className={`${text.meta} px-4 py-3`}>{run.id.slice(-12)}</td>
                    <td className="px-4 py-3 text-[13px] text-[var(--ds-gray-900)]">
                      {new Date(run.created_at_ms).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 font-mono text-[13px] text-[var(--ds-gray-900)]">
                      {runDuration(run)}
                    </td>
                  </tr>
                  {openId === run.id ? (
                    <tr className="border-b border-[var(--ds-gray-alpha-300)]">
                      <td colSpan={5} className="bg-[var(--ds-background-200)] p-4">
                        {detail?.run.id === run.id ? (
                          <RunDetailBody detail={detail} />
                        ) : detailError ? (
                          <DetailError error={detailError} />
                        ) : (
                          <Skeleton className="h-24 rounded-[7px]" />
                        )}
                      </td>
                    </tr>
                  ) : null}
                </Fragment>
              ))}
            </tbody>
          </table>
          {runs.data?.length === 0 ? (
            <p className={`${text.small} p-4`}>No runs yet. Start one from the Ops page.</p>
          ) : null}
        </div>
      </Surface>
    </div>
  );
}

/* Output (or the error) next to the run's metrics. */
function RunDetailBody({ detail }: { detail: RunDetail }) {
  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_280px]">
      <JsonView value={detail.run.error_message ?? detail.run.output} />
      <div>
        <p className={text.eyebrowSm}>Metrics</p>
        <ul className="mt-2 space-y-1.5">
          {detail.metrics.map((m) => (
            <li key={m.seq} className="flex justify-between gap-3 font-mono text-[12px]">
              <span className="text-[var(--ds-gray-900)]">{m.key}</span>
              <span className="text-[var(--ds-gray-1000)]">{formatMetric(m.value, m.unit)}</span>
            </li>
          ))}
          {detail.metrics.length === 0 ? <li className={text.meta}>No metrics recorded</li> : null}
        </ul>
      </div>
    </div>
  );
}

function DetailError({ error }: { error: unknown }) {
  return (
    <Card tone="warning" depth="flat">
      Couldn't load this run: {describeError(error)}
    </Card>
  );
}
