import { Fragment, useState } from "react";
import { RefreshCw } from "lucide-react";
import type { RunDetail } from "@spl/sdk";
import { Button, Card, SectionHeader, Surface, cn, text } from "@spl/ui";
import { client, errorMessage, useLoad } from "../api";
import { JsonView } from "../components/json-view";
import { PageIntro } from "../components/page-intro";
import { RunStatusSignal, runDuration } from "../components/run-status";

export function RunsPage() {
  const runs = useLoad(() => client.runs({ limit: 100 }));
  const [openId, setOpenId] = useState<string>();
  const [detail, setDetail] = useState<RunDetail>();

  async function toggle(id: string) {
    if (openId === id) {
      setOpenId(undefined);
      return;
    }
    setOpenId(id);
    setDetail(undefined);
    setDetail(await client.runDetail(id));
  }

  return (
    <div className="space-y-6">
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
                    onClick={() => void toggle(run.id)}
                    className={cn(
                      "cursor-pointer border-b border-[var(--ds-gray-alpha-300)] transition-colors hover:bg-[var(--ds-gray-100)]",
                      openId === run.id && "bg-[var(--ds-gray-100)]",
                    )}
                  >
                    <td className="px-4 py-3">
                      <RunStatusSignal status={run.status} />
                    </td>
                    <td className="px-4 py-3 font-mono text-[13px] text-[var(--ds-gray-1000)]">{run.op_id}</td>
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
                          <div className="grid gap-4 xl:grid-cols-[1fr_280px]">
                            <JsonView value={detail.run.error_message ?? detail.run.output} />
                            <div>
                              <p className={text.eyebrowSm}>Metrics</p>
                              <ul className="mt-2 space-y-1.5">
                                {detail.metrics.map((m) => (
                                  <li key={m.seq} className="flex justify-between gap-3 font-mono text-[12px]">
                                    <span className="text-[var(--ds-gray-900)]">{m.key}</span>
                                    <span className="text-[var(--ds-gray-1000)]">
                                      {Number(m.value.toFixed(4))} {m.unit ?? ""}
                                    </span>
                                  </li>
                                ))}
                                {detail.metrics.length === 0 ? (
                                  <li className={text.meta}>No metrics recorded</li>
                                ) : null}
                              </ul>
                            </div>
                          </div>
                        ) : (
                          <div className="ds-skeleton h-24 rounded-[7px]" />
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
