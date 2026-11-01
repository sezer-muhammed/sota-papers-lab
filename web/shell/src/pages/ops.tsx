import { useEffect, useState } from "react";
import { Play } from "lucide-react";
import type { RunRecord } from "@spl/sdk";
import { Badge, Button, Card, SectionHeader, Surface, cn, text } from "@spl/ui";
import { client, errorMessage, useLoad } from "../api";
import { JsonView } from "../components/json-view";
import { PageIntro } from "../components/page-intro";
import { RunStatusSignal, runDuration } from "../components/run-status";

export function OpsPage() {
  const ops = useLoad(() => client.ops());
  const [selectedId, setSelectedId] = useState<string>();
  const [input, setInput] = useState("{}");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<RunRecord>();
  const [runError, setRunError] = useState<string>();

  const selected = ops.data?.find((op) => op.id === selectedId) ?? ops.data?.[0];

  useEffect(() => {
    setInput("{}");
    setResult(undefined);
    setRunError(undefined);
  }, [selected?.id]);

  async function run() {
    if (!selected) return;
    setRunning(true);
    setRunError(undefined);
    try {
      setResult(await client.run(selected.id, JSON.parse(input)));
    } catch (e) {
      setRunError(errorMessage(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="space-y-6">
      <PageIntro
        eyebrow="Ops"
        title="Run any op"
        lead="Every app exposes typed ops. Pick one, give it JSON input, and the host records the run, its metrics and the environment it ran in."
      />
      {ops.error ? (
        <Card tone="warning" depth="flat">
          {errorMessage(ops.error)}
        </Card>
      ) : null}
      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <Surface tone="raised" className="h-fit overflow-hidden">
          <ul>
            {ops.data?.map((op) => (
              <li key={op.id} className="border-b border-[var(--ds-gray-alpha-300)] last:border-b-0">
                <button
                  type="button"
                  onClick={() => setSelectedId(op.id)}
                  className={cn(
                    "w-full px-4 py-3 text-left outline-none transition-colors hover:bg-[var(--ds-gray-100)] focus-visible:shadow-[var(--ds-focus-ring)]",
                    op.id === selected?.id && "bg-[var(--ds-gray-100)]",
                  )}
                >
                  <span className="block font-mono text-[13px] font-semibold text-[var(--ds-gray-1000)]">
                    {op.id}
                  </span>
                  <span className="mt-0.5 block text-[13px] leading-5 text-[var(--ds-gray-900)]">
                    {op.summary}
                  </span>
                </button>
              </li>
            ))}
            {ops.data?.length === 0 ? (
              <li className={`${text.small} p-4`}>No apps linked into this host.</li>
            ) : null}
          </ul>
        </Surface>

        {selected ? (
          <Surface>
            <SectionHeader
              eyebrow={selected.id}
              title={selected.summary}
              action={
                <Button variant="primary" icon={Play} onClick={() => void run()} disabled={running}>
                  {running ? "Running" : "Run"}
                </Button>
              }
            />
            <div className="grid gap-4 p-4 xl:grid-cols-2">
              <label className="block">
                <span className={text.eyebrowSm}>Input · JSON</span>
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  spellCheck={false}
                  rows={8}
                  className="mt-2 w-full rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] p-3 font-mono text-[12px] leading-5 text-[var(--ds-gray-1000)] outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
                />
              </label>
              <div>
                <span className={text.eyebrowSm}>Input schema</span>
                <JsonView value={selected.input_schema} className="mt-2 max-h-[190px]" />
              </div>
            </div>
            {runError || result ? (
              <div className="border-t border-[var(--ds-gray-alpha-400)] p-4">
                {runError ? (
                  <Card tone="warning" depth="flat">
                    {runError}
                  </Card>
                ) : result ? (
                  <div className="space-y-3">
                    <div className="flex flex-wrap items-center gap-3">
                      <RunStatusSignal status={result.status} />
                      <Badge>{runDuration(result)}</Badge>
                      <span className={text.meta}>{result.id}</span>
                    </div>
                    {result.error_message ? (
                      <Card tone="warning" depth="flat">
                        {result.error_message}
                      </Card>
                    ) : (
                      <JsonView value={result.output} />
                    )}
                  </div>
                ) : null}
              </div>
            ) : null}
          </Surface>
        ) : null}
      </div>
    </div>
  );
}
