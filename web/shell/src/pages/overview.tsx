import { Cpu, RefreshCw } from "lucide-react";
import type { GpuReport, Health } from "@spl/sdk";
import { Badge, Button, Card, SectionHeader, StatusSignal, Surface, text } from "@spl/ui";
import { errorMessage } from "../api";
import { PageIntro } from "../components/page-intro";

type HealthState = {
  data?: Health;
  error?: unknown;
  loading: boolean;
  reload: () => Promise<void>;
};

const GIB = 1024 ** 3;

export function OverviewPage({ health }: { health: HealthState }) {
  const env = health.data?.env as { os?: string; arch?: string; gpu?: GpuReport } | undefined;
  const gpu = env?.gpu;

  return (
    <div className="space-y-6">
      <PageIntro
        eyebrow="Overview"
        title="Lab host"
        lead="The GPUs, ops and runs on the host this page is connected to. Every app registers its ops here, so each can run alone, inside a pipeline, or under a benchmark."
        action={
          <Button icon={RefreshCw} onClick={() => void health.reload()} disabled={health.loading}>
            Refresh
          </Button>
        }
      />

      {health.error ? (
        <Card tone="warning" depth="flat">
          <p className="text-[14px] font-medium">Can't reach the host</p>
          <p className="mt-1 text-[14px] leading-6">{errorMessage(health.error)}</p>
        </Card>
      ) : null}

      <div className="stagger grid gap-3 sm:grid-cols-3">
        <Stat label="Ops registered" value={health.data ? String(health.data.ops) : "—"} />
        <Stat label="CUDA driver" value={gpu?.driver_version ?? (health.data ? "None" : "—")} />
        <Stat label="GPUs" value={gpu ? String(gpu.devices.length) : "—"} />
      </div>

      <Surface className="reveal">
        <SectionHeader
          eyebrow="Hardware"
          title="GPUs"
          summary={
            env ? `${env.os ?? "?"} · ${env.arch ?? "?"} · spl ${health.data?.version}` : undefined
          }
        />
        <div className="p-4">
          {gpu && gpu.devices.length > 0 ? (
            <ul className="divide-y divide-[var(--ds-gray-alpha-300)]">
              {gpu.devices.map((device) => (
                <li key={device.ordinal} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-3">
                  <div className="grid h-10 w-10 place-items-center rounded-[8px] border border-[var(--ds-gray-alpha-300)] bg-[var(--ds-background-200)]">
                    <Cpu aria-hidden="true" className="h-5 w-5 text-[var(--ds-gray-800)]" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[15px] font-semibold text-[var(--ds-gray-1000)]">
                      {device.name}
                    </p>
                    <p className={text.meta}>
                      GPU {device.ordinal} · {device.multiprocessors} SMs
                    </p>
                  </div>
                  <Badge tone="blue">sm {device.compute_capability}</Badge>
                  <span className="font-mono text-[13px] text-[var(--ds-gray-900)]">
                    {(device.total_memory_bytes / GIB).toFixed(1)} GiB
                  </span>
                  <StatusSignal variant="dot" color="var(--ds-green-700)" />
                </li>
              ))}
            </ul>
          ) : (
            <Card tone="muted" depth="flat">
              <p className="text-[14px] font-medium text-[var(--ds-gray-1000)]">No CUDA devices</p>
              <p className="mt-1 text-[14px] leading-6 text-[var(--ds-gray-900)]">
                {gpu?.error ??
                  "Waiting for the host. Ops run on a Linux machine with an NVIDIA driver; this page can run anywhere."}
              </p>
            </Card>
          )}
        </div>
      </Surface>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Card depth="deep">
      <p className={text.eyebrowSm}>{label}</p>
      <p className="mt-2 font-mono text-[28px] font-semibold leading-8 text-[var(--ds-gray-1000)]">
        {value}
      </p>
    </Card>
  );
}
