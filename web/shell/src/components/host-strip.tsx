import { RefreshCw } from "lucide-react";
import type { GpuInfo, GpuReport, Health } from "@spl/sdk";
import { Button, Skeleton, StatusSignal, Surface, cn, text } from "@spl/ui";
import { describeError, type Loaded } from "../api";
import { readEnv, type HomeHostState } from "../lib/env";
import { formatGiB } from "../lib/format";

const CUDA_TITLE = "Highest CUDA version the installed NVIDIA driver supports";

function stripSignal(state: HomeHostState, gpus: number): { label: string; color: string } {
  switch (state) {
    case "connecting":
      return { label: "Connecting", color: "var(--ds-gray-700)" };
    case "online":
      return { label: gpus > 1 ? `${gpus} GPUs detected` : "GPU detected", color: "var(--ds-green-700)" };
    case "no-gpu":
      return { label: "No CUDA device", color: "var(--ds-amber-700)" };
    case "offline":
      return { label: "Host unreachable", color: "var(--ds-red-700)" };
  }
}

function noGpuSentence(gpu: GpuReport | undefined) {
  const reason = gpu?.error?.trim().replace(/\.+$/, "");
  return `${reason || "The driver loaded but reported no devices"}. Ops that need CUDA will fail on this host.`;
}

/* One device reads as itself; several read as a summary (the list below the
   strip has each one). */
function summarizeGpus(devices: GpuInfo[]) {
  const [first] = devices;
  if (!first) return undefined;
  if (devices.length === 1) {
    return {
      name: first.name,
      compute: `sm ${first.compute_capability}`,
      memory: formatGiB(first.total_memory_bytes),
      sms: String(first.multiprocessors),
    };
  }
  const sameName = devices.every((device) => device.name === first.name);
  const computes = [...new Set(devices.map((device) => device.compute_capability))];
  const memory = devices.reduce((sum, device) => sum + device.total_memory_bytes, 0);
  const sms = devices.reduce((sum, device) => sum + device.multiprocessors, 0);
  return {
    name: sameName ? `${devices.length} × ${first.name}` : `${devices.length} GPUs`,
    compute: `sm ${computes.join(", ")}`,
    memory: `${formatGiB(memory)} total`,
    sms: `${sms} total`,
  };
}

/* The GPU and system this page is connected to, from public /v1/health, so
   it still reads on a token-protected host. Target of #/host and the header pill. */
export function HostStrip({
  health,
  refreshAll,
  state,
}: {
  health: Loaded<Health>;
  /** Reloads the host's health. */
  refreshAll: () => void;
  state: HomeHostState;
}) {
  const env = readEnv(health.data?.env);
  const devices = env.gpu?.devices ?? [];
  const gpu = summarizeGpus(devices);
  const stale = state === "offline" && health.data !== undefined;
  const signal = stripSignal(state, devices.length);
  const system = [env.os, env.arch].filter(Boolean).join(" · ");
  // aria-disabled, not disabled: disabling the focused button would drop focus to <body>.
  const refresh = () => {
    if (!health.loading) refreshAll();
  };

  return (
    <Surface
      tone="flat"
      id="host"
      tabIndex={-1}
      aria-labelledby="host-title"
      aria-busy={state === "connecting" || undefined}
      className="reveal scroll-mt-4 overflow-hidden outline-none md:scroll-mt-24"
    >
      <h2 id="host-title" className="sr-only">
        This host
      </h2>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3">
        <span role="status">
          <StatusSignal variant="pill" color={signal.color}>
            {signal.label}
          </StatusSignal>
        </span>
        {/* Below sm the details get a full-width row under the pill and the
            button, so the GPU name is never squeezed beside the pill. */}
        <div className="order-last flex w-full min-w-0 flex-wrap items-center gap-x-5 gap-y-2 sm:order-none sm:w-auto sm:flex-1">
          {state === "connecting" ? <Skeleton className="h-4 w-72 max-w-full" /> : null}
          {state === "offline" && !health.data ? (
            <p className={cn(text.caption, "min-w-0")}>Can't reach the host: {describeError(health.error)}</p>
          ) : null}
          {state === "no-gpu" ? <p className={cn(text.caption, "min-w-0")}>{noGpuSentence(env.gpu)}</p> : null}
          {health.data ? (
            <>
              {stale ? <span className={text.meta}>Last known:</span> : null}
              <dl className="flex min-w-0 flex-wrap items-center gap-x-5 gap-y-1.5">
                {gpu ? (
                  <>
                    <Fact label="GPU" value={gpu.name} />
                    <Fact label="Compute" value={gpu.compute} />
                    <Fact label="Memory" value={gpu.memory} />
                    <Fact label="SMs" value={gpu.sms} className="hidden lg:flex" />
                  </>
                ) : null}
                <Fact label="CUDA" value={env.gpu?.driver_version ?? "—"} title={CUDA_TITLE} />
                <Fact label="System" value={system || "—"} className="hidden md:flex" />
                <Fact label="spl" value={health.data.version} className="hidden md:flex" />
              </dl>
            </>
          ) : null}
        </div>
        <div className="ml-auto">
          {state === "offline" ? (
            <Button size="sm" icon={RefreshCw} aria-disabled={health.loading} onClick={refresh}>
              Retry
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="sm"
              icon={RefreshCw}
              className="w-8 px-0"
              aria-disabled={health.loading}
              onClick={refresh}
            >
              <span className="sr-only">Refresh host status</span>
            </Button>
          )}
        </div>
      </div>
      {devices.length > 1 ? (
        <ul className="grid grid-cols-1 gap-1 border-t border-[var(--ds-gray-alpha-300)] px-4 py-2.5">
          {devices.map((device) => (
            <li key={device.ordinal} className={text.meta}>
              GPU {device.ordinal} · {device.name} · sm {device.compute_capability} ·{" "}
              {formatGiB(device.total_memory_bytes)} · {device.multiprocessors} SMs
            </li>
          ))}
        </ul>
      ) : null}
    </Surface>
  );
}

function Fact({
  className,
  label,
  title,
  value,
}: {
  className?: string;
  label: string;
  title?: string;
  value: string;
}) {
  return (
    <div className={cn("flex min-w-0 items-baseline gap-1.5", className)}>
      <dt className={text.eyebrowSm}>{label}</dt>
      <dd className={cn(text.mono, "min-w-0 truncate")} title={title ?? value}>
        {value}
      </dd>
    </div>
  );
}
