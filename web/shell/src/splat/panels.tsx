import { useState } from "react";
import { Download, Square } from "lucide-react";
import type { ArtifactRef, RunDetail } from "@spl/sdk";
import { isFinished } from "@spl/sdk";
import type { TrainOutput } from "@spl/bindings/image-to-3d/TrainOutput";
import { Button, ButtonLink, ProgressBar, SectionHeader, Surface, cn, text } from "@spl/ui";
import { client, hasToken } from "../api";
import { RunStatusSignal, runDuration } from "../components/run-status";
import { downloadArtifact } from "../retouch/artifact";
import { RECONSTRUCT_OP, liveNumbers, type ReconstructOutput } from "./reconstruct";
import { ReconstructResult, StageBar } from "./reconstruct-panels";

export const mib = (bytes: number) =>
  bytes < 2 ** 20 ? `${Math.max(1, Math.round(bytes / 1024))} KiB` : `${(bytes / 2 ** 20).toFixed(bytes < 10 * 2 ** 20 ? 1 : 0)} MiB`;

/** Numbers the op attaches to its progress (`cx.progress_with`), shown as
    they come: iteration, Gaussians, PSNR, ... */
function progressFacts(data: unknown): [string, string][] {
  if (!data || typeof data !== "object") return [];
  return Object.entries(data as Record<string, unknown>).flatMap(([k, v]) =>
    typeof v === "number" && Number.isFinite(v)
      ? [[k.replaceAll("_", " "), Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3)] as [string, string]]
      : typeof v === "string"
        ? [[k.replaceAll("_", " "), v] as [string, string]]
        : [],
  );
}

export function RunPanel({
  detail,
  error,
  onCancel,
  snapshot,
}: {
  detail?: RunDetail;
  error?: string;
  onCancel: () => void;
  snapshot?: { revision: number; info: unknown };
}) {
  const run = detail?.run;
  const going = run ? !isFinished(run.status) : false;
  const reconstruct = run?.op_id === RECONSTRUCT_OP;
  const succeeded = run?.status === "succeeded";
  const out = succeeded && !reconstruct ? (run.output as TrainOutput | null) : null;
  const rout = succeeded && reconstruct ? (run.output as ReconstructOutput | null) : null;
  const live = liveNumbers(detail?.progress?.data);
  const facts = reconstruct ? [] : progressFacts(detail?.progress?.data);
  const wantsMesh = Boolean((run?.input as { mesh?: boolean } | undefined)?.mesh);
  const snapshotIter = progressFacts(snapshot?.info).find(([k]) => k === "iteration")?.[1];

  return (
    <Surface aria-labelledby="splat-run-title">
      <SectionHeader
        titleId="splat-run-title"
        eyebrow="Run"
        title={run ? (reconstruct ? "Reconstruction" : "Training") : "Waiting for the run"}
        action={
          going ? (
            <Button size="sm" icon={Square} onClick={onCancel}>
              Cancel
            </Button>
          ) : null
        }
      />
      <div className="space-y-4 p-4" aria-live="polite">
        {run ? (
          <div className="flex flex-wrap items-center gap-3">
            <RunStatusSignal status={run.status} />
            <a className={cn(text.link, "font-mono text-[13px]")} href={`#/runs/${encodeURIComponent(run.id)}`}>
              {run.id.slice(0, 12)}
            </a>
            <span className={text.meta}>{runDuration(run)}</span>
          </div>
        ) : null}
        {detail?.waiting_for ? (
          <p className={text.caption}>
            Queued: waiting for {detail.waiting_for === "gpu" ? "the GPU (another op holds it)" : "a free run slot"}.
          </p>
        ) : null}
        {reconstruct && (going || succeeded) ? <StageBar live={live} mesh={wantsMesh} finished={succeeded} /> : null}
        {going && detail?.progress ? (
          <div className="space-y-1.5">
            <ProgressBar label="Overall progress" value={detail.progress.fraction} />
            <p className={text.meta}>
              {detail.progress.message} · {Math.round(detail.progress.fraction * 100)}%
            </p>
          </div>
        ) : going ? (
          <ProgressBar label="Overall progress" />
        ) : null}
        {facts.length ? <Facts rows={facts} /> : null}
        {snapshot ? (
          <p className={text.meta}>
            Viewer shows snapshot {snapshot.revision}
            {snapshotIter ? ` (iteration ${snapshotIter})` : ""}
            {going ? "; it updates as training goes." : "."}
          </p>
        ) : null}
        {run?.status === "failed" ? (
          <p role="alert" className={text.caption}>
            {run.error_message ?? "The run failed."}
          </p>
        ) : null}
        {error ? (
          <p role="alert" className={text.caption}>
            {error}
          </p>
        ) : null}
        {going && detail?.logs.length ? (
          <pre className="max-h-32 overflow-auto rounded-[6px] bg-[var(--ds-background-200)] p-2 font-mono text-[12px] leading-5 text-[var(--ds-gray-900)]">
            {detail.logs.slice(-6).join("\n")}
          </pre>
        ) : null}
        {out ? <Result out={out} /> : null}
        {rout ? <ReconstructResult out={rout} /> : null}
      </div>
    </Surface>
  );
}

function Facts({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3">
      {rows.map(([k, v]) => (
        <div key={k} className="min-w-0">
          <dt className={text.meta}>{k}</dt>
          <dd className="truncate font-mono text-[14px] text-[var(--ds-gray-1000)]">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function Result({ out }: { out: TrainOutput }) {
  const rows: [string, string][] = [
    ["views", `${out.scene.train_views} train / ${out.scene.test_views} test`],
    ["size", `${out.scene.width}×${out.scene.height}`],
  ];
  if (out.summary) {
    rows.push(
      ["train time", `${out.summary.train_s.toFixed(1)} s`],
      ["iterations", out.summary.iterations.toLocaleString()],
      ["gaussians", out.summary.final_gaussians.toLocaleString()],
      ["peak VRAM", `${(out.summary.peak_vram_bytes / 2 ** 30).toFixed(2)} GiB`],
    );
  }
  if (out.eval) rows.push(["PSNR", `${out.eval.psnr.toFixed(2)} dB`], ["SSIM", out.eval.ssim.toFixed(4)]);
  const all: [string, string, ArtifactRef | undefined][] = [
    ["SPZ", "scene.spz", out.spz],
    ["PLY", "scene.ply", out.ply],
    ["COLMAP", "cameras.zip", out.colmap],
  ];
  const files = all.filter((f): f is [string, string, ArtifactRef] => Boolean(f[2]));

  return (
    <div className="space-y-4 border-t border-[var(--ds-gray-alpha-400)] pt-4">
      <Facts rows={rows} />
      {files.length ? (
        <div className="flex flex-wrap gap-2">
          {files.map(([label, name, ref]) => (
            <ArtifactDownload key={label} label={label} name={name} file={ref} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** A real link; on a token-protected host the click fetches with the header. */
export function ArtifactDownload({ file, label, name }: { file: ArtifactRef; label: string; name: string }) {
  const [busy, setBusy] = useState(false);
  return (
    <ButtonLink
      size="sm"
      icon={Download}
      href={client.artifactUrl(file.id, { download: name })}
      download={name}
      onClick={(event) => {
        if (!hasToken()) return;
        event.preventDefault();
        if (busy) return;
        setBusy(true);
        downloadArtifact(file.id, name).finally(() => setBusy(false));
      }}
    >
      {label} <span className={text.meta}>{mib(file.size_bytes)}</span>
    </ButtonLink>
  );
}
