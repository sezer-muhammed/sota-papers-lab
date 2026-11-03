import { useEffect, useRef, useState } from "react";
import { ChevronDown, Clock, Download, Square } from "lucide-react";
import type { ArtifactRef, RunDetail } from "@spl/sdk";
import { isFinished } from "@spl/sdk";
import type { TrainOutput } from "@spl/bindings/image-to-3d/TrainOutput";
import { Button, ButtonLink, ProgressBar, SectionHeader, Surface, cn, text } from "@spl/ui";
import { client, hasToken } from "../api";
import { RunStatusSignal, runDuration } from "../components/run-status";
import { downloadArtifact } from "../retouch/artifact";
import { RECONSTRUCT_OP, liveNumbers, type ReconstructOutput } from "./reconstruct";
import { ReconstructResult, StageBar } from "./reconstruct-panels";
import { useRuns } from "./history";
import { duration, framesOf, gpuWait, presetOf, queuePosition, remainingSeconds, stageModel, stageOf, waitHeadline, type FlowStage } from "./progress";

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
        {detail && going ? <RunWait detail={detail} /> : null}
        {reconstruct && (going || succeeded) ? <StageBar live={live} mesh={wantsMesh} finished={succeeded} /> : null}
        {going && detail?.progress ? (
          <div className="space-y-1.5">
            <ProgressBar label="Overall progress" value={detail.progress.fraction} />
            {gpuWait(detail.progress.data) ? null : (
              <p className={text.meta}>
                {detail.progress.message} · {Math.round(detail.progress.fraction * 100)}%
              </p>
            )}
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
        {rout ? (
          // Its files are in the Downloads panel next to this one.
          <ReconstructResult out={{ ...rout, spz: undefined, ply: undefined, glb: undefined, cameras: undefined, colmap: undefined, transforms: undefined }} />
        ) : null}
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

  return (
    <div className="space-y-4 border-t border-[var(--ds-gray-alpha-400)] pt-4">
      <Facts rows={rows} />
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

const GB = 1e9;

/** What a going run waits for, in plain words, with the elapsed time and an
    ETA from this host's finished runs; the raw numbers sit behind Details. */
function RunWait({ detail }: { detail: RunDetail }) {
  const run = detail.run;
  const { runs } = useRuns();
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const wait = gpuWait(detail.progress?.data);
  const queued = run.status === "queued";
  const position = queued && runs ? queuePosition(runs, run.id) : undefined;
  const stage = queued ? undefined : stageOf(detail.progress?.message, detail.progress?.data);
  // When the current stage began, as this page saw it.
  const since = useRef<{ stage?: FlowStage; at: number }>({ at: Date.now() });
  if (since.current.stage !== stage) since.current = { stage, at: Date.now() };
  const headline = waitHeadline({ queued, reason: detail.waiting_for, wait, position });
  const waitedS = wait?.waitedS ?? (queued ? (now - run.created_at_ms) / 1000 : undefined);
  const left =
    runs && !queued && !wait
      ? remainingSeconds({
          model: stageModel(runs.filter((r) => r.id !== run.id), presetOf(run)),
          frames: framesOf(run) ?? 100,
          stage,
          stageElapsedS: (now - since.current.at) / 1000,
          message: detail.progress?.message,
          mesh: Boolean((run.input as { mesh?: boolean } | null)?.mesh),
        })
      : undefined;
  const elapsedS = run.started_at_ms ? (now - run.started_at_ms) / 1000 : undefined;

  return (
    <div data-testid="splat-run-wait" className="space-y-1.5">
      {headline ? (
        <p role="status" className="text-[14px] text-[var(--ds-gray-1000)]">
          {headline}
        </p>
      ) : null}
      <p className={cn(text.meta, "flex flex-wrap items-center gap-x-3 gap-y-1")}>
        <Clock aria-hidden="true" className="h-3.5 w-3.5" />
        {headline && waitedS !== undefined ? <span>Waiting {duration(waitedS)}</span> : null}
        {!headline && elapsedS !== undefined ? <span>Elapsed {duration(elapsedS)}</span> : null}
        {left !== undefined ? <span data-testid="splat-eta">About {duration(left)} left</span> : null}
        {left === undefined && !headline ? <span>Time left appears once this host has finished a run</span> : null}
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          className={cn(text.link, "ml-auto inline-flex items-center gap-1")}
        >
          Details <ChevronDown aria-hidden="true" className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")} />
        </button>
      </p>
      {open ? (
        <dl data-testid="splat-run-details" className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-[6px] bg-[var(--ds-background-200)] p-2 font-mono text-[12px] text-[var(--ds-gray-900)]">
          <dt>status</dt>
          <dd>{run.status}{detail.waiting_for ? ` (waits for ${detail.waiting_for})` : ""}</dd>
          {stage ? (<><dt>stage</dt><dd>{stage}</dd></>) : null}
          {detail.progress?.message ? (<><dt>message</dt><dd className="break-words">{detail.progress.message}</dd></>) : null}
          {wait?.needBytes !== undefined ? (<><dt>needs</dt><dd>{(wait.needBytes / GB).toFixed(2)} GB</dd></>) : null}
          {wait?.freeBytes !== undefined ? (<><dt>free now</dt><dd>{(wait.freeBytes / GB).toFixed(2)} GB</dd></>) : null}
          {wait?.timeoutS !== undefined ? (<><dt>gives up after</dt><dd>{duration(wait.timeoutS)}</dd></>) : null}
          {position ? (<><dt>queue position</dt><dd>{position}</dd></>) : null}
        </dl>
      ) : null}
    </div>
  );
}
