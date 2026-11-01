import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ArrowLeft, CircleStop, Download, Eraser, WandSparkles } from "lucide-react";
import { ActionBar, Button, ButtonLink, Card, Surface, cn, text } from "@spl/ui";
import { client, describeError, isUnauthorized, useLoad } from "../api";
import { RetryNotice } from "../components/notice";
import { PageIntro } from "../components/page-intro";
import { TokenPrompt } from "../components/token-prompt";
import { plural } from "../lib/format";
import { DropZone, useFileDrop } from "../retouch/drop-zone";
import { finalFiles, isActive, needsRetouch, photoPhase, queueOrder, type Phase, type Photo } from "../retouch/model";
import { OptionsPanel } from "../retouch/options-panel";
import { PhotoCard } from "../retouch/photo-card";
import { PhotoDetail } from "../retouch/photo-detail";
import { OPS, type AddResult } from "../retouch/session";
import { useRetouch } from "../retouch/use-retouch";

/* raw_retouch's page: add RAW files, retouch them all at the chosen
   parallelism, then review each photo (#/retouch/<key>) and download. The
   session lives outside the page (use-retouch.ts), so work continues while
   you are elsewhere and a reload picks it up again. */
export function RetouchPage({ photoKey }: { photoKey?: string }) {
  const ops = useLoad(() => client.ops());
  const { state, session } = useRetouch();
  const [added, setAdded] = useState("");
  const locked = state.locked || isUnauthorized(ops.error);
  const linked = ops.data ? ops.data.some((op) => op.id === OPS.auto) : undefined;
  const canAdd = !locked && linked !== false;

  const addFiles = useCallback((files: File[]) => setAdded(describeAdd(session.addFiles(files))), [session]);
  const dragging = useFileDrop(addFiles, canAdd);
  const positions = useMemo(() => queuePositions(state.photos), [state.photos]);

  // Mounted from the start, so screen readers announce every change.
  const announcer = (
    <p aria-live="polite" className="sr-only">
      {state.announcement?.text}
    </p>
  );

  if (locked) {
    return (
      <div className="space-y-6">
        <Intro />
        <TokenPrompt />
      </div>
    );
  }

  if (photoKey !== undefined) {
    const photo = state.photos.find((p) => p.key === photoKey);
    return (
      <>
        {photo ? (
          <PhotoDetail
            key={photo.key}
            photo={photo}
            photos={state.photos}
            position={positions.get(photo.key)}
            session={session}
          />
        ) : (
          <NotInSession />
        )}
        {announcer}
      </>
    );
  }

  const photos = state.photos;

  return (
    <div className="space-y-6">
      <Intro
        action={
          photos.length > 0 ? (
            <ClearSession active={photos.filter(isActive).length} onClear={() => session.clear()} />
          ) : undefined
        }
      />

      {ops.error && !isUnauthorized(ops.error) ? (
        <RetryNotice title="Couldn't check this host's apps." onRetry={() => void ops.reload()}>
          {describeError(ops.error)}
        </RetryNotice>
      ) : null}
      {linked === false ? (
        <Card tone="warning" depth="flat" role="status">
          <p className={cn(text.small, "text-inherit")}>
            This host does not run the raw_retouch app, so photos can't be retouched here. Build spl with the
            raw-retouch feature and restart it.
          </p>
        </Card>
      ) : null}
      {state.trouble ? (
        <Card tone="warning" depth="flat" density="compact" role="status">
          <p className={cn(text.caption, "text-inherit")}>
            Lost contact with the host ({state.trouble}). Retrying; running retouches continue on the host.
          </p>
        </Card>
      ) : null}

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_340px] xl:items-start">
        <div className="min-w-0 space-y-4">
          <DropZone compact={photos.length > 0} dragging={dragging} disabled={!canAdd} onFiles={addFiles} />
          <p role="status" className={cn(text.caption, !added && "sr-only")}>
            {added}
          </p>
          {photos.length > 0 ? (
            <PhotosPanel canAdd={canAdd} positions={positions} />
          ) : (
            <HowItWorks />
          )}
        </div>
        <OptionsPanel
          className="xl:sticky xl:top-24 xl:max-h-[calc(100vh-7rem)] xl:overflow-y-auto"
          options={state.options}
          onChange={(patch) => session.setOptions(patch)}
          onReset={() => session.resetOptions()}
        />
      </div>
      {announcer}
    </div>
  );
}

function Intro({ action }: { action?: ReactNode }) {
  return (
    <PageIntro
      eyebrow="App · raw_retouch"
      title="Retouch RAW photos"
      lead="Each photo is developed on the GPU, searched over Lightroom's ten Basic sliders, judged by Claude, and exported as a full-resolution JPEG with an XMP sidecar."
      action={action}
    />
  );
}

const PHASE_ORDER: Phase[] = ["done", "working", "queued", "uploading", "reading", "ready", "failed", "cancelled"];

/* The toolbar (counts, Retouch all, Cancel all, Download all) and the grid. */
function PhotosPanel({ canAdd, positions }: { canAdd: boolean; positions: Map<string, number> }) {
  const { state, session } = useRetouch();
  const photos = state.photos;
  const counts = new Map<Phase, number>();
  for (const photo of photos) {
    const phase = photoPhase(photo);
    counts.set(phase, (counts.get(phase) ?? 0) + 1);
  }
  const summary = [
    plural(photos.length, "photo"),
    ...PHASE_ORDER.flatMap((phase) => (counts.get(phase) ? [`${counts.get(phase)} ${phase}`] : [])),
  ].join(" · ");
  const pending = photos.filter(needsRetouch).length;
  const active = photos.filter(isActive).length;
  const jpegs = photos.filter((p) => finalFiles(p).jpeg).length;
  const xmps = photos.filter((p) => finalFiles(p).xmp).length;
  const downloading = state.downloading;

  return (
    <Surface tone="flat" aria-labelledby="retouch-photos-title" className="overflow-hidden">
      <div className="flex flex-col gap-3 border-b border-[var(--ds-gray-alpha-400)] p-4">
        <div className="min-w-0">
          <h2 id="retouch-photos-title" className="text-[22px] font-semibold leading-7 text-[var(--ds-gray-1000)]">
            Photos
          </h2>
          <p className={cn(text.meta, "mt-1")}>{summary}</p>
        </div>
        <ActionBar>
          <Button
            variant="primary"
            icon={WandSparkles}
            disabled={pending === 0 || !canAdd}
            onClick={() => session.retouchAll()}
          >
            Retouch all{pending ? ` (${pending})` : ""}
          </Button>
          {active > 0 ? (
            <Button icon={CircleStop} onClick={() => session.cancelAll()}>
              Cancel all
            </Button>
          ) : null}
          <Button
            icon={Download}
            disabled={jpegs === 0 || downloading !== undefined}
            onClick={() => void session.downloadAll("jpeg")}
          >
            Download all JPEGs{jpegs ? ` (${jpegs})` : ""}
          </Button>
          <Button
            icon={Download}
            disabled={xmps === 0 || downloading !== undefined}
            onClick={() => void session.downloadAll("xmp")}
          >
            Download all XMPs{xmps ? ` (${xmps})` : ""}
          </Button>
        </ActionBar>
      </div>
      <p
        role="status"
        className={cn(
          text.caption,
          downloading ? "border-b border-[var(--ds-gray-alpha-300)] px-4 py-2" : "sr-only",
        )}
      >
        {downloading
          ? `Downloading ${downloading.done + 1} of ${downloading.total} ${downloading.kind === "jpeg" ? "JPEGs" : "XMPs"}… Allow multiple downloads if the browser asks.`
          : ""}
      </p>
      <ul className="grid grid-cols-1 gap-3 p-3 sm:grid-cols-2 lg:grid-cols-3">
        {photos.map((photo) => (
          <li key={photo.key} className="flex min-w-0">
            <PhotoCard
              photo={photo}
              position={positions.get(photo.key)}
              session={session}
              canRetryUpload={session.hasFile(photo.key)}
            />
          </li>
        ))}
      </ul>
    </Surface>
  );
}

const steps = [
  { title: "Add photos", body: "Drop your ARW files or a folder. Each one is read for its camera, lens and exposure." },
  {
    title: "Retouch all",
    body: "The search tries thousands of slider settings per photo; Claude plans looks and picks between finalists.",
  },
  {
    title: "Review and download",
    body: "Compare before and after, adjust the sliders if you like, then take the JPEGs and XMP sidecars.",
  },
];

function HowItWorks() {
  return (
    <ol className="grid grid-cols-1 gap-3 md:grid-cols-3">
      {steps.map((step, index) => (
        <li key={step.title}>
          <Card tone="muted" depth="flat" className="h-full">
            <p className={text.eyebrowSm}>Step {index + 1}</p>
            <h2 className="mt-1 text-[15px] font-semibold text-[var(--ds-gray-1000)]">{step.title}</h2>
            <p className={cn(text.caption, "mt-1")}>{step.body}</p>
          </Card>
        </li>
      ))}
    </ol>
  );
}

function NotInSession() {
  return (
    <div className="space-y-4">
      <h1 className={cn(text.pageTitle, "text-[var(--ds-gray-1000)]")}>Photo not found</h1>
      <Card tone="muted" depth="flat" className="flex flex-wrap items-center justify-between gap-3">
        <p className={text.small}>This photo is not in the session any more. It may have been removed or cleared.</p>
        <ButtonLink href="#/retouch" size="sm" icon={ArrowLeft}>
          All photos
        </ButtonLink>
      </Card>
    </div>
  );
}

/* Two steps, so a stray click never wipes a session: Clear session, then
   Clear (or Keep). The question times out after a few seconds. */
function ClearSession({ active, onClear }: { active: number; onClear: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const restoreFocus = useRef(false);

  useEffect(() => {
    if (!confirming) {
      // Keep (or the timeout) removes the focused button: hand focus back.
      if (restoreFocus.current) box.current?.querySelector("button")?.focus();
      restoreFocus.current = false;
      return;
    }
    const timer = window.setTimeout(() => {
      restoreFocus.current = box.current?.contains(document.activeElement) ?? false;
      setConfirming(false);
    }, 8000);
    return () => window.clearTimeout(timer);
  }, [confirming]);

  return (
    <div ref={box} role="group" aria-label="Clear session" className="flex flex-wrap items-center gap-2">
      {confirming ? (
        <>
          <span className={text.caption}>
            {active
              ? `Forget every photo and cancel ${active} ${active === 1 ? "retouch" : "retouches"}?`
              : "Forget every photo?"}
          </span>
          <Button
            autoFocus
            onClick={() => {
              setConfirming(false);
              onClear();
            }}
          >
            Clear
          </Button>
          <Button
            variant="ghost"
            onClick={() => {
              restoreFocus.current = true;
              setConfirming(false);
            }}
          >
            Keep
          </Button>
        </>
      ) : (
        <Button icon={Eraser} onClick={() => setConfirming(true)}>
          Clear session
        </Button>
      )}
    </div>
  );
}

function queuePositions(photos: Photo[]): Map<string, number> {
  const queued = photos.filter((p) => p.auto.status === "queued").sort((a, b) => queueOrder(a) - queueOrder(b));
  return new Map(queued.map((p, index) => [p.key, index + 1]));
}

const listNames = (names: string[]) =>
  names.length > 3 ? `${names.slice(0, 3).join(", ")} and ${names.length - 3} more` : names.join(", ");

function describeAdd(result: AddResult): string {
  const parts: string[] = [];
  if (result.added) parts.push(`Added ${plural(result.added, "photo")}.`);
  if (result.retried) parts.push(`Uploading ${plural(result.retried, "interrupted photo")} again.`);
  if (result.duplicates) parts.push(`Skipped ${plural(result.duplicates, "photo")} already in this session.`);
  if (result.rejected.length) {
    const one = result.rejected.length === 1;
    parts.push(
      `Skipped ${plural(result.rejected.length, "file")} that ${one ? "is not a RAW photo" : "are not RAW photos"}: ${listNames(result.rejected)}.`,
    );
  }
  return parts.join(" ") || "No files were added.";
}
