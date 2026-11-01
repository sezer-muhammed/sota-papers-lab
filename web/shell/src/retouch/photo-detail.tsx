import { useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowLeft, ChevronLeft, ChevronRight, CircleStop, FileDown, Undo2 } from "lucide-react";
import type { AutoOutput } from "@spl/bindings/raw-retouch/AutoOutput";
import type { CandidateSummary } from "@spl/bindings/raw-retouch/CandidateSummary";
import {
  ActionBar,
  Badge,
  Button,
  ButtonLink,
  Card,
  ProgressBar,
  SegmentedControl,
  Skeleton,
  Surface,
  cn,
  text,
} from "@spl/ui";
import { prefersReducedMotion } from "../lib/scroll";
import { CompareView, type CompareSide } from "./compare-view";
import { FileList, XmpNote } from "./downloads";
import {
  cameraLine,
  exposureLine,
  formatBytes,
  formatCaptured,
  formatMs,
  formatPercent,
  formatScore,
  sameParams,
  uprightSize,
} from "./format";
import { ArtifactImage } from "./images";
import {
  filesOf,
  isActive,
  neutralId,
  photoInfo,
  photoPhase,
  rankedCandidates,
  thumbnailId,
  type AutoResult,
  type ExportState,
  type Photo,
} from "./model";
import { AdjustSliders, ParamsTable } from "./params";
import { ScopesPanel, useTwoColumns } from "./scopes/scopes-panel";
import type { RetouchSession } from "./session";
import { PhaseSignal, PhotoActions, PhotoProgress, phaseColor } from "./status";

const photoHref = (key: string) => `#/retouch/${encodeURIComponent(key)}`;

/* One photo, large: before/after, the candidates, Claude's reasoning, the
   Develop panels (the chosen values, or sliders to adjust them), the scopes of
   what is compared, downloads and timings. Keyed by photo, so view state
   starts fresh for each one. */
export function PhotoDetail({
  photo,
  photos,
  position,
  session,
}: {
  photo: Photo;
  photos: Photo[];
  position?: number;
  session: RetouchSession;
}) {
  const [candidateId, setCandidateId] = useState<string>();
  const [mode, setMode] = useState<"auto" | "manual">("auto");
  const { key } = photo;
  const result = photo.result?.output;
  const info = photoInfo(photo);
  const size = info ? uprightSize(info) : undefined;
  const ratio = size && size.height > 0 ? size.width / size.height : 1.5;
  const manual = photo.manual;
  const canAdjust = photo.artifactId !== undefined && (result !== undefined || photo.inspect.status === "done");
  const candidate = result && candidateId ? result.candidates.find((c) => c.id === candidateId) : undefined;

  // A finished re-run replaces the manual edit; keep the Manual view usable.
  useEffect(() => {
    if (mode === "manual" && !manual && canAdjust) session.startAdjust(key);
  }, [mode, manual, canAdjust, session, key]);

  // A new result may not have the candidate that was selected.
  useEffect(() => {
    if (candidateId && !result?.candidates.some((c) => c.id === candidateId)) setCandidateId(undefined);
  }, [candidateId, result]);

  const index = photos.findIndex((p) => p.key === key);
  const prev = index > 0 ? photos[index - 1] : undefined;
  const next = index >= 0 ? photos[index + 1] : undefined;
  usePhotoKeys(prev?.key, next?.key);

  const before: CompareSide = { id: neutralId(photo), label: "As shot" };
  let after: CompareSide | undefined;
  let note: string | undefined;
  if (mode === "manual" && manual) {
    after = { id: manual.preview?.id, label: "Manual" };
    const stale = !manual.preview || !sameParams(manual.preview.params, manual.params);
    note = manual.error ? "Preview failed" : stale ? "Rendering…" : undefined;
  } else if (result) {
    after = candidate
      ? { id: candidate.preview.id, label: candidate.label }
      : { id: result.after.id, label: chosenLabel(result) };
  }

  const twoColumns = useTwoColumns();
  const scopes = before.id || after?.id ? <ScopesPanel before={before} after={after} info={info} /> : null;

  const captured = info ? formatCaptured(info.captured_at) : undefined;
  const facts = [
    info ? exposureLine(info) : undefined,
    size ? `${size.width} × ${size.height}` : undefined,
    captured,
    formatBytes(photo.size),
  ].filter(Boolean);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <ButtonLink href="#/retouch" size="sm" variant="ghost" icon={ArrowLeft}>
          All photos
        </ButtonLink>
        <nav aria-label="Photo" className="flex items-center gap-1">
          <StepLink target={prev} direction="previous" />
          <span className={cn(text.meta, "px-1 tabular-nums")}>
            {index + 1} / {photos.length}
          </span>
          <StepLink target={next} direction="next" />
        </nav>
      </div>

      <header className="space-y-1">
        <p className={text.eyebrowSm}>Photo</p>
        <h1 className={cn(text.bandTitle, "break-all font-mono text-[var(--ds-gray-1000)]")}>{photo.filename}</h1>
        {info ? <p className={text.small}>{cameraLine(info)}</p> : null}
        <p className={text.meta}>{facts.join(" · ")}</p>
      </header>

      <Surface tone="flat" className="flex flex-col gap-3 p-3 sm:flex-row sm:items-center">
        {/* A wrapper, so the pill keeps its width when the bar stacks on phones. */}
        <div className="shrink-0">
          <PhaseSignal photo={photo} variant="pill" />
        </div>
        <div className="min-w-0 flex-1">
          <PhotoProgress photo={photo} position={position} detailed />
        </div>
        <ActionBar>
          <PhotoActions
            photo={photo}
            session={session}
            canRetryUpload={session.hasFile(key)}
            primary
            onRemoved={() => {
              window.location.hash = "#/retouch";
            }}
          />
        </ActionBar>
      </Surface>

      {photos.length > 1 ? <Filmstrip photos={photos} current={key} /> : null}

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-5">
          {/* While adjusting, the preview stays in view beside the sliders on
              wide screens (with its scopes: the photo shrinks to leave them
              room), and nothing sits between them on narrow ones. */}
          <div
            className={cn(
              "space-y-5",
              mode === "manual" &&
                "xl:sticky xl:top-20 xl:z-10 xl:[--compare-max-h:max(16rem,min(72vh,100vh_-_440px))]",
            )}
          >
            {before.id || after?.id ? (
              <CompareView key={`${key}-${mode}`} before={before} after={after} ratio={ratio} note={note} />
            ) : (
              <NoImage photo={photo} ratio={ratio} />
            )}
            {scopes && twoColumns ? scopes : null}
          </div>
          {result && mode === "auto" ? (
            <CandidateStrip output={result} selected={candidateId} onSelect={setCandidateId} />
          ) : null}
          {photo.result && mode === "auto" ? <ClaudePanel result={photo.result} /> : null}
        </div>

        <div className="space-y-5">
          <DevelopPanel
            photo={photo}
            mode={mode}
            onMode={setMode}
            canAdjust={canAdjust}
            candidate={candidate}
            session={session}
          />
          {scopes && !twoColumns ? scopes : null}
          <DownloadsPanel photo={photo} />
          {photo.result ? <TimingsPanel result={photo.result} /> : null}
        </div>
      </div>
    </div>
  );
}

function chosenLabel(output: AutoOutput) {
  return output.candidates.find((c) => c.id === output.chosen)?.label ?? "Auto";
}

/** ← and → step through the photos, unless a control that uses the arrows has focus. */
function usePhotoKeys(prev: string | undefined, next: string | undefined) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable], [role='slider'], [role='radio']")) return;
      const to = event.key === "ArrowLeft" ? prev : next;
      if (!to) return;
      event.preventDefault();
      window.location.hash = photoHref(to);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [prev, next]);
}

function StepLink({ direction, target }: { direction: "previous" | "next"; target?: Photo }) {
  const Icon = direction === "previous" ? ChevronLeft : ChevronRight;
  const label = direction === "previous" ? "Previous photo" : "Next photo";
  if (!target) {
    return (
      <Button size="sm" variant="ghost" icon={Icon} disabled className="w-8 px-0">
        <span className="sr-only">{label}</span>
      </Button>
    );
  }
  return (
    <ButtonLink
      href={photoHref(target.key)}
      size="sm"
      variant="ghost"
      icon={Icon}
      className="w-8 px-0"
      title={`${label} (${direction === "previous" ? "←" : "→"})`}
    >
      <span className="sr-only">
        {label}: {target.filename}
      </span>
    </ButtonLink>
  );
}

function NoImage({ photo, ratio }: { photo: Photo; ratio: number }) {
  const phase = photoPhase(photo);
  return (
    <div
      className="mx-auto grid w-full place-items-center overflow-hidden rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)]"
      style={{ aspectRatio: ratio, maxWidth: `calc(72vh * ${ratio})` }}
    >
      {phase === "uploading" || phase === "reading" ? (
        <Skeleton className="h-full w-full rounded-none" />
      ) : (
        <p className={cn(text.caption, "p-4 text-center")}>No preview for this photo.</p>
      )}
    </div>
  );
}

/* Every photo of the session as a strip of small links; the current one is
   scrolled into view (sideways only, the page stays put). */
function Filmstrip({ current, photos }: { current: string; photos: Photo[] }) {
  const list = useRef<HTMLOListElement>(null);

  useEffect(() => {
    const strip = list.current;
    const item = strip?.querySelector<HTMLElement>("[aria-current='page']");
    if (!strip || !item) return;
    strip.scrollTo({
      left: item.offsetLeft - strip.clientWidth / 2 + item.clientWidth / 2,
      behavior: prefersReducedMotion() ? "instant" : "smooth",
    });
  }, [current]);

  return (
    <nav aria-label="Photos in this session">
      <ol ref={list} className="relative flex gap-2 overflow-x-auto pb-2">
        {photos.map((p) => {
          const phase = photoPhase(p);
          const active = p.key === current;
          return (
            <li key={p.key} className="shrink-0 p-0.5">
              <a
                href={photoHref(p.key)}
                aria-current={active ? "page" : undefined}
                title={p.filename}
                className={cn(
                  "relative block h-14 w-20 overflow-hidden rounded-[6px] border bg-[var(--ds-background-200)] outline-none transition-opacity focus-visible:shadow-[var(--ds-focus-ring)] sm:h-16 sm:w-24",
                  active
                    ? "border-[var(--ds-gray-1000)] shadow-[0_0_0_1px_var(--ds-gray-1000)]"
                    : "border-[var(--ds-gray-alpha-400)] opacity-75 hover:opacity-100",
                )}
              >
                <ArtifactImage id={thumbnailId(p)} alt="" fit="cover" />
                <span
                  aria-hidden="true"
                  className="absolute bottom-1 left-1 h-2.5 w-2.5 rounded-full border border-white/90 shadow-[0_0_0_1px_rgb(0_0_0_/_0.35)]"
                  style={{ background: phaseColor[phase] }}
                />
                <span className="sr-only">
                  {p.filename}, {phase}
                </span>
              </a>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

/* The finalists as small renders: Claude's rank, the objective score, and
   which one won. Selecting one puts it on the "after" side. */
function CandidateStrip({
  onSelect,
  output,
  selected,
}: {
  onSelect: (id: string | undefined) => void;
  output: AutoOutput;
  selected?: string;
}) {
  const candidates = rankedCandidates(output);
  if (candidates.length === 0) return null;
  const shown = candidates.find((c) => c.id === selected);

  return (
    <section aria-labelledby="candidates-title" className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h2 id="candidates-title" className="text-[15px] font-semibold text-[var(--ds-gray-1000)]">
          Candidates
        </h2>
        <p className={text.meta}>{candidates.length} rendered · select one to compare</p>
      </div>
      <ul className="flex gap-3 overflow-x-auto p-0.5 pb-2">
        {candidates.map((c) => (
          <CandidateTile
            key={c.id}
            candidate={c}
            chosen={c.id === output.chosen}
            active={selected ? selected === c.id : c.id === output.chosen}
            onSelect={() => onSelect(c.id === output.chosen ? undefined : c.id)}
          />
        ))}
      </ul>
      {shown?.note ? (
        <p className={text.caption}>
          <span className="font-medium text-[var(--ds-gray-1000)]">{shown.label}:</span> {shown.note}
        </p>
      ) : null}
    </section>
  );
}

function CandidateTile({
  active,
  candidate,
  chosen,
  onSelect,
}: {
  active: boolean;
  candidate: CandidateSummary;
  chosen: boolean;
  onSelect: () => void;
}) {
  return (
    <li className="w-36 shrink-0 sm:w-40">
      <button
        type="button"
        aria-pressed={active}
        onClick={onSelect}
        title={candidate.note ?? undefined}
        className={cn(
          "block w-full overflow-hidden rounded-[7px] border bg-[var(--ds-background-100)] text-left outline-none transition-colors focus-visible:shadow-[var(--ds-focus-ring)]",
          active
            ? "border-[var(--ds-gray-1000)] shadow-[0_0_0_1px_var(--ds-gray-1000)]"
            : "border-[var(--ds-gray-alpha-400)] hover:border-[var(--ds-gray-alpha-500)]",
        )}
      >
        <span className="relative block aspect-[3/2] overflow-hidden bg-[var(--ds-background-200)]">
          <ArtifactImage id={candidate.preview.id} alt="" />
        </span>
        <span className="block space-y-1 p-2">
          <span className="flex min-h-6 flex-wrap items-center gap-1">
            {candidate.rank != null ? <Badge className="font-mono">#{candidate.rank}</Badge> : null}
            {chosen ? <Badge tone="green">Chosen</Badge> : null}
          </span>
          <span className="block truncate text-[13px] font-medium text-[var(--ds-gray-1000)]">{candidate.label}</span>
          <span className={cn(text.meta, "block truncate")}>
            {candidate.source} · {formatScore(candidate.score)}
          </span>
        </span>
      </button>
    </li>
  );
}

function PanelHeader({ action, title }: { action?: ReactNode; title: string }) {
  return (
    <div className="flex min-h-12 flex-wrap items-center justify-between gap-x-3 gap-y-2 border-b border-[var(--ds-gray-alpha-400)] px-4 py-2">
      <h2 className="text-[15px] font-semibold text-[var(--ds-gray-1000)]">{title}</h2>
      {action}
    </div>
  );
}

/* Claude's part: its read of the scene, why the winner won, how stable that
   ranking was when the order was swapped, and the refinement's A/B. */
function ClaudePanel({ result }: { result: AutoResult }) {
  const judge = result.output.judge;
  if (!judge) {
    return (
      <Card tone="muted" depth="flat">
        <p className={text.small}>Claude was off for this run, so the objective score picked the edit.</p>
      </Card>
    );
  }
  const refineAsked = result.input?.refine !== false;
  const refinement = judge.refined ? "Won the blind A/B" : refineAsked ? "Not kept" : "Off";

  return (
    <Surface tone="flat" aria-labelledby="claude-title" className="overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-x-3 border-b border-[var(--ds-gray-alpha-400)] px-4 py-3">
        <h2 id="claude-title" className="text-[15px] font-semibold text-[var(--ds-gray-1000)]">
          Claude's judgement
        </h2>
        <span className={text.meta}>{judge.model}</span>
      </div>
      <div className="space-y-4 p-4">
        {judge.error ? (
          <Card tone="warning" depth="flat" density="compact">
            <p className={cn(text.caption, "text-inherit")}>
              Claude did not finish, so the objective score chose: {judge.error}
            </p>
          </Card>
        ) : null}
        {judge.scene ? <Note label="Scene">{judge.scene}</Note> : null}
        {judge.reasons ? <Note label="Why this one">{judge.reasons}</Note> : null}
        {judge.refine_note ? <Note label="Refinement">{judge.refine_note}</Note> : null}
        <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
          <Fact
            label="Agreement"
            value={judge.agreement != null ? formatPercent(judge.agreement) : "—"}
            title="Share of the order-swapped rankings that agreed on the winner"
          />
          <Fact label="Refinement" value={refinement} />
          <Fact
            label="Calls"
            value={
              !Number.isFinite(judge.calls)
                ? "—"
                : judge.cache_hits
                  ? `${judge.calls} (${judge.cache_hits} cached)`
                  : String(judge.calls)
            }
          />
          <Fact
            label="Cost"
            value={Number.isFinite(judge.cost_usd) ? `$${judge.cost_usd.toFixed(2)}` : "—"}
            title="As claude -p reports it; on a subscription it counts against quota"
          />
        </dl>
      </div>
    </Surface>
  );
}

function Note({ children, label }: { children: ReactNode; label: string }) {
  return (
    <div>
      <p className={text.eyebrowSm}>{label}</p>
      <p className={cn(text.small, "mt-1 whitespace-pre-line")}>{children}</p>
    </div>
  );
}

function Fact({ label, title, value }: { label: string; title?: string; value: string }) {
  return (
    <div className="min-w-0" title={title}>
      <dt className={text.eyebrowSm}>{label}</dt>
      <dd className={cn(text.mono, "mt-0.5 truncate")}>{value}</dd>
    </div>
  );
}

/* Lightroom's Develop panels (Basic, Tone Curve, Color Mixer, Detail, Lens
   Corrections, Calibration). Auto: the values shown in the comparison (the
   chosen edit, a selected candidate, or as shot before a retouch). Manual:
   sliders with a live preview, Reset, and a full-size export. */
function DevelopPanel({
  canAdjust,
  candidate,
  mode,
  onMode,
  photo,
  session,
}: {
  canAdjust: boolean;
  candidate?: CandidateSummary;
  mode: "auto" | "manual";
  onMode: (mode: "auto" | "manual") => void;
  photo: Photo;
  session: RetouchSession;
}) {
  const result = photo.result?.output;
  const info = photoInfo(photo);
  const neutral = photo.inspect.status === "done" ? photo.inspect.output.neutral : undefined;
  const asShot = info ? { temperature: info.as_shot_temperature, tint: info.as_shot_tint } : undefined;

  let body: ReactNode;
  if (mode === "manual") {
    body = photo.manual ? (
      <ManualEditor photo={photo} session={session} asShot={asShot} />
    ) : (
      <Skeleton className="h-64" />
    );
  } else {
    const shown = candidate?.params ?? result?.params ?? neutral;
    const caption = candidate
      ? `Candidate: ${candidate.label}`
      : result
        ? `Chosen: ${chosenLabel(result)}`
        : isActive(photo)
          ? "As shot, while the retouch looks for an edit."
          : "As shot. Retouch to find an edit.";
    body = shown ? (
      <div className="space-y-3">
        <p className={text.caption}>{caption}</p>
        <ParamsTable params={shown} caption={caption} asShot={asShot} />
      </div>
    ) : (
      <p className={text.caption}>The values appear once the RAW file has been read.</p>
    );
  }

  return (
    <Surface tone="flat" className="overflow-hidden">
      <PanelHeader
        title="Develop"
        action={
          <SegmentedControl
            label="Edit mode"
            value={mode}
            disabled={!canAdjust}
            options={[
              { value: "auto", label: "Auto" },
              { value: "manual", label: "Adjust" },
            ]}
            onChange={(next) => {
              if (next === "manual") session.startAdjust(photo.key);
              onMode(next);
            }}
          />
        }
      />
      <div className="p-4">{body}</div>
    </Surface>
  );
}

function ManualEditor({
  asShot,
  photo,
  session,
}: {
  asShot?: { temperature: number; tint: number };
  photo: Photo;
  session: RetouchSession;
}) {
  const manual = photo.manual!;
  const exporting = manual.export?.status === "running";
  const stale = !manual.preview || !sameParams(manual.preview.params, manual.params);
  const atBase = sameParams(manual.params, manual.base);
  const status = manual.error
    ? manual.error
    : stale
      ? "Rendering the preview…"
      : "Preview is up to date.";

  return (
    <div className="space-y-4">
      <AdjustSliders
        asShot={asShot}
        base={manual.base}
        params={manual.params}
        onChange={(params) => session.setManual(photo.key, params)}
      />
      <p aria-live="polite" className={text.caption}>
        {status}
      </p>
      <ActionBar>
        <Button size="sm" icon={Undo2} disabled={atBase} onClick={() => session.resetManual(photo.key)}>
          {photo.result ? "Reset to auto" : "Reset to as shot"}
        </Button>
        <Button
          size="sm"
          variant="primary"
          icon={FileDown}
          disabled={exporting}
          onClick={() => void session.exportFull(photo.key)}
        >
          Export full size
        </Button>
      </ActionBar>
      {manual.export ? <ExportStatus photo={photo} exported={manual.export} session={session} /> : null}
    </div>
  );
}

function ExportStatus({ exported, photo, session }: { exported: ExportState; photo: Photo; session: RetouchSession }) {
  const changed = photo.manual ? !sameParams(exported.params, photo.manual.params) : false;

  if (exported.status === "running") {
    return (
      <div className="space-y-2 rounded-[7px] border border-[var(--ds-gray-alpha-400)] p-3">
        <div className="flex items-center justify-between gap-3">
          <p className={cn(text.caption, "min-w-0 truncate")}>
            {exported.cancelling ? "Cancelling…" : exported.progress?.message || "Rendering at full size"}
          </p>
          <Button
            size="sm"
            variant="ghost"
            icon={CircleStop}
            aria-disabled={!exported.runId || exported.cancelling || undefined}
            onClick={() => session.cancelExport(photo.key)}
          >
            Cancel<span className="sr-only"> export</span>
          </Button>
        </div>
        <ProgressBar label={`Full-size export of ${photo.filename}`} value={exported.progress?.fraction} />
      </div>
    );
  }
  if (exported.status === "failed") return <p className={text.caption}>Export failed: {exported.error}</p>;
  if (exported.status === "cancelled") return <p className={text.caption}>Export cancelled.</p>;

  const files = filesOf(exported.output, photo, "manual");
  return (
    <div className="space-y-2 rounded-[7px] border border-[var(--ds-gray-alpha-400)] p-3">
      <p className={text.eyebrowSm}>Your export</p>
      {changed ? <p className={text.caption}>The sliders changed after this export. Export again to include the changes.</p> : null}
      <FileList files={files} />
      <p className={text.meta}>"Download all" uses this export for this photo.</p>
    </div>
  );
}

function DownloadsPanel({ photo }: { photo: Photo }) {
  const result = photo.result;
  const files = result ? filesOf(result.output, photo, "auto") : undefined;
  const any = files && (files.jpeg || files.xmp || files.tiff);
  const manualExport = photo.manual?.export?.status === "succeeded";

  return (
    <Surface tone="flat" className="overflow-hidden">
      <PanelHeader title="Downloads" />
      <div className="space-y-3 px-4 py-2">
        {!result ? (
          <p className={cn(text.caption, "py-2")}>The JPEG and XMP appear here when the retouch is done.</p>
        ) : any ? (
          <>
            <FileList files={files} />
            {files.xmp ? <XmpNote className="pb-2" /> : null}
          </>
        ) : (
          <p className={cn(text.caption, "py-2")}>
            This run exported no files: JPEG and XMP were off. Use Adjust, then Export full size.
          </p>
        )}
        {manualExport ? (
          <p className={cn(text.meta, "pb-2")}>"Download all" takes your manual export for this photo.</p>
        ) : null}
      </div>
    </Surface>
  );
}

function TimingsPanel({ result }: { result: AutoResult }) {
  const t = result.output.timings;
  const input = result.input;
  const rows: [string, number | undefined][] = [
    ["Decode", t?.decode_ms],
    ["Develop", t?.develop_ms],
    ["Search", t?.search_ms],
    ["Claude", t?.judge_ms],
    ["Export", t?.export_ms],
  ];
  const settings = input
    ? [
        input.use_claude === false ? "Claude off" : input.model ? `Claude ${input.model}` : undefined,
        input.use_claude !== false && input.plan ? "plan" : undefined,
        input.use_claude !== false && input.refine ? "refine" : undefined,
        input.evals_per_look ? `${input.evals_per_look} evals per look` : undefined,
      ].filter(Boolean)
    : [];

  return (
    <Surface tone="flat" className="overflow-hidden">
      <PanelHeader title="Timings" action={<span className={text.mono}>{formatMs(t?.total_ms)}</span>} />
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 p-4">
        {rows.map(([label, ms]) => (
          <div key={label} className="flex items-baseline justify-between gap-2">
            <dt className={text.eyebrowSm}>{label}</dt>
            <dd className={cn(text.mono, "tabular-nums")}>{formatMs(ms)}</dd>
          </div>
        ))}
      </dl>
      {settings.length ? <p className={cn(text.meta, "border-t border-[var(--ds-gray-alpha-300)] px-4 py-2")}>{settings.join(" · ")}</p> : null}
    </Surface>
  );
}
