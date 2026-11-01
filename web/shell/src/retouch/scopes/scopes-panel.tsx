import { useId, useState, useSyncExternalStore, type ReactNode } from "react";
import type { RawInfo } from "@spl/bindings/raw-retouch/RawInfo";
import { SegmentedControl, Skeleton, StatusSignal, Surface, cn, text } from "@spl/ui";
import type { CompareSide } from "../compare-view";
import { cameraLine, exposureLine } from "../format";
import {
  HIGHLIGHT_LIMIT,
  SHADOW_LIMIT,
  clipLevel,
  dominantBand,
  formatLevel,
  formatShare,
  type ClipLevel,
  type ClipStats,
  type HistogramScale,
  type Scopes,
} from "./compute";
import { Histogram, LAYER_FILL, OVERLAY_DASH, type Channels } from "./histogram";
import { SAMPLE_EDGE, useScopes, type ScopesState } from "./use-scopes";
import { Vectorscope } from "./vectorscope";

/** One image's scopes, or before and after side by side, or the before's
    luma outlined over the after. */
type Source = "before" | "after" | "side" | "overlay";

/* Status lives in dots: grey for no clipping to speak of, amber for some,
   red for a lot (see clipLevel). */
const CLIP_COLOR: Record<ClipLevel, string> = {
  none: "var(--ds-gray-700)",
  some: "var(--ds-amber-700)",
  heavy: "var(--ds-red-700)",
};

/* Channels and scale are a viewer's habit: kept per browser. */
const PREFS_KEY = "spl-retouch-scopes";
type Prefs = { channels: Channels; scale: HistogramScale };

function loadPrefs(): Prefs {
  try {
    const saved = JSON.parse(window.localStorage.getItem(PREFS_KEY) ?? "null") as Partial<Prefs> | null;
    return { channels: saved?.channels === "luma" ? "luma" : "rgb", scale: saved?.scale === "log" ? "log" : "linear" };
  } catch {
    return { channels: "rgb", scale: "linear" };
  }
}

function savePrefs(prefs: Prefs) {
  try {
    window.localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch {
    /* private mode: not remembered */
  }
}

/** Tailwind's xl breakpoint, where the photo page has two columns. */
const TWO_COLUMNS = "(min-width: 80rem)";

function subscribeColumns(onChange: () => void): () => void {
  const media = window.matchMedia?.(TWO_COLUMNS);
  media?.addEventListener("change", onChange);
  return () => media?.removeEventListener("change", onChange);
}

/** Whether the photo page shows its two columns. There the scopes sit under
    the photo (and stay in view with it while adjusting); stacked, they follow
    the Basic panel, so nothing comes between the photo and the sliders. */
export function useTwoColumns(): boolean {
  return useSyncExternalStore(subscribeColumns, () => window.matchMedia?.(TWO_COLUMNS).matches ?? true, () => true);
}

/* View-only scopes of what the comparison shows, measured in the browser
   from the preview JPEGs: Lightroom's RGB histogram (or luma) with clipping,
   luma statistics, and a hue and saturation vectorscope. Before is the
   as-shot render, after the chosen edit, a selected candidate, or the live
   manual preview. Before sits on the left and after on the right, as in the
   comparison: side by side (the default), or "Overlay" outlines the before's
   luma over the after. */
export function ScopesPanel({
  after,
  before,
  className,
  info,
  initial = "side",
}: {
  after?: CompareSide;
  before: CompareSide;
  className?: string;
  info?: RawInfo;
  /** The first view: side by side, or the shorter overlay where height is tight. */
  initial?: Source;
}) {
  const titleId = useId();
  const [source, setSource] = useState<Source>(initial);
  const [prefs, setPrefs] = useState(loadPrefs);
  const beforeScopes = useScopes(before.id);
  const afterScopes = useScopes(after?.id);

  const view: Source = after ? source : "before";
  const updating =
    (view !== "after" && Boolean(beforeScopes.scopes) && beforeScopes.pending) ||
    (view !== "before" && Boolean(afterScopes.scopes) && afterScopes.pending);

  const updatePrefs = (next: Partial<Prefs>) => {
    const merged = { ...prefs, ...next };
    setPrefs(merged);
    savePrefs(merged);
  };

  const shown = !after
    ? before.label
    : view === "before"
      ? `Before · ${before.label}`
      : view === "after"
        ? `After · ${after.label}`
        : view === "side"
          ? `Before · ${before.label} | After · ${after.label}`
          : `After · ${after.label}, before dashed`;

  return (
    <Surface tone="flat" aria-labelledby={titleId} className={cn("overflow-hidden", className)}>
      <div className="flex min-h-12 flex-wrap items-center justify-between gap-x-3 gap-y-2 border-b border-[var(--ds-gray-alpha-400)] px-4 py-2">
        <div className="flex min-w-0 items-baseline gap-2">
          <h2 id={titleId} className="text-[15px] font-semibold text-[var(--ds-gray-1000)]">
            Scopes
          </h2>
          <span
            className={cn(text.meta, "min-w-0 truncate")}
            title={`Measured in your browser from the 8-bit preview JPEG, sampled at ${SAMPLE_EDGE} px on the long side.`}
          >
            {shown}
            {updating ? " · updating…" : ""}
          </span>
        </div>
        {after ? (
          <SegmentedControl
            label="Scopes show"
            value={source}
            onChange={setSource}
            options={[
              { value: "before", label: "Before" },
              { value: "side", label: "Side by side" },
              { value: "overlay", label: "Overlay" },
              { value: "after", label: "After" },
            ]}
          />
        ) : null}
      </div>

      {view === "side" && after ? (
        <div className="grid gap-x-6 gap-y-5 p-4 md:grid-cols-2">
          <ScopeView
            heading={`Before · ${before.label}`}
            state={beforeScopes}
            prefs={prefs}
            info={info}
            stacked
          />
          <ScopeView heading={`After · ${after.label}`} state={afterScopes} prefs={prefs} stacked />
        </div>
      ) : (
        <ScopeView
          state={view === "before" ? beforeScopes : afterScopes}
          overlay={view === "overlay" ? beforeScopes.scopes : undefined}
          shown={shown}
          prefs={prefs}
          info={info}
        />
      )}

      <div className="flex flex-wrap items-center gap-2 px-4 pb-4">
        <SegmentedControl
          label="Histogram channels"
          value={prefs.channels}
          onChange={(channels) => updatePrefs({ channels })}
          options={[
            { value: "rgb", label: "RGB" },
            { value: "luma", label: "Luma" },
          ]}
        />
        <SegmentedControl
          label="Histogram scale"
          value={prefs.scale}
          onChange={(scale) => updatePrefs({ scale })}
          options={[
            { value: "linear", label: "Linear" },
            { value: "log", label: "Log" },
          ]}
        />
        {view === "overlay" ? <Legend channels={prefs.channels} /> : null}
      </div>

      {info ? (
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-t border-[var(--ds-gray-alpha-300)] px-4 py-2">
          <p className={cn(text.meta, "min-w-0 truncate")} title={cameraLine(info)}>
            {cameraLine(info)}
          </p>
          {exposureLine(info) ? <p className={cn(text.mono, "tabular-nums")}>{exposureLine(info)}</p> : null}
        </div>
      ) : null}
    </Surface>
  );
}

/* One image's histogram, clipping, vectorscope and statistics; with
   `overlay`, the before's values beside the after's ("31% → 38%").
   `stacked` puts the vectorscope under the histogram (a half-width column). */
function ScopeView({
  heading,
  info,
  overlay,
  prefs,
  shown,
  stacked = false,
  state,
}: {
  heading?: string;
  info?: RawInfo;
  overlay?: Scopes;
  prefs: Prefs;
  shown?: string;
  stacked?: boolean;
  state: ScopesState;
}) {
  const main = state.scopes;
  const name = heading ?? shown ?? "";
  const band = main ? dominantBand(main.bands) : undefined;
  const hue = !main ? "—" : band ? `Mostly ${band.name} (${formatShare(band.share)})` : "Nearly neutral";
  const plotLabel = main
    ? `${prefs.channels === "rgb" ? "RGB" : "Luma"} histogram, ${prefs.scale} scale: ${name}. ` +
      `Mean luma ${formatLevel(main.meanLuma)}, shadows clipped ${formatShare(main.shadows.any)}, ` +
      `highlights clipped ${formatShare(main.highlights.any)}.`
    : "Histogram";

  return (
    <div className={cn("grid min-w-0 gap-4", stacked ? "" : "p-4 pb-3 lg:grid-cols-[minmax(0,1fr)_auto]")}>
      <div className="min-w-0 space-y-2">
        {heading ? <p className={cn(text.eyebrowSm, "truncate normal-case")}>{heading}</p> : null}
        <div className="relative h-28 overflow-hidden rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] sm:h-32">
          {main ? (
            <Histogram
              main={main}
              overlay={overlay}
              channels={prefs.channels}
              scale={prefs.scale}
              label={plotLabel}
              className="absolute inset-0 h-full w-full"
            />
          ) : state.failed ? (
            <Unavailable />
          ) : (
            <Skeleton className="absolute inset-0 rounded-none" />
          )}
        </div>
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
          <ClipReadout kind="shadows" main={main?.shadows} before={overlay?.shadows} />
          <ClipReadout kind="highlights" main={main?.highlights} before={overlay?.highlights} />
        </div>
      </div>

      <div className="flex flex-wrap items-start gap-4 sm:flex-nowrap">
        <figure className="w-36 shrink-0">
          <Vectorscope
            scopes={main}
            label={
              main
                ? `Hue and saturation scope: ${name}. ${hue}, mean saturation ${formatLevel(main.meanChroma)}.`
                : "Hue and saturation scope"
            }
          />
          <figcaption className={cn(text.meta, "mt-1 text-center")}>{hue}</figcaption>
        </figure>
        <dl
          className={cn(
            "grid min-w-36 flex-1 grid-cols-2 gap-x-4 gap-y-3",
            !stacked && "lg:w-40 lg:flex-none lg:grid-cols-1",
          )}
        >
          <Stat
            label="Mean luma"
            title="Average Rec. 709 luma of the preview, 0 (black) to 100% (white)"
            main={main?.meanLuma}
            before={overlay?.meanLuma}
          />
          <Stat label="Median luma" title="Half the pixels are darker than this" main={main?.medianLuma} before={overlay?.medianLuma} />
          <Stat
            label="Saturation"
            title="Mean chroma: how far the pixels are from grey, 0 to 100%"
            main={main?.meanChroma}
            before={overlay?.meanChroma}
          />
          {info ? (
            <div
              className="min-w-0"
              title="Pixels clipped in at least one channel in the RAW file itself; no edit brings these back"
            >
              <dt className={text.eyebrowSm}>RAW clipped</dt>
              <dd className={cn(text.mono, "mt-0.5 tabular-nums")}>{formatShare(info.raw_clipped_fraction)}</dd>
            </div>
          ) : null}
        </dl>
      </div>
    </div>
  );
}

function Unavailable() {
  return (
    <p className={cn(text.caption, "absolute inset-0 grid place-items-center p-2 text-center")}>
      No scopes for this image.
    </p>
  );
}

/** "38%" or, against a before value, "31% → 38%". */
function Change({ before, format, main }: { before?: number; format: (value: number) => string; main: number }) {
  if (before === undefined) return <>{format(main)}</>;
  return (
    <>
      <span className="text-[var(--ds-gray-900)]">{format(before)}</span>
      <span aria-hidden="true"> → </span>
      <span className="sr-only"> before, </span>
      {format(main)}
      <span className="sr-only"> after</span>
    </>
  );
}

function Stat({ before, label, main, title }: { before?: number; label: string; main?: number; title: string }) {
  return (
    <div className="min-w-0" title={title}>
      <dt className={text.eyebrowSm}>{label}</dt>
      <dd className={cn(text.mono, "mt-0.5 tabular-nums")}>
        {main === undefined ? "—" : <Change before={before} main={main} format={formatLevel} />}
      </dd>
    </div>
  );
}

/* Clipping at either end, as Lightroom's corner triangles: the shadows' dot
   on the left, the highlights' on the right. */
function ClipReadout({ before, kind, main }: { before?: ClipStats; kind: "shadows" | "highlights"; main?: ClipStats }) {
  const shadows = kind === "shadows";
  const level = main ? clipLevel(main.any) : "none";
  const title = main
    ? `${formatShare(main.any)} of pixels have a channel at ${shadows ? `${SHADOW_LIMIT} or below` : `${HIGHLIGHT_LIMIT} or above`} ` +
      `(R ${formatShare(main.channels[0])}, G ${formatShare(main.channels[1])}, B ${formatShare(main.channels[2])}); ` +
      `${formatShare(main.all)} are pure ${shadows ? "black" : "white"}.`
    : undefined;

  return (
    <div className={cn("flex items-center gap-1.5", !shadows && "flex-row-reverse")} title={title}>
      <StatusSignal variant="dot" color={CLIP_COLOR[level]} />
      <span className={text.eyebrowSm}>
        {shadows ? "Shadows" : "Highlights"}
        <span className="sr-only"> clipped</span>
      </span>
      <span className={cn(text.mono, "tabular-nums")}>
        {main ? <Change before={before?.any} main={main.any} format={formatShare} /> : "—"}
      </span>
    </div>
  );
}

function Legend({ channels }: { channels: Channels }) {
  return (
    <span className={cn(text.meta, "ml-auto flex flex-wrap items-center gap-x-3 gap-y-1")}>
      <LegendItem
        swatch={
          <svg width="16" height="4" className="overflow-visible">
            <line x1="0" y1="2" x2="16" y2="2" stroke="var(--ds-gray-1000)" strokeWidth="1.5" strokeDasharray={OVERLAY_DASH} />
          </svg>
        }
      >
        Before, luma
      </LegendItem>
      <LegendItem
        swatch={
          <span
            className="h-2.5 w-3 rounded-[2px]"
            style={{ background: channels === "rgb" ? LAYER_FILL.neutral : "var(--ds-gray-alpha-500)" }}
          />
        }
      >
        After
      </LegendItem>
    </span>
  );
}

function LegendItem({ children, swatch }: { children: ReactNode; swatch: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span aria-hidden="true" className="inline-flex">
        {swatch}
      </span>
      {children}
    </span>
  );
}

