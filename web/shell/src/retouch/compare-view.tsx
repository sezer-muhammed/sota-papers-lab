import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent,
  type ReactNode,
} from "react";
import { ChevronsLeftRight } from "lucide-react";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import type { Region } from "@spl/bindings/raw-retouch/Region";
import { Button, SegmentedControl, Skeleton, cn, text } from "@spl/ui";
import { loadArtifactImage } from "./artifact";
import { PARAM_KEYS, paramValue } from "./format";
import { ImageUnavailable, useDecodedImage } from "./images";
import type { RegionRender } from "./session";
import * as Z from "./zoom";

/** One side of the comparison: its display-size render, a label, and the
    settings it was rendered with (for 1:1 crops when zoomed in). */
export type CompareSide = { id?: string; label: string; params?: Params };

/** Renders a 1:1 crop of the photo (session.renderRegion for this photo). */
export type RenderTile = (params: Params, region: Region, signal: AbortSignal) => Promise<RegionRender>;

type Mode = "slider" | "side";

const MODE_KEY = "spl-retouch-compare";
/** A pan or zoom settles this long before crops are requested. */
const TILE_DEBOUNCE_MS = 220;
/** Crops rendering at once (the host keeps one slot for interactive work). */
const TILE_INFLIGHT = 2;
/** Decoded crops kept across photos; each is up to 2048 × 2048. */
const TILE_LIMIT = 16;
/** More crops than this per side (a wide view at 50%) and the preview stays. */
const TILES_PER_VIEW = 4;

function loadMode(): Mode {
  try {
    return window.localStorage.getItem(MODE_KEY) === "side" ? "side" : "slider";
  } catch {
    return "slider";
  }
}

function saveMode(mode: Mode) {
  try {
    window.localStorage.setItem(MODE_KEY, mode);
  } catch {
    /* private mode: not remembered */
  }
}

/* ---- 1:1 crops, shared by every viewer: content is fixed by the RAW and
   the settings, so a crop stays valid as long as both match. ---- */

type Tile = Z.Rect & { image: HTMLImageElement };
const tileCache: { key: string; tile: Tile }[] = [];

const tileKey = (source: string, params: Params) =>
  `${source}|${PARAM_KEYS.map((key) => paramValue(params, key)).join(",")}`;

function tilesOf(key: string): Tile[] {
  return tileCache.filter((entry) => entry.key === key).map((entry) => entry.tile);
}

function rememberTile(key: string, tile: Tile) {
  tileCache.push({ key, tile });
  while (tileCache.length > TILE_LIMIT) tileCache.shift();
}

function intersection(a: Z.Rect, b: Z.Rect): Z.Rect {
  const x = Math.max(a.x, b.x);
  const y = Math.max(a.y, b.y);
  return { x, y, w: Math.max(0, Math.min(a.x + a.w, b.x + b.w) - x), h: Math.max(0, Math.min(a.y + a.h, b.y + b.h) - y) };
}

/** Draws the part of `image` (which covers `covers` of the photo) that the
    frame shows. */
function blit(
  context: CanvasRenderingContext2D,
  image: HTMLImageElement,
  covers: Z.Rect,
  visible: Z.Rect,
  p: Z.Placement,
  frame: Z.Size,
) {
  const part = intersection(covers, visible);
  if (part.w <= 0 || part.h <= 0 || covers.w <= 0 || covers.h <= 0) return;
  const kx = image.naturalWidth / covers.w;
  const ky = image.naturalHeight / covers.h;
  const at = Z.framePoint(p, frame, part);
  context.drawImage(
    image,
    (part.x - covers.x) * kx,
    (part.y - covers.y) * ky,
    part.w * kx,
    part.h * ky,
    at.x,
    at.y,
    part.w * p.scale,
    part.h * p.scale,
  );
}

type Layer = { preview?: HTMLImageElement; tiles: Tile[] };

/** The preview (smooth: it is a stand-in, magnified), then the 1:1 crops on
    top, with visible pixels above 100%. */
function drawLayer(context: CanvasRenderingContext2D, layer: Layer, p: Z.Placement, frame: Z.Size, full: Z.Size) {
  const visible = Z.visibleRegion(p, frame, full);
  if (layer.preview) {
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    blit(context, layer.preview, { x: 0, y: 0, w: full.width, h: full.height }, visible, p, frame);
  }
  context.imageSmoothingEnabled = p.zoom <= 1;
  for (const tile of layer.tiles) if (Z.intersects(tile, visible)) blit(context, tile.image, tile, visible, p, frame);
}

function prepare(canvas: HTMLCanvasElement | null, frame: Z.Size, dpr: number): CanvasRenderingContext2D | undefined {
  if (!canvas || frame.width <= 0 || frame.height <= 0) return undefined;
  const width = Math.max(1, Math.round(frame.width * dpr));
  const height = Math.max(1, Math.round(frame.height * dpr));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return undefined;
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, width, height);
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return context;
}

type Gesture =
  | { kind: "divider" }
  | { kind: "pan"; x: number; y: number; start: Z.Placement }
  | { kind: "pinch"; distance: number; mid: Z.Point; start: Z.Placement };

/*
 * Before and after, before on the left: a slider (the before image left of
 * a draggable divider) or side by side. Both zoom and pan together, from Fit
 * to 800%: the buttons, pinch or Ctrl/⌘ + scroll, double-click (Fit ⇄ 100%),
 * or the keyboard (+, −, 0, 1, arrows) on the focused photo. Up to the
 * preview's own resolution the preview is drawn; beyond it the visible part
 * is rendered at full resolution on the host (`renderTile`) for both sides,
 * with the magnified preview standing in until each crop arrives. Above
 * 100% the crops are drawn without smoothing, so single pixels show.
 */
export function CompareView({
  after,
  before,
  className,
  full: fullSize,
  note,
  ratio,
  renderTile,
  tileSource,
}: {
  after?: CompareSide;
  before: CompareSide;
  className?: string;
  /** Upright full-resolution size of the photo. */
  full?: Z.Size;
  /** A short state shown over the photo, e.g. "Rendering…". */
  note?: string;
  /** Width / height of the upright photo, so the frame is sized before it loads. */
  ratio: number;
  renderTile?: RenderTile;
  /** Identifies the RAW for the crop cache (its artifact id). */
  tileSource?: string;
}) {
  const split = after !== undefined;
  const [mode, setMode] = useState<Mode>(loadMode);
  const side = split && mode === "side";
  const [view, setView] = useState<Z.View>(Z.FIT);
  const [position, setPosition] = useState(50);
  const [frame, setFrame] = useState<Z.Size>({ width: 0, height: 0 });
  const [tileVersion, setTileVersion] = useState(0);
  const [tileBusy, setTileBusy] = useState(false);
  const [tileFailed, setTileFailed] = useState(false);
  const beforeImage = useDecodedImage(before.id);
  const afterImage = useDecodedImage(after?.id);
  const firstPane = useRef<HTMLDivElement>(null);
  const panes = useRef<(HTMLDivElement | null)[]>([]);
  const canvases = useRef<(HTMLCanvasElement | null)[]>([]);
  const handle = useRef<HTMLDivElement>(null);
  const gesture = useRef<Gesture | null>(null);
  const pointers = useRef(new Map<number, Z.Point>());
  const inflight = useRef(new Map<string, { controller: AbortController; key: string; rect: Z.Rect }>());
  /** Crops not to ask for again: they failed, or came back smaller. */
  const skipTiles = useRef(new Set<string>());
  const safeRatio = Number.isFinite(ratio) && ratio > 0 ? ratio : 1.5;
  const dpr = typeof window !== "undefined" && window.devicePixelRatio > 0 ? window.devicePixelRatio : 1;

  const anyImage = afterImage.image ?? beforeImage.image;
  const full: Z.Size =
    fullSize && fullSize.width > 0 && fullSize.height > 0
      ? fullSize
      : anyImage
        ? { width: anyImage.naturalWidth, height: anyImage.naturalHeight }
        : { width: Math.round(safeRatio * 1000), height: 1000 };
  const p = Z.place(view, frame, full, dpr);
  const pannable = Z.canPan(p, frame, full);
  const current = useRef({ p, frame, full, dpr, pannable });
  current.current = { p, frame, full, dpr, pannable };

  const beforeKey = tileSource && before.params ? tileKey(tileSource, before.params) : undefined;
  const afterKey = tileSource && after?.params ? tileKey(tileSource, after.params) : undefined;

  // Every pane has the same size; measure the first.
  useLayoutEffect(() => {
    const element = firstPane.current;
    if (!element) return;
    const measure = () => {
      const rect = element.getBoundingClientRect();
      setFrame((old) => (old.width === rect.width && old.height === rect.height ? old : { width: rect.width, height: rect.height }));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [side]);

  // Draw.
  useLayoutEffect(() => {
    const beforeLayer: Layer = { preview: beforeImage.image, tiles: beforeKey ? tilesOf(beforeKey) : [] };
    const afterLayer: Layer = { preview: afterImage.image, tiles: afterKey ? tilesOf(afterKey) : [] };
    if (side) {
      const left = prepare(canvases.current[0] ?? null, frame, dpr);
      if (left) drawLayer(left, beforeLayer, p, frame, full);
      const right = prepare(canvases.current[1] ?? null, frame, dpr);
      if (right) drawLayer(right, afterLayer, p, frame, full);
      return;
    }
    const context = prepare(canvases.current[0] ?? null, frame, dpr);
    if (!context) return;
    if (!split) {
      drawLayer(context, beforeLayer, p, frame, full);
      return;
    }
    drawLayer(context, afterLayer, p, frame, full);
    context.save();
    context.beginPath();
    context.rect(0, 0, (frame.width * position) / 100, frame.height);
    context.clip();
    drawLayer(context, beforeLayer, p, frame, full);
    context.restore();
  });

  // Ask the host for 1:1 crops of what is in view, once the view settles.
  const wantTiles =
    renderTile !== undefined &&
    anyImage !== undefined &&
    !p.fit &&
    Z.needsTiles(p, { width: anyImage.naturalWidth, height: anyImage.naturalHeight }, full);
  const sides = [
    before.params && beforeKey ? { key: beforeKey, params: before.params } : undefined,
    after?.params && afterKey ? { key: afterKey, params: after.params } : undefined,
  ].filter((s): s is { key: string; params: Params } => s !== undefined);
  const sidesKey = sides.map((s) => s.key).join("\n");

  useEffect(() => {
    const visible = Z.visibleRegion(p, frame, full);
    const keys = new Set(sides.map((s) => s.key));
    // Stale requests: settings that changed, or a part no longer in view.
    for (const [id, job] of inflight.current) {
      if (!wantTiles || !keys.has(job.key) || !Z.intersects(job.rect, visible)) {
        job.controller.abort();
        inflight.current.delete(id);
      }
    }
    setTileBusy(inflight.current.size > 0);
    if (!wantTiles || !renderTile || visible.w <= 0) return;
    const timer = window.setTimeout(() => {
      const centre = { x: visible.x + visible.w / 2, y: visible.y + visible.h / 2 };
      const queue: { id: string; key: string; params: Params; rect: Z.Rect; distance: number }[] = [];
      for (const s of sides) {
        const have: Z.Rect[] = [
          ...tilesOf(s.key),
          ...[...inflight.current.values()].filter((job) => job.key === s.key).map((job) => job.rect),
        ];
        if (Z.covered(visible, have)) continue;
        const rects = Z.tilesFor(visible, full);
        if (rects.length > TILES_PER_VIEW) continue;
        for (const rect of rects) {
          const id = `${s.key}|${rect.x},${rect.y},${rect.w},${rect.h}`;
          if (skipTiles.current.has(id) || inflight.current.has(id)) continue;
          if (Z.covered(intersection(rect, visible), have)) continue;
          const distance = Math.hypot(rect.x + rect.w / 2 - centre.x, rect.y + rect.h / 2 - centre.y);
          queue.push({ id, key: s.key, params: s.params, rect, distance });
        }
      }
      queue.sort((a, b) => a.distance - b.distance);
      for (const job of queue.slice(0, Math.max(0, TILE_INFLIGHT - inflight.current.size))) {
        const controller = new AbortController();
        inflight.current.set(job.id, { controller, key: job.key, rect: job.rect });
        const region = { x: job.rect.x, y: job.rect.y, w: job.rect.w, h: job.rect.h };
        renderTile(job.params, region, controller.signal)
          .then(async (rendered) => {
            const image = await loadArtifactImage(rendered.id);
            if (controller.signal.aborted) return;
            const w = rendered.width > 0 ? rendered.width : region.w;
            const h = rendered.height > 0 ? rendered.height : region.h;
            rememberTile(job.key, { x: region.x, y: region.y, w, h, image });
            // A host that cut the crop short would be asked again forever.
            if (w !== region.w || h !== region.h) skipTiles.current.add(job.id);
            setTileFailed(false);
          })
          .catch(() => {
            if (controller.signal.aborted) return;
            skipTiles.current.add(job.id);
            setTileFailed(true);
          })
          .finally(() => {
            if (inflight.current.get(job.id)?.controller === controller) inflight.current.delete(job.id);
            setTileVersion((v) => v + 1);
          });
      }
      setTileBusy(inflight.current.size > 0);
    }, TILE_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
    // `p`, `frame`, `full` and `sides` are read through these primitives.
  }, [wantTiles, renderTile, p.zoom, p.cx, p.cy, frame.width, frame.height, full.width, full.height, sidesKey, tileVersion]);

  // Leaving the photo cancels what is still rendering.
  useEffect(() => {
    const jobs = inflight.current;
    return () => {
      for (const job of jobs.values()) job.controller.abort();
      jobs.clear();
    };
  }, []);

  // Pinch on a trackpad (Ctrl + wheel) or Ctrl/⌘ + scroll zooms; once zoomed
  // in, scrolling pans. Native and not passive, so the page stays put.
  useEffect(() => {
    const elements = panes.current.filter((e): e is HTMLDivElement => e !== null);
    const onWheel = (event: WheelEvent) => {
      const { p: now, frame: f, full: size, dpr: ratioNow, pannable: canMove } = current.current;
      const pane = event.currentTarget as HTMLElement;
      const rect = pane.getBoundingClientRect();
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? rect.height : 1;
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        const factor = Math.exp(-event.deltaY * unit * 0.004);
        const at = { x: event.clientX - rect.left, y: event.clientY - rect.top };
        setView(Z.zoomAround(now, now.zoom * factor, f, size, ratioNow, at));
      } else if (canMove) {
        event.preventDefault();
        setView(Z.panBy(now, -event.deltaX * unit, -event.deltaY * unit));
      }
    };
    for (const element of elements) element.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      for (const element of elements) element.removeEventListener("wheel", onWheel);
    };
  }, [side, split]);

  const zoomTo = (zoom: number | "fit", at?: Z.Point) => {
    if (zoom === "fit") setView(Z.FIT);
    else setView(Z.zoomAround(p, zoom, frame, full, dpr, at));
  };

  const moveDivider = (clientX: number) => {
    const rect = panes.current[0]?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    setPosition(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)));
  };

  const local = (event: { clientX: number; clientY: number; currentTarget: Element }): Z.Point => {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    event.currentTarget.setPointerCapture(event.pointerId);
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()] as [Z.Point, Z.Point];
      const rect = event.currentTarget.getBoundingClientRect();
      gesture.current = {
        kind: "pinch",
        distance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)),
        mid: { x: (a.x + b.x) / 2 - rect.left, y: (a.y + b.y) / 2 - rect.top },
        start: p,
      };
      return;
    }
    if (pointers.current.size > 2) return;
    // Cancelling pointerdown keeps focus where it is and starts no text selection.
    event.preventDefault();
    if (split && !side && !pannable) {
      gesture.current = { kind: "divider" };
      moveDivider(event.clientX);
      handle.current?.focus({ preventScroll: true });
    } else if (pannable) {
      gesture.current = { kind: "pan", x: event.clientX, y: event.clientY, start: p };
      event.currentTarget.focus({ preventScroll: true });
    } else {
      gesture.current = null;
      event.currentTarget.focus({ preventScroll: true });
    }
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!pointers.current.has(event.pointerId)) return;
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const g = gesture.current;
    if (!g) return;
    if (g.kind === "divider") moveDivider(event.clientX);
    else if (g.kind === "pan") setView(Z.panBy(g.start, event.clientX - g.x, event.clientY - g.y));
    else if (pointers.current.size >= 2) {
      const [a, b] = [...pointers.current.values()] as [Z.Point, Z.Point];
      const rect = event.currentTarget.getBoundingClientRect();
      const mid = { x: (a.x + b.x) / 2 - rect.left, y: (a.y + b.y) / 2 - rect.top };
      const zoom = (g.start.zoom * Math.hypot(a.x - b.x, a.y - b.y)) / g.distance;
      const zoomed = Z.zoomAround(g.start, zoom, frame, full, dpr, g.mid);
      // Then follow the fingers' midpoint.
      setView(zoomed.fit ? zoomed : Z.panBy(Z.place(zoomed, frame, full, dpr), mid.x - g.mid.x, mid.y - g.mid.y));
    }
  };

  const onPointerEnd = (event: PointerEvent<HTMLDivElement>) => {
    pointers.current.delete(event.pointerId);
    const g = gesture.current;
    if (g?.kind === "pinch") {
      const rest = [...pointers.current.values()][0];
      gesture.current = rest ? { kind: "pan", x: rest.x, y: rest.y, start: p } : null;
    } else if (pointers.current.size === 0) {
      gesture.current = null;
    }
  };

  const onDoubleClick = (event: ReactMouseEvent<HTMLDivElement>) => {
    const at = local(event);
    if (p.fit || p.zoom < 0.999) zoomTo(1, at);
    else zoomTo("fit");
  };

  const onFrameKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const stepX = frame.width * 0.15;
    const stepY = frame.height * 0.15;
    const pans: Record<string, [number, number]> = {
      ArrowLeft: [stepX, 0],
      ArrowRight: [-stepX, 0],
      ArrowUp: [0, stepY],
      ArrowDown: [0, -stepY],
    };
    let handled = true;
    if (event.key === "+" || event.key === "=") zoomTo(Z.stepZoom(p.zoom, Z.fitZoom(frame, full, dpr), 1));
    else if (event.key === "-" || event.key === "_") zoomTo(Z.stepZoom(p.zoom, Z.fitZoom(frame, full, dpr), -1));
    else if (event.key === "0") zoomTo("fit");
    else if (event.key === "1") zoomTo(1);
    else if (pans[event.key] && pannable) {
      const [dx, dy] = pans[event.key]!;
      setView(Z.panBy(p, dx, dy));
    } else handled = false;
    if (handled) {
      event.preventDefault();
      // The page binds the arrow keys to the previous and next photo.
      event.stopPropagation();
    }
  };

  const onHandleKey = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 10 : 2;
    const next: Record<string, number> = {
      ArrowLeft: position - step,
      ArrowDown: position - step,
      ArrowRight: position + step,
      ArrowUp: position + step,
      PageDown: position - 10,
      PageUp: position + 10,
      Home: 0,
      End: 100,
    };
    const value = next[event.key];
    if (value === undefined) return;
    event.preventDefault();
    event.stopPropagation();
    setPosition(Math.min(100, Math.max(0, value)));
  };

  const onHandlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.focus({ preventScroll: true });
    gesture.current = { kind: "divider" };
    moveDivider(event.clientX);
  };

  const zoomText = p.fit ? `Fit · ${Z.zoomLabel(p.zoom)}` : Z.zoomLabel(p.zoom);
  const instructions = "Plus and minus zoom, 0 fits, 1 shows 100%, arrow keys pan when zoomed in.";
  const loadingAll = !beforeImage.image && !afterImage.image && !beforeImage.failed && !afterImage.failed;
  const status = tileBusy ? "Rendering 1:1…" : tileFailed && wantTiles ? "1:1 unavailable" : undefined;

  const pane = (index: number, content: ReactNode, label: string) => (
    <div
      ref={(element) => {
        panes.current[index] = element;
        if (index === 0) firstPane.current = element;
      }}
      tabIndex={0}
      role="group"
      aria-roledescription="zoomable photo"
      aria-label={`${label}. Zoom ${zoomText}. ${instructions}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      onDoubleClick={onDoubleClick}
      onKeyDown={onFrameKey}
      className={cn(
        "relative w-full select-none overflow-hidden rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] outline-none focus-visible:shadow-[var(--ds-focus-ring)]",
        pannable ? "cursor-grab active:cursor-grabbing" : split && !side ? "cursor-ew-resize" : "cursor-zoom-in",
      )}
      style={{ aspectRatio: safeRatio, touchAction: pannable ? "none" : "pan-y" }}
    >
      <canvas
        ref={(element) => {
          canvases.current[index] = element;
        }}
        role="img"
        aria-label={label}
        className="absolute inset-0 h-full w-full"
        style={{ imageRendering: p.zoom > 1 ? "pixelated" : "auto" }}
      />
      {content}
    </div>
  );

  const chip = "pointer-events-none absolute max-w-[45%] truncate rounded-[5px] bg-black/60 px-2 py-1 text-[12px] font-medium text-white backdrop-blur-sm";
  const footer = (
    <>
      <span className={cn(chip, "bottom-2 right-2 font-mono")}>{zoomText}</span>
      {status ? <span className={cn(chip, "bottom-2 left-2 font-mono")}>{status}</span> : null}
    </>
  );
  const pendingNote = note ?? (split && (beforeImage.pending || afterImage.pending) && !loadingAll ? "Loading…" : undefined);

  return (
    <figure
      className={cn("mx-auto w-full space-y-2", className)}
      // Tall photos stay within the viewport: the width follows from a 72vh
      // height cap, or the --compare-max-h a parent sets.
      style={{ maxWidth: `calc(var(--compare-max-h, 72vh) * ${side ? safeRatio * 2 : safeRatio} + ${side ? 8 : 0}px)` }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        {split ? (
          <SegmentedControl
            label="Compare as"
            value={mode}
            onChange={(next) => {
              setMode(next);
              saveMode(next);
            }}
            options={[
              { value: "slider", label: "Slider" },
              { value: "side", label: "Side by side" },
            ]}
          />
        ) : (
          <span />
        )}
        <div role="group" aria-label="Zoom" className="flex flex-wrap items-center gap-1">
          <ZoomButton active={p.fit} onClick={() => zoomTo("fit")}>
            Fit
          </ZoomButton>
          {Z.ZOOM_STEPS.map((step) => (
            <ZoomButton key={step} active={!p.fit && Math.abs(p.zoom - step) < 0.001} onClick={() => zoomTo(step)}>
              {Z.zoomLabel(step)}
            </ZoomButton>
          ))}
        </div>
      </div>

      {side ? (
        <div className="grid grid-cols-2 gap-2">
          {pane(
            0,
            <>
              <Placeholder state={beforeImage} />
              <span className={cn(chip, "left-2 top-2")}>Before · {before.label}</span>
              {footer}
            </>,
            `Before: ${before.label}`,
          )}
          {pane(
            1,
            <>
              <Placeholder state={afterImage} />
              <span className={cn(chip, "left-2 top-2")}>After · {after!.label}</span>
              {pendingNote ? <span className={cn(chip, "bottom-2 left-1/2 -translate-x-1/2 font-mono")}>{pendingNote}</span> : null}
            </>,
            `After: ${after!.label}`,
          )}
        </div>
      ) : (
        pane(
          0,
          <>
            <Placeholder state={split ? afterImage : beforeImage} />
            {split ? (
              <div className="pointer-events-none absolute inset-y-0" style={{ left: `${position}%` }}>
                <div className="absolute inset-y-0 -ml-px w-0.5 bg-white/90 shadow-[0_0_0_1px_rgb(0_0_0_/_0.25)]" />
                <div
                  ref={handle}
                  role="slider"
                  tabIndex={0}
                  aria-label="Before and after divider"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(position)}
                  aria-valuetext={`${Math.round(position)}% before`}
                  onKeyDown={onHandleKey}
                  onPointerDown={onHandlePointerDown}
                  onPointerMove={(event) => gesture.current?.kind === "divider" && moveDivider(event.clientX)}
                  onPointerUp={() => (gesture.current = null)}
                  onPointerCancel={() => (gesture.current = null)}
                  onDoubleClick={(event) => {
                    event.stopPropagation();
                    setPosition(50);
                  }}
                  className="pointer-events-auto absolute top-1/2 grid h-9 w-9 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize touch-none place-items-center rounded-full border border-black/20 bg-white text-black shadow-[0_2px_8px_rgb(0_0_0_/_0.35)] outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
                >
                  <ChevronsLeftRight aria-hidden="true" className="h-4 w-4" />
                </div>
              </div>
            ) : null}
            <span className={cn(chip, "left-2 top-2")}>{split ? `Before · ${before.label}` : before.label}</span>
            {split ? <span className={cn(chip, "right-2 top-2")}>After · {after.label}</span> : null}
            {footer}
            {pendingNote ? (
              <span aria-hidden="true" className={cn(chip, "bottom-2 left-1/2 -translate-x-1/2 font-mono")}>
                {pendingNote}
              </span>
            ) : null}
          </>,
          split ? `Before, ${before.label}, left of the divider; after, ${after.label}, right of it` : before.label,
        )
      )}
      <figcaption className={cn(text.meta, "normal-case")}>
        Before left, after right. Pinch, Ctrl/⌘ + scroll or the buttons zoom; drag to pan; double-click for 100%.
      </figcaption>
    </figure>
  );
}

function ZoomButton({ active, children, onClick }: { active: boolean; children: ReactNode; onClick: () => void }) {
  return (
    <Button
      size="sm"
      variant={active ? "secondary" : "ghost"}
      aria-pressed={active}
      onClick={onClick}
      className={cn("min-w-11 px-2 font-mono tabular-nums", active && "border-[var(--ds-gray-1000)]")}
    >
      {children}
    </Button>
  );
}

function Placeholder({ state }: { state: { image?: HTMLImageElement; failed: boolean } }) {
  if (state.image) return null;
  if (state.failed) return <ImageUnavailable />;
  return <Skeleton className="absolute inset-0 rounded-none" />;
}
