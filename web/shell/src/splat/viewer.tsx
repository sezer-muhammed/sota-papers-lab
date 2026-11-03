import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { Box, ChevronLeft, ChevronRight, Crop, Eye, Maximize2, Move3d, Orbit, RotateCcw, RotateCw, X } from "lucide-react";
import { Button, IconTile, RangeInput, Switch, cn, text } from "@spl/ui";
import type { Frustum } from "./cameras";
import type { Box as CropBox, SplatEngine, SplatFormat, ViewMode } from "./engine";
import { Wipe } from "./wipe";

/** Bytes to show. A new `key` replaces the scene; `live` keeps the view (a
    newer snapshot of the same run). */
export type SplatSource = { key: string; bytes: ArrayBuffer; format: SplatFormat; live?: boolean };

/** The posing stage's point cloud (.spz). */
export type PointsSource = { key: string; bytes: ArrayBuffer };

/** The photo behind camera `index`, for thumbnails and the wipe. */
export type PhotoFor = (index: number) => Promise<Blob | undefined>;

type ViewerState = "starting" | "empty" | "loading" | "ready" | "failed";

/** Thumbnails drawn at most, and fetched at once. */
const MAX_THUMBS = 300;
const THUMB_PARALLEL = 4;
const THUMB_WIDTH = 160;
/** Height of the strip's thumbnails, in CSS pixels. */
const STRIP_HEIGHT = 56;

/* The in-browser viewer: drag to orbit, right-drag to pan, wheel to zoom;
   or fly (F): WASD, Q/E, drag to look, Shift faster, wheel sets the speed.
   Room-scale captures start in fly mode.
   Click a camera frustum (or step through them) to fly into it and wipe
   between the render and the photo. three.js and Spark load on first use
   (a separate chunk). */
/** A small JPEG of a thumbnail bitmap for the camera strip. */
async function stripThumb(bitmap: ImageBitmap): Promise<Blob | undefined> {
  const h = STRIP_HEIGHT * 2;
  const w = Math.max(1, Math.round((bitmap.width / bitmap.height) * h));
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d")?.drawImage(bitmap, 0, 0, w, h);
  return new Promise((resolve) => c.toBlob((b) => resolve(b ?? undefined), "image/jpeg", 0.8));
}

export function SplatViewer({
  cameras,
  className,
  photoFor,
  points,
  source,
}: {
  cameras: Frustum[];
  className?: string;
  photoFor?: PhotoFor;
  points?: PointsSource;
  source?: SplatSource;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [engine, setEngine] = useState<SplatEngine>();
  const [state, setState] = useState<ViewerState>("starting");
  const [error, setError] = useState<string>();
  const [count, setCount] = useState<number>();
  const [pointCount, setPointCount] = useState<number>();
  const [showFrustums, setShowFrustums] = useState(true);
  const [showPoints, setShowPoints] = useState(false);
  const [turntable, setTurntable] = useState(false);
  const [crop, setCrop] = useState<{ base: CropBox; size: [number, number, number] }>();
  const [viewing, setViewing] = useState<number>();
  const [photo, setPhoto] = useState<string>();
  const [photoOpacity, setPhotoOpacity] = useState(100);
  /** Object URLs of the strip's thumbnails, by camera index. */
  const [strip, setStrip] = useState<(string | undefined)[]>([]);
  const stripRef = useRef<HTMLDivElement>(null);
  const lastKey = useRef<string>(undefined);
  const down = useRef<{ x: number; y: number }>(undefined);
  const [mode, setMode] = useState<ViewMode>("orbit");
  const [flySpeed, setFlySpeed] = useState(1);
  /** The user picked a mode; stop choosing it from the camera layout. */
  const chose = useRef(false);

  useEffect(() => {
    let live = true;
    let made: SplatEngine | undefined;
    import("./engine").then(
      ({ SplatEngine }) => {
        if (!live || !canvas.current) return;
        try {
          made = new SplatEngine(canvas.current);
          made.onLeaveCamera = () => setViewing(undefined);
          made.onTurntableStop = () => setTurntable(false);
          made.onFlySpeed = setFlySpeed;
          setEngine(made);
        } catch (e) {
          setState("failed");
          setError(e instanceof Error ? e.message : String(e));
        }
      },
      () => live && (setState("failed"), setError("The viewer did not load.")),
    );
    return () => {
      live = false;
      made?.dispose();
    };
  }, []);

  // Theme switches change the background token.
  useEffect(() => {
    if (!engine) return;
    const again = () => engine.applyTheme();
    const observer = new MutationObserver(again);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", again);
    return () => {
      observer.disconnect();
      media.removeEventListener("change", again);
    };
  }, [engine]);

  useEffect(() => {
    if (!engine) return;
    if (!source) {
      if (!points && !cameras.length) engine.clear();
      lastKey.current = undefined;
      setCount(undefined);
      setState("empty");
      return;
    }
    let live = true;
    const reframe = !source.live || lastKey.current === undefined;
    setState((s) => (s === "ready" && source.live ? s : "loading"));
    engine.load(source.bytes, source.format, reframe).then(
      (n) => {
        if (!live) return;
        lastKey.current = source.key;
        setCount(n);
        setState("ready");
        setError(undefined);
      },
      (e: unknown) => {
        if (!live) return;
        setState("failed");
        setError(e instanceof Error ? e.message : String(e));
      },
    );
    return () => {
      live = false;
    };
    // `points` and `cameras` only matter for the empty case.
  }, [engine, source]);

  useEffect(() => {
    if (!engine || !points) return;
    let live = true;
    engine.loadPoints(points.bytes).then(
      (n) => live && setPointCount(n),
      () => undefined,
    );
    return () => {
      live = false;
    };
  }, [engine, points]);

  useEffect(() => {
    engine?.setPointsVisible(showPoints);
  }, [engine, showPoints]);

  // Cameras and their thumbnails.
  useEffect(() => {
    if (!engine) return;
    engine.setCameras(cameras);
    setViewing(undefined);
    if (!chose.current && cameras.length) setMode(engine.roomScale() ? "fly" : "orbit");
    setStrip([]);
    if (!photoFor || !cameras.length) return;
    let live = true;
    const urls: string[] = [];
    // Strip thumbnails arrive in batches, not one render each.
    const pending = new Map<number, string>();
    let flush: ReturnType<typeof setTimeout> | undefined;
    const publish = () => {
      flush = undefined;
      const done = [...pending];
      pending.clear();
      setStrip((old) => {
        const next = [...old];
        for (const [i, url] of done) next[i] = url;
        return next;
      });
    };
    const queue = cameras.slice(0, MAX_THUMBS).map((_, i) => i);
    const worker = async () => {
      for (let i = queue.shift(); i !== undefined && live; i = queue.shift()) {
        try {
          const blob = await photoFor(i);
          if (!blob || !live) continue;
          const bitmap = await createImageBitmap(blob, { resizeWidth: THUMB_WIDTH, resizeQuality: "medium" });
          if (!live) continue;
          const small = await stripThumb(bitmap);
          engine.setThumbnail(i, bitmap);
          if (!small || !live) continue;
          const url = URL.createObjectURL(small);
          urls.push(url);
          pending.set(i, url);
          flush ??= setTimeout(publish, 150);
        } catch {
          /* a missing thumbnail leaves the plain frustum */
        }
      }
    };
    void Promise.all(Array.from({ length: THUMB_PARALLEL }, worker));
    return () => {
      live = false;
      if (flush) clearTimeout(flush);
      urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [engine, cameras, photoFor]);

  useEffect(() => {
    engine?.setFrustumsVisible(showFrustums);
  }, [engine, showFrustums, cameras]);

  useEffect(() => {
    engine?.setTurntable(turntable);
    if (turntable) setMode("orbit");
  }, [engine, turntable]);

  useEffect(() => {
    engine?.setMode(mode);
    if (mode === "fly") setTurntable(false);
  }, [engine, mode]);

  const pickMode = (m: ViewMode) => {
    chose.current = true;
    setMode(m);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLElement>) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    // Arrows move the fly camera, so they step cameras only in orbit or
    // while looking through one.
    const arrows = mode === "orbit" || viewing !== undefined;
    if (cameras.length && (e.key === "]" || (arrows && e.key === "ArrowRight"))) {
      e.preventDefault();
      look((viewing ?? -1) + 1);
    } else if (cameras.length && (e.key === "[" || (arrows && e.key === "ArrowLeft"))) {
      e.preventDefault();
      look((viewing ?? 0) - 1);
    } else if (e.key === "Escape" && viewing !== undefined) {
      e.preventDefault();
      engine?.exitCamera();
    } else if (e.code === "KeyF" && e.target === canvas.current) {
      e.preventDefault();
      pickMode(mode === "fly" ? "orbit" : "fly");
    }
  };

  useEffect(() => {
    if (!engine) return;
    if (!crop) return engine.setCrop(undefined);
    const { base, size } = crop;
    engine.setCrop({ center: base.center, half: [0, 1, 2].map((k) => base.half[k]! * size[k]!) as CropBox["half"] });
  }, [engine, crop]);

  // The photo of the camera being looked through, for the wipe.
  useEffect(() => {
    setPhoto(undefined);
    if (viewing === undefined || !photoFor) return;
    let url: string | undefined;
    let live = true;
    photoFor(viewing).then(
      (blob) => {
        if (!live || !blob) return;
        url = URL.createObjectURL(blob);
        setPhoto(url);
      },
      () => undefined,
    );
    return () => {
      live = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [viewing, photoFor]);

  const look = (i: number) => {
    if (!engine || !cameras.length) return;
    const n = ((i % cameras.length) + cameras.length) % cameras.length;
    setTurntable(false);
    engine.flyTo(n);
    setViewing(n);
  };

  // Keep the strip's current thumbnail in sight.
  useEffect(() => {
    if (viewing === undefined) return;
    stripRef.current?.querySelector(`[data-camera="${viewing}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [viewing]);

  // A click (not a drag) on a frustum flies into it.
  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    down.current = { x: e.clientX, y: e.clientY };
  };
  const onPointerUp = (e: PointerEvent<HTMLCanvasElement>) => {
    const d = down.current;
    down.current = undefined;
    if (!engine || !d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return;
    const hit = engine.pickCamera(e.clientX, e.clientY);
    if (hit !== undefined) look(hit);
  };

  const hasScene = state === "ready" || Boolean(points) || cameras.length > 0;
  const facts = [
    count ? `${count.toLocaleString()} splats` : null,
    !count && pointCount ? `${pointCount.toLocaleString()} points` : null,
    cameras.length ? `${cameras.length} cameras` : null,
  ].filter(Boolean);

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      <div className="relative overflow-hidden rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)]">
        <canvas
          ref={canvas}
          data-testid="splat-canvas"
          data-state={state}
          data-splats={count ?? 0}
          data-points={pointCount ?? 0}
          data-viewing={viewing ?? ""}
          data-mode={mode}
          tabIndex={0}
          aria-label={
            mode === "fly"
              ? "3D view, fly mode. W A S D to move, Q and E down and up, drag to look, Shift for speed, scroll for fly speed, F for orbit. Click a camera to look through it."
              : "3D view. Drag to orbit, right-drag or shift-drag to pan, scroll to zoom, F to fly. Click a camera to look through it."
          }
          className="block aspect-[4/3] w-full touch-none outline-none sm:aspect-[16/9]"
          onPointerDown={onPointerDown}
          onPointerUp={onPointerUp}
          onKeyDown={onKeyDown}
        />
        {hasScene ? (
          <p
            data-testid="splat-keys"
            className="pointer-events-none absolute bottom-3 left-3 rounded-[6px] bg-black/60 px-2 py-1 font-mono text-[11px] text-white"
          >
            {mode === "fly"
              ? `WASD move · Q/E down/up · drag look · Shift fast · wheel speed ×${flySpeed.toFixed(flySpeed < 1 ? 2 : 1)} · two fingers move · F orbit`
              : "Drag orbit · right-drag pan · wheel zoom · F fly"}
          </p>
        ) : null}
        {viewing !== undefined && photo ? (
          <Wipe
            photo={photo}
            opacity={photoOpacity / 100}
            label={`Camera ${viewing + 1}`}
            onKeyDown={onKeyDown}
            onLeave={() => engine?.exitCamera()}
          />
        ) : null}
        {viewing !== undefined ? (
          <p
            data-testid="splat-camera-label"
            className="pointer-events-none absolute bottom-12 left-3 max-w-[70%] truncate rounded-[6px] bg-black/60 px-2 py-1 font-mono text-[12px] text-white"
          >
            {`Camera ${viewing + 1} / ${cameras.length}`}
            {cameras[viewing]?.name ? ` · ${cameras[viewing]!.name}` : ""}
          </p>
        ) : null}
        {hasScene ? null : (
          <div className="pointer-events-none absolute inset-0 grid place-items-center p-6 text-center">
            <div className="flex flex-col items-center gap-3">
              <IconTile icon={Box} />
              <p className={text.small} role={state === "failed" ? "alert" : "status"}>
                {state === "failed"
                  ? `The viewer failed: ${error ?? "unknown error"}`
                  : state === "loading"
                    ? "Loading splats…"
                    : state === "starting"
                      ? "Starting the viewer…"
                      : "Drop photos or a video to reconstruct, or open a .ply / .spz to view it here."}
              </p>
            </div>
          </div>
        )}
        {state === "failed" && hasScene ? (
          <p role="alert" className="absolute left-3 top-3 rounded-[6px] bg-black/60 px-2 py-1 text-[12px] text-white">
            The viewer failed: {error ?? "unknown error"}
          </p>
        ) : null}
        {state === "loading" && hasScene ? (
          <p role="status" className="pointer-events-none absolute left-3 top-3 rounded-[6px] bg-black/60 px-2 py-1 font-mono text-[12px] uppercase text-white">
            Loading splats…
          </p>
        ) : null}
      </div>

      {cameras.length ? (
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Cameras" onKeyDown={onKeyDown}>
          <Button size="sm" icon={ChevronLeft} disabled={!engine} onClick={() => look((viewing ?? 0) - 1)}>
            <span className="sr-only">Previous camera ([)</span>
          </Button>
          <Button size="sm" icon={Eye} disabled={!engine} onClick={() => look(viewing ?? 0)}>
            {viewing === undefined ? "Look through camera 1" : `Camera ${viewing + 1} of ${cameras.length}`}
          </Button>
          <Button size="sm" icon={ChevronRight} disabled={!engine} onClick={() => look((viewing ?? -1) + 1)}>
            <span className="sr-only">Next camera (])</span>
          </Button>
          {viewing !== undefined ? (
            <Button size="sm" icon={X} onClick={() => engine?.exitCamera()}>
              {mode === "fly" ? "Back to Fly" : "Back to Orbit"}
            </Button>
          ) : null}
          {viewing !== undefined && photo ? (
            <label className="flex items-center gap-2">
              <span className={text.caption}>
                Photo <span className="font-mono">{photoOpacity}%</span>
              </span>
              <RangeInput min={0} max={100} value={photoOpacity} aria-label="Photo opacity" onValueChange={setPhotoOpacity} />
            </label>
          ) : null}
          <span className={text.meta}>
            {photoFor ? "[ and ] step cameras, Esc goes back; drag the split to compare render and photo." : "[ and ] step cameras, Esc goes back."}
          </span>
        </div>
      ) : null}
      {cameras.length && photoFor ? (
        <div
          ref={stripRef}
          data-testid="splat-camera-strip"
          role="listbox"
          aria-label="Look through a camera"
          className="flex gap-1.5 overflow-x-auto pb-1"
          onKeyDown={onKeyDown}
        >
          {cameras.slice(0, MAX_THUMBS).map((cam, i) => (
            <button
              key={i}
              type="button"
              role="option"
              aria-selected={viewing === i}
              data-camera={i}
              title={`Camera ${i + 1}${cam.name ? ` · ${cam.name}` : ""}`}
              onClick={() => look(i)}
              className={cn(
                "relative shrink-0 overflow-hidden rounded-[6px] border bg-[var(--ds-background-100)] outline-none focus-visible:shadow-[var(--ds-focus-ring)]",
                viewing === i ? "border-[var(--ds-blue-700)] ring-2 ring-[var(--ds-blue-700)]" : "border-[var(--ds-gray-alpha-400)]",
              )}
              style={{ height: STRIP_HEIGHT, width: Math.round(STRIP_HEIGHT * cam.aspect) }}
            >
              {strip[i] ? <img src={strip[i]} alt="" loading="lazy" draggable={false} className="h-full w-full object-cover" /> : null}
              <span className="absolute bottom-0 left-0 rounded-tr-[4px] bg-black/60 px-1 font-mono text-[10px] text-white">{i + 1}</span>
            </button>
          ))}
        </div>
      ) : null}

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Switch
          className="justify-start gap-3"
          label="Cameras"
          checked={showFrustums}
          disabled={!cameras.length}
          onCheckedChange={setShowFrustums}
        />
        <Switch
          className="justify-start gap-3"
          label="Points"
          checked={showPoints || (!count && Boolean(pointCount))}
          disabled={!pointCount || !count}
          onCheckedChange={setShowPoints}
        />
        <div className="flex gap-1" role="group" aria-label="Camera controls">
          <Button size="sm" icon={Orbit} aria-pressed={mode === "orbit"} disabled={!engine} onClick={() => pickMode("orbit")}>
            Orbit
          </Button>
          <Button size="sm" icon={Move3d} aria-pressed={mode === "fly"} disabled={!engine} onClick={() => pickMode("fly")}>
            Fly
          </Button>
        </div>
        <Switch className="justify-start gap-3" label="Turntable" checked={turntable} disabled={!engine} onCheckedChange={setTurntable} />
        <p className={cn(text.meta, "font-mono")}>{facts.join(" · ")}</p>
        <div className="ml-auto flex flex-wrap gap-2">
          <Button
            size="sm"
            icon={Crop}
            aria-pressed={Boolean(crop)}
            disabled={!engine || !hasScene}
            onClick={() => {
              if (crop) return setCrop(undefined);
              const base = engine?.bounds();
              if (base) setCrop({ base, size: [1, 1, 1] });
            }}
          >
            {crop ? "Remove crop" : "Crop"}
          </Button>
          <Button size="sm" icon={RotateCcw} disabled={!engine} onClick={() => engine?.frame()}>
            Reset view
          </Button>
          <Button size="sm" icon={Maximize2} disabled={!engine} onClick={() => void canvas.current?.parentElement?.requestFullscreen?.()}>
            Full screen
          </Button>
        </div>
      </div>

      {crop ? (
        <fieldset className="grid gap-3 rounded-[8px] border border-[var(--ds-gray-alpha-400)] p-3 sm:grid-cols-3">
          <legend className={cn(text.meta, "px-1")}>Crop box (view only; downloads stay whole)</legend>
          {(["Width", "Height", "Depth"] as const).map((label, k) => (
            <label key={label} className="flex flex-col gap-1.5">
              <span className={text.caption}>
                {label} <span className="font-mono">{Math.round(crop.size[k]! * 100)}%</span>
              </span>
              <RangeInput
                min={5}
                max={100}
                value={Math.round(crop.size[k]! * 100)}
                aria-label={`Crop ${label.toLowerCase()}`}
                onValueChange={(v) => {
                  const size = [...crop.size] as [number, number, number];
                  size[k] = v / 100;
                  setCrop({ ...crop, size });
                }}
              />
            </label>
          ))}
        </fieldset>
      ) : null}
      {turntable ? (
        <p className={cn(text.meta, "flex items-center gap-1.5")}>
          <RotateCw aria-hidden="true" className="h-3.5 w-3.5" /> Turning about the scene centre; drag to stop.
        </p>
      ) : null}
    </div>
  );
}
