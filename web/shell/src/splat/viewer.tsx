import { useEffect, useRef, useState, type PointerEvent } from "react";
import { Box, ChevronLeft, ChevronRight, Crop, Eye, Maximize2, RotateCcw, RotateCw, X } from "lucide-react";
import { Button, IconTile, RangeInput, Switch, cn, text } from "@spl/ui";
import type { Frustum } from "./cameras";
import type { Box as CropBox, SplatEngine, SplatFormat } from "./engine";
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

/* The in-browser viewer: drag to orbit, right-drag to pan, wheel to zoom.
   Click a camera frustum (or step through them) to fly into it and wipe
   between the render and the photo. three.js and Spark load on first use
   (a separate chunk). */
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
  const lastKey = useRef<string>(undefined);
  const down = useRef<{ x: number; y: number }>(undefined);

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
    if (!photoFor || !cameras.length) return;
    let live = true;
    const queue = cameras.slice(0, MAX_THUMBS).map((_, i) => i);
    const worker = async () => {
      for (let i = queue.shift(); i !== undefined && live; i = queue.shift()) {
        try {
          const blob = await photoFor(i);
          if (!blob || !live) continue;
          const bitmap = await createImageBitmap(blob, { resizeWidth: THUMB_WIDTH, resizeQuality: "medium" });
          if (live) engine.setThumbnail(i, bitmap);
        } catch {
          /* a missing thumbnail leaves the plain frustum */
        }
      }
    };
    void Promise.all(Array.from({ length: THUMB_PARALLEL }, worker));
    return () => {
      live = false;
    };
  }, [engine, cameras, photoFor]);

  useEffect(() => {
    engine?.setFrustumsVisible(showFrustums);
  }, [engine, showFrustums, cameras]);

  useEffect(() => {
    engine?.setTurntable(turntable);
  }, [engine, turntable]);

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
          aria-label="3D view. Drag to orbit, right-drag or shift-drag to pan, scroll to zoom. Click a camera to look through it."
          className="block aspect-[4/3] w-full touch-none outline-none sm:aspect-[16/9]"
          onPointerDown={onPointerDown}
          onPointerUp={onPointerUp}
        />
        {viewing !== undefined && photo ? (
          <Wipe photo={photo} label={`Camera ${viewing + 1}`} onLeave={() => engine?.leaveCamera()} />
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
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Cameras">
          <Button size="sm" icon={ChevronLeft} disabled={!engine} onClick={() => look((viewing ?? 0) - 1)}>
            <span className="sr-only">Previous camera</span>
          </Button>
          <Button size="sm" icon={Eye} disabled={!engine} onClick={() => look(viewing ?? 0)}>
            {viewing === undefined ? "Look through camera 1" : `Camera ${viewing + 1} of ${cameras.length}`}
          </Button>
          <Button size="sm" icon={ChevronRight} disabled={!engine} onClick={() => look((viewing ?? -1) + 1)}>
            <span className="sr-only">Next camera</span>
          </Button>
          {viewing !== undefined ? (
            <Button size="sm" icon={X} onClick={() => engine?.leaveCamera()}>
              Orbit
            </Button>
          ) : null}
          <span className={text.meta}>{photoFor ? "Drag the split to compare render and photo." : null}</span>
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
