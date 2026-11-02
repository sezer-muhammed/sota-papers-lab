import { useEffect, useRef, useState } from "react";
import { Box, Maximize2, RotateCcw } from "lucide-react";
import { Button, IconTile, Switch, cn, text } from "@spl/ui";
import type { Frustum } from "./cameras";
import type { SplatEngine, SplatFormat } from "./engine";

/** Bytes to show. A new `key` replaces the scene; `live` keeps the view (a
    newer snapshot of the same run). */
export type SplatSource = { key: string; bytes: ArrayBuffer; format: SplatFormat; live?: boolean };

type ViewerState = "starting" | "empty" | "loading" | "ready" | "failed";

/* The in-browser viewer: drag to orbit, right-drag to pan, wheel to zoom.
   three.js and Spark load on first use (a separate chunk). */
export function SplatViewer({
  cameras,
  className,
  source,
}: {
  cameras: Frustum[];
  className?: string;
  source?: SplatSource;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [engine, setEngine] = useState<SplatEngine>();
  const [state, setState] = useState<ViewerState>("starting");
  const [error, setError] = useState<string>();
  const [count, setCount] = useState<number>();
  const [showFrustums, setShowFrustums] = useState(true);
  const lastKey = useRef<string>(undefined);

  useEffect(() => {
    let live = true;
    let made: SplatEngine | undefined;
    import("./engine").then(
      ({ SplatEngine }) => {
        if (!live || !canvas.current) return;
        try {
          made = new SplatEngine(canvas.current);
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
      engine.clear();
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
  }, [engine, source]);

  useEffect(() => {
    engine?.setCameras(cameras);
    engine?.setFrustumsVisible(showFrustums);
  }, [engine, cameras, showFrustums]);

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      <div className="relative overflow-hidden rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)]">
        <canvas
          ref={canvas}
          data-testid="splat-canvas"
          data-state={state}
          data-splats={count ?? 0}
          aria-label="3D view. Drag to orbit, right-drag or shift-drag to pan, scroll to zoom."
          className="block aspect-[4/3] w-full touch-none outline-none sm:aspect-[16/9]"
        />
        {state === "ready" ? null : (
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
                      : cameras.length
                        ? "Cameras loaded. Splats appear here as training goes."
                        : "Train a scene or open a .ply / .spz to view it here."}
              </p>
            </div>
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Switch
          className="justify-start gap-3"
          label="Camera frustums"
          checked={showFrustums}
          disabled={!cameras.length}
          onCheckedChange={setShowFrustums}
        />
        <p className={cn(text.meta, "font-mono")}>
          {count ? `${count.toLocaleString()} splats` : null}
          {count && cameras.length ? " · " : null}
          {cameras.length ? `${cameras.length} cameras` : null}
        </p>
        <div className="ml-auto flex gap-2">
          <Button size="sm" icon={RotateCcw} disabled={!engine} onClick={() => engine?.frame()}>
            Reset view
          </Button>
          <Button
            size="sm"
            icon={Maximize2}
            disabled={!engine}
            onClick={() => void canvas.current?.requestFullscreen?.()}
          >
            Full screen
          </Button>
        </div>
      </div>
    </div>
  );
}
