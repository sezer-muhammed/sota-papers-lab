import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { ChevronsLeftRight } from "lucide-react";
import { Skeleton, cn } from "@spl/ui";
import { ImageUnavailable, useDecodedSrc } from "./images";

export type CompareSide = { id?: string; label: string };

const clamp = (value: number) => Math.min(100, Math.max(0, value));

/* Before/after with a draggable divider: the after image fills the frame and
   the before image is clipped to the left of the divider. Drag anywhere on
   the photo, or focus the handle and use the arrow keys (Shift for bigger
   steps, Home/End for all before/after). Without an `after`, one image. */
export function CompareView({
  after,
  before,
  className,
  note,
  ratio,
}: {
  after?: CompareSide;
  before: CompareSide;
  className?: string;
  /** A short state shown over the photo, e.g. "Rendering…". */
  note?: string;
  /** Width / height of the upright photo, so the frame is sized before it loads. */
  ratio: number;
}) {
  const [position, setPosition] = useState(50);
  const frame = useRef<HTMLDivElement>(null);
  const handle = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const split = after !== undefined;
  const base = useDecodedSrc(split ? after.id : before.id);
  const top = useDecodedSrc(split ? before.id : undefined);
  const safeRatio = Number.isFinite(ratio) && ratio > 0 ? ratio : 1.5;

  const moveTo = (clientX: number) => {
    const rect = frame.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    setPosition(clamp(((clientX - rect.left) / rect.width) * 100));
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!split || event.button !== 0) return;
    // Cancelling pointerdown suppresses the compatibility mousedown, whose
    // default would blur the handle focused below (and start a text selection).
    event.preventDefault();
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    moveTo(event.clientX);
    handle.current?.focus({ preventScroll: true });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
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
    // The page binds the arrow keys to the previous and next photo.
    event.stopPropagation();
    setPosition(clamp(value));
  };

  const loading = !base.src && !base.failed;

  return (
    <figure
      className={cn("mx-auto w-full", className)}
      // Tall photos stay within the viewport: the width follows from a 72vh
      // height cap, or the --compare-max-h a parent sets (the photo page
      // lowers it so the scopes fit beside the sliders).
      style={{ maxWidth: `calc(var(--compare-max-h, 72vh) * ${safeRatio})` }}
    >
      <div
        ref={frame}
        onPointerDown={onPointerDown}
        onPointerMove={(event) => dragging.current && moveTo(event.clientX)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
        onDoubleClick={() => split && setPosition(50)}
        className={cn(
          "relative w-full touch-pan-y select-none overflow-hidden rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)]",
          split && "cursor-ew-resize",
        )}
        style={{ aspectRatio: safeRatio }}
      >
        {base.src ? (
          <img
            src={base.src}
            alt={split ? `After: ${after.label}` : before.label}
            draggable={false}
            className="absolute inset-0 h-full w-full object-contain"
          />
        ) : null}
        {split && top.src ? (
          <img
            src={top.src}
            alt={`Before: ${before.label}`}
            draggable={false}
            className="absolute inset-0 h-full w-full object-contain"
            style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
          />
        ) : null}
        {loading ? <Skeleton className="absolute inset-0 rounded-none" /> : null}
        {base.failed ? <ImageUnavailable /> : null}

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
              onKeyDown={onKeyDown}
              className="pointer-events-auto absolute top-1/2 grid h-9 w-9 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize place-items-center rounded-full border border-black/20 bg-white text-black shadow-[0_2px_8px_rgb(0_0_0_/_0.35)] outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
            >
              <ChevronsLeftRight aria-hidden="true" className="h-4 w-4" />
            </div>
          </div>
        ) : null}

        <span className="pointer-events-none absolute left-2 top-2 max-w-[45%] truncate rounded-[5px] bg-black/60 px-2 py-1 text-[12px] font-medium text-white backdrop-blur-sm">
          {split ? `Before · ${before.label}` : before.label}
        </span>
        {split ? (
          <span className="pointer-events-none absolute right-2 top-2 max-w-[45%] truncate rounded-[5px] bg-black/60 px-2 py-1 text-[12px] font-medium text-white backdrop-blur-sm">
            After · {after.label}
          </span>
        ) : null}
        {note || (split && (base.pending || top.pending) && !loading) ? (
          <span
            aria-hidden="true"
            className="pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 rounded-[5px] bg-black/60 px-2 py-1 font-mono text-[12px] text-white backdrop-blur-sm"
          >
            {note ?? "Loading…"}
          </span>
        ) : null}
      </div>
    </figure>
  );
}
