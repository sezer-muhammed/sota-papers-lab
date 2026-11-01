import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { ChevronsLeftRight } from "lucide-react";
import { cn } from "@spl/ui";
import { exampleImage, type Example } from "./examples";

const clamp = (n: number) => Math.min(100, Math.max(0, n));

/* A movable split between two renders of one photo: BEFORE (as shot) on the
   left, AFTER (the auto edit) on the right. Drag or click anywhere on the
   photo; the handle is a slider for keyboards and screen readers (arrows,
   Page Up/Down, Home, End). On touch a vertical swipe still scrolls the
   page: only horizontal drags move the split. */
export function Compare({
  example,
  priority = false,
  className,
}: {
  example: Example;
  /** The hero: load at once and first. */
  priority?: boolean;
  className?: string;
}) {
  const [split, setSplit] = useState(50);
  const frame = useRef<HTMLDivElement>(null);
  const handle = useRef<HTMLDivElement>(null);
  // Touch: move only once the finger travels sideways, or on a tap.
  const drag = useRef<{ id: number; x: number; moved: boolean } | null>(null);

  const at = (clientX: number) => {
    const box = frame.current?.getBoundingClientRect();
    if (box && box.width > 0) setSplit(clamp(((clientX - box.left) / box.width) * 100));
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    drag.current = { id: e.pointerId, x: e.clientX, moved: e.pointerType === "mouse" };
    if (e.pointerType === "mouse") {
      e.preventDefault(); // no image drag or text selection
      e.currentTarget.setPointerCapture(e.pointerId);
      at(e.clientX);
      handle.current?.focus({ preventScroll: true });
    }
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    if (!d.moved) {
      if (Math.abs(e.clientX - d.x) < 6) return;
      d.moved = true;
      e.currentTarget.setPointerCapture(e.pointerId);
    }
    at(e.clientX);
  };
  const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    if (!d.moved) at(e.clientX); // a tap
    drag.current = null;
  };
  const onPointerCancel = () => {
    drag.current = null; // the browser took the gesture (a vertical scroll)
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 10 : 2;
    const next: Record<string, number> = {
      ArrowLeft: split - step,
      ArrowDown: split - step,
      ArrowRight: split + step,
      ArrowUp: split + step,
      PageDown: split - 10,
      PageUp: split + 10,
      Home: 0,
      End: 100,
    };
    if (!(e.key in next)) return;
    e.preventDefault();
    setSplit(clamp(next[e.key]!));
  };

  const shown = Math.round(split);
  const img = "pointer-events-none absolute inset-0 h-full w-full select-none object-cover";
  const chip =
    "pointer-events-none absolute top-3 rounded-[6px] bg-black/60 px-2 py-1 font-mono text-[12px] uppercase leading-4 text-white transition-opacity duration-150";

  return (
    <div
      ref={frame}
      className={cn(
        "relative cursor-ew-resize touch-pan-y select-none overflow-hidden rounded-[8px] bg-[var(--ds-gray-200)]",
        className,
      )}
      style={{ aspectRatio: `${example.width} / ${example.height}` }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
    >
      <img
        src={exampleImage(example.key, "after")}
        alt={`After: ${example.subject}, edited with the ${example.look} look.`}
        width={example.width}
        height={example.height}
        loading={priority ? "eager" : "lazy"}
        fetchPriority={priority ? "high" : "auto"}
        decoding="async"
        draggable={false}
        className={img}
      />
      <img
        src={exampleImage(example.key, "before")}
        alt={`Before: ${example.subject}, as shot.`}
        width={example.width}
        height={example.height}
        loading={priority ? "eager" : "lazy"}
        decoding="async"
        draggable={false}
        className={img}
        style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
      />

      <span aria-hidden="true" className={cn(chip, "left-3", split < 14 && "opacity-0")}>
        Before
      </span>
      <span aria-hidden="true" className={cn(chip, "right-3", split > 86 && "opacity-0")}>
        After
      </span>

      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-y-0 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_0_1px_rgb(0_0_0/0.18)]"
        style={{ left: `${split}%` }}
      />
      <div
        ref={handle}
        role="slider"
        tabIndex={0}
        aria-label={`Before and after, ${example.camera}`}
        aria-orientation="horizontal"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={shown}
        aria-valuetext={`${shown}% before, ${100 - shown}% after`}
        onKeyDown={onKeyDown}
        className="absolute top-1/2 flex h-10 w-10 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize items-center justify-center rounded-full border border-black/10 bg-white text-neutral-900 shadow-[0_2px_8px_rgb(0_0_0/0.25)] outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
        style={{ left: `${split}%` }}
      >
        <ChevronsLeftRight aria-hidden="true" className="h-4 w-4" />
      </div>
    </div>
  );
}
