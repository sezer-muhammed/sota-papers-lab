import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { ChevronsLeftRight, X } from "lucide-react";
import { cn } from "@spl/ui";

const clamp = (n: number) => Math.min(100, Math.max(0, n));

/* Over the viewer while it looks through a camera: the photo on the left
   of a movable split, the render on the right. The photo fills the height
   and is centred, which lines up with the render because the viewer took
   the camera's vertical field of view. Drag, click or use the handle (a
   slider: arrows, Page Up/Down, Home, End), like the Home page's compare. */
export function Wipe({ label, onLeave, photo }: { label: string; onLeave: () => void; photo: string }) {
  const [split, setSplit] = useState(50);
  const frame = useRef<HTMLDivElement>(null);
  const dragging = useRef<number>(undefined);

  const at = (clientX: number) => {
    const box = frame.current?.getBoundingClientRect();
    if (box && box.width > 0) setSplit(clamp(((clientX - box.left) / box.width) * 100));
  };
  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    e.preventDefault();
    dragging.current = e.pointerId;
    e.currentTarget.setPointerCapture(e.pointerId);
    at(e.clientX);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (dragging.current === e.pointerId) at(e.clientX);
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
    if (e.key === "Escape") return onLeave();
    if (!(e.key in next)) return;
    e.preventDefault();
    setSplit(clamp(next[e.key]!));
  };

  const shown = Math.round(split);
  const chip =
    "pointer-events-none absolute top-3 rounded-[6px] bg-black/60 px-2 py-1 font-mono text-[12px] uppercase leading-4 text-white transition-opacity duration-150";

  return (
    <div
      ref={frame}
      data-testid="splat-wipe"
      className="absolute inset-0 cursor-ew-resize touch-none select-none overflow-hidden"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={() => (dragging.current = undefined)}
      onPointerCancel={() => (dragging.current = undefined)}
    >
      <div className="pointer-events-none absolute inset-0" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
        <img
          src={photo}
          alt={`${label}: the photo`}
          draggable={false}
          className="absolute left-1/2 top-0 h-full w-auto max-w-none -translate-x-1/2 select-none"
        />
      </div>
      <span aria-hidden="true" className={cn(chip, "left-3", split < 14 && "opacity-0")}>
        Photo
      </span>
      <span aria-hidden="true" className={cn(chip, "right-14", split > 80 && "opacity-0")}>
        Render
      </span>
      <button
        type="button"
        onClick={onLeave}
        onPointerDown={(e) => e.stopPropagation()}
        className="absolute right-3 top-3 grid h-8 w-8 place-items-center rounded-[6px] bg-black/60 text-white outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
      >
        <X aria-hidden="true" className="h-4 w-4" />
        <span className="sr-only">Close the comparison and orbit</span>
      </button>
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-y-0 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_0_1px_rgb(0_0_0/0.18)]"
        style={{ left: `${split}%` }}
      />
      <div
        role="slider"
        tabIndex={0}
        aria-label={`${label}: photo and render`}
        aria-orientation="horizontal"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={shown}
        aria-valuetext={`${shown}% photo, ${100 - shown}% render`}
        onKeyDown={onKeyDown}
        className="absolute top-1/2 flex h-10 w-10 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize items-center justify-center rounded-full border border-black/10 bg-white text-neutral-900 shadow-[0_2px_8px_rgb(0_0_0/0.25)] outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
        style={{ left: `${split}%` }}
      >
        <ChevronsLeftRight aria-hidden="true" className="h-4 w-4" />
      </div>
    </div>
  );
}
