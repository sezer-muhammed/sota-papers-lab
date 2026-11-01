/* The zoomable viewer's geometry, without the DOM. Image coordinates are
   upright full-resolution pixels; frame coordinates are CSS pixels from the
   frame's top left. Zoom counts device pixels per image pixel, so 100% (1)
   is one photo pixel on one screen pixel, as in Lightroom, on any display. */

export type Size = { width: number; height: number };
export type Point = { x: number; y: number };
export type Rect = { x: number; y: number; w: number; h: number };

/** Fit the frame, or a zoom with the image point at the frame's centre. */
export type View = { fit: true } | { fit: false; zoom: number; cx: number; cy: number };

/** The zoom buttons besides Fit: 50% to 800%. */
export const ZOOM_STEPS = [0.5, 1, 2, 4, 8] as const;
export const MAX_ZOOM = 8;
/** The host's largest 1:1 crop per request. */
export const MAX_TILE = 2048;

export const FIT: View = { fit: true };

/** The view resolved for a frame: zoom (device px per image px), scale
    (CSS px per image px) and the centre, which keeps the image covering the
    frame where it is larger, and centred on an axis where it is smaller. */
export type Placement = { zoom: number; scale: number; cx: number; cy: number; fit: boolean };

const valid = (size: Size) => size.width > 0 && size.height > 0;

export function fitZoom(frame: Size, image: Size, dpr: number): number {
  if (!valid(frame) || !valid(image)) return 1;
  return Math.min(frame.width / image.width, frame.height / image.height) * dpr;
}

/** From a little under Fit (or 50%) to 800%. */
export function clampZoom(zoom: number, fit: number): number {
  const min = Math.min(fit, ZOOM_STEPS[0]);
  return Math.min(MAX_ZOOM, Math.max(min, Number.isFinite(zoom) ? zoom : fit));
}

function clampAxis(centre: number, span: number, length: number): number {
  if (span >= length) return length / 2;
  return Math.min(length - span / 2, Math.max(span / 2, centre));
}

export function place(view: View, frame: Size, image: Size, dpr: number): Placement {
  const fit = fitZoom(frame, image, dpr);
  const zoom = view.fit ? fit : clampZoom(view.zoom, fit);
  const scale = zoom / dpr;
  if (view.fit || !valid(frame)) {
    return { zoom, scale, cx: image.width / 2, cy: image.height / 2, fit: view.fit };
  }
  return {
    zoom,
    scale,
    cx: clampAxis(view.cx, frame.width / scale, image.width),
    cy: clampAxis(view.cy, frame.height / scale, image.height),
    fit: false,
  };
}

/** Whether the image is larger than the frame, so dragging pans it. */
export function canPan(p: Placement, frame: Size, image: Size): boolean {
  return image.width * p.scale > frame.width + 0.5 || image.height * p.scale > frame.height + 0.5;
}

/** The image point under a frame point. */
export function imagePoint(p: Placement, frame: Size, point: Point): Point {
  return {
    x: p.cx + (point.x - frame.width / 2) / p.scale,
    y: p.cy + (point.y - frame.height / 2) / p.scale,
  };
}

/** The frame point over an image point. */
export function framePoint(p: Placement, frame: Size, point: Point): Point {
  return {
    x: (point.x - p.cx) * p.scale + frame.width / 2,
    y: (point.y - p.cy) * p.scale + frame.height / 2,
  };
}

/** Zooms to `zoom`, keeping the image point under the frame point `at`
    where it is (the frame's centre when not given). */
export function zoomAround(p: Placement, zoom: number, frame: Size, image: Size, dpr: number, at?: Point): View {
  const fit = fitZoom(frame, image, dpr);
  const next = clampZoom(zoom, fit);
  if (Math.abs(next - fit) <= fit * 0.005) return FIT;
  const anchor = at ?? { x: frame.width / 2, y: frame.height / 2 };
  const under = imagePoint(p, frame, anchor);
  const scale = next / dpr;
  return {
    fit: false,
    zoom: next,
    cx: under.x - (anchor.x - frame.width / 2) / scale,
    cy: under.y - (anchor.y - frame.height / 2) / scale,
  };
}

/** Moves the view by a drag of `dx`, `dy` CSS pixels. */
export function panBy(p: Placement, dx: number, dy: number): View {
  return { fit: false, zoom: p.zoom, cx: p.cx - dx / p.scale, cy: p.cy - dy / p.scale };
}

/** The next zoom step in or out from `zoom` (Fit counts as its own value). */
export function stepZoom(zoom: number, fit: number, direction: 1 | -1): number {
  const steps = [...new Set([...ZOOM_STEPS, fit])].sort((a, b) => a - b);
  if (direction > 0) return steps.find((s) => s > zoom * 1.001) ?? MAX_ZOOM;
  return [...steps].reverse().find((s) => s < zoom * 0.999) ?? steps[0]!;
}

/** The part of the image the frame shows. */
export function visibleRegion(p: Placement, frame: Size, image: Size): Rect {
  const halfW = frame.width / 2 / p.scale;
  const halfH = frame.height / 2 / p.scale;
  const x0 = Math.max(0, p.cx - halfW);
  const y0 = Math.max(0, p.cy - halfH);
  const x1 = Math.min(image.width, p.cx + halfW);
  const y1 = Math.min(image.height, p.cy + halfH);
  return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
}

/** Whether the preview is magnified on screen here, so 1:1 crops show more. */
export function needsTiles(p: Placement, preview: Size, image: Size): boolean {
  if (!valid(preview) || !valid(image)) return false;
  return p.zoom > (preview.width / image.width) * 1.02;
}

const floorTo = (value: number, grid: number) => Math.floor(value / grid) * grid;
const ceilTo = (value: number, grid: number) => Math.ceil(value / grid) * grid;

/** The 1:1 crops to request for a visible region: grown by a margin (so a
    small pan stays inside) and snapped to a grid (so nearby views ask for
    the same crops), then cut into pieces the host takes (2048 px a side). */
export function tilesFor(visible: Rect, image: Size, grid = 128, max = MAX_TILE): Rect[] {
  if (visible.w <= 0 || visible.h <= 0 || !valid(image)) return [];
  const margin = Math.max(visible.w, visible.h) <= max / 2 ? 0.25 : 0;
  const x0 = Math.max(0, floorTo(visible.x - visible.w * margin, grid));
  const y0 = Math.max(0, floorTo(visible.y - visible.h * margin, grid));
  const x1 = Math.min(image.width, ceilTo(visible.x + visible.w * (1 + margin), grid));
  const y1 = Math.min(image.height, ceilTo(visible.y + visible.h * (1 + margin), grid));
  const cols = Math.max(1, Math.ceil((x1 - x0) / max));
  const rows = Math.max(1, Math.ceil((y1 - y0) / max));
  const w = Math.ceil((x1 - x0) / cols);
  const h = Math.ceil((y1 - y0) / rows);
  const tiles: Rect[] = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const x = x0 + col * w;
      const y = y0 + row * h;
      const tw = Math.min(w, x1 - x);
      const th = Math.min(h, y1 - y);
      if (tw > 0 && th > 0) tiles.push({ x, y, w: tw, h: th });
    }
  }
  return tiles;
}

const inside = (rect: Rect, x: number, y: number) =>
  x >= rect.x - 0.5 && x <= rect.x + rect.w + 0.5 && y >= rect.y - 0.5 && y <= rect.y + rect.h + 0.5;

/** Whether `rects` together cover `target`, checked on a 9 × 9 grid of points. */
export function covered(target: Rect, rects: Rect[]): boolean {
  if (target.w <= 0 || target.h <= 0) return true;
  if (rects.length === 0) return false;
  for (let i = 0; i <= 8; i += 1) {
    for (let j = 0; j <= 8; j += 1) {
      const x = target.x + (target.w * i) / 8;
      const y = target.y + (target.h * j) / 8;
      if (!rects.some((rect) => inside(rect, x, y))) return false;
    }
  }
  return true;
}

export function intersects(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

/** A label for a zoom: "Fit" or "100%". */
export function zoomLabel(zoom: number): string {
  return `${Math.round(zoom * 100)}%`;
}
