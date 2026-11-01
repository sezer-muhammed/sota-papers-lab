/* Lightroom's parametric (Region) tone curve, for the Tone Curve panel's
   small picture: identity plus one smooth bump per region slider, anchored at
   black and white. It illustrates what the four sliders do; the renderer's
   own curve (apps/raw-retouch/src/render/tone_curve.rs) decides the pixels.
   Pure and DOM-free (no imports), so `node --test` runs it as is. */

/** The splits between the regions, fixed at Lightroom's defaults (25/50/75). */
export const CURVE_SPLITS = [0.25, 0.5, 0.75] as const;

/** The four region sliders, -100 to +100, as the Params fields hold them. */
export type CurveRegions = { shadows: number; darks: number; lights: number; highlights: number };

export type Region = keyof CurveRegions;

/* Each region lifts (or lowers) the curve with a raised-cosine bump over
   [lo, hi] that peaks at `peak`. Neighbouring regions overlap, as they do in
   Lightroom. `gain` is the lift at +100: no more than keeps one slider's
   curve rising (its steepest fall is gain * pi / 2 over the shorter side). */
type Bump = { lo: number; peak: number; hi: number; gain: number };

const bump = (lo: number, peak: number, hi: number): Bump => ({
  lo,
  peak,
  hi,
  gain: Math.min(0.16, (0.9 * 2 * Math.min(peak - lo, hi - peak)) / Math.PI),
});

const [S1, S2, S3] = CURVE_SPLITS;

export const CURVE_BUMPS: Record<Region, Bump> = {
  shadows: bump(0, S1 * 0.7, S2),
  darks: bump(0, (S1 + S2) / 2, S3),
  lights: bump(S1, (S2 + S3) / 2, 1),
  highlights: bump(S2, S3 + (1 - S3) * 0.3, 1),
};

export const REGIONS: Region[] = ["highlights", "lights", "darks", "shadows"];

function weight(b: Bump, x: number): number {
  if (x <= b.lo || x >= b.hi) return 0;
  const t = x <= b.peak ? (x - b.lo) / (b.peak - b.lo) : (b.hi - x) / (b.hi - b.peak);
  return (1 - Math.cos(Math.PI * t)) / 2;
}

/** The curve at input `x` (0 black to 1 white, perceptual), clamped to [0, 1]. */
export function curveAt(r: CurveRegions, x: number): number {
  let y = x;
  for (const region of REGIONS) {
    const amount = Number.isFinite(r[region]) ? Math.max(-100, Math.min(100, r[region])) / 100 : 0;
    if (amount !== 0) y += amount * CURVE_BUMPS[region].gain * weight(CURVE_BUMPS[region], x);
  }
  return Math.min(1, Math.max(0, y));
}

const round = (v: number) => Math.round(v * 100) / 100;

/** `n + 1` points of the curve in an SVG box `size` wide and tall (y down). */
export function curvePoints(r: CurveRegions, size: number, n = 64): [number, number][] {
  return Array.from({ length: n + 1 }, (_, i) => {
    const x = i / n;
    return [round(x * size), round((1 - curveAt(r, x)) * size)];
  });
}

/** The curve as an SVG path in a `size` box. */
export function curvePath(r: CurveRegions, size: number, n = 64): string {
  return curvePoints(r, size, n)
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${x} ${y}`)
    .join(" ");
}

/** Where one region's slider can take the curve, the others as they are:
    the closed area between its -100 and +100 curves (Lightroom shades it
    while that slider is in use). */
export function regionRangePath(r: CurveRegions, region: Region, size: number, n = 64): string {
  const up = curvePoints({ ...r, [region]: 100 }, size, n);
  const down = curvePoints({ ...r, [region]: -100 }, size, n).reverse();
  return [...up, ...down].map(([x, y], i) => `${i === 0 ? "M" : "L"}${x} ${y}`).join(" ") + " Z";
}

/** Whether every region slider is at zero (the curve is the identity). */
export const isFlat = (r: CurveRegions) => REGIONS.every((region) => Math.round(r[region]) === 0);
