/* The parametric Tone Curve as the panel draws it: Lightroom's Region mode,
   four sliders over input regions split at 25, 50 and 75 % (the app keeps
   Lightroom's default splits). Each slider lifts or lowers the curve around
   its region's centre and a monotone cubic joins the nodes, with 0 and 1
   pinned, which is how Lightroom's curve behaves. It is the panel's picture
   of the settings, not the renderer's exact curve. Import-free, so
   `node --test web/shell/test` runs it as is. */

/** Lightroom's default region splits, as fractions of the input range. */
export const CURVE_SPLITS = [0.25, 0.5, 0.75] as const;

export type CurveRegion = "shadows" | "darks" | "lights" | "highlights";

/** The four regions from dark to light, with their input span. */
export const CURVE_REGIONS: { region: CurveRegion; from: number; to: number }[] = [
  { region: "shadows", from: 0, to: CURVE_SPLITS[0] },
  { region: "darks", from: CURVE_SPLITS[0], to: CURVE_SPLITS[1] },
  { region: "lights", from: CURVE_SPLITS[1], to: CURVE_SPLITS[2] },
  { region: "highlights", from: CURVE_SPLITS[2], to: 1 },
];

/** Slider values, -100 to 100. */
export type CurveSliders = Record<CurveRegion, number>;

/* How far a slider at ±100 moves the curve at its region's centre. Smaller
   for the outer regions, whose ends are pinned; small enough everywhere that
   the nodes stay in increasing order whatever the sliders say. */
const REACH: Record<CurveRegion, number> = { shadows: 0.09, darks: 0.12, lights: 0.12, highlights: 0.09 };

/** The curve's nodes: the ends, and each region's centre moved by its slider. */
export function curveNodes(sliders: CurveSliders): [number, number][] {
  const nodes: [number, number][] = [[0, 0]];
  for (const { region, from, to } of CURVE_REGIONS) {
    const x = (from + to) / 2;
    const amount = Math.max(-100, Math.min(100, Number.isFinite(sliders[region]) ? sliders[region] : 0)) / 100;
    nodes.push([x, x + REACH[region] * amount]);
  }
  nodes.push([1, 1]);
  return nodes;
}

/** A monotone cubic (PCHIP) through increasing nodes, clamped to [0, 1]. */
export function monotoneCubic(nodes: [number, number][]): (x: number) => number {
  const n = nodes.length;
  const xs = nodes.map((node) => node[0]);
  const ys = nodes.map((node) => node[1]);
  const h: number[] = [];
  const slope: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    h.push(xs[i + 1]! - xs[i]!);
    slope.push((ys[i + 1]! - ys[i]!) / h[i]!);
  }
  const tangent = new Array<number>(n).fill(0);
  tangent[0] = slope[0]!;
  tangent[n - 1] = slope[n - 2]!;
  for (let i = 1; i < n - 1; i++) {
    const a = slope[i - 1]!;
    const b = slope[i]!;
    if (a * b <= 0) continue;
    // Fritsch–Butland weighted harmonic mean: keeps each piece monotone.
    const w1 = 2 * h[i]! + h[i - 1]!;
    const w2 = h[i]! + 2 * h[i - 1]!;
    tangent[i] = (w1 + w2) / (w1 / a + w2 / b);
  }
  return (x: number) => {
    const t = Math.max(0, Math.min(1, x));
    let i = 0;
    while (i < n - 2 && t > xs[i + 1]!) i++;
    const u = (t - xs[i]!) / h[i]!;
    const u2 = u * u;
    const u3 = u2 * u;
    const y =
      (2 * u3 - 3 * u2 + 1) * ys[i]! +
      (u3 - 2 * u2 + u) * h[i]! * tangent[i]! +
      (-2 * u3 + 3 * u2) * ys[i + 1]! +
      (u3 - u2) * h[i]! * tangent[i + 1]!;
    return Math.max(0, Math.min(1, y));
  };
}

/** The curve for these sliders, input to output, both 0 to 1. */
export function toneCurve(sliders: CurveSliders): (x: number) => number {
  return monotoneCubic(curveNodes(sliders));
}

/** An SVG path of the curve in a `size` square, y pointing down. */
export function curvePath(sliders: CurveSliders, size = 100, samples = 64): string {
  const curve = toneCurve(sliders);
  const points: string[] = [];
  for (let k = 0; k <= samples; k++) {
    const x = k / samples;
    points.push(`${(x * size).toFixed(2)} ${((1 - curve(x)) * size).toFixed(2)}`);
  }
  return `M${points.join("L")}`;
}
