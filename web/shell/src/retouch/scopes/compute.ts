/* Scopes for an 8-bit sRGB image, computed in the browser from the preview
   JPEGs the page already shows: RGB and luma histograms, clipping, luma
   statistics, and a hue and saturation density for the vectorscope. Pure
   and DOM-free (no imports), so `node --test web/shell/test` runs it as is. */

export const BINS = 256;
/** Cells across the vectorscope's diameter. */
export const SCOPE_CELLS = 128;
/** Code values that count as clipped. One step inside the limit, because
    JPEG rounding turns a clipped 255 into 254 (and 0 into 1) here and there. */
export const SHADOW_LIMIT = 1;
export const HIGHLIGHT_LIMIT = 254;
/** Below this chroma a pixel is neutral: it has no hue band. */
export const NEUTRAL_CHROMA = 0.08;

/** Lightroom's HSL / Color Mixer bands and their centre hues in degrees. */
export const HUE_BANDS = [
  { name: "red", hue: 0 },
  { name: "orange", hue: 30 },
  { name: "yellow", hue: 60 },
  { name: "green", hue: 120 },
  { name: "aqua", hue: 180 },
  { name: "blue", hue: 240 },
  { name: "purple", hue: 270 },
  { name: "magenta", hue: 300 },
] as const;

export type HueBand = (typeof HUE_BANDS)[number]["name"];

export type ClipStats = {
  /** Share of pixels with at least one channel at the limit. */
  any: number;
  /** Share with all three at the limit: pure black, or pure white. */
  all: number;
  /** Share per channel, red, green, blue. */
  channels: [number, number, number];
};

export type Scopes = {
  pixels: number;
  red: Uint32Array;
  green: Uint32Array;
  blue: Uint32Array;
  /** Rec. 709 luma of the encoded values, as Lightroom's readouts use. */
  luma: Uint32Array;
  shadows: ClipStats;
  highlights: ClipStats;
  /** 0 to 1. */
  meanLuma: number;
  medianLuma: number;
  /** Mean chroma (max − min channel), 0 to 1. */
  meanChroma: number;
  /** Chroma-weighted share of each HUE_BANDS entry; all 0 for a neutral image. */
  bands: number[];
  /** Pixel counts on a cells × cells grid, row 0 at the top: hue is the angle
      (red to the right, yellow up and right, counter-clockwise), chroma the
      distance from the centre. */
  scope: Uint32Array;
  cells: number;
  scopeMax: number;
};

/* Hue at quarter-degree steps: cos/sin and the band for each, so the pixel
   loop does no trigonometry. */
const ANGLE_STEPS = 1440;
const COS = new Float64Array(ANGLE_STEPS);
const SIN = new Float64Array(ANGLE_STEPS);
const BAND_OF = new Uint8Array(ANGLE_STEPS);
// sin(180°) is 1e-16, not 0: snap, so the axes land in the middle cells.
const snap = (value: number) => (Math.abs(value) < 1e-12 ? 0 : value);
for (let k = 0; k < ANGLE_STEPS; k++) {
  const degrees = (k * 360) / ANGLE_STEPS;
  COS[k] = snap(Math.cos((degrees * Math.PI) / 180));
  SIN[k] = snap(Math.sin((degrees * Math.PI) / 180));
  BAND_OF[k] = nearestBand(degrees);
}

function nearestBand(degrees: number): number {
  let best = 0;
  let bestDistance = Infinity;
  HUE_BANDS.forEach((band, index) => {
    const d = Math.abs(degrees - band.hue) % 360;
    const distance = Math.min(d, 360 - d);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}

/** The band a hue (degrees) falls in. */
export function hueBand(degrees: number): HueBand {
  const wrapped = ((degrees % 360) + 360) % 360;
  return HUE_BANDS[nearestBand(wrapped)]!.name;
}

/** Rec. 709 luma in integer arithmetic (the weights sum to 256, so white stays 255). */
export const lumaOf = (r: number, g: number, b: number) => (54 * r + 183 * g + 19 * b) >> 8;

/** Scopes of RGBA pixels (alpha ignored), e.g. ImageData.data. */
export function computeScopes(rgba: ArrayLike<number>, cells = SCOPE_CELLS): Scopes {
  const red = new Uint32Array(BINS);
  const green = new Uint32Array(BINS);
  const blue = new Uint32Array(BINS);
  const luma = new Uint32Array(BINS);
  const scope = new Uint32Array(cells * cells);
  const bandWeight = new Float64Array(HUE_BANDS.length);
  const low = [0, 0, 0];
  const high = [0, 0, 0];
  let lowAny = 0;
  let lowAll = 0;
  let highAny = 0;
  let highAll = 0;
  let lumaSum = 0;
  let chromaSum = 0;
  const half = cells / 2;
  const centre = Math.floor(half) * cells + Math.floor(half);
  const pixels = Math.floor(rgba.length / 4);

  for (let i = 0; i < pixels; i++) {
    const o = i * 4;
    const r = rgba[o]!;
    const g = rgba[o + 1]!;
    const b = rgba[o + 2]!;
    red[r]! += 1;
    green[g]! += 1;
    blue[b]! += 1;
    const y = lumaOf(r, g, b);
    luma[y]! += 1;
    lumaSum += y;

    const lr = r <= SHADOW_LIMIT;
    const lg = g <= SHADOW_LIMIT;
    const lb = b <= SHADOW_LIMIT;
    if (lr || lg || lb) {
      lowAny++;
      if (lr && lg && lb) lowAll++;
      if (lr) low[0]! += 1;
      if (lg) low[1]! += 1;
      if (lb) low[2]! += 1;
    }
    const hr = r >= HIGHLIGHT_LIMIT;
    const hg = g >= HIGHLIGHT_LIMIT;
    const hb = b >= HIGHLIGHT_LIMIT;
    if (hr || hg || hb) {
      highAny++;
      if (hr && hg && hb) highAll++;
      if (hr) high[0]! += 1;
      if (hg) high[1]! += 1;
      if (hb) high[2]! += 1;
    }

    const max = r > g ? (r > b ? r : b) : g > b ? g : b;
    const min = r < g ? (r < b ? r : b) : g < b ? g : b;
    const c = max - min;
    if (c === 0) {
      scope[centre]! += 1;
      continue;
    }
    // HSV hue in sixths of a turn.
    let h = max === r ? (g - b) / c : max === g ? (b - r) / c + 2 : (r - g) / c + 4;
    if (h < 0) h += 6;
    const k = Math.round((h * ANGLE_STEPS) / 6) % ANGLE_STEPS;
    const radius = c / 255;
    chromaSum += radius;
    if (radius >= NEUTRAL_CHROMA) bandWeight[BAND_OF[k]!]! += radius;
    const col = Math.min(cells - 1, Math.floor((1 + radius * COS[k]!) * half));
    const row = Math.min(cells - 1, Math.floor((1 - radius * SIN[k]!) * half));
    scope[row * cells + col]! += 1;
  }

  const share = (count: number) => (pixels > 0 ? count / pixels : 0);
  const clip = (any: number, all: number, channels: number[]): ClipStats => ({
    any: share(any),
    all: share(all),
    channels: [share(channels[0]!), share(channels[1]!), share(channels[2]!)],
  });
  const bandTotal = bandWeight.reduce((sum, w) => sum + w, 0);
  let scopeMax = 0;
  for (const count of scope) if (count > scopeMax) scopeMax = count;

  return {
    pixels,
    red,
    green,
    blue,
    luma,
    shadows: clip(lowAny, lowAll, low),
    highlights: clip(highAny, highAll, high),
    meanLuma: pixels > 0 ? lumaSum / pixels / 255 : 0,
    medianLuma: quantile(luma, 0.5) / 255,
    meanChroma: share(chromaSum),
    bands: Array.from(bandWeight, (w) => (bandTotal > 0 ? w / bandTotal : 0)),
    scope,
    cells,
    scopeMax,
  };
}

/** The smallest bin at or below which `q` of the counts fall (0 when empty). */
export function quantile(hist: ArrayLike<number>, q: number): number {
  let total = 0;
  for (let i = 0; i < hist.length; i++) total += hist[i]!;
  if (total === 0) return 0;
  const target = q * total;
  let running = 0;
  for (let i = 0; i < hist.length; i++) {
    running += hist[i]!;
    if (running >= target) return i;
  }
  return hist.length - 1;
}

/** The band with the most colour, and its share; none for a neutral image. */
export function dominantBand(bands: number[]): { name: HueBand; share: number } | undefined {
  let best = -1;
  bands.forEach((share, index) => {
    if (share > 0 && (best < 0 || share > bands[best]!)) best = index;
  });
  return best >= 0 ? { name: HUE_BANDS[best]!.name, share: bands[best]! } : undefined;
}

/* ---- Histogram drawing: pure geometry for an SVG with viewBox 0 0 256 100. */

export const PLOT_HEIGHT = 100;
export type HistogramScale = "linear" | "log";

/** Counts as shares of the pixels. */
export function shares(hist: ArrayLike<number>, pixels: number): number[] {
  return Array.from(hist, (count) => (pixels > 0 ? count / pixels : 0));
}

/** A light [1, 2, 1] blur for display. The end bins (pure black and white,
    where clipping piles up) neither blur nor bleed into their neighbours. */
export function smooth(values: number[]): number[] {
  const last = values.length - 1;
  return values.map((value, i) => {
    if (i === 0 || i === last) return value;
    const left = i - 1 === 0 ? value : values[i - 1]!;
    const right = i + 1 === last ? value : values[i + 1]!;
    return (left + 2 * value + right) / 4;
  });
}

/** The value the plot's top stands for: the tallest bin of any series,
    leaving out the end bins so a clipping spike doesn't flatten the rest
    (it is drawn to the top instead). Never 0. */
export function peakOf(series: number[][]): number {
  let interior = 0;
  let any = 0;
  for (const values of series) {
    values.forEach((value, i) => {
      if (value > any) any = value;
      if (i > 0 && i < values.length - 1 && value > interior) interior = value;
    });
  }
  return interior > 0 ? interior : any > 0 ? any : 1;
}

/** Shares scaled to 0..1 of the plot height. Log compresses the tall bins so
    the sparse ends (a few clipped pixels) stay visible. */
export function scaled(values: number[], peak: number, scale: HistogramScale): number[] {
  if (scale === "linear") return values.map((v) => Math.min(1, v / peak));
  // About one pixel of a 512 px sample per unit inside the log.
  const gain = 1e5;
  const top = Math.log1p(peak * gain);
  return values.map((v) => Math.min(1, Math.log1p(v * gain) / top));
}

const round2 = (value: number) => Math.round(value * 100) / 100;

function points(heights: number[]): string[] {
  const n = heights.length;
  const width = BINS;
  const y = (h: number) => round2(PLOT_HEIGHT * (1 - Math.max(0, Math.min(1, h))));
  if (n === 0) return [];
  const step = width / n;
  const out = [`0 ${y(heights[0]!)}`];
  heights.forEach((h, i) => out.push(`${round2((i + 0.5) * step)} ${y(h)}`));
  out.push(`${width} ${y(heights[n - 1]!)}`);
  return out;
}

/** A closed area under the curve through the bin centres. */
export function areaPath(heights: number[]): string {
  const pts = points(heights);
  if (pts.length === 0) return "";
  return `M0 ${PLOT_HEIGHT}L${pts.join("L")}L${BINS} ${PLOT_HEIGHT}Z`;
}

/** The curve alone, for an outline. */
export function linePath(heights: number[]): string {
  const pts = points(heights);
  return pts.length ? `M${pts.join("L")}` : "";
}

export type Layer = "red" | "green" | "blue" | "yellow" | "cyan" | "magenta" | "neutral";

/** Lightroom's overlapping channels as opaque layers, painted in order: each
    channel, then each pair's overlap (the area under the lower of the two),
    then the area all three share. Where exactly one channel reaches, its own
    colour shows; where two, their mix; where all three, the neutral. */
export function rgbLayers(r: number[], g: number[], b: number[]): { layer: Layer; heights: number[] }[] {
  const min2 = (a: number[], c: number[]) => a.map((v, i) => Math.min(v, c[i]!));
  const rg = min2(r, g);
  return [
    { layer: "red", heights: r },
    { layer: "green", heights: g },
    { layer: "blue", heights: b },
    { layer: "yellow", heights: rg },
    { layer: "cyan", heights: min2(g, b) },
    { layer: "magenta", heights: min2(r, b) },
    { layer: "neutral", heights: min2(rg, b) },
  ];
}

/** The colour showing at a point of the RGB plot: the one the painter order
    leaves on top (undefined above every channel). For tests and readouts. */
export function layerAt(r: number, g: number, b: number, height: number): Layer | undefined {
  const inR = r >= height;
  const inG = g >= height;
  const inB = b >= height;
  if (inR && inG && inB) return "neutral";
  if (inR && inB) return "magenta";
  if (inG && inB) return "cyan";
  if (inR && inG) return "yellow";
  if (inB) return "blue";
  if (inG) return "green";
  if (inR) return "red";
  return undefined;
}

/* ---- Readouts. */

export type ClipLevel = "none" | "some" | "heavy";

/** Under 0.1 % reads as none; from 1 % it is heavy. */
export function clipLevel(share: number): ClipLevel {
  if (!(share >= 0.001)) return "none";
  return share >= 0.01 ? "heavy" : "some";
}

/** "0%", "<0.01%", "0.42%", "3.4%", "23%". */
export function formatShare(share: number): string {
  if (!Number.isFinite(share) || share <= 0) return "0%";
  const percent = share * 100;
  if (percent < 0.01) return "<0.01%";
  if (percent < 1) return `${percent.toFixed(2)}%`;
  if (percent < 10) return `${percent.toFixed(1)}%`;
  return `${Math.round(percent)}%`;
}

/** A 0..1 level as a whole percentage: "42%". */
export const formatLevel = (value: number) => `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;

/* ---- Vectorscope image. */

/** RGB (0..255) of an HSL colour; h in degrees, s and l 0..1. */
export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [f(0), f(8), f(4)];
}

/** Dot lightness per theme: darker on a light surface, lighter on a dark one. */
export const SCOPE_LIGHTNESS = { light: 0.42, dark: 0.6 } as const;

/** RGBA pixels for the vectorscope grid: each occupied cell in its own hue,
    saturated by its distance from the centre, with opacity rising with the
    log of its count (sparse cells stay visible). Empty cells stay
    transparent, so the surface shows through. */
export function scopeImage(
  scope: ArrayLike<number>,
  cells: number,
  max: number,
  lightness: number = SCOPE_LIGHTNESS.light,
): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(cells * cells * 4);
  if (max <= 0) return out;
  const half = cells / 2;
  const top = Math.log1p(max);
  for (let row = 0; row < cells; row++) {
    for (let col = 0; col < cells; col++) {
      const count = scope[row * cells + col]!;
      if (count === 0) continue;
      const x = (col + 0.5) / half - 1;
      const y = 1 - (row + 0.5) / half;
      const radius = Math.min(1, Math.hypot(x, y));
      const hue = ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
      const [r, g, b] = hslToRgb(hue, Math.min(1, radius * 1.6), lightness);
      const o = (row * cells + col) * 4;
      out[o] = r;
      out[o + 1] = g;
      out[o + 2] = b;
      out[o + 3] = Math.round(255 * (0.4 + 0.6 * Math.pow(Math.log1p(count) / top, 0.7)));
    }
  }
  return out;
}

/** Width and height that fit `edge` on the long side, never upscaled. */
export function fitWithin(width: number, height: number, edge: number): { width: number; height: number } {
  if (!(width > 0 && height > 0)) return { width: 0, height: 0 };
  const factor = Math.min(1, edge / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * factor)), height: Math.max(1, Math.round(height * factor)) };
}
