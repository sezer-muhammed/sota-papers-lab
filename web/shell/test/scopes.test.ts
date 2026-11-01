/* Unit tests for the Retouch scopes' pure maths. Node runs the TypeScript
   directly (type stripping): `pnpm --filter @spl/shell test`. */
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  BINS,
  HIGHLIGHT_LIMIT,
  PLOT_HEIGHT,
  SCOPE_CELLS,
  SHADOW_LIMIT,
  areaPath,
  clipLevel,
  computeScopes,
  dominantBand,
  fitWithin,
  formatLevel,
  formatShare,
  hslToRgb,
  hueBand,
  layerAt,
  linePath,
  lumaOf,
  peakOf,
  quantile,
  rgbLayers,
  scaled,
  scopeImage,
  smooth,
} from "../src/retouch/scopes/compute.ts";

/** RGBA bytes for a list of RGB pixels (alpha 255). */
function rgba(pixels: [number, number, number][]): Uint8ClampedArray {
  const out = new Uint8ClampedArray(pixels.length * 4);
  pixels.forEach(([r, g, b], i) => out.set([r, g, b, 255], i * 4));
  return out;
}

const sum = (values: ArrayLike<number>) => Array.from(values).reduce((a, b) => a + b, 0);
const close = (actual: number, expected: number, eps = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= eps, `${actual} is not within ${eps} of ${expected}`);

test("a flat grey image bins at one level, unclipped and neutral", () => {
  const s = computeScopes(rgba(Array.from({ length: 100 }, () => [128, 128, 128])));
  assert.equal(s.pixels, 100);
  for (const hist of [s.red, s.green, s.blue, s.luma]) {
    assert.equal(hist.length, BINS);
    assert.equal(hist[128], 100);
    assert.equal(sum(hist), 100);
  }
  close(s.meanLuma, 128 / 255);
  close(s.medianLuma, 128 / 255);
  assert.equal(s.shadows.any, 0);
  assert.equal(s.highlights.any, 0);
  assert.equal(s.meanChroma, 0);
  assert.deepEqual(s.bands, [0, 0, 0, 0, 0, 0, 0, 0]);
  assert.equal(dominantBand(s.bands), undefined);
  // Every pixel sits in the centre cell of the vectorscope.
  assert.equal(s.scopeMax, 100);
  assert.equal(s.scope[Math.floor(SCOPE_CELLS / 2) * SCOPE_CELLS + Math.floor(SCOPE_CELLS / 2)], 100);
});

test("an empty image gives zeros, not NaN", () => {
  const s = computeScopes(new Uint8ClampedArray(0));
  assert.equal(s.pixels, 0);
  assert.equal(s.meanLuma, 0);
  assert.equal(s.medianLuma, 0);
  assert.equal(s.shadows.any, 0);
  assert.equal(s.highlights.all, 0);
  assert.equal(s.scopeMax, 0);
});

test("clipping counts any channel and all channels, per side", () => {
  const s = computeScopes(
    rgba([
      [0, 0, 0], // pure black
      [255, 255, 255], // pure white
      [255, 10, 10], // red clipped only
      [128, 128, 128],
      [SHADOW_LIMIT, 200, 200], // at the shadow limit: counts
      [SHADOW_LIMIT + 1, 200, 200], // just above: doesn't
      [HIGHLIGHT_LIMIT - 1, 20, 20], // just below the highlight limit: doesn't
    ]),
  );
  const n = 7;
  close(s.shadows.any, 2 / n);
  close(s.shadows.all, 1 / n);
  assert.deepEqual(s.shadows.channels, [2 / n, 1 / n, 1 / n]);
  close(s.highlights.any, 2 / n);
  close(s.highlights.all, 1 / n);
  assert.deepEqual(s.highlights.channels, [2 / n, 1 / n, 1 / n]);
});

test("luma uses Rec. 709 weights and keeps white at 255", () => {
  assert.equal(lumaOf(255, 255, 255), 255);
  assert.equal(lumaOf(0, 0, 0), 0);
  assert.equal(lumaOf(255, 0, 0), 53);
  assert.equal(lumaOf(0, 255, 0), 182);
  assert.equal(lumaOf(0, 0, 255), 18);
  const s = computeScopes(rgba([[0, 0, 0], [255, 255, 255], [255, 255, 255]]));
  assert.equal(s.luma[0], 1);
  assert.equal(s.luma[255], 2);
  close(s.medianLuma, 1);
});

test("hue is the angle (red right, counter-clockwise) and chroma the radius", () => {
  const half = SCOPE_CELLS / 2;
  const cellOf = (pixel: [number, number, number]) => {
    const s = computeScopes(rgba([pixel]));
    const index = s.scope.findIndex((count) => count > 0);
    return { row: Math.floor(index / SCOPE_CELLS), col: index % SCOPE_CELLS };
  };
  const red = cellOf([255, 0, 0]);
  assert.deepEqual(red, { row: half, col: SCOPE_CELLS - 1 });
  const green = cellOf([0, 255, 0]); // 120°: up and left
  assert.ok(green.row < half && green.col < half, JSON.stringify(green));
  const blue = cellOf([0, 0, 255]); // 240°: down and left
  assert.ok(blue.row > half && blue.col < half, JSON.stringify(blue));
  const cyan = cellOf([0, 255, 255]); // 180°: left
  assert.deepEqual(cyan, { row: half, col: 0 });
  // Half the chroma lands half way out.
  const dullRed = cellOf([191, 64, 64]);
  assert.equal(dullRed.row, half);
  assert.ok(Math.abs(dullRed.col - (half + half / 2)) <= 1, JSON.stringify(dullRed));
});

test("colour falls into Lightroom's HSL bands, weighted by chroma", () => {
  const s = computeScopes(rgba([[255, 128, 0], [255, 128, 0], [0, 0, 255], [120, 120, 120]]));
  const band = dominantBand(s.bands);
  assert.equal(band?.name, "orange");
  close(band!.share, 2 / 3, 1e-3);
  close(sum(s.bands), 1);
  assert.equal(hueBand(0), "red");
  assert.equal(hueBand(350), "red");
  assert.equal(hueBand(-10), "red");
  assert.equal(hueBand(29), "orange");
  assert.equal(hueBand(100), "green");
  assert.equal(hueBand(200), "aqua");
  assert.equal(hueBand(262), "purple");
  assert.equal(hueBand(305), "magenta");
});

test("quantile finds the bin where the share is reached", () => {
  assert.equal(quantile([0, 2, 2], 0.5), 1);
  assert.equal(quantile([0, 2, 2], 0.51), 2);
  assert.equal(quantile([5, 0, 0], 0.5), 0);
  assert.equal(quantile([0, 0, 0], 0.5), 0);
});

test("smoothing blurs the interior and leaves the clipped ends alone", () => {
  assert.deepEqual(smooth([10, 0, 4, 0, 10]), [10, 1, 2, 1, 10]);
  assert.deepEqual(smooth([]), []);
  assert.deepEqual(smooth([3]), [3]);
});

test("the peak ignores the end bins unless that is all there is", () => {
  assert.equal(peakOf([[5, 1, 2, 9]]), 2);
  assert.equal(peakOf([[5, 1, 2, 9], [0, 3, 0, 0]]), 3);
  assert.equal(peakOf([[3, 0, 0, 4]]), 4);
  assert.equal(peakOf([[0, 0, 0]]), 1);
});

test("scales map to 0..1, clamp spikes, and log lifts the small bins", () => {
  assert.deepEqual(scaled([0, 0.5, 1, 2], 1, "linear"), [0, 0.5, 1, 1]);
  const log = scaled([0, 1e-5, 0.01, 0.02], 0.02, "log");
  assert.equal(log[0], 0);
  close(log[3]!, 1);
  assert.ok(log[1]! > 1e-5 / 0.02, "log should lift a sparse bin above its linear height");
  assert.ok(log[1]! < log[2]! && log[2]! < log[3]!, "log stays monotone");
});

test("painting the RGB layers in order leaves the right overlap colour on top", () => {
  // A small deterministic generator, so a failure reproduces.
  let seed = 7;
  const next = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const n = 64;
  const r = Array.from({ length: n }, next);
  const g = Array.from({ length: n }, next);
  const b = Array.from({ length: n }, next);
  const layers = rgbLayers(r, g, b);
  assert.deepEqual(
    layers.map((l) => l.layer),
    ["red", "green", "blue", "yellow", "cyan", "magenta", "neutral"],
  );
  for (let i = 0; i < n; i++) {
    for (const height of [0.05, 0.25, 0.5, 0.75, 0.95]) {
      let top: string | undefined;
      for (const { layer, heights } of layers) if (heights[i]! >= height) top = layer;
      assert.equal(top, layerAt(r[i]!, g[i]!, b[i]!, height), `bin ${i} at ${height}`);
    }
  }
});

test("paths cover the full width in viewBox units", () => {
  const full = Array.from({ length: BINS }, () => 1);
  const area = areaPath(full);
  assert.ok(area.startsWith(`M0 ${PLOT_HEIGHT}L0 0L0.5 0`), area.slice(0, 30));
  assert.ok(area.endsWith(`L${BINS} 0L${BINS} ${PLOT_HEIGHT}Z`), area.slice(-30));
  assert.equal(area.split("L").length - 1, BINS + 3);
  const line = linePath([0, 0.5]);
  assert.equal(line, `M0 ${PLOT_HEIGHT}L64 ${PLOT_HEIGHT}L192 50L${BINS} 50`);
  assert.equal(areaPath([]), "");
});

test("clip levels and share formatting", () => {
  assert.equal(clipLevel(0), "none");
  assert.equal(clipLevel(0.0009), "none");
  assert.equal(clipLevel(0.001), "some");
  assert.equal(clipLevel(0.0099), "some");
  assert.equal(clipLevel(0.01), "heavy");
  assert.equal(clipLevel(Number.NaN), "none");
  assert.equal(formatShare(0), "0%");
  assert.equal(formatShare(0.00005), "<0.01%");
  assert.equal(formatShare(0.0042), "0.42%");
  assert.equal(formatShare(0.034), "3.4%");
  assert.equal(formatShare(0.234), "23%");
  assert.equal(formatLevel(0.424), "42%");
  assert.equal(formatLevel(1.2), "100%");
});

test("vectorscope pixels: hue colours, transparent where empty", () => {
  assert.deepEqual(hslToRgb(0, 1, 0.5), [255, 0, 0]);
  assert.deepEqual(hslToRgb(120, 1, 0.5), [0, 255, 0]);
  assert.deepEqual(hslToRgb(240, 1, 0.5), [0, 0, 255]);
  assert.deepEqual(hslToRgb(0, 0, 0.5), [128, 128, 128]);

  const s = computeScopes(rgba([[128, 128, 128], [255, 0, 0], [255, 0, 0]]));
  const image = scopeImage(s.scope, s.cells, s.scopeMax);
  assert.equal(image.length, s.cells * s.cells * 4);
  const at = (row: number, col: number) => Array.from(image.slice((row * s.cells + col) * 4, (row * s.cells + col) * 4 + 4));
  const half = s.cells / 2;
  const [cr, cg, cb, ca] = at(half, half);
  assert.ok(Math.abs(cr! - cg!) < 8 && Math.abs(cg! - cb!) < 8, "the centre is near grey");
  assert.ok(ca! > 0);
  const [rr, rg, , ra] = at(half, s.cells - 1);
  assert.ok(rr! > 200 && rg! < 60, "the right edge is red");
  assert.equal(ra, 255, "the densest cell is opaque");
  assert.equal(at(0, 0)[3], 0, "an empty cell is transparent");
  assert.equal(sum(scopeImage(new Uint32Array(16), 4, 0)), 0);
});

test("samples fit the long edge and never upscale", () => {
  assert.deepEqual(fitWithin(6000, 4000, 512), { width: 512, height: 341 });
  assert.deepEqual(fitWithin(4000, 6000, 512), { width: 341, height: 512 });
  assert.deepEqual(fitWithin(300, 200, 512), { width: 300, height: 200 });
  assert.deepEqual(fitWithin(0, 200, 512), { width: 0, height: 0 });
});

test("a 512 px sample computes in well under a frame budget", () => {
  const width = 512;
  const height = 341;
  const data = new Uint8ClampedArray(width * height * 4);
  let seed = 1;
  for (let i = 0; i < data.length; i++) {
    seed = (seed * 16807) % 2147483647;
    data[i] = seed & 255;
  }
  computeScopes(data); // warm up
  const start = performance.now();
  const s = computeScopes(data);
  const ms = performance.now() - start;
  assert.equal(s.pixels, width * height);
  assert.equal(sum(s.red), width * height);
  assert.equal(sum(s.scope), width * height);
  // Generous for slow CI machines; typically a few milliseconds.
  assert.ok(ms < 200, `took ${ms.toFixed(1)} ms`);
  console.log(`computeScopes on ${width}×${height}: ${ms.toFixed(1)} ms`);
});
