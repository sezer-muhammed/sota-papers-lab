/* Unit tests for the Tone Curve panel's preview maths. Node runs the
   TypeScript directly (type stripping): `pnpm --filter @spl/shell test`. */
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  CURVE_REGIONS,
  curveNodes,
  curvePath,
  monotoneCubic,
  toneCurve,
  type CurveSliders,
} from "../src/retouch/develop/tone-curve.ts";

const flat: CurveSliders = { shadows: 0, darks: 0, lights: 0, highlights: 0 };
const grid = Array.from({ length: 101 }, (_, k) => k / 100);

test("neutral sliders draw the identity", () => {
  const curve = toneCurve(flat);
  for (const x of grid) assert.ok(Math.abs(curve(x) - x) < 1e-9, `x=${x}`);
});

test("the ends stay pinned whatever the sliders say", () => {
  for (const v of [-100, 100]) {
    const curve = toneCurve({ shadows: v, darks: v, lights: v, highlights: v });
    assert.equal(curve(0), 0);
    assert.equal(curve(1), 1);
  }
});

test("each slider moves its own region the way it says", () => {
  for (const { region, from, to } of CURVE_REGIONS) {
    const centre = (from + to) / 2;
    const up = toneCurve({ ...flat, [region]: 100 });
    const down = toneCurve({ ...flat, [region]: -100 });
    assert.ok(up(centre) > centre + 0.05, `${region} +100 lifts its centre`);
    assert.ok(down(centre) < centre - 0.05, `${region} -100 lowers its centre`);
    // Opposite ends of the range move far less than the region itself.
    const far = centre < 0.5 ? 0.9 : 0.1;
    assert.ok(Math.abs(up(far) - far) < Math.abs(up(centre) - centre) / 2, `${region} stays local`);
  }
});

test("the curve never turns back, even at the extremes", () => {
  const extremes = [-100, 100];
  for (const shadows of extremes)
    for (const darks of extremes)
      for (const lights of extremes)
        for (const highlights of extremes) {
          const curve = toneCurve({ shadows, darks, lights, highlights });
          let last = -Infinity;
          for (let k = 0; k <= 1000; k++) {
            const y = curve(k / 1000);
            assert.ok(y >= last - 1e-12, `monotone at ${k / 1000} for ${[shadows, darks, lights, highlights]}`);
            assert.ok(y >= 0 && y <= 1);
            last = y;
          }
        }
});

test("nodes stay in order and out-of-range or missing values are clamped", () => {
  const nodes = curveNodes({ shadows: 500, darks: -500, lights: Number.NaN, highlights: 0 });
  for (let i = 1; i < nodes.length; i++) assert.ok(nodes[i]![1] > nodes[i - 1]![1]);
  assert.deepEqual(nodes[3], [0.625, 0.625]);
  assert.deepEqual(curveNodes({ ...flat, shadows: 100 })[1], curveNodes({ ...flat, shadows: 500 })[1]);
});

test("the monotone cubic passes through its nodes", () => {
  const nodes: [number, number][] = [
    [0, 0],
    [0.3, 0.5],
    [0.6, 0.55],
    [1, 1],
  ];
  const f = monotoneCubic(nodes);
  for (const [x, y] of nodes) assert.ok(Math.abs(f(x) - y) < 1e-9);
});

test("the SVG path spans the square with y pointing down", () => {
  const d = curvePath(flat, 100, 4);
  assert.equal(d, "M0.00 100.00L25.00 75.00L50.00 50.00L75.00 25.00L100.00 0.00");
});
