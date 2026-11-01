/* Unit tests for the zoomable viewer's geometry: `pnpm --filter @spl/shell test`. */
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  FIT,
  MAX_TILE,
  canPan,
  covered,
  fitZoom,
  framePoint,
  imagePoint,
  needsTiles,
  panBy,
  place,
  stepZoom,
  tilesFor,
  visibleRegion,
  zoomAround,
  type View,
} from "../src/retouch/zoom.ts";

const image = { width: 6000, height: 4000 };
const frame = { width: 900, height: 600 };

test("Fit shows the whole photo, centred", () => {
  const p = place(FIT, frame, image, 2);
  assert.equal(p.zoom, 0.3);
  assert.equal(p.scale, 0.15);
  assert.deepEqual([p.cx, p.cy], [3000, 2000]);
  assert.deepEqual(visibleRegion(p, frame, image), { x: 0, y: 0, w: 6000, h: 4000 });
  assert.equal(canPan(p, frame, image), false);
  assert.equal(fitZoom(frame, image, 1), 0.15);
});

test("100% is one photo pixel per device pixel", () => {
  const view: View = { fit: false, zoom: 1, cx: 3000, cy: 2000 };
  const one = place(view, frame, image, 1);
  assert.equal(one.scale, 1);
  assert.deepEqual(visibleRegion(one, frame, image), { x: 2550, y: 1700, w: 900, h: 600 });
  const retina = place(view, frame, image, 2);
  assert.equal(retina.scale, 0.5);
  assert.deepEqual(visibleRegion(retina, frame, image), { x: 2100, y: 1400, w: 1800, h: 1200 });
});

test("the centre stays where the photo covers the frame", () => {
  const p = place({ fit: false, zoom: 1, cx: 0, cy: 99999 }, frame, image, 1);
  assert.deepEqual([p.cx, p.cy], [450, 3700]);
  assert.ok(canPan(p, frame, image));
});

test("zooming keeps the point under the cursor in place", () => {
  const start = place({ fit: false, zoom: 1, cx: 3000, cy: 2000 }, frame, image, 1);
  const cursor = { x: 200, y: 150 };
  const under = imagePoint(start, frame, cursor);
  const view = zoomAround(start, 4, frame, image, 1, cursor);
  assert.equal(view.fit, false);
  const after = place(view, frame, image, 1);
  const back = framePoint(after, frame, under);
  assert.ok(Math.abs(back.x - cursor.x) < 1e-6 && Math.abs(back.y - cursor.y) < 1e-6);
  // Zooming to Fit's own value is Fit; past 800% stops at 800%.
  assert.deepEqual(zoomAround(start, fitZoom(frame, image, 1), frame, image, 1), FIT);
  const max = zoomAround(start, 50, frame, image, 1);
  assert.equal(max.fit ? 0 : max.zoom, 8);
});

test("dragging pans by the drag over the scale", () => {
  const p = place({ fit: false, zoom: 2, cx: 3000, cy: 2000 }, frame, image, 1);
  const moved = panBy(p, 100, -50);
  assert.deepEqual(moved, { fit: false, zoom: 2, cx: 2950, cy: 2025 });
});

test("zoom steps include Fit", () => {
  assert.equal(stepZoom(0.3, 0.3, 1), 0.5);
  assert.equal(stepZoom(1, 0.3, 1), 2);
  assert.equal(stepZoom(1, 0.3, -1), 0.5);
  assert.equal(stepZoom(0.5, 0.3, -1), 0.3);
  assert.equal(stepZoom(8, 0.3, 1), 8);
});

test("crops are asked for only past the preview's resolution", () => {
  const preview = { width: 2048, height: 1365 };
  assert.equal(needsTiles(place(FIT, frame, image, 2), preview, image), false);
  assert.equal(needsTiles(place({ fit: false, zoom: 1, cx: 3000, cy: 2000 }, frame, image, 2), preview, image), true);
});

test("crops cover the view, fit the host's limit and stay in the photo", () => {
  const small = { x: 3000, y: 2000, w: 450, h: 300 };
  const tiles = tilesFor(small, image);
  assert.equal(tiles.length, 1);
  assert.ok(covered(small, tiles));
  const big = { x: 100, y: 50, w: 5000, h: 3000 };
  const many = tilesFor(big, image);
  assert.ok(many.length > 1);
  assert.ok(covered(big, many));
  for (const t of many) {
    assert.ok(t.w <= MAX_TILE && t.h <= MAX_TILE);
    assert.ok(t.x >= 0 && t.y >= 0 && t.x + t.w <= image.width && t.y + t.h <= image.height);
    assert.ok(Number.isInteger(t.x) && Number.isInteger(t.y) && Number.isInteger(t.w) && Number.isInteger(t.h));
  }
  assert.equal(covered(big, [tiles[0]!]), false);
});
