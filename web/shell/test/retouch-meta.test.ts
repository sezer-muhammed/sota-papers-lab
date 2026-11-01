/* Unit tests for the photo card's date and place and the edit instructions:
   `pnpm --filter @spl/shell test`. */
import assert from "node:assert/strict";
import { test } from "node:test";
import { formatCaptured, parseCaptured, placeLine } from "../src/retouch/format.ts";
import {
  DEFAULT_OPTIONS,
  autoInput,
  cleanInstruction,
  normalizeOptions,
  resolveInstruction,
} from "../src/retouch/options.ts";

test("capture times read as the camera's clock, with or without an offset", () => {
  const iso = parseCaptured("2026-11-01T14:32:05+03:00");
  assert.equal(iso?.wall.toISOString(), "2026-11-01T14:32:05.000Z");
  assert.equal(iso?.offset, "+03:00");
  assert.equal(parseCaptured("2026:11:01 14:32:05")?.wall.toISOString(), "2026-11-01T14:32:05.000Z");
  assert.equal(parseCaptured("2026-11-01T14:32")?.offset, undefined);
  assert.equal(parseCaptured("0000:00:00 00:00:00"), undefined);
  assert.equal(parseCaptured("yesterday"), undefined);
  assert.equal(parseCaptured(null), undefined);
});

test("capture times format in the browser's language", () => {
  assert.equal(formatCaptured("2026-11-01T14:32:05+03:00", "tr-TR"), "1 Kasım 2026 14:32");
  assert.match(formatCaptured("2026:11:01 14:32:05", "en-GB") ?? "", /1 November 2026.*14:32/);
  assert.equal(formatCaptured(null), undefined);
});

test("the place is the looked-up name, else the coordinates", () => {
  assert.equal(placeLine({ place: "İzmir, Türkiye", gps: { lat: 38.42, lon: 27.14, alt_m: null } }), "İzmir, Türkiye");
  assert.equal(placeLine({ place: null, gps: { lat: 38.4237, lon: -27.1428, alt_m: 12 } }), "38.4237° N, 27.1428° W");
  assert.equal(placeLine({ place: null, gps: null }), undefined);
  // A host from before GPS: the fields are missing.
  assert.equal(placeLine({} as never), undefined);
});

test("a photo's own box wins over the default box; empty sends nothing", () => {
  assert.equal(cleanInstruction("   "), undefined);
  assert.equal(cleanInstruction("  siyah beyaz  "), "siyah beyaz");
  assert.equal(cleanInstruction("x".repeat(600))?.length, 500);
  assert.equal(resolveInstruction("", "high contrast"), "high contrast");
  assert.equal(resolveInstruction("black and white", "high contrast"), "black and white");
  assert.equal(resolveInstruction(undefined, " "), undefined);
  const input = autoInput(DEFAULT_OPTIONS, "raw", "a.ARW", " cinematic ");
  assert.equal(input.instruction, "cinematic");
  assert.equal("instruction" in autoInput(DEFAULT_OPTIONS, "raw", "a.ARW", ""), false);
});

test("PNG export is on by default, 8-bit unless 16 is chosen", () => {
  const input = autoInput(DEFAULT_OPTIONS, "raw", "a.ARW");
  assert.equal(input.export.png, true);
  assert.equal(input.export.png_bits, 8);
  assert.equal(normalizeOptions({ pngBits: 16 }).pngBits, 16);
  assert.equal(normalizeOptions({ pngBits: 12 }).pngBits, 8);
  // Retouches leave a host slot free: at most two run together.
  assert.equal(normalizeOptions({ parallel: 4 }).parallel, 2);
});
