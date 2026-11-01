/* Unit tests for the Develop settings helpers that read the generated
   PARAM_SPECS table: `pnpm --filter @spl/shell test`. */
import assert from "node:assert/strict";
import { test } from "node:test";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import {
  BASIC_KEYS,
  PARAM_KEYS,
  completeParams,
  formatParam,
  isDefault,
  paramValue,
  sameParams,
  snapParam,
  specOf,
} from "../src/retouch/format.ts";

/** A run recorded before the extra panels: the Basic panel only. */
const basicOnly = {
  temperature: 5200,
  tint: 4,
  exposure: 0.35,
  contrast: 10,
  highlights: -30,
  shadows: 20,
  whites: 5,
  blacks: -5,
  vibrance: 15,
  saturation: 0,
} as Params;

test("the Basic panel comes first and every setting has a spec", () => {
  assert.deepEqual(BASIC_KEYS, Object.keys(basicOnly));
  assert.equal(PARAM_KEYS.length, new Set(PARAM_KEYS).size);
  for (const key of PARAM_KEYS) assert.equal(specOf(key).key, key);
});

test("older runs read as Lightroom's defaults", () => {
  assert.equal(paramValue(basicOnly, "sharpening"), 40);
  assert.equal(paramValue(basicOnly, "color_nr"), 25);
  assert.equal(paramValue(basicOnly, "hue_red"), 0);
  const full = completeParams(basicOnly);
  for (const key of PARAM_KEYS) assert.equal(typeof full[key], "number", key);
  assert.equal(full.sharpen_radius, 1);
  assert.equal(full.exposure, 0.35);
  // Already complete: the same object back.
  assert.equal(completeParams(full), full);
  assert.ok(sameParams(basicOnly, full));
});

test("every setting counts when comparing", () => {
  const full = completeParams(basicOnly);
  assert.ok(sameParams(full, { ...full }));
  assert.ok(!sameParams(full, { ...full, luminance_magenta: 1 }));
  assert.ok(!sameParams(full, { ...full, lens_ca: 1 }));
  assert.ok(!sameParams(full, undefined));
});

test("values snap to Lightroom's range and step", () => {
  assert.equal(snapParam("exposure", 0.3449), 0.34);
  assert.equal(snapParam("exposure", 9), 5);
  assert.equal(snapParam("temperature", 60000), 50000);
  assert.equal(snapParam("temperature", 5234.6), 5235);
  assert.equal(snapParam("sharpen_radius", 1.26), 1.3);
  assert.equal(snapParam("sharpen_radius", 0.1), 0.5);
  assert.equal(snapParam("sharpening", 151), 150);
  assert.equal(snapParam("texture", -12.4), -12);
  assert.equal(snapParam("lens_profile", 0.7), 1);
});

test("values read the way Lightroom shows them", () => {
  assert.equal(formatParam("temperature", 5234.6), "5235 K");
  assert.equal(formatParam("exposure", 0.35), "+0.35 EV");
  assert.equal(formatParam("exposure", -0), "0.00 EV");
  assert.equal(formatParam("clarity", 12), "+12");
  assert.equal(formatParam("dehaze", -8), "-8");
  assert.equal(formatParam("sharpening", 40), "40");
  assert.equal(formatParam("sharpen_radius", 1), "1.0");
  assert.equal(formatParam("sharpen_radius", 1.26), "1.3");
  assert.equal(formatParam("lens_profile", 1), "On");
  assert.equal(formatParam("lens_ca", 0), "Off");
});

test("defaults: Lightroom's own, or as shot for white balance", () => {
  assert.ok(isDefault(specOf("sharpening"), 40));
  assert.ok(!isDefault(specOf("sharpening"), 30));
  assert.ok(isDefault(specOf("hue_aqua"), 0.2));
  assert.ok(!isDefault(specOf("lens_profile"), 1));
  const asShot = { temperature: 5234.6, tint: 3.2 };
  assert.ok(isDefault(specOf("temperature"), 5235, asShot));
  assert.ok(!isDefault(specOf("temperature"), 5600, asShot));
  assert.ok(!isDefault(specOf("tint"), 10, asShot));
  // Without the as-shot values white balance has no default to differ from.
  assert.ok(isDefault(specOf("temperature"), 5600));
});
