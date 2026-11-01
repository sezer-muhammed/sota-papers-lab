/* Unit tests for the Retouch page's Develop panel logic: the PARAM_SPECS
   helpers, the section layout and the tone-curve picture. Node runs the
   TypeScript directly (type stripping): `pnpm --filter @spl/shell test`. */
import assert from "node:assert/strict";
import { test } from "node:test";
import { PARAM_SPECS } from "@spl/bindings/raw-retouch/paramSpecs";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import { CURVE_BUMPS, REGIONS, curveAt, curvePath, isFlat, regionRangePath } from "../src/retouch/develop/curve.ts";
import {
  HAS_MIXER,
  MIXER_BANDS,
  MIXER_KINDS,
  SECTIONS,
  mixerKey,
  sectionKeys,
  shortLabel,
  trackOf,
} from "../src/retouch/develop/sections.ts";
import {
  BASIC_KEYS,
  PARAM_KEYS,
  awayFromDefault,
  completeParams,
  formatParam,
  sameParams,
  snapParam,
  specOf,
} from "../src/retouch/format.ts";
import { HUE_BANDS } from "../src/retouch/scopes/compute.ts";

/** Lightroom's defaults with an as-shot white balance, like the host's Params::neutral. */
function neutral(temperature = 5200, tint = 3): Params {
  const out: Record<string, number> = {};
  for (const spec of PARAM_SPECS) out[spec.key] = spec.neutral ?? 0;
  return { ...out, temperature, tint } as Params;
}

test("every Params field is in exactly one section, in a section of its group", () => {
  const seen = sectionKeys({ id: "", title: "", groups: [], blocks: SECTIONS.flatMap((s) => s.blocks) });
  assert.deepEqual([...seen].sort(), [...PARAM_KEYS].sort());
  assert.equal(new Set(seen).size, seen.length);
  for (const section of SECTIONS) {
    for (const key of sectionKeys(section)) assert.ok(section.groups.includes(specOf(key).group), key);
  }
  assert.deepEqual(
    SECTIONS.map((s) => s.title),
    ["Basic", "Tone Curve", "Color Mixer", "Detail", "Lens Corrections", "Calibration"],
  );
});

test("the Basic section follows Lightroom: white balance, tone, presence", () => {
  const basic = SECTIONS.find((s) => s.id === "basic")!;
  assert.deepEqual(
    basic.blocks.map((b) => b.title),
    ["White balance", "Tone", "Presence"],
  );
  assert.deepEqual(basic.blocks[2]!.keys, ["texture", "clarity", "dehaze", "vibrance", "saturation"]);
});

test("the Color Mixer has all 24 fields and the scopes' band hues", () => {
  assert.ok(HAS_MIXER);
  assert.deepEqual(
    MIXER_BANDS.map((b) => [b.band, b.hue]),
    HUE_BANDS.map((b) => [b.name, b.hue]),
  );
  const mixer = SECTIONS.find((s) => s.id === "colorMixer")!;
  assert.equal(sectionKeys(mixer).length, MIXER_BANDS.length * MIXER_KINDS.length);
  assert.equal(specOf(mixerKey("luminance", "aqua")).crs, "LuminanceAdjustmentAqua");
});

test("labels shorten under a heading; tracks tint the colour sliders only", () => {
  assert.equal(shortLabel("red_primary_hue", "Red Primary Hue"), "Hue");
  assert.equal(shortLabel("clarity", "Clarity"), "Clarity");
  assert.match(trackOf("hue_orange") ?? "", /hsl\(0 .*hsl\(30 .*hsl\(60 /);
  assert.match(trackOf("blue_primary_saturation") ?? "", /var\(--ds-gray-500\).*hsl\(240 /);
  assert.match(trackOf("temperature") ?? "", /--ds-blue-700.*--ds-amber-700/);
  assert.equal(trackOf("saturation"), undefined);
  assert.equal(trackOf("luminance_nr"), undefined);
  assert.equal(trackOf("contrast"), undefined);
});

test("snapParam rounds to each field's step and clamps to its range", () => {
  assert.equal(snapParam("exposure", 0.356), 0.36);
  assert.equal(snapParam("sharpen_radius", 1.26), 1.3);
  assert.equal(snapParam("sharpen_radius", 9), 3);
  assert.equal(snapParam("sharpening", 151), 150);
  assert.equal(snapParam("temperature", 60000), 50000);
  assert.equal(snapParam("temperature", 5123.6), 5124);
  assert.equal(snapParam("hue_red", -100.4), -100);
  assert.equal(snapParam("lens_ca", 0.7), 1);
});

test("formatParam shows units, signs and switches the way Lightroom does", () => {
  assert.equal(formatParam("temperature", 5123.4), "5123 K");
  assert.equal(formatParam("exposure", 0.35), "+0.35 EV");
  assert.equal(formatParam("exposure", -0.001), "0.00 EV");
  assert.equal(formatParam("clarity", 12), "+12");
  assert.equal(formatParam("hue_red", -8), "-8");
  assert.equal(formatParam("sharpening", 40), "40");
  assert.equal(formatParam("sharpen_radius", 1), "1.0");
  assert.equal(formatParam("lens_profile", 1), "On");
  assert.equal(formatParam("lens_ca", 0), "Off");
});

test("completeParams fills fields an older run lacks with Lightroom's defaults", () => {
  const old: Record<string, number> = {};
  for (const key of BASIC_KEYS) old[key] = 1;
  old.temperature = 5000;
  const full = completeParams(old as unknown as Params);
  assert.equal(full.temperature, 5000);
  assert.equal(full.contrast, 1);
  assert.equal(full.sharpening, 40);
  assert.equal(full.color_nr, 25);
  assert.equal(full.hue_red, 0);
  assert.equal(full.lens_profile, 0);
  const complete = neutral();
  assert.equal(completeParams(complete), complete);
  assert.ok(sameParams(full, completeParams(full)));
});

test("sameParams sees every field, and treats a missing one as its default", () => {
  const a = neutral();
  assert.ok(sameParams(a, { ...a }));
  assert.ok(!sameParams(a, { ...a, luminance_magenta: 1 }));
  assert.ok(!sameParams(a, { ...a, lens_ca: 1 }));
  const { sharpening: _drop, ...partial } = a;
  assert.ok(sameParams(a, partial as Params));
  assert.ok(!sameParams(a, undefined));
});

test("awayFromDefault compares with the spec default, white balance with as shot", () => {
  const asShot = { temperature: 5200.4, tint: 3 };
  assert.ok(!awayFromDefault("sharpening", 40));
  assert.ok(awayFromDefault("sharpening", 55));
  assert.ok(!awayFromDefault("color_nr", 25));
  assert.ok(awayFromDefault("clarity", -3));
  assert.ok(!awayFromDefault("temperature", 5200, asShot));
  assert.ok(awayFromDefault("temperature", 5450, asShot));
  assert.ok(!awayFromDefault("temperature", 5450));
  assert.ok(awayFromDefault("tint", 8, asShot));
});

const flat = { shadows: 0, darks: 0, lights: 0, highlights: 0 };

test("the tone curve is the identity at zero and stays anchored at black and white", () => {
  for (let i = 0; i <= 20; i++) assert.equal(curveAt(flat, i / 20), i / 20);
  assert.ok(isFlat(flat));
  const strong = { shadows: 100, darks: -100, lights: 100, highlights: -100 };
  assert.equal(curveAt(strong, 0), 0);
  assert.equal(curveAt(strong, 1), 1);
  assert.ok(!isFlat(strong));
});

test("each region slider moves its own part of the curve, the right way, and keeps it rising", () => {
  const probe: Record<string, number> = { shadows: 0.15, darks: 0.375, lights: 0.625, highlights: 0.85 };
  for (const region of REGIONS) {
    for (const amount of [-100, 100]) {
      const r = { ...flat, [region]: amount };
      const x = probe[region]!;
      assert.equal(Math.sign(curveAt(r, x) - x), Math.sign(amount), `${region} ${amount}`);
      let last = -Infinity;
      for (let i = 0; i <= 200; i++) {
        const y = curveAt(r, i / 200);
        assert.ok(y >= last - 1e-9, `${region} ${amount} falls at ${i / 200}`);
        last = y;
      }
    }
    const b = CURVE_BUMPS[region];
    assert.ok(b.lo < b.peak && b.peak < b.hi && b.gain > 0 && b.gain <= 0.16, region);
  }
  // Shadows leave the highlights alone, and the other way round.
  assert.equal(curveAt({ ...flat, shadows: 100 }, 0.8), 0.8);
  assert.equal(curveAt({ ...flat, highlights: 100 }, 0.2), 0.2);
});

test("curve paths fit the box", () => {
  const path = curvePath({ ...flat, lights: 50 }, 100);
  assert.match(path, /^M0 100 L/);
  assert.match(path, /L100 0$/);
  const band = regionRangePath(flat, "darks", 100);
  assert.match(band, / Z$/);
  for (const n of band.match(/-?\d+(\.\d+)?/g)!.map(Number)) assert.ok(n >= 0 && n <= 100);
});
