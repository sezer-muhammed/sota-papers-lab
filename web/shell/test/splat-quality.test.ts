/* Unit tests for the #/3d capture and result reports' pure helpers.
   `pnpm --filter @spl/shell test`. */
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  captureFacts,
  floaterSummary,
  longerPreset,
  readCapture,
  trainLongerInput,
  verdictHeadline,
  type ReconstructInput,
} from "../src/splat/reconstruct.ts";

const capture = {
  verdict: "poor",
  frames: 60,
  stats: {
    keyframes: "motion",
    decoded: 300,
    kept: 60,
    blurry_share: 0.42,
    kept_blurry_share: 0.1,
    still_share: 0.3,
    total_motion: 4.2,
    duplicates: 2,
    dark_clip_share: 0.01,
    bright_clip_share: 0.04,
  },
  baseline: 0.12,
  orbit_deg: 14.6,
  turn_deg: 80,
  problems: ["The camera turned in place instead of moving around the scene."],
  tips: ["Move around the object instead of turning in place."],
};

test("readCapture accepts the announcement and rejects junk", () => {
  assert.equal(readCapture(capture)?.verdict, "poor");
  assert.deepEqual(readCapture({ ...capture, problems: undefined })?.problems, []);
  assert.equal(readCapture(null), undefined);
  assert.equal(readCapture({ verdict: "ok" }), undefined);
});

test("captureFacts reads in plain numbers", () => {
  const f = Object.fromEntries(captureFacts(readCapture(capture)!));
  assert.equal(f.frames, "60 of 300");
  assert.equal(f.blurry, "42 %");
  assert.equal(f["standing still"], "30 %");
  assert.equal(f.clipped, "5 %");
  assert.equal(f.orbit, "15°");
  assert.equal(f["camera travel"], "0.12× distance");
  assert.match(verdictHeadline("poor"), /weak/);
});

test("train longer goes one preset up and keeps the media", () => {
  assert.equal(longerPreset("fast"), "quality");
  assert.equal(longerPreset("quality"), "big");
  assert.equal(longerPreset("big"), undefined);
  const input = { video: "abc", preset: "fast", mode: "auto", mesh: false, profile: "research", max_frames: 200 } as ReconstructInput;
  assert.deepEqual(trainLongerInput(input), { ...input, preset: "quality" });
  assert.equal(trainLongerInput({ ...input, preset: "big" }), undefined);
  assert.equal(trainLongerInput(undefined), undefined);
});

test("floaterSummary gives counts, share and the PSNR change", () => {
  const f = { before: 1_000_000, after: 830_000, unseen: 150_000, transparent: 15_000, huge: 5_000, min_views: 3, psnr_before: 24.1, psnr_after: 24.13 };
  assert.equal(floaterSummary(f), "170,000 of 1,000,000 Gaussians (17 %); held-out PSNR +0.03 dB");
  assert.equal(floaterSummary({ ...f, psnr_after: 23.9 }).endsWith("−0.20 dB"), true);
  assert.equal(floaterSummary({ ...f, psnr_before: undefined }), "170,000 of 1,000,000 Gaussians (17 %)");
});
