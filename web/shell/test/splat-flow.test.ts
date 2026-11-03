/* Unit tests for the #/3d run status (waits, stages, ETA) and downloads.
   `pnpm --filter @spl/shell test`. */
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  duration,
  framesOf,
  gpuWait,
  queuePosition,
  remainingSeconds,
  stageModel,
  stageOf,
  stageSeconds,
  trainingRate,
  waitHeadline,
} from "../src/splat/progress.ts";

const finished = (preset: string, frames: number, train: number) => ({
  status: "succeeded" as const,
  input: { preset, images: Array.from({ length: frames }, (_, i) => `a${i}`) },
  output: {
    frames,
    ingest_s: frames * 0.01,
    times: { engine_load_s: 1, engine_s: frames * 0.03, setup_s: 0.5, init_s: 0.5, train_s: train, export_s: 2 },
  },
});

test("the op's GPU-memory wait becomes plain words, numbers behind details", () => {
  const data = { stage: "posing", waiting_for_gpu_memory: true, need_bytes: 3.57e9, free_bytes: 1.53e9, waited_s: 12, timeout_s: 600 };
  const w = gpuWait(data);
  assert.deepEqual(w, { needBytes: 3.57e9, freeBytes: 1.53e9, waitedS: 12, timeoutS: 600 });
  assert.equal(gpuWait({ stage: "posing" }), undefined);
  const h = waitHeadline({ queued: false, wait: w });
  assert.match(h!, /GPU is busy with another job; your run starts automatically/);
  assert.doesNotMatch(h!, /GB/);
  assert.match(waitHeadline({ queued: true, reason: "gpu", position: 3 })!, /number 3 in the queue/);
  assert.doesNotMatch(waitHeadline({ queued: true, reason: "gpu", position: 1 })!, /number/);
  assert.equal(waitHeadline({ queued: false }), undefined);
});

test("stages come from the progress data or the op's messages", () => {
  assert.equal(stageOf("reading input"), "ingest");
  assert.equal(stageOf("engine 1/2: trunk"), "posing");
  assert.equal(stageOf("initializing"), "init");
  assert.equal(stageOf("training 10/7000: 5 Gaussians, 90 it/s, loss -"), "refine");
  assert.equal(stageOf("meshing"), "mesh");
  assert.equal(stageOf("storing exports"), "export");
  assert.equal(stageOf("waiting for GPU memory: ...", { stage: "init" }), "init");
  assert.equal(stageOf("something else"), undefined);
  assert.deepEqual(trainingRate("training 1200/7000: 812345 Gaussians, 95 it/s, loss 0.0123"), { done: 1200, total: 7000, perS: 95 });
});

test("the ETA scales measured stage times by frames, and trusts the trainer's own rate", () => {
  const runs = [finished("quality", 100, 150), finished("quality", 200, 300), finished("fast", 100, 40)];
  assert.equal(stageSeconds(runs[0]!.output)!.refine, 150);
  assert.equal(framesOf(runs[1]!), 200);
  const model = stageModel(runs, "quality");
  assert.ok(Math.abs(model.refine! - 1.5) < 1e-9, "median per-frame train seconds of quality runs");
  // At the start of posing for 100 frames: posing + init + refine + export.
  const left = remainingSeconds({ model, frames: 100, stage: "posing", stageElapsedS: 0, mesh: false })!;
  const posing = model.posing! * 100;
  assert.ok(Math.abs(left - (posing + model.init! * 100 + 150 + model.export! * 100)) < 1e-6);
  // In training the reported rate wins: 5800 iterations at 100 it/s.
  const t = remainingSeconds({ model, frames: 100, stage: "refine", stageElapsedS: 5, mesh: false, message: "training 1200/7000: 1 Gaussians, 100 it/s, loss -" })!;
  assert.ok(Math.abs(t - (58 + model.export! * 100)) < 1e-6);
  assert.equal(remainingSeconds({ model: {}, frames: 100, stage: "posing", stageElapsedS: 0, mesh: false }), undefined);
  // Fast runs only, as the preset matches.
  assert.ok(Math.abs(stageModel(runs, "fast").refine! - 0.4) < 1e-9);
});

test("queue position counts the queued runs oldest first", () => {
  const runs = [
    { id: "a", status: "running" as const, created_at_ms: 1 },
    { id: "b", status: "queued" as const, created_at_ms: 3 },
    { id: "c", status: "queued" as const, created_at_ms: 2 },
  ];
  assert.equal(queuePosition(runs, "b"), 2);
  assert.equal(queuePosition(runs, "c"), 1);
  assert.equal(queuePosition(runs, "a"), undefined);
});

test("durations read naturally", () => {
  assert.equal(duration(8.4), "8 s");
  assert.equal(duration(75), "1 min 15 s");
  assert.equal(duration(120), "2 min");
  assert.equal(duration(3700), "1 h 2 min");
});
