/* Unit tests for the COLMAP-free #/3d flow: chunked uploads, the drop
   zone's sorting, the stage bar, thumbnails and camera views.
   `pnpm --filter @spl/shell test`. */
import assert from "node:assert/strict";
import { test } from "node:test";
import { cameraView, parseCameras, pinholeFrustum } from "../src/splat/cameras.ts";
import { missingParts, partRange, uploadChunked, type ResumeStore, type UploadApi } from "../src/splat/chunked.ts";
import {
  liveNumbers,
  matchPhotos,
  mediaTypeOf,
  pickFiles,
  reconstructInput,
  stageStates,
  DEFAULT_RECONSTRUCT,
} from "../src/splat/reconstruct.ts";

/* ---- A fake /v1/uploads host ---- */

type Fake = UploadApi & { puts: number[]; bytes: () => Uint8Array; creates: number; failNext: Map<number, number> };

function fakeHost(): Fake {
  let session: { id: string; size: number; part_bytes: number; parts: number; data: Uint8Array; got: Set<number> } | undefined;
  const view = () => ({
    id: session!.id,
    size: session!.size,
    part_bytes: session!.part_bytes,
    parts: session!.parts,
    received: [...session!.got].sort((a, b) => a - b),
  });
  const fake: Fake = {
    puts: [],
    creates: 0,
    failNext: new Map(),
    bytes: () => session!.data,
    async createUpload(r) {
      fake.creates += 1;
      const part = r.part_bytes ?? 16;
      session = { id: `u${fake.creates}`, size: r.size, part_bytes: part, parts: Math.ceil(r.size / part), data: new Uint8Array(r.size), got: new Set() };
      return view();
    },
    async uploadStatus(id) {
      if (!session || session.id !== id) throw Object.assign(new Error("gone"), { status: 404 });
      return view();
    },
    async putUploadPart(id, n, part, o) {
      assert.equal(id, session!.id);
      fake.puts.push(n);
      const fails = fake.failNext.get(n) ?? 0;
      if (fails > 0) {
        fake.failNext.set(n, fails - 1);
        throw Object.assign(new Error("bad gateway"), { status: 502 });
      }
      const buf = new Uint8Array(await part.arrayBuffer());
      assert.equal(buf.length, partRange(session!, n)[1] - partRange(session!, n)[0]);
      session!.data.set(buf, n * session!.part_bytes);
      o.onProgress?.(buf.length);
      session!.got.add(n);
      return view();
    },
    async completeUpload() {
      if (session!.got.size !== session!.parts) throw Object.assign(new Error("missing"), { status: 409 });
      return { id: "artifact", kind: "blob", media_type: "video/mp4", size_bytes: session!.size } as never;
    },
  };
  return fake;
}

const memoryStore = (): ResumeStore & { map: Map<string, string> } => {
  const map = new Map<string, string>();
  return { map, get: (k) => map.get(k), set: (k, v) => void map.set(k, v), delete: (k) => void map.delete(k) };
};

const blob = (n: number) => new Blob([Uint8Array.from({ length: n }, (_, i) => (i * 7) % 256)]);
const noSleep = async () => undefined;

test("parts cover the file exactly", () => {
  const s = { size: 40, part_bytes: 16, parts: 3, received: [1] };
  assert.deepEqual([0, 1, 2].map((n) => partRange(s, n)), [[0, 16], [16, 32], [32, 40]]);
  assert.deepEqual(missingParts(s), [0, 2]);
});

test("a chunked upload sends every part and reports progress to the end", async () => {
  const host = fakeHost();
  const file = blob(100);
  const progress: number[] = [];
  const artifact = await uploadChunked(host, file, {
    filename: "walk.mp4",
    mediaType: "video/mp4",
    partBytes: 16,
    parallel: 3,
    onProgress: (sent) => progress.push(sent),
  });
  assert.equal(artifact.id, "artifact");
  assert.deepEqual([...host.puts].sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6]);
  assert.deepEqual(host.bytes(), new Uint8Array(await file.arrayBuffer()));
  assert.equal(progress.at(-1), 100);
  assert.ok(progress.every((p, i) => i === 0 || p >= progress[i - 1]!), "progress never goes back");
});

test("failed parts are retried, then the upload fails", async () => {
  const host = fakeHost();
  host.failNext.set(2, 2);
  await uploadChunked(host, blob(64), { filename: "a", mediaType: "x", partBytes: 16, sleep: noSleep });
  assert.equal(host.puts.filter((n) => n === 2).length, 3);

  const broken = fakeHost();
  broken.failNext.set(0, 99);
  await assert.rejects(
    uploadChunked(broken, blob(64), { filename: "a", mediaType: "x", partBytes: 16, tries: 3, sleep: noSleep }),
    /bad gateway/,
  );
  assert.equal(broken.puts.filter((n) => n === 0).length, 3);
});

test("a second try resumes the same session and sends only the missing parts", async () => {
  const host = fakeHost();
  const store = memoryStore();
  const file = blob(64);
  host.failNext.set(3, 99);
  const o = { filename: "a", mediaType: "x", partBytes: 16, tries: 1, parallel: 1, fingerprint: "f", store, sleep: noSleep };
  await assert.rejects(uploadChunked(host, file, o));
  assert.equal(store.map.get("f"), "u1");
  host.failNext.clear();
  host.puts.length = 0;
  await uploadChunked(host, file, o);
  assert.equal(host.creates, 1);
  assert.deepEqual(host.puts, [3]);
  assert.equal(store.map.has("f"), false, "a finished upload forgets its session");
});

test("a session the host lost is opened again", async () => {
  const host = fakeHost();
  const store = memoryStore();
  store.set("f", "gone");
  await uploadChunked(host, blob(20), { filename: "a", mediaType: "x", partBytes: 16, fingerprint: "f", store });
  assert.equal(host.creates, 1);
});

/* ---- Drop zone ---- */

const file = (name: string, type = "") => new File([new Uint8Array(4)], name, { type });

test("drops sort into photos or one video", () => {
  const photos = pickFiles([file("IMG_10.JPG"), file("IMG_2.jpg"), file("notes.txt"), file("clip.mov")]);
  assert.equal(photos.kind, "photos");
  if (photos.kind === "photos") {
    assert.deepEqual(photos.files.map((f) => f.name), ["IMG_2.jpg", "IMG_10.JPG"]);
    assert.deepEqual(photos.skipped, ["notes.txt", "clip.mov"]);
  }
  const video = pickFiles([file("walk.MP4"), file("b.mov", "video/quicktime")]);
  assert.equal(video.kind, "video");
  if (video.kind === "video") assert.equal(video.file.name, "walk.MP4");
  assert.equal(pickFiles([file("one.png")]).kind, "none");
  assert.equal(pickFiles([file("a.zip")]).kind, "none");
  assert.equal(pickFiles([file("x", "image/webp"), file("y", "image/webp")]).kind, "photos");
});

test("media types come from the extension when the browser gives none", () => {
  assert.equal(mediaTypeOf(file("a.MOV")), "video/quicktime");
  assert.equal(mediaTypeOf(file("a.jpeg")), "image/jpeg");
  assert.equal(mediaTypeOf(file("a.bin")), "application/octet-stream");
  assert.equal(mediaTypeOf(file("a.bin", "image/png")), "image/png");
});

test("the op input carries photos or a video and the options", () => {
  assert.deepEqual(reconstructInput({ images: ["a", "b"] }, DEFAULT_RECONSTRUCT), {
    images: ["a", "b"],
    mode: "auto",
    preset: "quality",
    mesh: false,
    max_images: 200,
  });
  const v = reconstructInput({ video: "v" }, { ...DEFAULT_RECONSTRUCT, mesh: true, mode: "object" });
  assert.equal(v.video, "v");
  assert.equal(v.images, undefined);
  assert.equal(v.mesh, true);
});

/* ---- Stage bar ---- */

test("live numbers read the op's progress leniently", () => {
  assert.deepEqual(liveNumbers({ stage: "refine", iteration: 1200, held_out_psnr: 24.5, gaussians: 900000 }), {
    stage: "refine",
    iteration: 1200,
    iterations: undefined,
    psnr: 24.5,
    gaussians: 900000,
    images: undefined,
  });
  assert.equal(liveNumbers({ stage: "dancing" }).stage, undefined);
  assert.deepEqual(liveNumbers(null), {});
});

test("stages before the current one are done; mesh only when asked", () => {
  const s = stageStates("init", { mesh: false, finished: false });
  assert.deepEqual(s, { ingest: "done", posing: "done", init: "active", refine: "todo", mesh: "skipped", export: "todo" });
  const done = stageStates(undefined, { mesh: true, finished: true });
  assert.ok(Object.values(done).every((v) => v === "done"));
});

/* ---- Thumbnails and views ---- */

test("cameras match photos by name, else by order", () => {
  assert.deepEqual(matchPhotos(["images/b.jpg", "a.JPG", undefined], ["a.jpg", "b.jpg", "c.jpg"]), [1, 0, undefined]);
  assert.deepEqual(matchPhotos([undefined, undefined], ["a.jpg", "b.jpg"]), [0, 1]);
  assert.deepEqual(matchPhotos([undefined], ["a.jpg", "b.jpg"]), [undefined]);
});

const I = [1, 0, 0, 0, 1, 0, 0, 0, 1] as [number, number, number, number, number, number, number, number, number];

test("camera names survive parsing", () => {
  const cams = parseCameras({ cameras: [{ r_cw: I, t_cw: [0, 0, 0], fx: 1, fy: 1, cx: 1, cy: 1, width: 2, height: 2, name: "a.jpg" }] });
  assert.equal(cams[0]?.name, "a.jpg");
  const tf = parseCameras({ fl_x: 1, w: 2, h: 2, frames: [{ file_path: "images/b.png", transform_matrix: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]] }] });
  assert.equal(tf[0]?.name, "images/b.png");
});

test("a camera looks down +z with image up along −y and its true field of view", () => {
  const v = cameraView(pinholeFrustum({ r_cw: I, t_cw: [0, 0, 0], fx: 50, fy: 50, cx: 50, cy: 50, width: 100, height: 100 }));
  const close = (a: number[], b: number[]) => a.forEach((x, i) => assert.ok(Math.abs(x - b[i]!) < 1e-9, `${a} vs ${b}`));
  close(v.forward, [0, 0, 1]);
  close(v.up, [0, -1, 0]);
  assert.ok(Math.abs(v.fovY - 90) < 1e-9);
});

test("a ring of inward-looking cameras focuses on its centre with their shared up", async () => {
  const { focusPoint, meanUp } = await import("../src/splat/cameras.ts");
  // Cameras on a circle in the x-z plane at height 0, looking at (0, 0, 5); image up is −y.
  const ring = Array.from({ length: 8 }, (_, k) => {
    const t = (k / 8) * 2 * Math.PI;
    const c: [number, number, number] = [3 * Math.sin(t), 0, 5 - 3 * Math.cos(t)];
    const fwd = [-Math.sin(t), 0, Math.cos(t)];
    // Rows of R are the camera axes in world: x = y × z, y = (0, 1, 0), z = forward.
    const x = [fwd[2]!, 0, -fwd[0]!];
    const r = [x[0]!, x[1]!, x[2]!, 0, 1, 0, fwd[0]!, fwd[1]!, fwd[2]!] as typeof I;
    const t_cw = [0, 1, 2].map((i) => -(r[i * 3]! * c[0] + r[i * 3 + 1]! * c[1] + r[i * 3 + 2]! * c[2])) as [number, number, number];
    return pinholeFrustum({ r_cw: r, t_cw, fx: 50, fy: 50, cx: 50, cy: 50, width: 100, height: 100 });
  });
  const p = focusPoint(ring)!;
  [0, 0, 5].forEach((v, i) => assert.ok(Math.abs(p[i]! - v) < 1e-9, `${p}`));
  const up = meanUp(ring)!;
  [0, -1, 0].forEach((v, i) => assert.ok(Math.abs(up[i]! - v) < 1e-9, `${up}`));
  // A forward-facing row has no focus.
  const row = [0, 1, 2].map((k) => pinholeFrustum({ r_cw: I, t_cw: [-k, 0, 0], fx: 50, fy: 50, cx: 50, cy: 50, width: 100, height: 100 }));
  assert.equal(focusPoint(row), undefined);
});
