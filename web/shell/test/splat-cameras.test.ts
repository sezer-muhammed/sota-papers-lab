/* Unit tests for the 3D viewer's camera frustums: `pnpm --filter @spl/shell test`. */
import assert from "node:assert/strict";
import { test } from "node:test";
import { frustumScale, parseCameras, pinholeFrustum } from "../src/splat/cameras.ts";

const close = (a: number[], b: number[]) =>
  a.forEach((v, i) => assert.ok(Math.abs(v - (b[i] ?? NaN)) < 1e-9, `${a} vs ${b}`));

const I = [1, 0, 0, 0, 1, 0, 0, 0, 1] as const;
const cam = (t: [number, number, number]) => ({
  r_cw: [...I] as [number, number, number, number, number, number, number, number, number],
  t_cw: t,
  fx: 100,
  fy: 100,
  cx: 50,
  cy: 25,
  width: 100,
  height: 50,
});

test("identity pose: centre at the origin, corners on the z = 1 plane", () => {
  const f = pinholeFrustum(cam([0, 0, 0]));
  close(f.center, [0, 0, 0]);
  close(f.corners[0], [-0.5, -0.25, 1]);
  close(f.corners[2], [0.5, 0.25, 1]);
  assert.equal(f.aspect, 2);
});

test("centre is −Rᵀt for a rotated pose", () => {
  // 90° about y: camera z (forward) points along world −x... R rows map world → cam.
  const r = [0, 0, -1, 0, 1, 0, 1, 0, 0] as [number, number, number, number, number, number, number, number, number];
  const f = pinholeFrustum({ ...cam([1, 2, 3]), r_cw: r });
  // Rᵀ·(−t) = (−3, −2, 1)
  close(f.center, [-3, -2, 1]);
});

test("RenderCamera list, wrapped list and junk", () => {
  assert.equal(parseCameras([cam([0, 0, 0]), { nope: 1 }]).length, 1);
  assert.equal(parseCameras({ cameras: [cam([0, 0, 0]), cam([1, 0, 0])] }).length, 2);
  assert.deepEqual(parseCameras("x"), []);
  assert.deepEqual(parseCameras(null), []);
});

test("transforms.json agrees with the same COLMAP camera", () => {
  // COLMAP camera at world (1, 2, 3) with identity rotation: OpenGL c2w has
  // y and z axes negated.
  const colmap = pinholeFrustum(cam([-1, -2, -3]));
  const [gl] = parseCameras({
    fl_x: 100,
    w: 100,
    h: 50,
    frames: [
      {
        transform_matrix: [
          [1, 0, 0, 1],
          [0, -1, 0, 2],
          [0, 0, -1, 3],
          [0, 0, 0, 1],
        ],
      },
    ],
  });
  assert.ok(gl);
  close(gl.center, colmap.center);
  gl.corners.forEach((c, i) => close(c, colmap.corners[i]!));
});

test("frustum scale follows the camera spacing", () => {
  const fs = [0, 1, 2, 3].map((x) => pinholeFrustum(cam([-x * 2, 0, 0])));
  assert.ok(Math.abs(frustumScale(fs) - 1.2) < 1e-9);
  assert.equal(frustumScale(fs.slice(0, 1)), 0.1);
});
