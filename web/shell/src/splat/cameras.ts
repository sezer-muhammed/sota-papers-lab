/* Camera frustums for the viewer, in COLMAP world coordinates (x right,
   y down, z forward). Pure geometry, no three.js, so it is unit tested.

   A `cameras` artifact may hold either
   - our RenderCamera list (`[{ r_cw, t_cw, fx, fy, cx, cy, width, height }]`,
     or `{ cameras: [...] }`), world → camera, or
   - a nerfstudio/Instant-NGP transforms.json (`frames[].transform_matrix`,
     camera → world in OpenGL axes: y up, z backward). */

export type Vec3 = [number, number, number];

/** One camera: its centre and the four image corners at depth 1, in world
    coordinates. */
export type Frustum = {
  center: Vec3;
  corners: [Vec3, Vec3, Vec3, Vec3];
  aspect: number;
  /** The photo's file name, when the cameras artifact gives one. */
  name?: string;
};

type M3 = [number, number, number, number, number, number, number, number, number];

type Pinhole = {
  r_cw: M3;
  t_cw: Vec3;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
  name?: string;
};

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isNums = (v: unknown, n: number): v is number[] =>
  Array.isArray(v) && v.length === n && v.every(isNum);

function isPinhole(v: unknown): v is Pinhole {
  if (!v || typeof v !== "object") return false;
  const c = v as Record<string, unknown>;
  return (
    isNums(c.r_cw, 9) &&
    isNums(c.t_cw, 3) &&
    ["fx", "fy", "cx", "cy", "width", "height"].every((k) => isNum(c[k]))
  );
}

/** world = Rᵀ (x − t) for a world → camera pose (R row-major). */
function camToWorld(r: M3, t: Vec3, x: Vec3): Vec3 {
  const d: Vec3 = [x[0] - t[0], x[1] - t[1], x[2] - t[2]];
  return [
    r[0] * d[0] + r[3] * d[1] + r[6] * d[2],
    r[1] * d[0] + r[4] * d[1] + r[7] * d[2],
    r[2] * d[0] + r[5] * d[1] + r[8] * d[2],
  ];
}

export function pinholeFrustum(c: Pinhole): Frustum {
  const ray = (u: number, v: number): Vec3 => [(u - c.cx) / c.fx, (v - c.cy) / c.fy, 1];
  const pts: [Vec3, Vec3, Vec3, Vec3] = [
    ray(0, 0),
    ray(c.width, 0),
    ray(c.width, c.height),
    ray(0, c.height),
  ];
  return {
    center: camToWorld(c.r_cw, c.t_cw, [0, 0, 0]),
    corners: pts.map((p) => camToWorld(c.r_cw, c.t_cw, p)) as Frustum["corners"],
    aspect: c.width / c.height,
    ...(c.name ? { name: c.name } : {}),
  };
}

/** A file name from the keys exporters use. */
function nameOf(v: unknown): string | undefined {
  if (!v || typeof v !== "object") return undefined;
  const o = v as Record<string, unknown>;
  const n = o.name ?? o.image_name ?? o.image ?? o.file_path;
  return typeof n === "string" && n ? n : undefined;
}

const named = (list: unknown[]) =>
  list.filter(isPinhole).map((c) => pinholeFrustum({ ...c, name: nameOf(c) }));

/* transforms.json: c2w in OpenGL axes. Flip y and z of the camera axes to get
   a COLMAP-style c2w, then invert to w2c. */
type Row = [number, number, number, number];

function transformsFrames(json: Record<string, unknown>): Pinhole[] {
  const frames = json.frames;
  if (!Array.isArray(frames)) return [];
  const out: Pinhole[] = [];
  for (const frame of frames) {
    const f = { ...json, ...(frame as Record<string, unknown>) };
    const m = f.transform_matrix;
    if (!Array.isArray(m) || m.length < 3 || !m.slice(0, 3).every((row) => isNums(row, 4))) continue;
    const w = isNum(f.w) ? f.w : 0;
    const h = isNum(f.h) ? f.h : 0;
    const fx = isNum(f.fl_x) ? f.fl_x : NaN;
    const fy = isNum(f.fl_y) ? f.fl_y : fx;
    if (!(w > 0 && h > 0 && fx > 0)) continue;
    const [r0, r1, r2] = m as [Row, Row, Row];
    // w2c rotation = (c2w with camera y, z negated)ᵀ: rows are the c2w columns.
    const r: M3 = [r0[0], r1[0], r2[0], -r0[1], -r1[1], -r2[1], -r0[2], -r1[2], -r2[2]];
    const c: Vec3 = [r0[3], r1[3], r2[3]];
    const t: Vec3 = [
      -(r[0] * c[0] + r[1] * c[1] + r[2] * c[2]),
      -(r[3] * c[0] + r[4] * c[1] + r[5] * c[2]),
      -(r[6] * c[0] + r[7] * c[1] + r[8] * c[2]),
    ];
    out.push({
      r_cw: r,
      t_cw: t,
      fx,
      fy,
      cx: isNum(f.cx) ? f.cx : w / 2,
      cy: isNum(f.cy) ? f.cy : h / 2,
      width: w,
      height: h,
      name: nameOf(frame),
    });
  }
  return out;
}

/** Frustums from a parsed cameras artifact; [] for anything unrecognised. */
export function parseCameras(json: unknown): Frustum[] {
  if (Array.isArray(json)) return named(json);
  if (!json || typeof json !== "object") return [];
  const o = json as Record<string, unknown>;
  if (Array.isArray(o.cameras)) return named(o.cameras);
  return transformsFrames(o).map(pinholeFrustum);
}

/** Median distance between neighbouring camera centres: the frustum draw
    scale, so frustums look alike in any scene unit. */
export function frustumScale(frustums: Frustum[]): number {
  if (frustums.length < 2) return 0.1;
  const gaps: number[] = [];
  for (let i = 1; i < frustums.length; i += 1) {
    const a = frustums[i - 1]!.center;
    const b = frustums[i]!.center;
    gaps.push(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
  }
  gaps.sort((x, y) => x - y);
  const median = gaps[gaps.length >> 1] ?? 0;
  return median > 0 ? median * 0.6 : 0.1;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const unit = (a: Vec3): Vec3 => {
  const l = len(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

/** How to look through a camera, in world coordinates: its viewing
    direction, the image's up (−y), and the vertical field of view in
    degrees (assuming a centred principal point). */
export function cameraView(f: Frustum): { forward: Vec3; up: Vec3; fovY: number } {
  const mid = (a: Vec3, b: Vec3): Vec3 => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  const [c0, c1, c2, c3] = f.corners;
  const top = sub(mid(c0, c1), f.center);
  const bottom = sub(mid(c2, c3), f.center);
  const forward = unit([(top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2, (top[2] + bottom[2]) / 2]);
  const up = unit(sub(top, bottom));
  const cos = (top[0] * bottom[0] + top[1] * bottom[1] + top[2] * bottom[2]) / (len(top) * len(bottom));
  const fovY = (Math.acos(Math.min(1, Math.max(-1, cos))) * 180) / Math.PI;
  return { forward, up, fovY };
}

/** The cameras' mean image-up direction: the scene's up for a hand-held
    capture, whatever axes the poses came in. */
export function meanUp(frustums: Frustum[]): Vec3 | undefined {
  const sum: Vec3 = [0, 0, 0];
  for (const f of frustums) {
    const u = cameraView(f).up;
    sum[0] += u[0];
    sum[1] += u[1];
    sum[2] += u[2];
  }
  return len(sum) > 1e-6 ? unit(sum) : undefined;
}

/** The point nearest every camera's optical axis (least squares): what a
    capture walking around something looks at. Undefined when the axes are
    near parallel (a forward-facing capture). */
export function focusPoint(frustums: Frustum[]): Vec3 | undefined {
  if (frustums.length < 2) return undefined;
  // Σ (I − d dᵀ) x = Σ (I − d dᵀ) c
  const a = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  const b: Vec3 = [0, 0, 0];
  for (const f of frustums) {
    const d = cameraView(f).forward;
    const c = f.center;
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        const m = (i === j ? 1 : 0) - d[i]! * d[j]!;
        a[i * 3 + j] = a[i * 3 + j]! + m;
        b[i] = b[i]! + m * c[j]!;
      }
    }
  }
  const det =
    a[0]! * (a[4]! * a[8]! - a[5]! * a[7]!) -
    a[1]! * (a[3]! * a[8]! - a[5]! * a[6]!) +
    a[2]! * (a[3]! * a[7]! - a[4]! * a[6]!);
  // Parallel axes make the system singular; relative to n³ for scale.
  if (Math.abs(det) < 1e-3 * frustums.length ** 3) return undefined;
  const col = (k: number, v: Vec3) => a.map((x, idx) => (idx % 3 === k ? v[Math.floor(idx / 3)]! : x));
  const det3 = (m: number[]) =>
    m[0]! * (m[4]! * m[8]! - m[5]! * m[7]!) - m[1]! * (m[3]! * m[8]! - m[5]! * m[6]!) + m[2]! * (m[3]! * m[7]! - m[4]! * m[6]!);
  return [det3(col(0, b)) / det, det3(col(1, b)) / det, det3(col(2, b)) / det];
}

/** How a capture was shot, for the viewer's default controls. `room` when
    the cameras look outward from among themselves (a walk through a room or
    street) or all the same way, rather than in at one object; `scale` is
    the cameras' median distance from their centroid (or the neighbour gap),
    the fly speed's unit. */
export function sceneLayout(frustums: Frustum[]): { room: boolean; scale: number } {
  if (frustums.length < 3) return { room: false, scale: frustumScale(frustums) / 0.6 };
  const n = frustums.length;
  const mid: Vec3 = [0, 0, 0];
  for (const f of frustums) for (let k = 0; k < 3; k += 1) mid[k] = mid[k]! + f.center[k]! / n;
  const radii = frustums.map((f) => len(sub(f.center, mid))).sort((x, y) => x - y);
  const spread = radii[n >> 1]!;
  const scale = spread > 0 ? spread : frustumScale(frustums) / 0.6;
  // Mean cosine between each view and the way out from the centroid:
  // about −1 circling an object, 0 or more walking about a room.
  let outward = 0;
  for (const f of frustums) {
    const away = sub(f.center, mid);
    const l = len(away);
    if (l < 1e-9) continue;
    const d = cameraView(f).forward;
    outward += (d[0] * away[0] + d[1] * away[1] + d[2] * away[2]) / l / n;
  }
  const focus = focusPoint(frustums);
  // An object sits well inside the ring of cameras that look at it.
  const inward = focus !== undefined && len(sub(focus, mid)) < spread * 0.75;
  return { room: outward > -0.35 || !inward, scale };
}
