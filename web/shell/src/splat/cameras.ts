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
export type Frustum = { center: Vec3; corners: [Vec3, Vec3, Vec3, Vec3]; aspect: number };

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
  };
}

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
    });
  }
  return out;
}

/** Frustums from a parsed cameras artifact; [] for anything unrecognised. */
export function parseCameras(json: unknown): Frustum[] {
  if (Array.isArray(json)) return json.filter(isPinhole).map(pinholeFrustum);
  if (!json || typeof json !== "object") return [];
  const o = json as Record<string, unknown>;
  if (Array.isArray(o.cameras)) return o.cameras.filter(isPinhole).map(pinholeFrustum);
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
