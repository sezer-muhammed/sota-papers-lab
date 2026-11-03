/* The page's side of `image_to_3d.reconstruct`, the COLMAP-free flow:
   unposed photos or one video in; cameras, a point cloud, live splat
   snapshots and the final exports out. Pure, so it is unit tested.

   The op's Rust types are not in `web/bindings` yet; `ReconstructInput`
   below is the contract this page sends and `ReconstructOutput` what it
   reads (missing fields are tolerated). Replace both with the ts-rs
   bindings once the op lands. */
import type { ArtifactRef } from "@spl/sdk";

export const RECONSTRUCT_OP = "image_to_3d.reconstruct";
export const GENERATE_OP = "image_to_3d.generate";

export type Mode = "auto" | "scene" | "object";
export type Preset = "fast" | "quality" | "big";
export type LicenceProfile = "research" | "commercial";
export type EngineSchedule = "dense" | "ours+sage";

export type ReconstructOptions = {
  mode: Mode;
  preset: Preset;
  mesh: boolean;
  profile: LicenceProfile;
  engineSchedule: EngineSchedule;
  /** Frames sampled from a video, or photos kept from a big set. */
  maxImages: number;
};

export const DEFAULT_RECONSTRUCT: ReconstructOptions = {
  mode: "auto", preset: "quality", mesh: false,
  profile: "research", engineSchedule: "dense", maxImages: 200,
};

export type ReconstructInput = {
  /** Photo artifacts, in the order the user gave them. */
  images?: string[];
  /** One video artifact. */
  video?: string;
  /** A photo folder under the host's dataset root (a sample scene). */
  dataset?: string;
  mode: Mode;
  preset: Preset;
  mesh: boolean;
  profile: LicenceProfile;
  engine_schedule?: string;
  /** Frame cap; the op's field is `max_frames` (photos are subsampled evenly above it). */
  max_frames: number;
};

export type ReconstructOutput = {
  spz?: ArtifactRef;
  ply?: ArtifactRef;
  cameras?: ArtifactRef;
  colmap?: ArtifactRef;
  transforms?: ArtifactRef;
  glb?: ArtifactRef;
  points?: ArtifactRef;
  eval?: { psnr?: number; ssim?: number; lpips?: number };
  summary?: { total_s?: number; final_gaussians?: number; images?: number };
  engine?: string;
  licence?: string;
};

export function reconstructInput(
  artifacts: { images?: string[]; video?: string; dataset?: string },
  o: ReconstructOptions,
): ReconstructInput {
  const source = artifacts.dataset
    ? { dataset: artifacts.dataset }
    : artifacts.video
      ? { video: artifacts.video }
      : { images: artifacts.images ?? [] };
  return {
    ...source,
    mode: o.mode,
    preset: o.preset,
    mesh: o.mesh,
    profile: o.profile,
    engine_schedule: o.engineSchedule === "dense" ? undefined : o.engineSchedule,
    max_frames: o.maxImages,
  };
}

/* ---- Sample scenes ---- */

/** Scenes the GPU host keeps under its dataset root (research-use
    datasets; never in the repo). Thumbnails are small WebPs of each
    folder's first photo, in `public/samples/<key>.webp`. */
export type SampleScene = { key: string; title: string; source: string; dataset: string; photos: number };

export const SAMPLE_SCENES: SampleScene[] = [
  { key: "garden", title: "Garden", source: "Mip-NeRF 360", dataset: "mipnerf360/garden/images_4", photos: 185 },
  { key: "bicycle", title: "Bicycle", source: "Mip-NeRF 360", dataset: "mipnerf360/bicycle/images_4", photos: 194 },
  { key: "kitchen", title: "Kitchen", source: "Mip-NeRF 360", dataset: "mipnerf360/kitchen/images_2", photos: 279 },
  { key: "room", title: "Room", source: "Mip-NeRF 360", dataset: "mipnerf360/room/images_2", photos: 311 },
  { key: "truck", title: "Truck", source: "Tanks and Temples", dataset: "tandt/truck/images", photos: 251 },
  { key: "playroom", title: "Playroom", source: "Deep Blending", dataset: "db/playroom/images", photos: 225 },
];

/** The run input for a sample scene: the page's options with the chosen preset. */
export function sampleInput(scene: SampleScene, preset: Preset, o: ReconstructOptions): ReconstructInput {
  return reconstructInput({ dataset: scene.dataset }, { ...o, mode: "scene", preset });
}

/* ---- Object generation (Pixal3D-mv sidecar) ---- */

export type GenerateInput = {
  images: string[];
  resolution?: number;
  max_tokens?: number;
};

export type GenerateOutput = {
  glb?: ArtifactRef;
  frames?: number;
  times?: { pose_s?: number; canon_s?: number; sidecar_s?: number; total_s?: number };
  peak_vram_mb?: number;
};

export function generateInput(imageIds: string[]): GenerateInput {
  return { images: imageIds, resolution: 1024 };
}

/* ---- Files the drop zone takes ---- */

const PHOTO = /\.(jpe?g|png|webp)$/i;
const VIDEO = /\.(mp4|mov|m4v)$/i;
const PHOTO_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const VIDEO_TYPES = new Set(["video/mp4", "video/quicktime", "video/x-m4v"]);

export type Picked =
  | { kind: "photos"; files: File[]; skipped: string[] }
  | { kind: "video"; file: File; skipped: string[] }
  | { kind: "none"; reason: string };

type Named = { name: string; type: string };

export function isPhoto(f: Named) {
  return PHOTO_TYPES.has(f.type) || PHOTO.test(f.name);
}

export function isVideo(f: Named) {
  return VIDEO_TYPES.has(f.type) || VIDEO.test(f.name);
}

/** Media type to record: the browser's, or one from the extension. */
export function mediaTypeOf(f: Named): string {
  if (f.type) return f.type;
  const ext = f.name.toLowerCase().split(".").pop() ?? "";
  return (
    { jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", webp: "image/webp", mp4: "video/mp4", mov: "video/quicktime", m4v: "video/x-m4v" }[
      ext
    ] ?? "application/octet-stream"
  );
}

/** Sorts a drop into photos or one video. Photos win over a video in a
    mixed drop; other files are listed as skipped.
    `objectMode` lowers the minimum to 1 photo and caps at 4. */
export function pickFiles<F extends File>(files: F[], objectMode = false): Picked {
  const photos = files.filter(isPhoto);
  const videos = files.filter(isVideo);
  const other = files.filter((f) => !isPhoto(f) && !isVideo(f)).map((f) => f.name);
  if (photos.length) {
    const skipped = [...other, ...videos.map((f) => f.name)];
    const minPhotos = objectMode ? 1 : 2;
    if (photos.length < minPhotos)
      return { kind: "none", reason: objectMode
        ? "Drop 1 to 4 photos of the object."
        : "Add at least 2 photos of the scene (20 to 200 work best)." };
    photos.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    const kept = objectMode ? photos.slice(0, 4) : photos;
    const extra = objectMode && photos.length > 4 ? photos.slice(4).map((f) => f.name) : [];
    return { kind: "photos", files: kept, skipped: [...skipped, ...extra] };
  }
  if (videos.length) return { kind: "video", file: videos[0]!, skipped: [...other, ...videos.slice(1).map((f) => f.name)] };
  return {
    kind: "none",
    reason: files.length ? "Those are not photos (JPEG, PNG, WebP) or a video (MP4, MOV)." : "Nothing was dropped.",
  };
}

/* ---- Stage bar ---- */

export const STAGES = ["ingest", "posing", "init", "refine", "mesh", "export"] as const;
export type Stage = (typeof STAGES)[number];

export const STAGE_LABEL: Record<Stage, string> = {
  ingest: "Ingest",
  posing: "Posing",
  init: "Init",
  refine: "Refine",
  mesh: "Mesh",
  export: "Export",
};

export type StageState = "done" | "active" | "todo" | "skipped";

/** The numbers the op attaches to its progress, read leniently. */
export type LiveNumbers = {
  stage?: Stage;
  iteration?: number;
  iterations?: number;
  psnr?: number;
  gaussians?: number;
  images?: number;
};

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);

export function liveNumbers(data: unknown): LiveNumbers {
  if (!data || typeof data !== "object") return {};
  const d = data as Record<string, unknown>;
  const stage = typeof d.stage === "string" && (STAGES as readonly string[]).includes(d.stage) ? (d.stage as Stage) : undefined;
  return {
    stage,
    iteration: num(d.iteration),
    iterations: num(d.iterations) ?? num(d.total_iterations),
    psnr: num(d.held_out_psnr) ?? num(d.test_psnr) ?? num(d.psnr),
    gaussians: num(d.gaussians) ?? num(d.num_gaussians) ?? num(d.gaussian_count),
    images: num(d.images) ?? num(d.frames),
  };
}

/** Each stage's state for the bar. Mesh is skipped unless asked for; a
    finished run shows every stage done. */
export function stageStates(stage: Stage | undefined, opts: { mesh: boolean; finished: boolean }): Record<Stage, StageState> {
  const at = stage ? STAGES.indexOf(stage) : -1;
  const out = {} as Record<Stage, StageState>;
  STAGES.forEach((s, i) => {
    if (s === "mesh" && !opts.mesh) out[s] = "skipped";
    else if (opts.finished || i < at) out[s] = "done";
    else if (i === at) out[s] = "active";
    else out[s] = "todo";
  });
  return out;
}

/* ---- Thumbnails ---- */

/** Which photo each camera shows: by a name in cameras.json when there is
    one (`name`, `image`, `image_name`, `file_path`), else by order. */
export function matchPhotos(cameraNames: (string | undefined)[], photoNames: string[]): (number | undefined)[] {
  const base = (s: string) => s.split(/[\\/]/).pop()!.toLowerCase();
  const byName = new Map(photoNames.map((n, i) => [base(n), i]));
  const named = cameraNames.some((n) => n && byName.has(base(n)));
  return cameraNames.map((n, i) => {
    if (named) return n ? byName.get(base(n)) : undefined;
    return cameraNames.length === photoNames.length ? i : undefined;
  });
}
