/* The page's side of `image_to_3d.reconstruct`, the COLMAP-free flow:
   unposed photos or one video in; cameras, a point cloud, live splat
   snapshots and the final exports out. Pure, so it is unit tested.

   The op's Rust types are not in `web/bindings` yet; `ReconstructInput`
   below is the contract this page sends and `ReconstructOutput` what it
   reads (missing fields are tolerated). Replace both with the ts-rs
   bindings once the op lands. */
import type { ArtifactRef } from "@spl/sdk";

export const RECONSTRUCT_OP = "image_to_3d.reconstruct";

export type Mode = "auto" | "scene" | "object";
export type Preset = "fast" | "quality";

export type ReconstructOptions = {
  mode: Mode;
  preset: Preset;
  mesh: boolean;
  /** Frames sampled from a video, or photos kept from a big set. */
  maxImages: number;
};

export const DEFAULT_RECONSTRUCT: ReconstructOptions = { mode: "auto", preset: "quality", mesh: false, maxImages: 200 };

export type ReconstructInput = {
  /** Photo artifacts, in the order the user gave them. */
  images?: string[];
  /** One video artifact. */
  video?: string;
  mode: Mode;
  preset: Preset;
  mesh: boolean;
  max_images: number;
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
};

export function reconstructInput(
  artifacts: { images?: string[]; video?: string },
  o: ReconstructOptions,
): ReconstructInput {
  return {
    ...(artifacts.video ? { video: artifacts.video } : { images: artifacts.images ?? [] }),
    mode: o.mode,
    preset: o.preset,
    mesh: o.mesh,
    max_images: o.maxImages,
  };
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
    mixed drop; other files are listed as skipped. */
export function pickFiles<F extends File>(files: F[]): Picked {
  const photos = files.filter(isPhoto);
  const videos = files.filter(isVideo);
  const other = files.filter((f) => !isPhoto(f) && !isVideo(f)).map((f) => f.name);
  if (photos.length) {
    const skipped = [...other, ...videos.map((f) => f.name)];
    if (photos.length < 2) return { kind: "none", reason: "Add at least 2 photos of the scene (20 to 200 work best)." };
    photos.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    return { kind: "photos", files: photos, skipped };
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
