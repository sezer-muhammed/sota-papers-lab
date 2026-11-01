import type { RunProgress, RunRecord } from "@spl/sdk";
import type { AutoInput } from "@spl/bindings/raw-retouch/AutoInput";
import type { AutoOutput } from "@spl/bindings/raw-retouch/AutoOutput";
import type { CandidateSummary } from "@spl/bindings/raw-retouch/CandidateSummary";
import type { InspectOutput } from "@spl/bindings/raw-retouch/InspectOutput";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import type { RawInfo } from "@spl/bindings/raw-retouch/RawInfo";
import type { RenderOutput } from "@spl/bindings/raw-retouch/RenderOutput";
import { BASIC_KEYS, completeParams, safeFileName, stemOf } from "./format";
import type { RetouchOptions } from "./options";

/* One photo's journey: upload → inspect (metadata and a neutral preview) →
   auto (search, Claude, export), plus optional manual edits. Each stage keeps
   its own state so a re-run never hides the result it would replace. */

export type UploadState =
  | { status: "waiting" }
  | { status: "uploading"; fraction: number }
  | { status: "done" }
  | { status: "failed"; error: string };

export type InspectState =
  | { status: "idle" }
  | { status: "running"; runId?: string }
  | { status: "done"; output: InspectOutput }
  | { status: "failed"; error: string; runId?: string };

export type AutoState =
  | { status: "idle" }
  /** Waiting for one of the "photos in parallel" slots (no run yet). */
  | { status: "queued"; order: number }
  /** A run id from an earlier visit whose state is not known yet. */
  | { status: "restoring"; runId: string }
  | {
      status: "running";
      /** Unset while startRun is in flight. */
      runId?: string;
      /** The host accepted the run but has not started it. */
      serverQueued: boolean;
      progress: RunProgress | null;
      logs: string[];
      startedAtMs: number | null;
      cancelling: boolean;
      /** Why the last cancel request failed. */
      notice?: string;
    }
  | { status: "succeeded"; runId: string }
  | { status: "failed"; error: string; runId?: string }
  | { status: "cancelled"; runId?: string };

export type AutoResult = {
  output: AutoOutput;
  runId: string;
  /** The input the run was started with, as the host recorded it. */
  input?: Partial<AutoInput>;
};

export type ExportState =
  | { status: "running"; runId?: string; params: Params; progress: RunProgress | null; cancelling: boolean }
  | { status: "succeeded"; runId: string; params: Params; output: RenderOutput }
  | { status: "failed"; runId?: string; params: Params; error: string }
  | { status: "cancelled"; runId?: string; params: Params };

export type ManualState = {
  /** What "Reset to auto" returns to: the chosen edit, or neutral before one exists. */
  base: Params;
  params: Params;
  /** The newest preview whose params were still current when it arrived. */
  preview?: { id: string; params: Params };
  error?: string;
  export?: ExportState;
};

export type Photo = {
  /** Local id, also the #/retouch/<key> route. */
  key: string;
  filename: string;
  size: number;
  artifactId?: string;
  upload: UploadState;
  inspect: InspectState;
  auto: AutoState;
  /** The newest successful auto run, kept while a re-run is queued or going. */
  result?: AutoResult;
  manual?: ManualState;
};

export type SessionState = {
  photos: Photo[];
  options: RetouchOptions;
  /** A request answered 401: the page asks for the token. */
  locked: boolean;
  /** The last error while polling the host, cleared when it answers again. */
  trouble?: string;
  /** For the polite live region; `id` changes with every message. */
  announcement?: { id: number; text: string };
  downloading?: { kind: "jpeg" | "xmp"; done: number; total: number };
};

/** What a photo's status chip says. */
export type Phase = "uploading" | "reading" | "ready" | "queued" | "working" | "done" | "failed" | "cancelled";

export function photoPhase(photo: Photo): Phase {
  const { auto, inspect, upload } = photo;
  if (upload.status === "waiting" || upload.status === "uploading") return "uploading";
  if (upload.status === "failed") return "failed";
  switch (auto.status) {
    case "queued":
      return "queued";
    case "restoring":
      return "working";
    case "running":
      return auto.serverQueued ? "queued" : "working";
    case "succeeded":
      return "done";
    case "failed":
      return "failed";
    case "cancelled":
      // Cancelling a re-run leaves the earlier result in place.
      return photo.result ? "done" : "cancelled";
    case "idle":
      break;
  }
  if (inspect.status === "idle" || inspect.status === "running") return "reading";
  if (inspect.status === "failed") return "failed";
  return "ready";
}

/** The error behind a "failed" chip. */
export function photoError(photo: Photo): string | undefined {
  if (photo.upload.status === "failed") return photo.upload.error;
  if (photo.auto.status === "failed") return photo.auto.error;
  if (photo.auto.status === "idle" && photo.inspect.status === "failed") return photo.inspect.error;
  return undefined;
}

/** Whether Retouch / Re-run applies: uploaded (or uploading) and not already
    queued or running. */
export function canRetouch(photo: Photo): boolean {
  if (photo.upload.status === "failed") return false;
  const status = photo.auto.status;
  return status === "idle" || status === "succeeded" || status === "failed" || status === "cancelled";
}

/** Whether "Retouch all" picks this photo up: it can run and is not done. */
export function needsRetouch(photo: Photo): boolean {
  return canRetouch(photo) && photoPhase(photo) !== "done";
}

export function isActive(photo: Photo): boolean {
  const status = photo.auto.status;
  return status === "queued" || status === "running" || status === "restoring";
}

export function queueOrder(photo: Photo): number {
  return photo.auto.status === "queued" ? photo.auto.order : Number.POSITIVE_INFINITY;
}

export function photoInfo(photo: Photo): RawInfo | undefined {
  return photo.result?.output.info ?? (photo.inspect.status === "done" ? photo.inspect.output.info : undefined);
}

/** The neutral render: the auto run's "before", or inspect's preview. */
export function neutralId(photo: Photo): string | undefined {
  return photo.result?.output.before.id ?? (photo.inspect.status === "done" ? photo.inspect.output.preview.id : undefined);
}

/** The grid's image: the chosen edit once there is one, else the neutral render. */
export function thumbnailId(photo: Photo): string | undefined {
  return photo.result?.output.after.id ?? neutralId(photo);
}

export type FileRef = { id: string; name: string; size: number };

export type FinalFiles = {
  source?: "manual" | "auto";
  jpeg?: FileRef;
  xmp?: FileRef;
  tiff?: FileRef;
};

/** The files "Download all" takes for a photo: a full-size manual export when
    there is one (the user's last word), else the auto run's exports. */
export function finalFiles(photo: Photo): FinalFiles {
  const exported = photo.manual?.export;
  if (exported?.status === "succeeded") return filesOf(exported.output, photo, "manual");
  if (photo.result) return filesOf(photo.result.output, photo, "auto");
  return {};
}

/** Download names for an export: `<stem>.jpg`, `<stem>.xmp`, `<stem>.tif`. */
export function filesOf(output: AutoOutput | RenderOutput, photo: Photo, source: "manual" | "auto"): FinalFiles {
  const stem = safeFileName(output.stem || stemOf(photo.filename));
  const ref = (artifact: { id: string; size_bytes: number } | null, ext: string): FileRef | undefined =>
    artifact ? { id: artifact.id, name: `${stem}.${ext}`, size: artifact.size_bytes } : undefined;
  return { source, jpeg: ref(output.jpeg, "jpg"), xmp: ref(output.xmp, "xmp"), tiff: ref(output.tiff, "tif") };
}

/** Candidates with Claude's ranking first (1 = favourite), then by score. */
export function rankedCandidates(output: AutoOutput): CandidateSummary[] {
  return output.candidates
    .slice()
    .sort((a, b) => {
      if (a.rank != null && b.rank != null) return a.rank - b.rank;
      if (a.rank != null) return -1;
      if (b.rank != null) return 1;
      return b.score - a.score;
    });
}

export function isFinished(status: RunRecord["status"]): boolean {
  return status === "succeeded" || status === "failed" || status === "cancelled";
}

/* Run outputs arrive as `unknown`: check the parts the page reads, so a host
   built from a different revision fails one photo with a message instead of
   breaking the page. */

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isRef(value: unknown): boolean {
  return isObject(value) && typeof value.id === "string";
}

/** Params with at least the Basic panel. Runs recorded before the other
    panels existed lack them; `completeParams` fills in Lightroom's defaults. */
export function isParams(value: unknown): value is Params {
  return isObject(value) && BASIC_KEYS.every((key) => typeof value[key] === "number");
}

export function asInspectOutput(value: unknown): InspectOutput | undefined {
  if (!isObject(value) || !isObject(value.info) || !isParams(value.neutral) || !isRef(value.preview)) return undefined;
  const output = value as InspectOutput;
  return { ...output, neutral: completeParams(output.neutral) };
}

function isCandidate(value: unknown): value is CandidateSummary {
  return (
    isObject(value) &&
    typeof value.id === "string" &&
    typeof value.label === "string" &&
    isParams(value.params) &&
    isRef(value.preview)
  );
}

/** An auto run's output, or undefined when its core is missing. Malformed
    candidates are dropped and a malformed judge reads as none, so one odd
    field costs a panel, not the page. */
export function asAutoOutput(value: unknown): AutoOutput | undefined {
  if (
    !isObject(value) ||
    !isObject(value.info) ||
    !isParams(value.params) ||
    !isRef(value.before) ||
    !isRef(value.after) ||
    !Array.isArray(value.candidates)
  ) {
    return undefined;
  }
  const output = value as AutoOutput;
  return {
    ...output,
    params: completeParams(output.params),
    candidates: output.candidates.filter(isCandidate).map((c) => ({ ...c, params: completeParams(c.params) })),
    judge: isObject(output.judge) ? output.judge : null,
    timings: isObject(output.timings) ? output.timings : { decode_ms: 0, develop_ms: 0, search_ms: 0, judge_ms: 0, export_ms: 0, total_ms: 0 },
  };
}

export function asRenderOutput(value: unknown): RenderOutput | undefined {
  return isObject(value) && isRef(value.preview) ? (value as RenderOutput) : undefined;
}

export function asAutoInput(value: unknown): Partial<AutoInput> | undefined {
  return isObject(value) ? (value as Partial<AutoInput>) : undefined;
}
