import type { Announcement } from "@spl/bindings/Announcement";
import type { ArtifactKind } from "@spl/bindings/ArtifactKind";
import type { ArtifactRef } from "@spl/bindings/ArtifactRef";
import type { Health } from "@spl/bindings/Health";
import type { OpDescriptor } from "@spl/bindings/OpDescriptor";
import type { RunDetail } from "@spl/bindings/RunDetail";
import type { RunRecord } from "@spl/bindings/RunRecord";
import type { RunRequest } from "@spl/bindings/RunRequest";
import type { RunStatus } from "@spl/bindings/RunStatus";
import type { UploadRequest } from "@spl/bindings/UploadRequest";
import type { UploadSession } from "@spl/bindings/UploadSession";

export type {
  Announcement,
  ArtifactKind,
  ArtifactRef,
  Health,
  OpDescriptor,
  RunDetail,
  RunRecord,
  RunRequest,
  RunStatus,
  UploadRequest,
  UploadSession,
};
export type { RunProgress } from "@spl/bindings/RunProgress";
export type { GpuInfo } from "@spl/bindings/GpuInfo";
export type { GpuReport } from "@spl/bindings/GpuReport";
export type { GpuUse } from "@spl/bindings/GpuUse";
export type { MetricRecord } from "@spl/bindings/MetricRecord";
export type { WaitReason } from "@spl/bindings/WaitReason";

/** Media types the hosts serve 3D results with (`spl_core::media`). */
export const MEDIA = {
  spz: "application/x-spz",
  ply: "application/ply",
  glb: "model/gltf-binary",
  json: "application/json",
} as const;

/** Whether a run has ended (succeeded, failed or cancelled). */
export function isFinished(status: RunStatus): boolean {
  return status !== "queued" && status !== "running";
}

export type WatchOptions = {
  /** Every poll's detail, including the last. */
  onDetail?: (detail: RunDetail) => void;
  /** Once per new artifact of a slot: the first announcement of the slot
      and every one with a higher revision. Fetch the artifact here. */
  onAnnounce?: (announcement: Announcement) => void;
  /** Milliseconds between polls while the run goes. Default 1000. */
  intervalMs?: number;
  signal?: AbortSignal;
};

export class SplError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export type SplClientOptions = {
  /** Origin of the host, e.g. "https://gpu-box:7070". Empty = same origin. */
  baseUrl?: string;
  /** Bearer token when the host sets SPL_AUTH_TOKEN. */
  token?: string;
};

/** One client for every host: local Linux box, cloud VM, or a dev proxy. */
export class SplClient {
  private readonly baseUrl: string;
  private readonly token?: string;

  constructor(options: SplClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "").replace(/\/$/, "");
    this.token = options.token;
  }

  health() {
    return this.request<Health>("GET", "/v1/health");
  }

  ops() {
    return this.request<OpDescriptor[]>("GET", "/v1/ops");
  }

  /** Runs an op to completion. A failed op still resolves, with status "failed". */
  run(op: string, input: unknown = {}) {
    const body: RunRequest = { op, input };
    return this.request<RunRecord>("POST", "/v1/runs", body);
  }

  /** Starts an op and resolves at once with the run "running", or "queued"
      while the host already runs its limit of ops (they start in order).
      Poll `runDetail(id)` for progress and the result. */
  startRun(op: string, input: unknown = {}) {
    const body: RunRequest = { op, input, wait: false };
    return this.request<RunRecord>("POST", "/v1/runs", body);
  }

  /** Asks a queued or running op to stop and resolves with the run as it is
      now; it ends as "cancelled" shortly after. Rejects with 409 when the run
      has already ended. */
  cancelRun(id: string) {
    return this.request<RunRecord>("POST", `/v1/runs/${encodeURIComponent(id)}/cancel`);
  }

  /** Uploads a file as an artifact (content-addressed; the same bytes give the same id).
      The host streams it to disk, so phone videos up to `Health.max_upload_bytes` work. */
  uploadArtifact(
    file: Blob,
    options: {
      filename?: string;
      mediaType?: string;
      /** Recorded kind, e.g. "splat" for a .spz to view; "blob" when omitted. */
      kind?: ArtifactKind;
      onProgress?: (fraction: number) => void;
      signal?: AbortSignal;
    } = {},
  ): Promise<ArtifactRef> {
    // XMLHttpRequest, not fetch: fetch cannot report upload progress.
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${this.baseUrl}/v1/artifacts`);
      xhr.setRequestHeader(
        "content-type",
        options.mediaType || file.type || "application/octet-stream",
      );
      const filename = options.filename ?? (file instanceof File ? file.name : undefined);
      if (filename) xhr.setRequestHeader("x-spl-filename", encodeURIComponent(filename));
      if (options.kind) xhr.setRequestHeader("x-spl-kind", options.kind);
      if (this.token) xhr.setRequestHeader("authorization", `Bearer ${this.token}`);
      xhr.responseType = "json";
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) options.onProgress?.(event.loaded / event.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as ArtifactRef);
        else reject(new SplError(xhr.status, xhr.response?.error ?? xhr.statusText));
      };
      xhr.onerror = () => reject(new TypeError("The host did not answer."));
      xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
      options.signal?.addEventListener("abort", () => xhr.abort(), { once: true });
      xhr.send(file);
    });
  }

  /** Opens a resumable chunked upload (`POST /v1/uploads`): the file goes up
      in parts of `part_bytes`, so no request passes a proxy's body limit. */
  createUpload(request: UploadRequest) {
    return this.request<UploadSession>("POST", "/v1/uploads", request);
  }

  /** The parts the host already holds, to resume an upload. */
  uploadStatus(id: string) {
    return this.request<UploadSession>("GET", `/v1/uploads/${encodeURIComponent(id)}`);
  }

  /** Sends part `index` (exactly its bytes); a part sent again replaces itself. */
  putUploadPart(
    id: string,
    index: number,
    part: Blob,
    options: { onProgress?: (loaded: number) => void; signal?: AbortSignal } = {},
  ): Promise<UploadSession> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", `${this.baseUrl}/v1/uploads/${encodeURIComponent(id)}/parts/${index}`);
      xhr.setRequestHeader("content-type", "application/octet-stream");
      if (this.token) xhr.setRequestHeader("authorization", `Bearer ${this.token}`);
      xhr.responseType = "json";
      xhr.upload.onprogress = (event) => options.onProgress?.(event.loaded);
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as UploadSession);
        else reject(new SplError(xhr.status, xhr.response?.error ?? xhr.statusText));
      };
      xhr.onerror = () => reject(new TypeError("The host did not answer."));
      xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
      options.signal?.addEventListener("abort", () => xhr.abort(), { once: true });
      xhr.send(part);
    });
  }

  /** Assembles the parts into an artifact; 409 while parts are missing. */
  completeUpload(id: string) {
    return this.request<ArtifactRef>("POST", `/v1/uploads/${encodeURIComponent(id)}/complete`);
  }

  /** Drops an upload and its parts. */
  abortUpload(id: string) {
    return this.request<void>("DELETE", `/v1/uploads/${encodeURIComponent(id)}`);
  }

  runs(filter: { op?: string; limit?: number } = {}) {
    const params = new URLSearchParams();
    if (filter.op) params.set("op", filter.op);
    if (filter.limit) params.set("limit", String(filter.limit));
    const query = params.size ? `?${params}` : "";
    return this.request<RunRecord[]>("GET", `/v1/runs${query}`);
  }

  runDetail(id: string) {
    return this.request<RunDetail>("GET", `/v1/runs/${encodeURIComponent(id)}`);
  }

  /** Polls a run until it ends and resolves with its last detail. Calls
      `onAnnounce` once per new intermediate artifact (a point cloud, then
      splat snapshots), so a viewer can swap each in as it lands. Network
      errors and 5xx answers retry with backoff (the host may be
      restarting); other errors reject. Rejects with an AbortError when
      `signal` fires. */
  async watchRun(id: string, options: WatchOptions = {}): Promise<RunDetail> {
    const interval = options.intervalMs ?? 1000;
    const seen = new Map<string, number>();
    let failures = 0;
    for (;;) {
      options.signal?.throwIfAborted();
      let detail: RunDetail;
      try {
        detail = await this.runDetail(id);
        failures = 0;
      } catch (error) {
        const transient = !(error instanceof SplError) || error.status >= 500;
        if (!transient) throw error;
        failures += 1;
        await sleep(Math.min(interval * 2 ** Math.min(failures, 4), 15_000), options.signal);
        continue;
      }
      options.onDetail?.(detail);
      for (const announcement of detail.announced) {
        if ((seen.get(announcement.slot) ?? 0) < announcement.revision) {
          seen.set(announcement.slot, announcement.revision);
          options.onAnnounce?.(announcement);
        }
      }
      if (isFinished(detail.run.status)) return detail;
      await sleep(interval, options.signal);
    }
  }

  /** With `download`, the host answers with Content-Disposition: attachment under that name. */
  artifactUrl(id: string, options: { download?: string } = {}) {
    const query = options.download ? `?download=${encodeURIComponent(options.download)}` : "";
    return `${this.baseUrl}/v1/artifacts/${encodeURIComponent(id)}${query}`;
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = {};
    if (body !== undefined) headers["content-type"] = "application/json";
    if (this.token) headers.authorization = `Bearer ${this.token}`;
    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new SplError(response.status, detail?.error ?? response.statusText);
    }
    if (response.status === 204) return undefined as T;
    return response.json() as Promise<T>;
  }
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", abort);
      resolve();
    }, ms);
    const abort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    signal?.addEventListener("abort", abort, { once: true });
  });
}
