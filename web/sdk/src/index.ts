import type { ArtifactRef } from "@spl/bindings/ArtifactRef";
import type { Health } from "@spl/bindings/Health";
import type { OpDescriptor } from "@spl/bindings/OpDescriptor";
import type { RunDetail } from "@spl/bindings/RunDetail";
import type { RunRecord } from "@spl/bindings/RunRecord";
import type { RunRequest } from "@spl/bindings/RunRequest";

export type { ArtifactRef, Health, OpDescriptor, RunDetail, RunRecord, RunRequest };
export type { ArtifactKind } from "@spl/bindings/ArtifactKind";
export type { RunProgress } from "@spl/bindings/RunProgress";
export type { GpuInfo } from "@spl/bindings/GpuInfo";
export type { GpuReport } from "@spl/bindings/GpuReport";
export type { MetricRecord } from "@spl/bindings/MetricRecord";
export type { RunStatus } from "@spl/bindings/RunStatus";

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

  /** Starts an op and resolves at once with the run in "running" state.
      Poll `runDetail(id)` for progress and the result. */
  startRun(op: string, input: unknown = {}) {
    const body: RunRequest = { op, input, wait: false };
    return this.request<RunRecord>("POST", "/v1/runs", body);
  }

  /** Asks a running op to stop; it ends as "cancelled". */
  cancelRun(id: string) {
    return this.request<RunRecord>("POST", `/v1/runs/${encodeURIComponent(id)}/cancel`);
  }

  /** Uploads a file as an artifact (content-addressed; the same bytes give the same id). */
  uploadArtifact(
    file: Blob,
    options: {
      filename?: string;
      mediaType?: string;
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
    return response.json() as Promise<T>;
  }
}
