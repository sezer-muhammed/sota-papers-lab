import type { Health } from "@spl/bindings/Health";
import type { OpDescriptor } from "@spl/bindings/OpDescriptor";
import type { RunDetail } from "@spl/bindings/RunDetail";
import type { RunRecord } from "@spl/bindings/RunRecord";
import type { RunRequest } from "@spl/bindings/RunRequest";

export type { Health, OpDescriptor, RunDetail, RunRecord, RunRequest };
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

  artifactUrl(id: string) {
    return `${this.baseUrl}/v1/artifacts/${encodeURIComponent(id)}`;
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
