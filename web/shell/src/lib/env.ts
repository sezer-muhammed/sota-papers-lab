import type { GpuReport, Health } from "@spl/sdk";
import { isUnauthorized, type Loaded } from "../api";

/* The host's environment snapshot. /v1/health and every run carry it as
   untyped JSON ({ os, arch, spl_version, gpu }); read it defensively. */
type HostEnv = { os?: string; arch?: string; splVersion?: string; gpu?: GpuReport };

export function readEnv(value: unknown): HostEnv {
  if (!value || typeof value !== "object") return {};
  const env = value as Record<string, unknown>;
  const gpu = env.gpu;
  const hasDevices =
    typeof gpu === "object" && gpu !== null && Array.isArray((gpu as { devices?: unknown }).devices);

  return {
    os: typeof env.os === "string" ? env.os : undefined,
    arch: typeof env.arch === "string" ? env.arch : undefined,
    splVersion: typeof env.spl_version === "string" ? env.spl_version : undefined,
    gpu: hasDevices ? (gpu as GpuReport) : undefined,
  };
}

export type HostState = "connecting" | "online" | "no-gpu" | "offline" | "locked";
/** The states Home can see: on a health 401 App shows the token prompt instead of Home. */
export type HomeHostState = Exclude<HostState, "locked">;

/* One reading of /v1/health for the header pill, the host strip, modes card 01
   and the CTA title. A failed reload wins over old data; a reload in flight
   keeps the last state, so it never drops back to "connecting". */
export function hostState(health: Loaded<Health>): HostState {
  return isUnauthorized(health.error) ? "locked" : homeHostState(health);
}

/** hostState for Home and its panels. A 401 never reaches them (see HomeHostState). */
export function homeHostState(health: Loaded<Health>): HomeHostState {
  if (health.error) return "offline";
  if (!health.data) return "connecting";
  return (readEnv(health.data.env).gpu?.devices.length ?? 0) > 0 ? "online" : "no-gpu";
}
