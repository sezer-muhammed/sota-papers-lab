/* Plain-language run status for #/3d: what a run waits for, which stage it
   is in, and how long the rest should take, from the stage timings of the
   runs this host already finished. Pure, so it is unit tested. */
import type { RunRecord } from "@spl/sdk";

export type FlowStage = "ingest" | "posing" | "init" | "refine" | "mesh" | "export";
export const FLOW_STAGES: readonly FlowStage[] = ["ingest", "posing", "init", "refine", "mesh", "export"];

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
const obj = (v: unknown) => (v && typeof v === "object" ? (v as Record<string, unknown>) : undefined);

/** The op's "waiting for GPU memory" progress data, when that is the state. */
export type GpuWait = { needBytes?: number; freeBytes?: number; waitedS?: number; timeoutS?: number };

export function gpuWait(data: unknown): GpuWait | undefined {
  const d = obj(data);
  if (!d || d.waiting_for_gpu_memory !== true) return undefined;
  return { needBytes: num(d.need_bytes), freeBytes: num(d.free_bytes), waitedS: num(d.waited_s), timeoutS: num(d.timeout_s) };
}

/** The stage a progress update belongs to: its `stage` field, else its message. */
export function stageOf(message: string | undefined, data?: unknown): FlowStage | undefined {
  const s = obj(data)?.stage;
  if (typeof s === "string" && (FLOW_STAGES as readonly string[]).includes(s)) return s as FlowStage;
  const m = (message ?? "").toLowerCase();
  if (m.startsWith("reading") || m.startsWith("ingest") || m.startsWith("decoding")) return "ingest";
  if (m.startsWith("engine") || m.startsWith("posing") || m.includes("vggt")) return "posing";
  if (m.startsWith("initializ") || m.startsWith("stage b") || m.startsWith("bundle")) return "init";
  if (m.startsWith("training") || m.startsWith("refin")) return "refine";
  if (m.startsWith("mesh")) return "mesh";
  if (m.startsWith("export") || m.startsWith("storing")) return "export";
  return undefined;
}

/** "training 1200/7000: 812345 Gaussians, 95 it/s, ..." → its numbers. */
export function trainingRate(message: string | undefined): { done: number; total: number; perS?: number } | undefined {
  const m = /training (\d+)\/(\d+)(?:.*?([\d.]+) it\/s)?/.exec(message ?? "");
  if (!m) return undefined;
  return { done: Number(m[1]), total: Number(m[2]), perS: m[3] ? Number(m[3]) : undefined };
}

/** Seconds each stage took in a finished reconstruction (its `times`). */
export function stageSeconds(output: unknown): Partial<Record<FlowStage, number>> | undefined {
  const o = obj(output);
  const t = obj(o?.times);
  if (!o || !t) return undefined;
  const sum = (...keys: string[]) => keys.reduce((a, k) => a + (num(t[k]) ?? 0), 0);
  const out: Partial<Record<FlowStage, number>> = {
    posing: sum("engine_load_s", "engine_s"),
    init: sum("setup_s", "init_s", "prune_s", "bootstrap_s", "geo_s", "ba_s"),
    refine: sum("train_s"),
    export: sum("export_s"),
  };
  const ingest = num(o.ingest_s);
  if (ingest !== undefined) out.ingest = ingest;
  return out;
}

/** Frames a run used or asked for. */
export function framesOf(run: Pick<RunRecord, "input" | "output">): number | undefined {
  const o = obj(run.output);
  const i = obj(run.input);
  return (
    num(o?.frames) ??
    num(obj(o?.summary)?.images) ??
    (Array.isArray(i?.images) ? i.images.length : undefined) ??
    num(i?.max_frames)
  );
}

export const presetOf = (run: Pick<RunRecord, "input">) => {
  const p = obj(run.input)?.preset;
  return typeof p === "string" ? p : undefined;
};

/** Per-frame seconds of each stage: the median over finished runs of the
    same preset (all presets when none match). */
export type StageModel = Partial<Record<FlowStage, number>>;

export function stageModel(runs: Pick<RunRecord, "status" | "input" | "output">[], preset?: string): StageModel {
  const done = runs.filter((r) => r.status === "succeeded" && stageSeconds(r.output));
  const same = done.filter((r) => presetOf(r) === preset);
  const pool = same.length ? same : done;
  const model: StageModel = {};
  for (const s of FLOW_STAGES) {
    const per = pool.flatMap((r) => {
      const secs = stageSeconds(r.output)?.[s];
      const n = framesOf(r) ?? 1;
      return secs !== undefined && secs > 0 ? [secs / Math.max(1, n)] : [];
    });
    if (per.length) model[s] = median(per);
  }
  return model;
}

function median(xs: number[]) {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

/** Seconds left: the rest of the current stage plus the later stages.
    Training reports its own rate, which wins over the model. Undefined
    when nothing is known about the stages still to run. */
export function remainingSeconds(opts: {
  model: StageModel;
  frames: number;
  stage: FlowStage | undefined;
  /** Seconds since the current stage began. */
  stageElapsedS: number;
  message?: string;
  mesh: boolean;
}): number | undefined {
  const { model, frames, stage, stageElapsedS, message, mesh } = opts;
  if (!stage) return undefined;
  const at = FLOW_STAGES.indexOf(stage);
  let left = 0;
  let known = false;
  const rate = stage === "refine" ? trainingRate(message) : undefined;
  if (rate?.perS && rate.perS > 0) {
    left += Math.max(0, rate.total - rate.done) / rate.perS;
    known = true;
  } else if (model[stage] !== undefined) {
    left += Math.max(0, model[stage]! * frames - stageElapsedS);
    known = true;
  }
  for (const s of FLOW_STAGES.slice(at + 1)) {
    if (s === "mesh" && !mesh) continue;
    if (model[s] !== undefined) {
      left += model[s]! * frames;
      known = true;
    }
  }
  return known ? left : undefined;
}

/** 1-based place of a queued run among the queued runs, oldest first. */
export function queuePosition(runs: Pick<RunRecord, "id" | "status" | "created_at_ms">[], id: string): number | undefined {
  const queued = runs.filter((r) => r.status === "queued").sort((a, b) => a.created_at_ms - b.created_at_ms);
  const i = queued.findIndex((r) => r.id === id);
  return i < 0 ? undefined : i + 1;
}

/** 75 → "1 min 15 s"; 8 → "8 s"; 3700 → "1 h 2 min". */
export function duration(s: number): string {
  const t = Math.max(0, Math.round(s));
  if (t < 60) return `${t} s`;
  if (t < 3600) return `${Math.floor(t / 60)} min${t % 60 ? ` ${t % 60} s` : ""}`;
  return `${Math.floor(t / 3600)} h ${Math.round((t % 3600) / 60)} min`;
}

/** The headline for a run that is not making progress yet, or undefined. */
export function waitHeadline(opts: { queued: boolean; reason?: "gpu" | "slot" | null; wait?: GpuWait; position?: number }): string | undefined {
  const { queued, reason, wait, position } = opts;
  const place = position && position > 1 ? ` You are number ${position} in the queue.` : "";
  if (queued && reason === "gpu") return `The GPU is busy with another job; your run starts automatically.${place}`;
  if (queued) return `The host is running its limit of jobs; your run starts automatically.${place}`;
  if (wait) return "The GPU is busy with another job; your run starts automatically when enough memory is free.";
  return undefined;
}
