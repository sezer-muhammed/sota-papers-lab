import type { AutoInput } from "@spl/bindings/raw-retouch/AutoInput";

/** The Retouch page's settings, saved per browser. They shape the next
    raw_retouch.auto runs; a run that has started keeps its own. */
export const EFFORTS = ["low", "medium", "high"] as const;
export type Effort = (typeof EFFORTS)[number];

export type RetouchOptions = {
  useClaude: boolean;
  model: "opus" | "sonnet";
  /** How long Claude thinks per call: lower answers sooner. */
  effort: Effort;
  plan: boolean;
  refine: boolean;
  jpeg: boolean;
  xmp: boolean;
  tiff16: boolean;
  jpegQuality: number;
  evalsPerLook: number;
  parallel: number;
};

export const DEFAULT_OPTIONS: RetouchOptions = {
  useClaude: true,
  model: "opus",
  effort: "medium",
  plan: true,
  refine: true,
  jpeg: true,
  xmp: true,
  tiff16: false,
  jpegQuality: 95,
  evalsPerLook: 1500,
  parallel: 2,
};

export const LIMITS = {
  jpegQuality: { min: 50, max: 100, step: 1 },
  evalsPerLook: { min: 100, max: 20000, step: 100 },
  parallel: { min: 1, max: 4 },
} as const;

const OPTIONS_KEY = "spl-retouch-options";

const clampInt = (value: unknown, min: number, max: number, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? Math.min(max, Math.max(min, Math.round(value))) : fallback;

const bool = (value: unknown, fallback: boolean) => (typeof value === "boolean" ? value : fallback);

/** Any stored or edited value made legal; unknown fields fall back to defaults. */
export function normalizeOptions(value: unknown): RetouchOptions {
  const v = (typeof value === "object" && value !== null ? value : {}) as Partial<Record<keyof RetouchOptions, unknown>>;
  const d = DEFAULT_OPTIONS;
  return {
    useClaude: bool(v.useClaude, d.useClaude),
    model: v.model === "sonnet" || v.model === "opus" ? v.model : d.model,
    effort: EFFORTS.find((e) => e === v.effort) ?? d.effort,
    plan: bool(v.plan, d.plan),
    refine: bool(v.refine, d.refine),
    jpeg: bool(v.jpeg, d.jpeg),
    xmp: bool(v.xmp, d.xmp),
    tiff16: bool(v.tiff16, d.tiff16),
    jpegQuality: clampInt(v.jpegQuality, LIMITS.jpegQuality.min, LIMITS.jpegQuality.max, d.jpegQuality),
    evalsPerLook: clampInt(v.evalsPerLook, LIMITS.evalsPerLook.min, LIMITS.evalsPerLook.max, d.evalsPerLook),
    parallel: clampInt(v.parallel, LIMITS.parallel.min, LIMITS.parallel.max, d.parallel),
  };
}

export type OptionsStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export function loadOptions(storage: OptionsStorage | undefined): RetouchOptions {
  try {
    const text = storage?.getItem(OPTIONS_KEY);
    return normalizeOptions(text ? JSON.parse(text) : undefined);
  } catch {
    return { ...DEFAULT_OPTIONS };
  }
}

export function saveOptions(storage: OptionsStorage | undefined, options: RetouchOptions) {
  try {
    storage?.setItem(OPTIONS_KEY, JSON.stringify(options));
  } catch {
    /* private mode or full: the options just aren't remembered */
  }
}

/** The raw_retouch.auto input for one photo under these options. */
export function autoInput(options: RetouchOptions, raw: string, filename: string): AutoInput {
  return {
    raw,
    filename,
    use_claude: options.useClaude,
    model: options.model,
    effort: options.effort,
    plan: options.useClaude && options.plan,
    refine: options.useClaude && options.refine,
    evals_per_look: options.evalsPerLook,
    export: {
      jpeg: options.jpeg,
      jpeg_quality: options.jpegQuality,
      tiff16: options.tiff16,
      xmp: options.xmp,
    },
  };
}
