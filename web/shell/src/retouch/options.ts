import type { AutoInput } from "@spl/bindings/raw-retouch/AutoInput";
import type { ExportOptions } from "@spl/bindings/raw-retouch/ExportOptions";

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
  png: boolean;
  /** Bits per channel of the PNG. */
  pngBits: 8 | 16;
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
  png: true,
  pngBits: 8,
  jpegQuality: 95,
  evalsPerLook: 1500,
  parallel: 2,
};

/** spl's default SPL_MAX_CONCURRENT_RUNS (the host does not report its own).
    Retouches leave one of these slots free, so previews, zoom tiles and
    exports never wait behind a retouch. */
export const HOST_SLOTS = 3;

export const LIMITS = {
  jpegQuality: { min: 50, max: 100, step: 1 },
  evalsPerLook: { min: 100, max: 20000, step: 100 },
  parallel: { min: 1, max: HOST_SLOTS - 1 },
} as const;

/** The longest edit instruction the host takes. */
export const INSTRUCTION_MAX = 500;

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
    png: bool(v.png, d.png),
    pngBits: v.pngBits === 16 ? 16 : v.pngBits === 8 ? 8 : d.pngBits,
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

/** An instruction as sent: trimmed, at most 500 characters, undefined when empty. */
export function cleanInstruction(value: string | null | undefined): string | undefined {
  const trimmed = (value ?? "").trim().slice(0, INSTRUCTION_MAX).trim();
  return trimmed || undefined;
}

/** What a photo's retouch follows: its own box, else the default box. */
export function resolveInstruction(own: string | undefined, fallback: string | undefined): string | undefined {
  return cleanInstruction(own) ?? cleanInstruction(fallback);
}

/** The export files these options ask for. A manual full-size export always
    takes the JPEG (`jpeg`), whatever the auto options say. */
export function exportOptions(options: RetouchOptions, jpeg = options.jpeg): ExportOptions {
  return {
    jpeg,
    jpeg_quality: options.jpegQuality,
    tiff16: options.tiff16,
    xmp: options.xmp,
    png: options.png,
    png_bits: options.pngBits,
  };
}

/** The raw_retouch.auto input for one photo under these options. */
export function autoInput(options: RetouchOptions, raw: string, filename: string, instruction?: string): AutoInput {
  const asked = cleanInstruction(instruction);
  return {
    raw,
    filename,
    use_claude: options.useClaude,
    model: options.model,
    effort: options.effort,
    plan: options.useClaude && options.plan,
    refine: options.useClaude && options.refine,
    evals_per_look: options.evalsPerLook,
    export: exportOptions(options),
    ...(asked ? { instruction: asked } : {}),
  };
}
