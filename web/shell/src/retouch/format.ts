import type { ParamSpec as SpecRow } from "@spl/bindings/raw-retouch/ParamSpec";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import { PARAM_SPECS } from "@spl/bindings/raw-retouch/paramSpecs";
import type { RawInfo } from "@spl/bindings/raw-retouch/RawInfo";

export type ParamKey = keyof Params;

/** One Params field as the host's PARAM_SPECS (apps/raw-retouch/src/types.rs)
    describes it: Lightroom's label, XMP name, panel, range, step, default. */
export type ParamSpec = Omit<SpecRow, "key"> & { key: ParamKey };

/** Every Params field, in the host's order. */
export const SPECS = PARAM_SPECS as ParamSpec[];

export const PARAM_KEYS: ParamKey[] = SPECS.map((spec) => spec.key);

/** The Basic panel: the fields every run has carried. Runs recorded before
    the other fields existed lack them (see completeParams). */
export const BASIC_KEYS: ParamKey[] = SPECS.filter((spec) => spec.group === "Basic").map((spec) => spec.key);

const BY_KEY = new Map(SPECS.map((spec) => [spec.key, spec]));

export function specOf(key: ParamKey): ParamSpec {
  const spec = BY_KEY.get(key);
  if (!spec) throw new Error(`No ParamSpec for ${key}`);
  return spec;
}

/** Lightroom's whole temperature range, which the kelvin field takes. */
export const TEMPERATURE_RANGE = { min: specOf("temperature").min, max: specOf("temperature").max } as const;

/** The temperature slider spans 2000 to 12000 K, where photos live. */
export const TEMPERATURE_SLIDER = { min: 2000, max: 12000, step: 50 } as const;

/** An on/off setting (Lens Corrections), stored as 0 or 1 like Lightroom. */
export const isSwitch = (spec: ParamSpec) => spec.min === 0 && spec.max === 1 && spec.step === 1;

/** Decimals the step shows: 2 for exposure, 1 for the sharpening radius. */
export const decimalsOf = (spec: ParamSpec) => (spec.step >= 1 ? 0 : Math.round(-Math.log10(spec.step)));

/** A field's value, or Lightroom's default where an older run lacks it. */
function valueOf(params: Params, spec: ParamSpec): number {
  const value = (params as Record<string, unknown>)[spec.key];
  return typeof value === "number" && Number.isFinite(value) ? value : (spec.neutral ?? 0);
}

/** `params` with every field: those a run recorded before they existed take
    Lightroom's defaults, as the host's serde defaults do. Complete input
    comes back as is. */
export function completeParams(params: Params): Params {
  const raw = params as Record<string, unknown>;
  const missing = SPECS.filter((spec) => {
    const value = raw[spec.key];
    return typeof value !== "number" || !Number.isFinite(value);
  });
  if (missing.length === 0) return params;
  const out: Record<string, unknown> = { ...raw };
  for (const spec of missing) out[spec.key] = spec.neutral ?? 0;
  return out as Params;
}

export function sameParams(a: Params | undefined, b: Params | undefined): boolean {
  if (!a || !b) return false;
  return SPECS.every((spec) => Math.abs(valueOf(a, spec) - valueOf(b, spec)) < 1e-6);
}

/** Clamped to the field's legal range and rounded to its step, the way
    Lightroom stores it (whole kelvin, 0.01 EV, 0.1 px, whole units): the
    host's `ParamSpec::quantize`. */
export function snapParam(key: ParamKey, value: number): number {
  const spec = specOf(key);
  const clamped = Math.min(spec.max, Math.max(spec.min, value));
  if (spec.step >= 1) return Math.round(clamped);
  const per = Math.round(1 / spec.step);
  return Math.round(clamped * per) / per;
}

/** Whether `value` is away from Lightroom's default: the spec's, or for white
    balance (whose default is as shot) the photo's as-shot value. False when
    there is nothing to compare with. */
export function awayFromDefault(
  key: ParamKey,
  value: number,
  asShot?: { temperature: number; tint: number },
): boolean {
  const spec = specOf(key);
  const reference = spec.neutral ?? (key === "temperature" || key === "tint" ? asShot?.[key] : undefined);
  if (reference == null || !Number.isFinite(reference) || !Number.isFinite(value)) return false;
  return Math.abs(snapParam(key, value) - snapParam(key, reference)) > 1e-6;
}

/** "+12", "-8", "0" (never "-0"); `decimals` fixed, so "+0.35" and "0.00". */
export function formatSigned(value: number, decimals = 0): string {
  const factor = 10 ** decimals;
  const rounded = Math.round(value * factor) / factor;
  if (rounded === 0) return (0).toFixed(decimals);
  return `${rounded > 0 ? "+" : "-"}${Math.abs(rounded).toFixed(decimals)}`;
}

/** A value as Lightroom shows it, with the unit where one helps: "5200 K",
    "+0.35 EV", "+12", "40", "1.0", "On". */
export function formatParam(key: ParamKey, value: number): string {
  if (key === "temperature") return `${Math.round(value)} K`;
  const spec = specOf(key);
  if (isSwitch(spec)) return value >= 0.5 ? "On" : "Off";
  const decimals = decimalsOf(spec);
  const factor = 10 ** decimals;
  const shown = spec.min < 0 ? formatSigned(value, decimals) : (Math.round(value * factor) / factor).toFixed(decimals);
  return key === "exposure" ? `${shown} EV` : shown;
}

const trim = (s: string) => s.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");

/** "1/250 s", "0.5 s", "2 s", "30 s". */
export function formatShutter(seconds: number | null): string | undefined {
  if (seconds == null || !(seconds > 0)) return undefined;
  if (seconds < 0.3) return `1/${Math.round(1 / seconds)} s`;
  return `${trim(seconds.toFixed(seconds < 10 ? 1 : 0))} s`;
}

/* Makes as cameras write them, as people say them. */
const MAKES: Record<string, string> = {
  SONY: "Sony",
  NIKON: "Nikon",
  "NIKON CORPORATION": "Nikon",
  CANON: "Canon",
  FUJIFILM: "Fujifilm",
  OLYMPUS: "Olympus",
  "OLYMPUS CORPORATION": "Olympus",
  "OLYMPUS IMAGING CORP.": "Olympus",
  "OM DIGITAL SOLUTIONS": "OM System",
  PANASONIC: "Panasonic",
  LEICA: "Leica",
  "LEICA CAMERA AG": "Leica",
  PENTAX: "Pentax",
  "RICOH IMAGING COMPANY, LTD.": "Ricoh",
  HASSELBLAD: "Hasselblad",
};

/** "Sony ILCE-7M3"; the model alone when it already names the make. */
export function cameraName(info: Pick<RawInfo, "make" | "model">): string {
  const rawMake = (info.make ?? "").trim();
  const model = (info.model ?? "").trim();
  const make = MAKES[rawMake.toUpperCase()] ?? rawMake;
  if (!make) return model || "Unknown camera";
  if (!model) return make;
  return model.toLowerCase().startsWith(make.toLowerCase()) ? model : `${make} ${model}`;
}

/** Camera and lens: "Sony ILCE-7M3 · FE 16-35mm F2.8 GM". */
export function cameraLine(info: RawInfo): string {
  return [cameraName(info), info.lens?.trim()].filter(Boolean).join(" · ");
}

/** "1/250 s · f/2.8 · ISO 400 · 35 mm", leaving out what the file lacks. */
export function exposureLine(info: RawInfo): string {
  return [
    formatShutter(info.exposure_time_s),
    info.f_number ? `f/${trim(info.f_number.toFixed(1))}` : undefined,
    info.iso ? `ISO ${info.iso}` : undefined,
    info.focal_length_mm ? `${Math.round(info.focal_length_mm)} mm` : undefined,
  ]
    .filter(Boolean)
    .join(" · ");
}

/** Width and height as the photo is shown (EXIF orientations 5 to 8 turn it). */
export function uprightSize(info: RawInfo): { width: number; height: number } {
  const turned = info.orientation >= 5 && info.orientation <= 8;
  return turned ? { width: info.height, height: info.width } : { width: info.width, height: info.height };
}

/** EXIF "2024:05:01 10:23:45" as "2024-05-01 10:23". */
export function formatCaptured(value: string | null): string | undefined {
  if (!value) return undefined;
  const m = /^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2})/.exec(value.trim());
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : value;
}

/** Decimal units, like the Finder: "48.2 MB". */
export function formatBytes(bytes: number): string {
  if (bytes < 1000) return `${bytes} B`;
  if (bytes < 1e6) return `${Math.round(bytes / 1e3)} kB`;
  if (bytes < 1e9) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

/** "640 ms", "2.35 s", "12.4 s", "3.2 min". */
export function formatMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms) || ms < 0) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
  if (ms < 3_600_000) return `${(ms / 60_000).toFixed(1)} min`;
  return `${(ms / 3_600_000).toFixed(1)} h`;
}

const scoreFormat = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 3 });

/** Objective scores have no fixed scale: three significant digits. */
export const formatScore = (score: number) => (Number.isFinite(score) ? scoreFormat.format(score) : "—");

export const formatPercent = (fraction: number) => `${Math.round(fraction * 100)}%`;

/** The file name without its extension: "DSC0001.ARW" → "DSC0001". */
export function stemOf(filename: string): string {
  const base = filename.split(/[\\/]/).pop() ?? filename;
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(0, dot) : base;
}

/** A download name the browser accepts as is on every OS. */
export function safeFileName(name: string): string {
  return name.replace(/[\\/:*?"<>|\u0000-\u001f]+/g, "_").trim() || "photo";
}
