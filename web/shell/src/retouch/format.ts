import type { Params } from "@spl/bindings/raw-retouch/Params";
import type { ParamSpec as BoundSpec } from "@spl/bindings/raw-retouch/ParamSpec";
import { PARAM_SPECS } from "@spl/bindings/raw-retouch/paramSpecs";
import type { RawInfo } from "@spl/bindings/raw-retouch/RawInfo";

export type ParamKey = keyof Params;

/** One Lightroom setting as the app's PARAM_SPECS table describes it: label,
    XMP name, panel, legal range and step, and Lightroom's default. */
export type ParamSpec = Omit<BoundSpec, "key"> & { key: ParamKey };

/** Every setting, in Params order (generated from the Rust table). */
export const SPECS = PARAM_SPECS as ParamSpec[];

const SPEC_OF = new Map(SPECS.map((spec) => [spec.key, spec]));

export function specOf(key: ParamKey): ParamSpec {
  return SPEC_OF.get(key)!;
}

export const PARAM_KEYS: ParamKey[] = SPECS.map((spec) => spec.key);

/** The Basic panel, which every host's Params carries. The other settings
    are missing from runs recorded before they existed and read as
    Lightroom's defaults (`paramValue`, `completeParams`), as on the host. */
export const BASIC_KEYS: ParamKey[] = SPECS.filter((spec) => spec.group === "Basic").map((spec) => spec.key);

/** Lightroom's whole temperature range, for the typed kelvin field. */
export const TEMPERATURE_RANGE = { min: specOf("temperature").min, max: specOf("temperature").max } as const;

/** `key` of `params`, or Lightroom's default when an older run lacks it. */
export function paramValue(params: Params, key: ParamKey): number {
  const value: unknown = params[key];
  return typeof value === "number" && Number.isFinite(value) ? value : (specOf(key).neutral ?? 0);
}

/** `params` with every setting present (older runs carry only the Basic panel). */
export function completeParams(params: Params): Params {
  if (PARAM_KEYS.every((key) => typeof params[key] === "number")) return params;
  const out = { ...params };
  for (const key of PARAM_KEYS) out[key] = paramValue(params, key);
  return out;
}

export function sameParams(a: Params | undefined, b: Params | undefined): boolean {
  if (!a || !b) return false;
  return PARAM_KEYS.every((key) => Math.abs(paramValue(a, key) - paramValue(b, key)) < 1e-6);
}

/** Decimal places of a step: 0.01 → 2, 0.1 → 1, 1 → 0. */
export function stepDecimals(step: number): number {
  return step < 1 ? Math.ceil(-Math.log10(step) - 1e-9) : 0;
}

/** Clamped to the setting's legal range and rounded to its step, the way
    Lightroom stores it (whole kelvin, 0.01 EV, 0.1 px, whole units). */
export function snapParam(key: ParamKey, value: number): number {
  const spec = specOf(key);
  const clamped = Math.min(spec.max, Math.max(spec.min, value));
  const factor = 10 ** stepDecimals(spec.step);
  return Math.round(clamped * factor) / factor;
}

/** Whether `value` is the setting's default: Lightroom's own, or for white
    balance (whose default is as shot) `asShot` when it is known. */
export function isDefault(spec: ParamSpec, value: number, asShot?: { temperature: number; tint: number }): boolean {
  let neutral = spec.neutral;
  if (neutral === null) {
    if (!asShot || (spec.key !== "temperature" && spec.key !== "tint")) return true;
    neutral = asShot[spec.key];
  }
  return snapParam(spec.key, value) === snapParam(spec.key, neutral);
}

/** "+12", "-8", "0" (never "-0"); `decimals` fixed, so "+0.35" and "0.00". */
export function formatSigned(value: number, decimals = 0): string {
  const factor = 10 ** decimals;
  const rounded = Math.round(value * factor) / factor;
  if (rounded === 0) return (0).toFixed(decimals);
  return `${rounded > 0 ? "+" : "-"}${Math.abs(rounded).toFixed(decimals)}`;
}

/** A setting as Lightroom shows it: signed for the bipolar sliders, with the
    unit where one helps, On/Off for the lens switches. */
export function formatParam(key: ParamKey, value: number): string {
  const spec = specOf(key);
  if (key === "temperature") return `${Math.round(value)} K`;
  if (spec.group === "Lens") return value >= 0.5 ? "On" : "Off";
  const decimals = stepDecimals(spec.step);
  const shown = spec.min < 0 ? formatSigned(value, decimals) : (Math.round(value * 10 ** decimals) / 10 ** decimals).toFixed(decimals);
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
