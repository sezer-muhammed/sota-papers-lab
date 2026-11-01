import type { Params } from "@spl/bindings/raw-retouch/Params";
import type { RawInfo } from "@spl/bindings/raw-retouch/RawInfo";

export type ParamKey = keyof Params;

export type ParamSpec = {
  key: ParamKey;
  label: string;
  group: "White balance" | "Tone" | "Presence";
  /** Slider range and step (Lightroom's own, except temperature: see below). */
  min: number;
  max: number;
  step: number;
  /** Half-width of the table bar; none for temperature, which has no centre. */
  bar?: number;
};

/* Lightroom Classic's Basic panel, in its order. The temperature slider spans
   2000 to 12000 K where photos live; the number field beside it takes
   Lightroom's full 2000 to 50000. */
export const PARAMS: ParamSpec[] = [
  { key: "temperature", label: "Temp", group: "White balance", min: 2000, max: 12000, step: 50 },
  { key: "tint", label: "Tint", group: "White balance", min: -150, max: 150, step: 1, bar: 150 },
  { key: "exposure", label: "Exposure", group: "Tone", min: -5, max: 5, step: 0.01, bar: 5 },
  { key: "contrast", label: "Contrast", group: "Tone", min: -100, max: 100, step: 1, bar: 100 },
  { key: "highlights", label: "Highlights", group: "Tone", min: -100, max: 100, step: 1, bar: 100 },
  { key: "shadows", label: "Shadows", group: "Tone", min: -100, max: 100, step: 1, bar: 100 },
  { key: "whites", label: "Whites", group: "Tone", min: -100, max: 100, step: 1, bar: 100 },
  { key: "blacks", label: "Blacks", group: "Tone", min: -100, max: 100, step: 1, bar: 100 },
  { key: "vibrance", label: "Vibrance", group: "Presence", min: -100, max: 100, step: 1, bar: 100 },
  { key: "saturation", label: "Saturation", group: "Presence", min: -100, max: 100, step: 1, bar: 100 },
];

export const PARAM_KEYS = PARAMS.map((spec) => spec.key);

export const TEMPERATURE_RANGE = { min: 2000, max: 50000 } as const;

export function sameParams(a: Params | undefined, b: Params | undefined): boolean {
  if (!a || !b) return false;
  return PARAM_KEYS.every((key) => Math.abs(a[key] - b[key]) < 1e-6);
}

/** Clamped to the slider's legal range and rounded to its step, the way
    Lightroom stores it (whole kelvin, 0.01 EV, whole units). */
export function snapParam(key: ParamKey, value: number): number {
  if (key === "temperature") {
    return Math.round(Math.min(TEMPERATURE_RANGE.max, Math.max(TEMPERATURE_RANGE.min, value)));
  }
  const spec = PARAMS.find((p) => p.key === key)!;
  const clamped = Math.min(spec.max, Math.max(spec.min, value));
  return key === "exposure" ? Math.round(clamped * 100) / 100 : Math.round(clamped);
}

/** "+12", "-8", "0" (never "-0"); `decimals` fixed, so "+0.35" and "0.00". */
export function formatSigned(value: number, decimals = 0): string {
  const factor = 10 ** decimals;
  const rounded = Math.round(value * factor) / factor;
  if (rounded === 0) return (0).toFixed(decimals);
  return `${rounded > 0 ? "+" : "-"}${Math.abs(rounded).toFixed(decimals)}`;
}

/** A slider value as Lightroom shows it, with the unit where one helps. */
export function formatParam(key: ParamKey, value: number): string {
  if (key === "temperature") return `${Math.round(value)} K`;
  if (key === "exposure") return `${formatSigned(value, 2)} EV`;
  return formatSigned(value);
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
