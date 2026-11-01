import type { RunRecord } from "@spl/sdk";

const SECOND = 1000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

export const plural = (n: number, word: string) => `${n} ${n === 1 ? word : `${word}s`}`;

/** "just now", "5 min ago", "3 h ago", "2 d ago", then a short date. */
export function relativeTime(ms: number, now: number) {
  const d = now - ms;
  if (d < 45 * SECOND) return "just now"; // also covers clock skew (negative)
  const min = Math.round(d / MINUTE);
  if (min < 60) return `${Math.max(1, min)} min ago`;
  const h = Math.round(d / HOUR);
  if (h < 24) return `${h} h ago`;
  const days = Math.round(d / DAY);
  if (days < 7) return `${days} d ago`;
  const date = new Date(ms);
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: date.getFullYear() === new Date(now).getFullYear() ? undefined : "numeric",
  });
}

export const absoluteTime = (ms: number) => new Date(ms).toLocaleString();

/** A run's wall time as value + unit ("640" ms, "1.25" s, "1.6" min, "2.5" h),
    or undefined while it has not finished. Every page formats durations with
    this (run-status.tsx runDuration joins the two). */
export function splitDuration(
  run: Pick<RunRecord, "started_at_ms" | "finished_at_ms">,
): { value: string; unit: string } | undefined {
  if (run.started_at_ms == null || run.finished_at_ms == null) return undefined;
  const ms = Math.max(0, run.finished_at_ms - run.started_at_ms);
  if (ms < SECOND) return { value: String(ms), unit: "ms" };
  if (ms < MINUTE) return { value: (ms / SECOND).toFixed(ms < 10 * SECOND ? 2 : 1), unit: "s" };
  if (ms < HOUR) return { value: (ms / MINUTE).toFixed(ms < 10 * MINUTE ? 1 : 0), unit: "min" };
  return { value: (ms / HOUR).toFixed(ms < 10 * HOUR ? 1 : 0), unit: "h" };
}

// Below 1000, 4 significant digits (32.46 dB, 0.1235 bpp). From 1000 up, every
// integer digit and at most one decimal, so a count (bits, params) is never
// rounded into exact-looking wrong digits.
const smallMetric = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 4 });
const largeMetric = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

/** A metric value with its unit, the same on Home and the Runs page. */
export const formatMetric = (value: number, unit: string | null) =>
  (Math.abs(value) >= 1000 ? largeMetric : smallMetric).format(value) + (unit ? ` ${unit}` : "");

export const formatGiB = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(1)} GiB`;

/** "NVIDIA GeForce RTX 4070 SUPER" → "RTX 4070 SUPER". */
export const shortGpuName = (name: string) => name.replace(/^NVIDIA\s+(GeForce\s+)?/i, "");
