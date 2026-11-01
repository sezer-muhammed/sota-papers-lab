import { Fragment, useId } from "react";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import { RangeInput, cn, text } from "@spl/ui";
import { NumberField } from "./number-field";
import { PARAMS, TEMPERATURE_RANGE, formatParam, formatSigned, snapParam, type ParamKey, type ParamSpec } from "./format";

const GROUPS = ["White balance", "Tone", "Presence"] as const;

/* White balance sliders show their scale the way Lightroom does: blue to
   amber for temperature, green to magenta for tint. Token colours only. */
const TRACKS: Partial<Record<ParamKey, string>> = {
  temperature: "linear-gradient(to right, var(--ds-blue-700), var(--ds-gray-300), var(--ds-amber-700))",
  tint: "linear-gradient(to right, var(--ds-green-700), var(--ds-gray-300), var(--ds-pink-700))",
};

/* Lightroom's Basic panel as numbers: signed values, a bar from the centre
   for the bipolar sliders, and the as-shot white balance for reference. */
export function ParamsTable({
  asShot,
  caption,
  params,
}: {
  asShot?: { temperature: number; tint: number };
  caption: string;
  params: Params;
}) {
  return (
    <div>
      <table className="w-full border-collapse">
        <caption className="sr-only">{caption}</caption>
        <tbody>
          {GROUPS.map((group, index) => (
            <Fragment key={group}>
              <tr>
                <th
                  colSpan={3}
                  scope="colgroup"
                  className={cn(text.eyebrowSm, "pb-1 text-left font-normal", index > 0 && "pt-3")}
                >
                  {group}
                </th>
              </tr>
              {PARAMS.filter((spec) => spec.group === group).map((spec) => (
                <tr key={spec.key} className="h-7">
                  <th scope="row" className="w-24 text-left text-[13px] font-normal text-[var(--ds-gray-900)]">
                    {spec.label}
                  </th>
                  <td className="px-3">
                    <ValueBar spec={spec} value={params[spec.key]} />
                  </td>
                  <td className="w-20 whitespace-nowrap text-right font-mono text-[13px] tabular-nums text-[var(--ds-gray-1000)]">
                    {formatParam(spec.key, params[spec.key])}
                  </td>
                </tr>
              ))}
            </Fragment>
          ))}
        </tbody>
      </table>
      {asShot ? (
        <p className={cn(text.meta, "mt-3")}>
          As shot: {Math.round(asShot.temperature)} K, tint {formatSigned(asShot.tint)}
        </p>
      ) : null}
    </div>
  );
}

/* Bipolar sliders: a bar from the centre, as wide as the value's share of
   the range. Temperature has no centre: a mark on the white-balance scale. */
function ValueBar({ spec, value }: { spec: ParamSpec; value: number }) {
  if (!spec.bar) {
    const share = Math.min(1, Math.max(0, (value - spec.min) / (spec.max - spec.min)));
    return (
      <span aria-hidden="true" className="relative block h-1.5 rounded-full" style={{ background: TRACKS.temperature }}>
        <span
          className="absolute -top-[3px] h-3 w-1 -translate-x-1/2 rounded-full border border-[var(--ds-gray-alpha-600)] bg-[var(--ds-background-100)]"
          style={{ left: `${share * 100}%` }}
        />
      </span>
    );
  }
  const share = Math.max(-1, Math.min(1, value / spec.bar));
  return (
    <span aria-hidden="true" className="relative block h-1.5 rounded-full bg-[var(--ds-gray-alpha-200)]">
      <span className="absolute inset-y-0 left-1/2 w-px bg-[var(--ds-gray-alpha-500)]" />
      <span
        className="absolute inset-y-0 rounded-full bg-[var(--ds-gray-900)]"
        style={share >= 0 ? { left: "50%", width: `${share * 50}%` } : { right: "50%", width: `${-share * 50}%` }}
      />
    </span>
  );
}

/* The ten sliders, live. Double-click a slider's name to put it back to
   `base` (Lightroom's gesture); temperature also takes a typed kelvin value
   across Lightroom's whole 2000 to 50000 range. */
export function AdjustSliders({
  base,
  disabled = false,
  onChange,
  params,
}: {
  base: Params;
  disabled?: boolean;
  onChange: (params: Params) => void;
  params: Params;
}) {
  const prefix = useId();
  const set = (key: ParamKey, value: number) => {
    const next = snapParam(key, value);
    if (next !== params[key]) onChange({ ...params, [key]: next });
  };

  return (
    <div className="space-y-4">
      {GROUPS.map((group, index) => (
        <div key={group} role="group" aria-labelledby={`${prefix}-group-${index}`} className="space-y-2.5">
          <p id={`${prefix}-group-${index}`} className={text.eyebrowSm}>
            {group}
          </p>
          {PARAMS.filter((spec) => spec.group === group).map((spec) => {
            const id = `${prefix}-${spec.key}`;
            const value = params[spec.key];
            const changed = Math.abs(value - base[spec.key]) > 1e-6;
            return (
              <div key={spec.key}>
                <div className="flex items-center justify-between gap-3">
                  <label
                    htmlFor={id}
                    title="Double-click to reset"
                    onDoubleClick={() => set(spec.key, base[spec.key])}
                    className={cn(
                      "select-none text-[13px]",
                      changed ? "font-medium text-[var(--ds-gray-1000)]" : "text-[var(--ds-gray-900)]",
                    )}
                  >
                    {spec.label}
                  </label>
                  {spec.key === "temperature" ? (
                    <NumberField
                      label="Temperature in kelvin"
                      value={Math.round(value)}
                      min={TEMPERATURE_RANGE.min}
                      max={TEMPERATURE_RANGE.max}
                      disabled={disabled}
                      onCommit={(kelvin) => set("temperature", kelvin)}
                      className="h-7 w-20"
                    />
                  ) : (
                    <output htmlFor={id} className="font-mono text-[13px] tabular-nums text-[var(--ds-gray-1000)]">
                      {formatParam(spec.key, value)}
                    </output>
                  )}
                </div>
                <RangeInput
                  id={id}
                  min={spec.min}
                  max={spec.max}
                  step={spec.step}
                  origin={spec.bar ? 0 : undefined}
                  track={TRACKS[spec.key]}
                  value={Math.min(spec.max, Math.max(spec.min, value))}
                  disabled={disabled}
                  aria-valuetext={formatParam(spec.key, value)}
                  onValueChange={(next) => set(spec.key, next)}
                  className="mt-1"
                />
              </div>
            );
          })}
        </div>
      ))}
      <p className={text.meta}>Double-click a slider's name to reset it.</p>
    </div>
  );
}
