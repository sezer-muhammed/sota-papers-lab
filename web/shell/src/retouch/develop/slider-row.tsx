import { useId, type ReactNode } from "react";
import { RangeInput, Switch, cn } from "@spl/ui";
import { TEMPERATURE_RANGE, TEMPERATURE_SLIDER, formatParam, type ParamSpec } from "../format";
import { NumberField } from "../number-field";
import { SetMark } from "./section";
import { trackOf } from "./sections";

/* Half the range thumb's width (.ds-range: 14px), so a mark under the track
   lines up with the thumb's centre at the same value. */
const THUMB_HALF = 7;

export type RowProps = {
  spec: ParamSpec;
  /** The visible label. */
  label: string;
  /** The accessible name when the label alone is ambiguous on the page
      (default: the spec's label). */
  name?: string;
  value: number;
  /** The auto edit's value: double-click (or Delete) goes back to it. */
  base: number;
  /** The auto edit set this away from Lightroom's default. */
  marked?: string;
  disabled?: boolean;
  onChange: (value: number) => void;
};

/* One live slider: label (double-click resets to the auto value), value,
   and the range; when the value has left the auto edit's, a tick under the
   track shows where that was. Temperature also takes typed kelvin across
   Lightroom's whole range. */
export function SliderRow({
  base,
  disabled = false,
  label,
  marked,
  name: accessible,
  onChange,
  onFocusChange,
  onHover,
  spec,
  value,
}: RowProps & {
  /** The pointer entered (true) or left (false) the row. */
  onHover?: (on: boolean) => void;
  /** Focus entered (true) or left (false) the row. */
  onFocusChange?: (on: boolean) => void;
}) {
  const id = useId();
  const name = accessible ?? spec.label;
  const markId = `${id}-mark`;
  const changed = Math.abs(value - base) > 1e-6;
  const range = spec.key === "temperature" ? TEMPERATURE_SLIDER : spec;
  const share = (v: number) => (Math.min(range.max, Math.max(range.min, v)) - range.min) / (range.max - range.min || 1);
  const reset = () => {
    if (changed) onChange(base);
  };

  return (
    <div
      onPointerEnter={onHover && (() => onHover(true))}
      onPointerLeave={onHover && (() => onHover(false))}
      onFocus={onFocusChange && (() => onFocusChange(true))}
      onBlur={onFocusChange && (() => onFocusChange(false))}
    >
      <div className="flex min-h-7 items-center justify-between gap-3">
        <span className="flex min-w-0 items-center gap-1.5">
          <label
            htmlFor={id}
            title="Double-click to reset to the auto value"
            onDoubleClick={reset}
            className={cn(
              "min-w-0 select-none truncate text-[13px]",
              changed ? "font-medium text-[var(--ds-gray-1000)]" : "text-[var(--ds-gray-900)]",
            )}
          >
            {label}
          </label>
          {marked ? <SetMark id={markId} title={marked} /> : null}
        </span>
        {spec.key === "temperature" ? (
          <NumberField
            label="Temperature in kelvin"
            value={Math.round(value)}
            min={TEMPERATURE_RANGE.min}
            max={TEMPERATURE_RANGE.max}
            disabled={disabled}
            onCommit={onChange}
            className="h-7 w-20"
          />
        ) : (
          <output htmlFor={id} className="font-mono text-[13px] tabular-nums text-[var(--ds-gray-1000)]">
            {formatParam(spec.key, value)}
          </output>
        )}
      </div>
      <div className="relative">
        <RangeInput
          id={id}
          min={range.min}
          max={range.max}
          step={range.step}
          origin={spec.min < 0 ? 0 : undefined}
          track={trackOf(spec.key)}
          value={Math.min(range.max, Math.max(range.min, value))}
          disabled={disabled}
          aria-label={name !== label ? name : undefined}
          aria-describedby={marked ? markId : undefined}
          aria-valuetext={formatParam(spec.key, value)}
          aria-keyshortcuts="Delete"
          onValueChange={onChange}
          onKeyDown={(event) => {
            if (event.key === "Delete" || event.key === "Backspace") {
              event.preventDefault();
              reset();
            }
          }}
        />
        {changed ? (
          <span
            aria-hidden="true"
            title={`Auto: ${formatParam(spec.key, base)}`}
            className="pointer-events-none absolute bottom-0 h-1.5 w-0.5 -translate-x-1/2 rounded-full bg-[var(--ds-gray-700)]"
            style={{ left: `calc(${THUMB_HALF}px + (100% - ${2 * THUMB_HALF}px) * ${share(base)})` }}
          />
        ) : null}
      </div>
    </div>
  );
}

/* An on/off setting (Lens Corrections), with what it does underneath and,
   once switched away from the auto edit, what that had. */
export function SwitchRow({
  base,
  description,
  disabled = false,
  label,
  marked,
  onChange,
  value,
}: RowProps & { description?: ReactNode }) {
  const on = value >= 0.5;
  const changed = on !== base >= 0.5;
  return (
    <Switch
      checked={on}
      disabled={disabled}
      onCheckedChange={(checked) => onChange(checked ? 1 : 0)}
      label={
        <span className="inline-flex items-center gap-1.5">
          {label}
          {marked ? <SetMark title={marked} /> : null}
        </span>
      }
      description={
        <>
          {description}
          {changed ? <span className="font-medium text-[var(--ds-gray-1000)]"> Auto: {on ? "off" : "on"}.</span> : null}
        </>
      }
    />
  );
}
