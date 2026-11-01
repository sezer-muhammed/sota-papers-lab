import { useId, type ReactNode } from "react";
import { ChevronRight, Undo2 } from "lucide-react";
import { Button, RangeInput, Switch, cn, text } from "@spl/ui";
import { NumberField } from "../number-field";
import { TEMPERATURE_RANGE, formatParam } from "../format";
import { TEMPERATURE_SLIDER, trackOf, type Row, type Section } from "./layout";
import { toggleSection, useDevelopPrefs } from "./prefs";

/** Read with a marked setting's name: what the dot means. */
export const AUTO_NOTE = "set by the auto edit";

/* The mark of a setting the auto edit moved off Lightroom's default: a dot
   before its name. A fixed slot, so names line up with and without it. */
export function AutoMark({ on }: { on: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "inline-block size-1.5 shrink-0 rounded-full align-middle",
        on ? "bg-[var(--ds-gray-1000)]" : "bg-transparent",
      )}
    />
  );
}

/** The marks explained, under the panels. */
export function MarkLegend({ children }: { children?: ReactNode }) {
  return (
    <div className={cn(text.meta, "space-y-1")}>
      <p>
        <AutoMark on />
        <span className="ml-1.5">Set by the auto edit, away from Lightroom's default.</span>
      </p>
      {children ? <p>{children}</p> : null}
    </div>
  );
}

/* One Lightroom panel that opens and closes; which ones are open is kept per
   browser. The header counts what is set: the auto edit's marks, and in the
   editor what you changed. */
export function DevelopSection({
  children,
  edited = 0,
  marked,
  onReset,
  section,
}: {
  children: ReactNode;
  /** Settings changed from the auto edit (editor only). */
  edited?: number;
  /** Settings away from Lightroom's default. */
  marked: number;
  /** Puts the panel's settings back (editor only); offered once one changed. */
  onReset?: () => void;
  section: Section;
}) {
  const open = useDevelopPrefs().open[section.id];
  const bodyId = useId();
  const counts = [
    marked ? `${marked} auto` : undefined,
    edited ? `${edited} edited` : undefined,
  ].filter(Boolean);
  return (
    <section className="-mx-4 border-t border-[var(--ds-gray-alpha-400)] px-4 first:border-t-0">
      <h3>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={() => toggleSection(section.id)}
          className="-mx-2 flex min-h-11 w-[calc(100%+1rem)] items-center gap-2 rounded-[6px] px-2 text-left outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
        >
          <ChevronRight
            aria-hidden="true"
            className={cn(
              "size-4 shrink-0 text-[var(--ds-gray-700)] transition-transform motion-reduce:transition-none",
              open && "rotate-90",
            )}
          />
          <span className="flex-1 text-[14px] font-semibold text-[var(--ds-gray-1000)]">{section.title}</span>
          <span className={cn(text.meta, "flex items-center gap-1.5")}>
            {marked ? <AutoMark on /> : null}
            {counts.length ? counts.join(" · ") : "default"}
          </span>
        </button>
      </h3>
      <div id={bodyId} hidden={!open} className="pb-4 pt-1">
        {open ? children : null}
        {open && onReset && edited > 0 ? (
          <div className="mt-3 flex justify-end">
            <Button size="sm" variant="ghost" icon={Undo2} onClick={onReset}>
              Reset {section.title}
            </Button>
          </div>
        ) : null}
      </div>
    </section>
  );
}

/** A sub-heading inside a panel ("Tone", "Red Primary"). */
export function BlockTitle({ children, id }: { children: ReactNode; id?: string }) {
  return (
    <p id={id} className={text.eyebrowSm}>
      {children}
    </p>
  );
}

/* One slider, live. The dot marks a value the auto edit set; the name turns
   bold once you move it, and a tick on the track shows where the auto value
   was. Double-click the name (Lightroom's gesture) or press Delete on the
   slider to put the auto value back. Temperature also takes a typed kelvin
   value across Lightroom's whole range. */
export function SliderRow({
  base,
  baseName,
  disabled,
  marked,
  onActive,
  onChange,
  row,
  value,
}: {
  /** The auto value: what a reset returns to. */
  base: number;
  /** What `base` is called in the reset hint ("auto value"). */
  baseName: string;
  disabled?: boolean;
  marked: boolean;
  /** Hover or focus, for panels that point at what the slider does. */
  onActive?: (active: boolean) => void;
  onChange: (value: number) => void;
  row: Row;
  value: number;
}) {
  const id = useId();
  const { label, name, spec } = row;
  const isTemperature = spec.key === "temperature";
  const range = isTemperature ? TEMPERATURE_SLIDER : spec;
  const changed = Math.abs(value - base) > 1e-6;
  const reset = () => onChange(base);
  const at = (v: number) => (Math.min(range.max, Math.max(range.min, v)) - range.min) / (range.max - range.min || 1);
  const defaultValue = spec.neutral === null ? undefined : spec.neutral;
  const atDefault = defaultValue !== undefined && Math.abs(value - defaultValue) < 1e-6;
  const fullName = `${name}${marked ? `, ${AUTO_NOTE}` : ""}`;

  return (
    <div
      onPointerEnter={onActive && (() => onActive(true))}
      onPointerLeave={onActive && (() => onActive(false))}
      onFocus={onActive && (() => onActive(true))}
      onBlur={onActive && (() => onActive(false))}
    >
      <div className="flex items-center justify-between gap-3">
        <label
          htmlFor={id}
          title={`Double-click to reset to the ${baseName} (${formatParam(spec.key, base)})`}
          onDoubleClick={disabled ? undefined : reset}
          className={cn(
            "flex min-w-0 select-none items-center gap-1.5 text-[13px]",
            changed ? "font-medium text-[var(--ds-gray-1000)]" : "text-[var(--ds-gray-900)]",
          )}
        >
          <AutoMark on={marked} />
          <span className="truncate">{label}</span>
        </label>
        {isTemperature ? (
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
          <output
            htmlFor={id}
            className={cn(
              "shrink-0 font-mono text-[13px] tabular-nums",
              atDefault ? "text-[var(--ds-gray-700)]" : "text-[var(--ds-gray-1000)]",
            )}
          >
            {formatParam(spec.key, value)}
          </output>
        )}
      </div>
      <div className="relative mt-1">
        {changed ? (
          // Where the auto value sits; under the slider, so the thumb covers it.
          <span
            aria-hidden="true"
            className="pointer-events-none absolute top-1 h-3 w-0.5 -translate-x-1/2 rounded-full bg-[var(--ds-gray-700)]"
            style={{ left: `calc(7px + (100% - 14px) * ${at(base)})` }}
          />
        ) : null}
        <RangeInput
          id={id}
          min={range.min}
          max={range.max}
          step={range.step}
          origin={spec.min < 0 ? 0 : undefined}
          track={trackOf(spec.key)}
          value={Math.min(range.max, Math.max(range.min, value))}
          disabled={disabled}
          aria-label={fullName}
          aria-valuetext={formatParam(spec.key, value)}
          onValueChange={onChange}
          onKeyDown={(event) => {
            if (event.key !== "Delete" && event.key !== "Backspace") return;
            event.preventDefault();
            reset();
          }}
          className="relative"
        />
      </div>
    </div>
  );
}

const SWITCH_NOTES: Record<string, string> = {
  lens_profile: "Distortion and vignetting from the lens profile.",
  lens_ca: "Colour fringes along high-contrast edges.",
};

/** A Lens Corrections switch (stored as 0 or 1, as Lightroom writes it). */
export function SwitchRow({
  disabled,
  marked,
  onChange,
  row,
  value,
}: {
  disabled?: boolean;
  marked: boolean;
  onChange: (value: number) => void;
  row: Row;
  value: number;
}) {
  return (
    <Switch
      checked={value >= 0.5}
      disabled={disabled}
      onCheckedChange={(on) => onChange(on ? row.spec.max : row.spec.min)}
      label={
        <span className="inline-flex items-center gap-1.5">
          <AutoMark on={marked} />
          <span>
            {row.label}
            {marked ? <span className="sr-only">, {AUTO_NOTE}</span> : null}
          </span>
        </span>
      }
      description={SWITCH_NOTES[row.spec.key]}
    />
  );
}
