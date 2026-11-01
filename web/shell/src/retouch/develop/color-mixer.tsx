import { useId } from "react";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import { SegmentedControl, cn, text } from "@spl/ui";
import { formatParam, specOf, type ParamKey } from "../format";
import { updatePrefs, useDevelopPrefs } from "./prefs";
import { BlockTitle, SetMark } from "./section";
import { MIXER_BANDS, MIXER_KINDS, bandColor, mixerKey, type MixerBand } from "./sections";
import { SliderRow } from "./slider-row";

type Marks = (key: ParamKey, value: number) => string | undefined;

/* Lightroom's Color Mixer, editable. HSL: one property (hue, saturation or
   luminance) for all eight bands. Color: pick a band by its chip, then its
   three sliders. Tracks are tinted by band; the view is kept per browser. */
export function MixerEditor({
  base,
  disabled,
  marks,
  onSet,
  params,
}: {
  base: Params;
  disabled?: boolean;
  marks: Marks;
  onSet: (key: ParamKey, value: number) => void;
  params: Params;
}) {
  const prefs = useDevelopPrefs();
  const bandTitle = useId();
  const row = (key: ParamKey, label: string) => (
    <SliderRow
      key={key}
      spec={specOf(key)}
      label={label}
      value={params[key]}
      base={base[key]}
      marked={marks(key, base[key])}
      disabled={disabled}
      onChange={(value) => onSet(key, value)}
    />
  );
  const band = MIXER_BANDS.find((b) => b.band === prefs.band) ?? MIXER_BANDS[0];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SegmentedControl
          label="Color Mixer view"
          value={prefs.mixer}
          options={[
            { value: "hsl", label: "HSL" },
            { value: "color", label: "Color" },
          ]}
          onChange={(mixer) => updatePrefs({ mixer })}
        />
        {prefs.mixer === "hsl" ? (
          <SegmentedControl
            label="Property"
            value={prefs.kind}
            options={MIXER_KINDS.map((k) => ({ value: k.kind, label: k.label }))}
            onChange={(kind) => updatePrefs({ kind })}
          />
        ) : null}
      </div>
      {prefs.mixer === "hsl" ? (
        <div role="group" aria-label={`${MIXER_KINDS.find((k) => k.kind === prefs.kind)?.label} by band`} className="space-y-2">
          {MIXER_BANDS.map((b) => row(mixerKey(prefs.kind, b.band), b.label))}
        </div>
      ) : (
        <>
          <BandPicker
            value={band.band}
            onChange={(next) => updatePrefs({ band: next })}
            marked={(b) => MIXER_KINDS.some(({ kind }) => marks(mixerKey(kind, b), base[mixerKey(kind, b)]))}
            edited={(b) => MIXER_KINDS.some(({ kind }) => Math.abs(params[mixerKey(kind, b)] - base[mixerKey(kind, b)]) > 1e-6)}
          />
          <div role="group" aria-labelledby={bandTitle} className="space-y-2">
            <BlockTitle id={bandTitle}>{band.label}</BlockTitle>
            {MIXER_KINDS.map((k) => row(mixerKey(k.kind, band.band), k.label))}
          </div>
        </>
      )}
    </div>
  );
}

/* Eight colour chips as a radio group (native radios, so arrow keys move
   between bands). A ring marks the selected band, a dot one the auto edit
   changed, and a band you changed has a dark rim. */
function BandPicker({
  edited,
  marked,
  onChange,
  value,
}: {
  edited: (band: MixerBand) => boolean;
  marked: (band: MixerBand) => boolean;
  onChange: (band: MixerBand) => void;
  value: MixerBand;
}) {
  const name = useId();
  return (
    <div role="radiogroup" aria-label="Band" className="grid grid-cols-8 gap-1">
      {MIXER_BANDS.map((b) => {
        const auto = marked(b.band);
        const mine = edited(b.band);
        return (
          <label key={b.band} className="relative flex cursor-pointer justify-center py-1" title={b.label}>
            <input
              type="radio"
              name={name}
              value={b.band}
              checked={value === b.band}
              onChange={() => onChange(b.band)}
              className="peer sr-only"
            />
            <span
              aria-hidden="true"
              className={cn(
                "h-7 w-7 rounded-full border transition-shadow motion-reduce:transition-none",
                "peer-checked:shadow-[0_0_0_2px_var(--ds-background-100),0_0_0_4px_var(--ds-gray-1000)] peer-focus-visible:shadow-[var(--ds-focus-ring)]",
                mine ? "border-2 border-[var(--ds-gray-1000)]" : "border-[var(--ds-gray-alpha-400)]",
              )}
              style={{ background: bandColor(b.hue) }}
            />
            {auto ? (
              <span
                aria-hidden="true"
                className="absolute left-1/2 top-0 ml-2.5 h-2 w-2 rounded-full border border-[var(--ds-background-100)] bg-[var(--ds-blue-700)]"
              />
            ) : null}
            <span className="sr-only">
              {b.label}
              {auto ? ", set by the auto edit" : ""}
              {mine ? ", changed" : ""}
            </span>
          </label>
        );
      })}
    </div>
  );
}

/* The Color Mixer read-only: hue, saturation and luminance for all eight
   bands at once. Values away from the default read at full strength. */
export function MixerTable({ caption, marks, params }: { caption: string; marks: Marks; params: Params }) {
  return (
    <table className="w-full table-fixed border-collapse">
      <caption className="sr-only">{caption}</caption>
      <thead>
        <tr>
          <td className="w-9" />
          {MIXER_BANDS.map((b) => {
            const auto = MIXER_KINDS.some(({ kind }) => marks(mixerKey(kind, b.band), params[mixerKey(kind, b.band)]));
            return (
              <th key={b.band} scope="col" className="pb-1.5 font-normal" title={b.label}>
                <span className="relative mx-auto block h-4 w-4">
                  <span
                    aria-hidden="true"
                    className="block h-4 w-4 rounded-full border border-[var(--ds-gray-alpha-400)]"
                    style={{ background: bandColor(b.hue) }}
                  />
                  {auto ? (
                    <span className="absolute -right-1 -top-1">
                      <SetMark title={`${b.label}: set by the auto edit`} />
                    </span>
                  ) : null}
                </span>
                <span className="sr-only">{b.label}</span>
              </th>
            );
          })}
        </tr>
      </thead>
      <tbody>
        {MIXER_KINDS.map((k) => (
          <tr key={k.kind} className="h-7">
            <th scope="row" className={cn(text.eyebrowSm, "text-left font-normal")}>
              <abbr title={k.label} className="no-underline">
                {k.short}
              </abbr>
            </th>
            {MIXER_BANDS.map((b) => {
              const key = mixerKey(k.kind, b.band);
              const set = marks(key, params[key]) !== undefined;
              return (
                <td
                  key={b.band}
                  className={cn(
                    "text-center font-mono text-[12px] tabular-nums",
                    set ? "font-medium text-[var(--ds-gray-1000)]" : "text-[var(--ds-gray-700)]",
                  )}
                >
                  {formatParam(key, params[key])}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
