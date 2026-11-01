import { useId } from "react";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import { SegmentedControl, cn, text } from "@spl/ui";
import { formatParam, isDefault, paramValue, specOf, type ParamKey } from "../format";
import type { HueBand } from "../scopes/compute";
import { AUTO_NOTE, AutoMark, SliderRow } from "./controls";
import { BANDS, MIXER_PROPERTIES, bandColor, mixerKey, rowOf, type Band } from "./layout";
import { setDevelopPrefs, useDevelopPrefs, type MixerMode } from "./prefs";

const MODES: { value: MixerMode; label: string }[] = [
  { value: "hsl", label: "HSL" },
  { value: "color", label: "Color" },
];

const bandSet = (params: Params, band: HueBand) =>
  MIXER_PROPERTIES.some((p) => {
    const key = mixerKey(p.value, band);
    return !isDefault(specOf(key), paramValue(params, key));
  });

/* Lightroom's Color Mixer, live. HSL: one property (Hue, Saturation or
   Luminance) across the eight bands. Color: pick a band's chip, then its
   three sliders. Tracks show what each slider does to its band. */
export function MixerSliders({
  base,
  baseName,
  disabled,
  onSet,
  params,
}: {
  base: Params;
  baseName: string;
  disabled?: boolean;
  onSet: (key: ParamKey, value: number) => void;
  params: Params;
}) {
  const prefs = useDevelopPrefs();
  const slider = (key: ParamKey, label: string) => (
    <SliderRow
      key={key}
      row={rowOf(specOf(key), label)}
      value={paramValue(params, key)}
      base={paramValue(base, key)}
      baseName={baseName}
      marked={!isDefault(specOf(key), paramValue(base, key))}
      disabled={disabled}
      onChange={(value) => onSet(key, value)}
    />
  );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className={text.eyebrowSm}>Adjust</span>
        <SegmentedControl
          label="Color Mixer adjusts"
          value={prefs.mixer}
          options={MODES}
          onChange={(mixer) => setDevelopPrefs((p) => ({ ...p, mixer }))}
        />
      </div>
      {prefs.mixer === "hsl" ? (
        <>
          <SegmentedControl
            label="Property"
            value={prefs.property}
            options={MIXER_PROPERTIES}
            onChange={(property) => setDevelopPrefs((p) => ({ ...p, property }))}
          />
          <div className="space-y-2.5">{BANDS.map((band) => slider(mixerKey(prefs.property, band.name), band.label))}</div>
        </>
      ) : (
        <>
          <BandChips
            value={prefs.band}
            marked={(band) => bandSet(base, band.name)}
            onChange={(band) => setDevelopPrefs((p) => ({ ...p, band }))}
          />
          <div className="space-y-2.5">
            {MIXER_PROPERTIES.map((p) => slider(mixerKey(p.value, prefs.band), p.label))}
          </div>
        </>
      )}
    </div>
  );
}

/* The eight bands as colour chips: a radio group (arrow keys move between
   them). A dot on a chip marks a band the auto edit changed. */
function BandChips({
  marked,
  onChange,
  value,
}: {
  marked: (band: Band) => boolean;
  onChange: (band: HueBand) => void;
  value: HueBand;
}) {
  const name = useId();
  const current = BANDS.find((b) => b.name === value);
  return (
    <div className="space-y-1.5">
      <div role="radiogroup" aria-label="Color" className="flex flex-wrap gap-2">
        {BANDS.map((band) => {
          const set = marked(band);
          return (
            <label key={band.name} className="relative flex" title={band.label}>
              <input
                type="radio"
                name={name}
                value={band.name}
                checked={band.name === value}
                onChange={() => onChange(band.name)}
                className="peer sr-only"
              />
              <span
                aria-hidden="true"
                className={cn(
                  "block size-7 cursor-pointer rounded-full border border-[var(--ds-gray-alpha-500)] transition-shadow",
                  "peer-checked:shadow-[0_0_0_2px_var(--ds-background-100),0_0_0_4px_var(--ds-gray-1000)]",
                  "peer-focus-visible:shadow-[var(--ds-focus-ring)]",
                )}
                style={{ background: bandColor(band.hue) }}
              />
              {set ? (
                <span
                  aria-hidden="true"
                  className="absolute -right-0.5 -top-0.5 size-2.5 rounded-full border-2 border-[var(--ds-background-100)] bg-[var(--ds-gray-1000)]"
                />
              ) : null}
              <span className="sr-only">
                {band.label}
                {set ? `, ${AUTO_NOTE}` : ""}
              </span>
            </label>
          );
        })}
      </div>
      <p className={text.meta}>{current?.label}</p>
    </div>
  );
}

/* The Color Mixer read-only: the eight bands down, hue, saturation and
   luminance across, values off the default in full ink and marked. */
export function MixerTable({ params }: { params: Params }) {
  return (
    <table className="w-full border-collapse">
      <caption className="sr-only">Color Mixer</caption>
      <thead>
        <tr>
          <th scope="col" className={cn(text.eyebrowSm, "pb-1 text-left font-normal")}>
            Band
          </th>
          {MIXER_PROPERTIES.map((p) => (
            <th key={p.value} scope="col" className={cn(text.eyebrowSm, "pb-1 text-right font-normal")}>
              <abbr title={p.label} className="no-underline">
                {p.label.slice(0, 3)}
              </abbr>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {BANDS.map((band) => (
          <tr key={band.name} className="h-7">
            <th scope="row" className="text-left text-[13px] font-normal text-[var(--ds-gray-900)]">
              <span className="inline-flex items-center gap-2">
                <span
                  aria-hidden="true"
                  className="size-3 shrink-0 rounded-full border border-[var(--ds-gray-alpha-500)]"
                  style={{ background: bandColor(band.hue) }}
                />
                {band.label}
              </span>
            </th>
            {MIXER_PROPERTIES.map((p) => {
              const key = mixerKey(p.value, band.name);
              const value = paramValue(params, key);
              const set = !isDefault(specOf(key), value);
              return (
                <td
                  key={p.value}
                  className={cn(
                    "w-16 whitespace-nowrap text-right font-mono text-[13px] tabular-nums",
                    set ? "text-[var(--ds-gray-1000)]" : "text-[var(--ds-gray-700)]",
                  )}
                >
                  <span className="inline-flex items-center justify-end gap-1.5">
                    <AutoMark on={set} />
                    {formatParam(key, value)}
                  </span>
                  {set ? <span className="sr-only">, {AUTO_NOTE}</span> : null}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
