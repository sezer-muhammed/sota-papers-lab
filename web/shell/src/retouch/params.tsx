import { Fragment, useId, useState } from "react";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import { cn, text } from "@spl/ui";
import { MixerEditor, MixerTable } from "./develop/color-mixer";
import type { Region } from "./develop/curve";
import { BlockTitle, DevelopSection, SetMark } from "./develop/section";
import {
  DESCRIPTIONS,
  HAS_MIXER,
  SECTIONS,
  accessibleName,
  sectionKeys,
  shortLabel,
  trackOf,
  type Section,
} from "./develop/sections";
import { SliderRow, SwitchRow } from "./develop/slider-row";
import { CURVE_REGION, CurvePreview } from "./develop/tone-curve";
import {
  TEMPERATURE_SLIDER,
  awayFromDefault,
  formatParam,
  formatSigned,
  isSwitch,
  snapParam,
  specOf,
  type ParamKey,
  type ParamSpec,
} from "./format";

type AsShot = { temperature: number; tint: number };

/** Why a value is marked, or undefined when it sits at Lightroom's default
    (white balance: as shot). */
type Marks = (key: ParamKey, value: number) => string | undefined;

function marksFor(asShot: AsShot | undefined, who: string): Marks {
  return (key, value) => {
    if (!awayFromDefault(key, value, asShot)) return undefined;
    const spec = specOf(key);
    const reference =
      spec.neutral != null
        ? `default ${formatParam(key, spec.neutral)}`
        : asShot && (key === "temperature" || key === "tint")
          ? `as shot ${formatParam(key, asShot[key])}`
          : "default";
    return `${who} (${reference})`;
  };
}

const edited = (params: Params, base: Params, key: ParamKey) => Math.abs(params[key] - base[key]) > 1e-6;

/* Lightroom's Develop panels as numbers, every setting the host has
   (PARAM_SPECS): Basic, Tone Curve with its curve, the Color Mixer as one
   grid, Detail, Lens Corrections and Calibration, each collapsible. Values
   away from Lightroom's default are marked and read at full strength. */
export function ParamsTable({
  asShot,
  caption,
  params,
}: {
  asShot?: AsShot;
  caption: string;
  params: Params;
}) {
  const marks = marksFor(asShot, "Away from Lightroom's default");

  return (
    <div>
      {SECTIONS.map((section) => {
        const marked = sectionKeys(section).filter((key) => marks(key, params[key])).length;
        return (
          <DevelopSection
            key={section.id}
            id={section.id}
            title={section.title}
            summary={marked ? `${marked} set` : "Default"}
          >
            {section.id === "toneCurve" ? <CurvePreview params={params} className="mb-3" /> : null}
            {section.id === "colorMixer" && HAS_MIXER ? (
              <MixerTable caption={`${section.title}: ${caption}`} marks={marks} params={params} />
            ) : (
              <ValuesTable section={section} caption={caption} marks={marks} params={params} />
            )}
            {section.id === "basic" && asShot ? (
              <p className={cn(text.meta, "mt-3")}>
                As shot: {Math.round(asShot.temperature)} K, tint {formatSigned(asShot.tint)}
              </p>
            ) : null}
          </DevelopSection>
        );
      })}
      <Legend />
    </div>
  );
}

function ValuesTable({
  caption,
  marks,
  params,
  section,
}: {
  caption: string;
  marks: Marks;
  params: Params;
  section: Section;
}) {
  return (
    <table className="w-full border-collapse">
      <caption className="sr-only">
        {section.title}: {caption}
      </caption>
      <tbody>
        {section.blocks.map((block, index) => (
          <Fragment key={block.title ?? index}>
            {block.title ? (
              <tr>
                <th
                  colSpan={3}
                  scope="colgroup"
                  className={cn(text.eyebrowSm, "pb-1 text-left font-normal", index > 0 && "pt-3")}
                >
                  {block.title}
                </th>
              </tr>
            ) : null}
            {block.keys.map((key) => (
              <ValueRow key={key} spec={specOf(key)} value={params[key]} mark={marks(key, params[key])} />
            ))}
          </Fragment>
        ))}
      </tbody>
    </table>
  );
}

function ValueRow({ mark, spec, value }: { mark?: string; spec: ParamSpec; value: number }) {
  const onOff = isSwitch(spec);
  return (
    <tr className="h-7">
      <th
        scope="row"
        colSpan={onOff ? 2 : 1}
        className={cn("text-left text-[13px] font-normal text-[var(--ds-gray-900)]", !onOff && "w-28")}
      >
        <span className="inline-flex items-center gap-1.5">
          {shortLabel(spec.key, spec.label)}
          {mark ? <SetMark title={mark} /> : null}
        </span>
      </th>
      {onOff ? null : (
        <td className="px-3">
          <ValueBar spec={spec} value={value} />
        </td>
      )}
      <td
        className={cn(
          "w-20 whitespace-nowrap text-right font-mono text-[13px] tabular-nums",
          mark ? "font-medium text-[var(--ds-gray-1000)]" : "text-[var(--ds-gray-700)]",
        )}
      >
        {formatParam(spec.key, value)}
      </td>
    </tr>
  );
}

/* Bipolar sliders: a bar from the centre, as wide as the value's share of
   the range. Others (Detail): a bar from the left. Temperature has no
   centre: a mark on the white-balance scale. */
function ValueBar({ spec, value }: { spec: ParamSpec; value: number }) {
  if (spec.key === "temperature") {
    const { max, min } = TEMPERATURE_SLIDER;
    const share = Math.min(1, Math.max(0, (value - min) / (max - min)));
    return (
      <span aria-hidden="true" className="relative block h-1.5 rounded-full" style={{ background: trackOf("temperature") }}>
        <span
          className="absolute -top-[3px] h-3 w-1 -translate-x-1/2 rounded-full border border-[var(--ds-gray-alpha-600)] bg-[var(--ds-background-100)]"
          style={{ left: `${share * 100}%` }}
        />
      </span>
    );
  }
  const bipolar = spec.min < 0;
  const span = bipolar ? Math.max(-spec.min, spec.max) : spec.max - spec.min;
  const share = Math.max(-1, Math.min(1, (bipolar ? value : value - spec.min) / (span || 1)));
  return (
    <span aria-hidden="true" className="relative block h-1.5 rounded-full bg-[var(--ds-gray-alpha-200)]">
      {bipolar ? <span className="absolute inset-y-0 left-1/2 w-px bg-[var(--ds-gray-alpha-500)]" /> : null}
      <span
        className="absolute inset-y-0 rounded-full bg-[var(--ds-gray-900)]"
        style={
          !bipolar
            ? { left: 0, width: `${share * 100}%` }
            : share >= 0
              ? { left: "50%", width: `${share * 50}%` }
              : { right: "50%", width: `${-share * 50}%` }
        }
      />
    </span>
  );
}

/* Every setting, live, in the same sections. Double-click a name (or press
   Delete on its slider) to put it back to `base`, the auto edit; a section
   with changes has its own Reset. Values the auto edit set away from
   Lightroom's default carry the mark. */
export function AdjustSliders({
  asShot,
  base,
  disabled = false,
  onChange,
  params,
}: {
  asShot?: AsShot;
  base: Params;
  disabled?: boolean;
  onChange: (params: Params) => void;
  params: Params;
}) {
  const prefix = useId();
  // The tone-curve region in use: the hovered slider's, else the focused one's.
  const [hovered, setHovered] = useState<Region>();
  const [focused, setFocused] = useState<Region>();
  const marks = marksFor(asShot, "Set by the auto edit");

  const set = (key: ParamKey, value: number) => {
    const next = snapParam(key, value);
    if (next !== params[key]) onChange({ ...params, [key]: next });
  };

  const row = (key: ParamKey) => {
    const spec = specOf(key);
    const props = {
      spec,
      label: shortLabel(key, spec.label),
      name: accessibleName(key, spec.label),
      value: params[key],
      base: base[key],
      marked: marks(key, base[key]),
      disabled,
      onChange: (value: number) => set(key, value),
    };
    if (isSwitch(spec)) return <SwitchRow key={key} {...props} description={DESCRIPTIONS[key]} />;
    const region = CURVE_REGION[key];
    const track = (set: typeof setHovered) => (on: boolean) =>
      set((current) => (on ? region : current === region ? undefined : current));
    return (
      <SliderRow
        key={key}
        {...props}
        onHover={region ? track(setHovered) : undefined}
        onFocusChange={region ? track(setFocused) : undefined}
      />
    );
  };

  return (
    <div>
      {SECTIONS.map((section) => {
        const keys = sectionKeys(section);
        const marked = keys.filter((key) => marks(key, base[key])).length;
        const changes = keys.filter((key) => edited(params, base, key));
        const summary = [marked ? `${marked} set` : "Default", changes.length ? `${changes.length} edited` : undefined]
          .filter(Boolean)
          .join(" · ");
        const reset = () => {
          const next = { ...params };
          for (const key of changes) next[key] = base[key];
          onChange(next);
        };
        return (
          <DevelopSection
            key={section.id}
            id={section.id}
            title={section.title}
            summary={summary}
            onReset={changes.length && !disabled ? reset : undefined}
          >
            {section.id === "toneCurve" ? (
              <CurvePreview params={params} base={base} active={hovered ?? focused} className="mb-3" />
            ) : null}
            {section.id === "colorMixer" && HAS_MIXER ? (
              <MixerEditor base={base} params={params} marks={marks} disabled={disabled} onSet={set} />
            ) : (
              <div className="space-y-4">
                {section.blocks.map((block, index) => {
                  const titleId = `${prefix}-${section.id}-${index}`;
                  return (
                    <div
                      key={block.title ?? index}
                      role="group"
                      aria-labelledby={block.title ? titleId : undefined}
                      className="space-y-2"
                    >
                      {block.title ? <BlockTitle id={titleId}>{block.title}</BlockTitle> : null}
                      {block.keys.map(row)}
                    </div>
                  );
                })}
              </div>
            )}
          </DevelopSection>
        );
      })}
      <Legend adjust />
    </div>
  );
}

function Legend({ adjust = false }: { adjust?: boolean }) {
  return (
    <div className={cn(text.meta, "mt-1 space-y-1 border-t border-[var(--ds-gray-alpha-400)] pt-3")}>
      <p className="flex items-baseline gap-1.5">
        <span aria-hidden="true" className="inline-block h-1.5 w-1.5 shrink-0 translate-y-[-1px] rounded-full bg-[var(--ds-blue-700)]" />
        <span>
          {adjust ? "Set by the auto edit" : "Away from Lightroom's default"} (white balance: away from as shot).
        </span>
      </p>
      {adjust ? <p>Double-click a name, or press Delete on its slider, to go back to the auto value.</p> : null}
    </div>
  );
}
