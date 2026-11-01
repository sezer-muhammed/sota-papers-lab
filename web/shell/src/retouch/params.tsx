import { Fragment, useId, useState } from "react";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import { cn, text } from "@spl/ui";
import { MixerSliders, MixerTable } from "./develop/color-mixer";
import { AUTO_NOTE, AutoMark, BlockTitle, DevelopSection, MarkLegend, SliderRow, SwitchRow } from "./develop/controls";
import {
  SECTIONS,
  TEMPERATURE_SLIDER,
  countSet,
  rowsOf,
  trackOf,
  type AsShot,
  type Block,
  type Section,
} from "./develop/layout";
import type { CurveRegion } from "./develop/tone-curve";
import { CURVE_KEY, ToneCurvePreview } from "./develop/tone-curve-preview";
import { formatParam, formatSigned, isDefault, paramValue, snapParam, type ParamKey, type ParamSpec } from "./format";

const REGION_OF = new Map<ParamKey, CurveRegion>(
  (Object.entries(CURVE_KEY) as [CurveRegion, ParamKey][]).map(([region, key]) => [key, region]),
);

/* Lightroom's Develop panels as numbers, read-only: Basic, Tone Curve,
   Color Mixer, Detail, Lens Corrections and Calibration, each one opening
   and closing (kept per browser). Bars from the centre for the bipolar
   sliders, values off Lightroom's default in full ink with a dot, and the
   as-shot white balance for reference. */
export function ParamsTable({ asShot, caption, params }: { asShot?: AsShot; caption: string; params: Params }) {
  return (
    <div>
      {SECTIONS.map((section) => (
        <DevelopSection key={section.id} section={section} marked={countSet(section, params, asShot)}>
          {section.id === "colorMixer" ? (
            <MixerTable params={params} />
          ) : (
            <div className="space-y-3">
              {section.id === "toneCurve" ? <ToneCurvePreview params={params} className="mx-auto max-w-44" /> : null}
              <ValuesTable section={section} caption={`${section.title}: ${caption}`} params={params} asShot={asShot} />
            </div>
          )}
        </DevelopSection>
      ))}
      <div className="mt-3 space-y-1">
        {asShot ? (
          <p className={text.meta}>
            As shot: {Math.round(asShot.temperature)} K, tint {formatSigned(asShot.tint)}
          </p>
        ) : null}
        <MarkLegend />
      </div>
    </div>
  );
}

function ValuesTable({
  asShot,
  caption,
  params,
  section,
}: {
  asShot?: AsShot;
  caption: string;
  params: Params;
  section: Section;
}) {
  return (
    <table className="w-full border-collapse">
      <caption className="sr-only">{caption}</caption>
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
            {block.rows.map(({ label, spec }) => {
              const value = paramValue(params, spec.key);
              const set = !isDefault(spec, value, asShot);
              return (
                <tr key={spec.key} className="h-7">
                  <th scope="row" className="w-28 text-left text-[13px] font-normal text-[var(--ds-gray-900)]">
                    <span className="flex items-center gap-1.5">
                      <AutoMark on={set} />
                      <span className="truncate" title={spec.label}>
                        {label}
                      </span>
                    </span>
                    {set ? <span className="sr-only">, {AUTO_NOTE}</span> : null}
                  </th>
                  <td className="px-3">
                    <ValueBar spec={spec} value={value} />
                  </td>
                  <td
                    className={cn(
                      "w-16 whitespace-nowrap text-right font-mono text-[13px] tabular-nums",
                      set ? "text-[var(--ds-gray-1000)]" : "text-[var(--ds-gray-700)]",
                    )}
                  >
                    {formatParam(spec.key, value)}
                  </td>
                </tr>
              );
            })}
          </Fragment>
        ))}
      </tbody>
    </table>
  );
}

/* Bipolar sliders: a bar from the centre, as wide as the value's share of
   the range. One-sided ones (Detail) fill from the left; temperature, which
   has no centre, is a mark on its scale; switches have no bar. The colour
   sliders' bars run over their tinted scale. */
function ValueBar({ spec, value }: { spec: ParamSpec; value: number }) {
  if (spec.group === "Lens") return null;
  // Temperature on its slider's scale, where photos live.
  const range = spec.key === "temperature" ? TEMPERATURE_SLIDER : spec;
  const share = (value - range.min) / (range.max - range.min || 1);
  const track = trackOf(spec.key);
  if (track) return <ScaleMark track={track} at={Math.min(1, Math.max(0, share))} />;
  if (spec.min >= 0) {
    return (
      <span aria-hidden="true" className="relative block h-1.5 rounded-full bg-[var(--ds-gray-alpha-200)]">
        <span
          className="absolute inset-y-0 left-0 rounded-full bg-[var(--ds-gray-900)]"
          style={{ width: `${Math.min(1, Math.max(0, share)) * 100}%` }}
        />
      </span>
    );
  }
  const half = Math.max(-1, Math.min(1, value / spec.max));
  return (
    <span aria-hidden="true" className="relative block h-1.5 rounded-full bg-[var(--ds-gray-alpha-200)]">
      <span className="absolute inset-y-0 left-1/2 w-px bg-[var(--ds-gray-alpha-500)]" />
      <span
        className="absolute inset-y-0 rounded-full bg-[var(--ds-gray-900)]"
        style={half >= 0 ? { left: "50%", width: `${half * 50}%` } : { right: "50%", width: `${-half * 50}%` }}
      />
    </span>
  );
}

function ScaleMark({ at, track }: { at: number; track: string }) {
  return (
    <span aria-hidden="true" className="relative block h-1.5 rounded-full" style={{ background: track }}>
      <span
        className="absolute -top-[3px] h-3 w-1 -translate-x-1/2 rounded-full border border-[var(--ds-gray-alpha-600)] bg-[var(--ds-background-100)]"
        style={{ left: `${at * 100}%` }}
      />
    </span>
  );
}

/* Every panel, live, starting from `base` (the auto edit, or as shot before
   one exists). Each change snaps to Lightroom's step and goes to `onChange`,
   which renders the preview once the sliders rest. */
export function AdjustSliders({
  asShot,
  base,
  baseName = "auto value",
  disabled = false,
  onChange,
  params,
}: {
  asShot?: AsShot;
  base: Params;
  /** What a reset returns to, in the hints: "auto value", "as-shot value". */
  baseName?: string;
  disabled?: boolean;
  onChange: (params: Params) => void;
  params: Params;
}) {
  const [region, setRegion] = useState<CurveRegion>();
  const set = (key: ParamKey, value: number) => {
    const next = snapParam(key, value);
    if (next !== paramValue(params, key)) onChange({ ...params, [key]: next });
  };
  const marked = (spec: ParamSpec) => !isDefault(spec, paramValue(base, spec.key), asShot);

  return (
    <div>
      {SECTIONS.map((section) => {
        const changed = rowsOf(section).filter(
          ({ spec }) => Math.abs(paramValue(params, spec.key) - paramValue(base, spec.key)) > 1e-6,
        );
        const reset = () => {
          const next = { ...params };
          for (const { spec } of changed) next[spec.key] = paramValue(base, spec.key);
          onChange(next);
        };
        return (
          <DevelopSection
            key={section.id}
            section={section}
            marked={countSet(section, base, asShot)}
            edited={changed.length}
            onReset={disabled ? undefined : reset}
          >
            {section.id === "colorMixer" ? (
              <MixerSliders params={params} base={base} baseName={baseName} disabled={disabled} onSet={set} />
            ) : (
              <div className="space-y-4">
                {section.id === "toneCurve" ? (
                  <ToneCurvePreview params={params} base={base} active={region} className="mx-auto max-w-44" />
                ) : null}
                {section.blocks.map((block, index) => (
                  <BlockSliders
                    key={block.title ?? index}
                    block={block}
                    params={params}
                    base={base}
                    baseName={baseName}
                    disabled={disabled}
                    marked={marked}
                    onSet={set}
                    onRegion={section.id === "toneCurve" ? setRegion : undefined}
                  />
                ))}
              </div>
            )}
          </DevelopSection>
        );
      })}
      <div className="mt-3">
        <MarkLegend>
          Bold: changed by you. Double-click a name, or press Delete on a slider, to reset it to the {baseName}.
        </MarkLegend>
      </div>
    </div>
  );
}

function BlockSliders({
  base,
  baseName,
  block,
  disabled,
  marked,
  onRegion,
  onSet,
  params,
}: {
  base: Params;
  baseName: string;
  block: Block;
  disabled: boolean;
  marked: (spec: ParamSpec) => boolean;
  onRegion?: (region: CurveRegion | undefined) => void;
  onSet: (key: ParamKey, value: number) => void;
  params: Params;
}) {
  const titleId = useId();
  return (
    <div role="group" aria-labelledby={block.title ? titleId : undefined} className="space-y-2.5">
      {block.title ? <BlockTitle id={titleId}>{block.title}</BlockTitle> : null}
      {block.rows.map((row) => {
        const { key } = row.spec;
        if (row.spec.group === "Lens") {
          return (
            <SwitchRow
              key={key}
              row={row}
              value={paramValue(params, key)}
              marked={marked(row.spec)}
              disabled={disabled}
              onChange={(value) => onSet(key, value)}
            />
          );
        }
        const region = REGION_OF.get(key);
        return (
          <SliderRow
            key={key}
            row={row}
            value={paramValue(params, key)}
            base={paramValue(base, key)}
            baseName={baseName}
            marked={marked(row.spec)}
            disabled={disabled}
            onChange={(value) => onSet(key, value)}
            onActive={onRegion && region ? (active) => onRegion(active ? region : undefined) : undefined}
          />
        );
      })}
    </div>
  );
}
