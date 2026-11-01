import type { Params } from "@spl/bindings/raw-retouch/Params";
import { cn, text } from "@spl/ui";
import { formatSigned, type ParamKey } from "../format";
import { CURVE_SPLITS, REGIONS, curvePath, regionRangePath, type CurveRegions, type Region } from "./curve";

/** The Tone Curve region a Params field moves. */
export const CURVE_REGION: Partial<Record<ParamKey, Region>> = {
  curve_highlights: "highlights",
  curve_lights: "lights",
  curve_darks: "darks",
  curve_shadows: "shadows",
};

const regionsOf = (p: Params): CurveRegions => ({
  shadows: p.curve_shadows,
  darks: p.curve_darks,
  lights: p.curve_lights,
  highlights: p.curve_highlights,
});

const NAMES: Record<Region, string> = { highlights: "Highlights", lights: "Lights", darks: "Darks", shadows: "Shadows" };

const SIZE = 100;
const GRID = CURVE_SPLITS.map((s) => s * SIZE);

/* The parametric curve the four region sliders make, Lightroom-style: the
   quarter grid at the splits, the straight line for reference, the auto
   edit's curve dashed once the sliders have left it, and, while a slider is
   in use, the band its region can move the curve through. */
export function CurvePreview({
  active,
  base,
  className,
  params,
}: {
  /** The region whose slider is in use. */
  active?: Region;
  /** The auto edit, drawn dashed when it differs. */
  base?: Params;
  className?: string;
  params: Params;
}) {
  const r = regionsOf(params);
  const b = base ? regionsOf(base) : undefined;
  const showBase = b !== undefined && REGIONS.some((region) => Math.round(b[region]) !== Math.round(r[region]));
  const label = `Tone curve: ${REGIONS.map((region) => `${NAMES[region]} ${formatSigned(r[region])}`).join(", ")}`;

  return (
    <div className={cn("flex flex-wrap items-end gap-x-4 gap-y-2", className)}>
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        role="img"
        aria-label={label}
        className="block aspect-square w-40 shrink-0 rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)]"
      >
        <g stroke="var(--ds-gray-alpha-300)" strokeWidth={1} vectorEffect="non-scaling-stroke">
          {GRID.map((g) => (
            <line key={`v${g}`} x1={g} x2={g} y1={0} y2={SIZE} vectorEffect="non-scaling-stroke" />
          ))}
          {GRID.map((g) => (
            <line key={`h${g}`} x1={0} x2={SIZE} y1={g} y2={g} vectorEffect="non-scaling-stroke" />
          ))}
        </g>
        {active ? <path d={regionRangePath(r, active, SIZE)} fill="var(--ds-gray-alpha-300)" stroke="none" /> : null}
        <line
          x1={0}
          y1={SIZE}
          x2={SIZE}
          y2={0}
          stroke="var(--ds-gray-alpha-500)"
          strokeWidth={1}
          strokeDasharray="2 3"
          vectorEffect="non-scaling-stroke"
        />
        {showBase ? (
          <path
            d={curvePath(b, SIZE)}
            fill="none"
            stroke="var(--ds-gray-700)"
            strokeWidth={1.25}
            strokeDasharray="4 3"
            vectorEffect="non-scaling-stroke"
          />
        ) : null}
        <path
          d={curvePath(r, SIZE)}
          fill="none"
          stroke="var(--ds-gray-1000)"
          strokeWidth={1.75}
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div className={cn(text.meta, "min-w-0 space-y-1 pb-0.5")}>
        <p>{active ? `${NAMES[active]}: the shaded band is its reach` : "Splits at 25 / 50 / 75"}</p>
        {showBase ? (
          <p className="flex items-center gap-1.5">
            <svg aria-hidden="true" width="16" height="4" className="shrink-0">
              <line x1="0" y1="2" x2="16" y2="2" stroke="var(--ds-gray-700)" strokeWidth="1.25" strokeDasharray="4 3" />
            </svg>
            Auto edit
          </p>
        ) : null}
      </div>
    </div>
  );
}
