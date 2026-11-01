import type { Params } from "@spl/bindings/raw-retouch/Params";
import { cn } from "@spl/ui";
import { formatSigned, paramValue } from "../format";
import { CURVE_REGIONS, CURVE_SPLITS, curvePath, type CurveRegion, type CurveSliders } from "./tone-curve";

export const CURVE_KEY: Record<CurveRegion, "curve_shadows" | "curve_darks" | "curve_lights" | "curve_highlights"> = {
  shadows: "curve_shadows",
  darks: "curve_darks",
  lights: "curve_lights",
  highlights: "curve_highlights",
};

export function curveSliders(params: Params): CurveSliders {
  return {
    shadows: paramValue(params, "curve_shadows"),
    darks: paramValue(params, "curve_darks"),
    lights: paramValue(params, "curve_lights"),
    highlights: paramValue(params, "curve_highlights"),
  };
}

const sameCurve = (a: CurveSliders, b: CurveSliders) =>
  CURVE_REGIONS.every(({ region }) => Math.abs(a[region] - b[region]) < 1e-6);

const SIZE = 100;
const GRID = CURVE_SPLITS.map((split) => split * SIZE);

/* The Tone Curve the region sliders make, drawn small: the identity as a
   dashed diagonal, the region splits as the grid, and the region of the
   slider under the pointer or focus shaded. In the editor, the auto edit's
   curve shows dashed once the sliders leave it. View only. */
export function ToneCurvePreview({
  active,
  base,
  className,
  params,
}: {
  active?: CurveRegion;
  base?: Params;
  className?: string;
  params: Params;
}) {
  const sliders = curveSliders(params);
  const baseSliders = base ? curveSliders(base) : undefined;
  const flat = CURVE_REGIONS.every(({ region }) => sliders[region] === 0);
  const description = flat
    ? "Tone curve: linear"
    : `Tone curve: ${CURVE_REGIONS.map(({ region }) => `${region} ${formatSigned(sliders[region])}`).join(", ")}`;
  const shaded = active ? CURVE_REGIONS.find((r) => r.region === active) : undefined;

  return (
    <svg
      role="img"
      aria-label={description}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      className={cn(
        "block aspect-square w-full rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)]",
        className,
      )}
    >
      {shaded ? (
        <rect
          x={shaded.from * SIZE}
          y={0}
          width={(shaded.to - shaded.from) * SIZE}
          height={SIZE}
          fill="var(--ds-gray-alpha-200)"
        />
      ) : null}
      <path
        d={GRID.map((g) => `M${g} 0V${SIZE}M0 ${SIZE - g}H${SIZE}`).join("")}
        stroke="var(--ds-gray-alpha-400)"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
      <path
        d={`M0 ${SIZE}L${SIZE} 0`}
        stroke="var(--ds-gray-alpha-500)"
        strokeWidth={1}
        strokeDasharray="3 3"
        vectorEffect="non-scaling-stroke"
      />
      {baseSliders && !sameCurve(baseSliders, sliders) ? (
        <path
          d={curvePath(baseSliders, SIZE)}
          fill="none"
          stroke="var(--ds-gray-700)"
          strokeWidth={1.25}
          strokeDasharray="4 3"
          vectorEffect="non-scaling-stroke"
        />
      ) : null}
      <path
        d={curvePath(sliders, SIZE)}
        fill="none"
        stroke="var(--ds-gray-1000)"
        strokeWidth={1.75}
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
