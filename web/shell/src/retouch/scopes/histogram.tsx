import { useMemo } from "react";
import {
  BINS,
  PLOT_HEIGHT,
  areaPath,
  linePath,
  peakOf,
  rgbLayers,
  scaled,
  shares,
  smooth,
  type HistogramScale,
  type Layer,
  type Scopes,
} from "./compute";

export type Channels = "rgb" | "luma";

/* Each overlap gets its own token colour, so the plot reads the same in both
   themes: one channel alone in its colour, two in their mix, all three in
   the neutral grey, the way Lightroom draws it. */
export const LAYER_FILL: Record<Layer, string> = {
  red: "var(--ds-red-700)",
  green: "var(--ds-green-700)",
  blue: "var(--ds-blue-700)",
  yellow: "var(--ds-amber-700)",
  cyan: "var(--ds-teal-700)",
  magenta: "var(--ds-pink-700)",
  neutral: "var(--ds-gray-700)",
};

/** The before image's luma as an outline over the main plot. */
export const OVERLAY_DASH = "4 3";

/* Lightroom's histogram as SVG: shadows on the left, highlights on the
   right, height per level. Stretched to its box, with hairline strokes. */
export function Histogram({
  channels,
  className,
  label,
  main,
  overlay,
  scale,
}: {
  channels: Channels;
  className?: string;
  /** What the plot shows, for screen readers. */
  label: string;
  main: Scopes;
  /** Drawn as a dashed luma outline on top (before, against after). */
  overlay?: Scopes;
  scale: HistogramScale;
}) {
  const plot = useMemo(() => {
    const series = (hist: Uint32Array, pixels: number) => smooth(shares(hist, pixels));
    const rgb =
      channels === "rgb"
        ? [series(main.red, main.pixels), series(main.green, main.pixels), series(main.blue, main.pixels)]
        : undefined;
    const luma = channels === "luma" ? series(main.luma, main.pixels) : undefined;
    const before = overlay ? series(overlay.luma, overlay.pixels) : undefined;
    // One scale for everything drawn, so before and after compare directly.
    const peak = peakOf([...(rgb ?? []), ...(luma ? [luma] : []), ...(before ? [before] : [])]);
    const fit = (values: number[]) => scaled(values, peak, scale);
    const lumaHeights = luma ? fit(luma) : undefined;
    return {
      layers: rgb
        ? rgbLayers(fit(rgb[0]!), fit(rgb[1]!), fit(rgb[2]!)).map(({ layer, heights }) => ({
            layer,
            d: areaPath(heights),
          }))
        : [],
      lumaArea: lumaHeights ? areaPath(lumaHeights) : undefined,
      lumaLine: lumaHeights ? linePath(lumaHeights) : undefined,
      overlay: before ? linePath(fit(before)) : undefined,
    };
  }, [channels, main, overlay, scale]);

  const quarter = BINS / 4;
  return (
    <svg
      role="img"
      aria-label={label}
      viewBox={`0 0 ${BINS} ${PLOT_HEIGHT}`}
      preserveAspectRatio="none"
      className={className}
    >
      <path
        d={`M${quarter} 0V${PLOT_HEIGHT}M${2 * quarter} 0V${PLOT_HEIGHT}M${3 * quarter} 0V${PLOT_HEIGHT}`}
        stroke="var(--ds-gray-alpha-400)"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
      {plot.layers.length ? (
        // Opacity on the group, after the layers are painted: translucent
        // against the surface without the layers bleeding into each other.
        <g opacity={0.92}>
          {plot.layers.map(({ layer, d }) => (
            <path key={layer} d={d} fill={LAYER_FILL[layer]} />
          ))}
        </g>
      ) : null}
      {plot.lumaArea ? <path d={plot.lumaArea} fill="var(--ds-gray-alpha-500)" /> : null}
      {plot.lumaLine ? (
        <path
          d={plot.lumaLine}
          fill="none"
          stroke="var(--ds-gray-1000)"
          strokeWidth={1.25}
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      ) : null}
      {plot.overlay ? (
        <path
          d={plot.overlay}
          fill="none"
          stroke="var(--ds-gray-1000)"
          strokeWidth={1.5}
          strokeDasharray={OVERLAY_DASH}
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      ) : null}
    </svg>
  );
}
