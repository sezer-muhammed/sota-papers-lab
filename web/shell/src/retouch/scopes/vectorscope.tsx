import { useEffect, useRef, useSyncExternalStore } from "react";
import { cn } from "@spl/ui";
import { SCOPE_LIGHTNESS, scopeImage, type Scopes } from "./compute";

/** Padding around the unit circle, in its radii (the rim's markers sit on it). */
const PAD = 0.1;
/** The canvas and the disc cover the unit circle inside that padding. */
const INSET = `${(PAD / (2 + 2 * PAD)) * 100}%`;
const SPAN = `${(2 / (2 + 2 * PAD)) * 100}%`;

/* The primaries and secondaries on the rim, as on a vectorscope's targets. */
const MARKERS = [
  { hue: 0, name: "Red", fill: "var(--ds-red-700)" },
  { hue: 60, name: "Yellow", fill: "var(--ds-amber-700)" },
  { hue: 120, name: "Green", fill: "var(--ds-green-700)" },
  { hue: 180, name: "Cyan", fill: "var(--ds-teal-700)" },
  { hue: 240, name: "Blue", fill: "var(--ds-blue-700)" },
  { hue: 300, name: "Magenta", fill: "var(--ds-pink-700)" },
];

/** The ColorChecker's light and dark skin patches both sit at about 18° in sRGB. */
export const SKIN_HUE = 18;

/* The dots are pixels, not tokens, so they follow the theme here: forced by
   <html data-theme> (see applyTheme), else the system's colour scheme. */
const DARK_QUERY = "(prefers-color-scheme: dark)";

function isDark(): boolean {
  const forced = document.documentElement.dataset.theme;
  if (forced === "dark" || forced === "light") return forced === "dark";
  return window.matchMedia?.(DARK_QUERY).matches ?? false;
}

function subscribeTheme(onChange: () => void): () => void {
  const media = window.matchMedia?.(DARK_QUERY);
  media?.addEventListener("change", onChange);
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return () => {
    media?.removeEventListener("change", onChange);
    observer.disconnect();
  };
}

const polar = (degrees: number, radius: number) => {
  const a = (degrees * Math.PI) / 180;
  // SVG's y grows downwards; hues go counter-clockwise from red on the right.
  return { x: radius * Math.cos(a), y: -radius * Math.sin(a) };
};

/* Hue around the circle (red right, yellow up-right, green up-left, cyan
   left, blue down-left, magenta down-right), saturation from the centre
   (neutral) out to the rim (a pure primary). Each dot is the photo's pixels
   at that hue and saturation, more opaque where there are more of them. */
export function Vectorscope({
  className,
  label,
  scopes,
}: {
  className?: string;
  /** What the scope shows, for screen readers. */
  label: string;
  scopes?: Scopes;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const dark = useSyncExternalStore(subscribeTheme, isDark, () => false);

  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext("2d");
    if (!element || !context) return;
    if (!scopes) {
      context.clearRect(0, 0, element.width, element.height);
      return;
    }
    // Setting the size also clears the canvas.
    element.width = scopes.cells;
    element.height = scopes.cells;
    const lightness = dark ? SCOPE_LIGHTNESS.dark : SCOPE_LIGHTNESS.light;
    const pixels = scopeImage(scopes.scope, scopes.cells, scopes.scopeMax, lightness);
    context.putImageData(new ImageData(pixels, scopes.cells, scopes.cells), 0, 0);
  }, [scopes, dark]);

  const skin = polar(SKIN_HUE, 1);
  const box = 1 + PAD;

  return (
    <div role="img" aria-label={label} className={cn("relative aspect-square", className)}>
      <div
        aria-hidden="true"
        className="absolute rounded-full border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)]"
        style={{ left: INSET, top: INSET, width: SPAN, height: SPAN }}
      />
      <canvas
        ref={canvas}
        aria-hidden="true"
        className="absolute rounded-full"
        style={{ left: INSET, top: INSET, width: SPAN, height: SPAN }}
      />
      <svg
        aria-hidden="true"
        viewBox={`${-box} ${-box} ${2 * box} ${2 * box}`}
        className="absolute inset-0 h-full w-full overflow-visible"
      >
        <g fill="none" strokeWidth={1}>
          <circle r={0.5} stroke="var(--ds-gray-alpha-500)" strokeDasharray="2 3" vectorEffect="non-scaling-stroke" />
          <path d="M-1 0H1M0 -1V1" stroke="var(--ds-gray-alpha-400)" vectorEffect="non-scaling-stroke" />
        </g>
        <line
          x1={0}
          y1={0}
          x2={skin.x}
          y2={skin.y}
          stroke="var(--ds-gray-alpha-700)"
          strokeWidth={1}
          strokeDasharray="3 2"
          vectorEffect="non-scaling-stroke"
        >
          <title>Skin tones</title>
        </line>
        {MARKERS.map((marker) => {
          const at = polar(marker.hue, 1);
          return (
            <circle
              key={marker.name}
              cx={at.x}
              cy={at.y}
              r={0.055}
              fill={marker.fill}
              stroke="var(--ds-background-100)"
              strokeWidth={1.5}
              vectorEffect="non-scaling-stroke"
            >
              <title>{marker.name}</title>
            </circle>
          );
        })}
      </svg>
    </div>
  );
}
