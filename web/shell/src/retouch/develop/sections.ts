/* How the Develop panel lays out the host's PARAM_SPECS: Lightroom Classic's
   panels in its order, the sub-headings inside them, shorter labels where a
   heading already says the rest, and the tinted slider tracks. Ranges,
   steps and defaults come from the specs. A field this table does not place
   still shows, at the end of its group's section (or in a section of its own
   for a new group), so a setting the host adds appears without UI work.
   Only package and type imports, so `node --test` runs it as is. */
import type { ParamGroup } from "@spl/bindings/raw-retouch/ParamGroup";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import { PARAM_SPECS } from "@spl/bindings/raw-retouch/paramSpecs";

type ParamKey = keyof Params;

export type SectionId = "basic" | "toneCurve" | "colorMixer" | "detail" | "lens" | "calibration";

export type Block = { title?: string; keys: ParamKey[] };

export type Section = {
  /** Stable: remembered per browser when the section is open. */
  id: string;
  title: string;
  groups: ParamGroup[];
  blocks: Block[];
};

/* Lightroom Classic's Develop panels, top to bottom. */
const LAYOUT: { id: SectionId; title: string; groups: ParamGroup[]; blocks: { title?: string; keys: string[] }[] }[] = [
  {
    id: "basic",
    title: "Basic",
    groups: ["Basic", "Presence"],
    blocks: [
      { title: "White balance", keys: ["temperature", "tint"] },
      { title: "Tone", keys: ["exposure", "contrast", "highlights", "shadows", "whites", "blacks"] },
      { title: "Presence", keys: ["texture", "clarity", "dehaze", "vibrance", "saturation"] },
    ],
  },
  {
    id: "toneCurve",
    title: "Tone Curve",
    groups: ["ToneCurve"],
    blocks: [{ title: "Region", keys: ["curve_highlights", "curve_lights", "curve_darks", "curve_shadows"] }],
  },
  // The Color Mixer draws its own grid of bands (color-mixer.tsx).
  { id: "colorMixer", title: "Color Mixer", groups: ["ColorMixer"], blocks: [] },
  {
    id: "detail",
    title: "Detail",
    groups: ["Detail"],
    blocks: [
      { title: "Sharpening", keys: ["sharpening", "sharpen_radius", "sharpen_detail", "sharpen_masking"] },
      { title: "Noise reduction", keys: ["luminance_nr", "color_nr"] },
    ],
  },
  { id: "lens", title: "Lens Corrections", groups: ["Lens"], blocks: [{ keys: ["lens_profile", "lens_ca"] }] },
  {
    id: "calibration",
    title: "Calibration",
    groups: ["Calibration"],
    blocks: [
      { title: "Shadows", keys: ["shadow_tint"] },
      { title: "Red primary", keys: ["red_primary_hue", "red_primary_saturation"] },
      { title: "Green primary", keys: ["green_primary_hue", "green_primary_saturation"] },
      { title: "Blue primary", keys: ["blue_primary_hue", "blue_primary_saturation"] },
    ],
  },
];

const specs = PARAM_SPECS as { key: ParamKey; group: ParamGroup }[];

/** Every field in exactly one section, in Lightroom's panel order. */
export const SECTIONS: Section[] = (() => {
  const placed = new Set<string>();
  const sections: Section[] = LAYOUT.map(({ blocks, groups, id, title }) => {
    const known = (key: string) => specs.some((s) => s.key === key && groups.includes(s.group));
    const out: Block[] = blocks
      .map((b) => ({ title: b.title, keys: b.keys.filter(known) as ParamKey[] }))
      .filter((b) => b.keys.length > 0);
    for (const b of out) for (const key of b.keys) placed.add(key);
    return { id, title, groups, blocks: out };
  });
  for (const spec of specs) {
    if (placed.has(spec.key)) continue;
    let section = sections.find((s) => s.groups.includes(spec.group));
    if (!section) {
      section = { id: spec.group, title: spec.group, groups: [spec.group], blocks: [] };
      sections.push(section);
    }
    // Unplaced fields go in a final untitled block (the Color Mixer's grid
    // takes its 24 from here).
    let last = section.blocks[section.blocks.length - 1];
    if (!last || last.title !== undefined) {
      last = { keys: [] };
      section.blocks.push(last);
    }
    last.keys.push(spec.key);
    placed.add(spec.key);
  }
  return sections;
})();

export const sectionKeys = (section: Section): ParamKey[] => section.blocks.flatMap((b) => b.keys);

/** Shorter labels where the block's heading says the rest. */
const LABELS: Partial<Record<ParamKey, string>> = {
  sharpening: "Amount",
  shadow_tint: "Tint",
  red_primary_hue: "Hue",
  red_primary_saturation: "Saturation",
  green_primary_hue: "Hue",
  green_primary_saturation: "Saturation",
  blue_primary_hue: "Hue",
  blue_primary_saturation: "Saturation",
  lens_profile: "Profile corrections",
  lens_ca: "Remove chromatic aberration",
};

export const shortLabel = (key: ParamKey, label: string) => LABELS[key] ?? label;

/** Accessible names where Lightroom's label alone is ambiguous on the page
    (two "Shadows" sliders, two "Detail"s, "Color" twice). */
const NAMES: Partial<Record<ParamKey, string>> = {
  curve_highlights: "Tone curve highlights",
  curve_lights: "Tone curve lights",
  curve_darks: "Tone curve darks",
  curve_shadows: "Tone curve shadows",
  sharpening: "Sharpening amount",
  sharpen_radius: "Sharpening radius",
  sharpen_detail: "Sharpening detail",
  sharpen_masking: "Sharpening masking",
  luminance_nr: "Luminance noise reduction",
  color_nr: "Color noise reduction",
};

export const accessibleName = (key: ParamKey, label: string) => NAMES[key] ?? label;

/** What a Lens Corrections switch does, under its label. */
export const DESCRIPTIONS: Partial<Record<ParamKey, string>> = {
  lens_profile: "Distortion and vignetting, from the lens profile.",
  lens_ca: "Colour fringes along high-contrast edges.",
};

/* ---- Color Mixer ---- */

/** Lightroom's eight hue bands and their centre hues in degrees (the same
    as the scopes' HUE_BANDS; a test keeps them equal). */
export const MIXER_BANDS = [
  { band: "red", label: "Red", hue: 0 },
  { band: "orange", label: "Orange", hue: 30 },
  { band: "yellow", label: "Yellow", hue: 60 },
  { band: "green", label: "Green", hue: 120 },
  { band: "aqua", label: "Aqua", hue: 180 },
  { band: "blue", label: "Blue", hue: 240 },
  { band: "purple", label: "Purple", hue: 270 },
  { band: "magenta", label: "Magenta", hue: 300 },
] as const;

export type MixerBand = (typeof MIXER_BANDS)[number]["band"];

export const MIXER_KINDS = [
  { kind: "hue", label: "Hue", short: "Hue" },
  { kind: "saturation", label: "Saturation", short: "Sat" },
  { kind: "luminance", label: "Luminance", short: "Lum" },
] as const;

export type MixerKind = (typeof MIXER_KINDS)[number]["kind"];

export const mixerKey = (kind: MixerKind, band: MixerBand) => `${kind}_${band}` as ParamKey;

/** Whether the host has all 24 Color Mixer fields; otherwise the section
    falls back to plain sliders. */
export const HAS_MIXER = MIXER_KINDS.every(({ kind }) =>
  MIXER_BANDS.every(({ band }) => specs.some((s) => s.key === mixerKey(kind, band))),
);

/* ---- Tracks ---- */

/* Hue swatches are data, like the vectorscope's dots: the colours a band or
   primary stands for, computed from its hue, not brand colours. */
const hsl = (hue: number, saturation = 75, lightness = 50) =>
  `hsl(${(((hue % 360) + 360) % 360).toFixed(0)} ${saturation}% ${lightness}%)`;

export const bandColor = (hue: number) => hsl(hue, 72, 52);

const gradient = (...stops: string[]) => `linear-gradient(to right, ${stops.join(", ")})`;

const WB_TEMPERATURE = gradient("var(--ds-blue-700)", "var(--ds-gray-300)", "var(--ds-amber-700)");
const WB_TINT = gradient("var(--ds-green-700)", "var(--ds-gray-300)", "var(--ds-pink-700)");

/** A hue slider: from the hue 30 degrees one way to 30 degrees the other. */
const hueTrack = (hue: number) => gradient(hsl(hue - 30), hsl(hue), hsl(hue + 30));
const saturationTrack = (hue: number) => gradient("var(--ds-gray-500)", hsl(hue));
const luminanceTrack = (hue: number) => gradient(hsl(hue, 70, 18), hsl(hue, 70, 50), hsl(hue, 70, 84));

const PRIMARY_HUE: Record<string, number> = { red: 0, green: 120, blue: 240 };

/** The track a slider shows instead of the neutral fill, if any: white
    balance and shadows tint as Lightroom draws them, and the colour bands
    and primaries in their own hues. */
export function trackOf(key: ParamKey): string | undefined {
  if (key === "temperature") return WB_TEMPERATURE;
  if (key === "tint" || key === "shadow_tint") return WB_TINT;
  const primary = /^(red|green|blue)_primary_(hue|saturation)$/.exec(key);
  if (primary) {
    const hue = PRIMARY_HUE[primary[1]!]!;
    return primary[2] === "hue" ? hueTrack(hue) : saturationTrack(hue);
  }
  const mixer = /^(hue|saturation|luminance)_([a-z]+)$/.exec(key);
  const band = mixer ? MIXER_BANDS.find((b) => b.band === mixer[2]) : undefined;
  if (mixer && band) {
    if (mixer[1] === "hue") return hueTrack(band.hue);
    if (mixer[1] === "saturation") return saturationTrack(band.hue);
    return luminanceTrack(band.hue);
  }
  return undefined;
}
