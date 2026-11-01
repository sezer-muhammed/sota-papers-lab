import type { ParamGroup } from "@spl/bindings/raw-retouch/ParamGroup";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import { SPECS, isDefault, paramValue, specOf, type ParamKey, type ParamSpec } from "../format";
import { HUE_BANDS, type HueBand } from "../scopes/compute";

/* Lightroom Classic's Develop panels, in its order, built from the app's
   PARAM_SPECS table: ranges, steps, defaults and labels come from there; this
   file only arranges the settings the way Lightroom does (sub-headings,
   shorter labels under a heading, the Color Mixer's bands). A setting the
   table gains later shows up at the end of its panel without a change here. */

export type SectionId = "basic" | "toneCurve" | "colorMixer" | "detail" | "lens" | "calibration";

/** A slider (or switch) as a panel shows it. `label` is shorter than the
    spec's when a heading carries the rest ("Hue" under "Red Primary");
    `name`, what assistive tech reads, is whole and unique on the page. */
export type Row = { spec: ParamSpec; label: string; name: string };
export type Block = { title?: string; rows: Row[] };
export type Section = { id: SectionId; title: string; groups: ParamGroup[]; blocks: Block[] };

/* Labels the Basic panel or the Color Mixer also use, made unique. */
const NAMES: Partial<Record<ParamKey, string>> = {
  curve_highlights: "Tone Curve Highlights",
  curve_lights: "Tone Curve Lights",
  curve_darks: "Tone Curve Darks",
  curve_shadows: "Tone Curve Shadows",
  sharpening: "Sharpening Amount",
  sharpen_radius: "Sharpening Radius",
  sharpen_detail: "Sharpening Detail",
  sharpen_masking: "Sharpening Masking",
  luminance_nr: "Luminance Noise Reduction",
  color_nr: "Color Noise Reduction",
};

export const rowOf = (spec: ParamSpec, label?: string): Row => ({
  spec,
  label: label ?? spec.label,
  name: NAMES[spec.key] ?? spec.label,
});

const row = (key: ParamKey, label?: string): Row => rowOf(specOf(key), label);

const SECTION_LAYOUT: Omit<Section, "blocks">[] = [
  { id: "basic", title: "Basic", groups: ["Basic", "Presence"] },
  { id: "toneCurve", title: "Tone Curve", groups: ["ToneCurve"] },
  { id: "colorMixer", title: "Color Mixer", groups: ["ColorMixer"] },
  { id: "detail", title: "Detail", groups: ["Detail"] },
  { id: "lens", title: "Lens Corrections", groups: ["Lens"] },
  { id: "calibration", title: "Calibration", groups: ["Calibration"] },
];

/* Lightroom's arrangement inside each panel. Texture, Clarity and Dehaze sit
   in Basic's Presence block, above Vibrance and Saturation. */
const BLOCKS: Record<Exclude<SectionId, "colorMixer">, Block[]> = {
  basic: [
    { title: "White Balance", rows: [row("temperature"), row("tint")] },
    {
      title: "Tone",
      rows: [row("exposure"), row("contrast"), row("highlights"), row("shadows"), row("whites"), row("blacks")],
    },
    {
      title: "Presence",
      rows: [row("texture"), row("clarity"), row("dehaze"), row("vibrance"), row("saturation")],
    },
  ],
  toneCurve: [
    {
      title: "Region",
      rows: [row("curve_highlights"), row("curve_lights"), row("curve_darks"), row("curve_shadows")],
    },
  ],
  detail: [
    {
      title: "Sharpening",
      rows: [
        row("sharpening", "Amount"),
        row("sharpen_radius"),
        row("sharpen_detail"),
        row("sharpen_masking"),
      ],
    },
    { title: "Noise Reduction", rows: [row("luminance_nr"), row("color_nr")] },
  ],
  lens: [{ rows: [row("lens_profile"), row("lens_ca")] }],
  calibration: [
    { title: "Shadows", rows: [row("shadow_tint", "Tint")] },
    { title: "Red Primary", rows: [row("red_primary_hue", "Hue"), row("red_primary_saturation", "Saturation")] },
    { title: "Green Primary", rows: [row("green_primary_hue", "Hue"), row("green_primary_saturation", "Saturation")] },
    { title: "Blue Primary", rows: [row("blue_primary_hue", "Hue"), row("blue_primary_saturation", "Saturation")] },
  ],
};

/* ---- Color Mixer ---- */

export type MixerProperty = "hue" | "saturation" | "luminance";
export const MIXER_PROPERTIES: { value: MixerProperty; label: string }[] = [
  { value: "hue", label: "Hue" },
  { value: "saturation", label: "Saturation" },
  { value: "luminance", label: "Luminance" },
];

export type Band = { name: HueBand; label: string; hue: number };

/** Lightroom's eight bands with their centre hues (shared with the scopes). */
export const BANDS: Band[] = HUE_BANDS.map((band) => ({
  name: band.name,
  label: band.name[0]!.toUpperCase() + band.name.slice(1),
  hue: band.hue,
}));

/** The Color Mixer setting of one band and property, e.g. `hue_red`. */
export function mixerKey(property: MixerProperty, band: HueBand): ParamKey {
  return `${property}_${band}` as ParamKey;
}

/* ---- Sections ---- */

function buildSections(): Section[] {
  const placed = new Set<ParamKey>();
  const sections = SECTION_LAYOUT.map((layout): Section => {
    const blocks: Block[] =
      layout.id === "colorMixer"
        ? MIXER_PROPERTIES.map((p) => ({
            title: p.label,
            rows: BANDS.map((band) => row(mixerKey(p.value, band.name), band.label)),
          }))
        : BLOCKS[layout.id].map((block) => ({ ...block, rows: block.rows.slice() }));
    for (const block of blocks) for (const r of block.rows) placed.add(r.spec.key);
    return { ...layout, blocks };
  });
  // Settings the table has and the arrangement above does not: last in their panel.
  for (const spec of SPECS) {
    if (placed.has(spec.key)) continue;
    const section = sections.find((s) => s.groups.includes(spec.group)) ?? sections[0]!;
    const last = section.blocks[section.blocks.length - 1];
    if (last && !last.title) last.rows.push(rowOf(spec));
    else section.blocks.push({ rows: [rowOf(spec)] });
  }
  return sections;
}

export const SECTIONS: Section[] = buildSections();

export const rowsOf = (section: Section): Row[] => section.blocks.flatMap((block) => block.rows);

export type AsShot = { temperature: number; tint: number };

/** How many of a section's settings `params` holds away from their default. */
export function countSet(section: Section, params: Params, asShot?: AsShot): number {
  return rowsOf(section).filter((r) => !isDefault(r.spec, paramValue(params, r.spec.key), asShot)).length;
}

/* ---- Slider tracks ----
   Lightroom tints the colour sliders by what they do. White balance and
   tints use token colours; the Color Mixer and the Calibration primaries
   show real hues (photo colours, like the vectorscope's dots). */

const hsl = (hue: number, saturation = 75, lightness = 50) => `hsl(${hue} ${saturation}% ${lightness}%)`;

/** A hue slider's track: the neighbouring hues it turns towards. */
export const hueTrack = (hue: number, reach = 30) =>
  `linear-gradient(to right, ${hsl(hue - reach)}, ${hsl(hue)}, ${hsl(hue + reach)})`;
/** From grey to the full colour. */
export const saturationTrack = (hue: number) => `linear-gradient(to right, ${hsl(hue, 0, 55)}, ${hsl(hue, 85)})`;
/** From dark to light in the colour. */
export const luminanceTrack = (hue: number) =>
  `linear-gradient(to right, ${hsl(hue, 70, 14)}, ${hsl(hue)}, ${hsl(hue, 70, 86)})`;

/** A band's swatch colour. */
export const bandColor = (hue: number) => hsl(hue, 72, 50);

const WB_TEMPERATURE = "linear-gradient(to right, var(--ds-blue-700), var(--ds-gray-300), var(--ds-amber-700))";
const GREEN_MAGENTA = "linear-gradient(to right, var(--ds-green-700), var(--ds-gray-300), var(--ds-pink-700))";

const MIXER_TRACK: Record<MixerProperty, (hue: number) => string> = {
  hue: (hue) => hueTrack(hue),
  saturation: saturationTrack,
  luminance: luminanceTrack,
};

const TRACKS = new Map<ParamKey, string>([
  ["temperature", WB_TEMPERATURE],
  ["tint", GREEN_MAGENTA],
  ["shadow_tint", GREEN_MAGENTA],
  ["red_primary_hue", hueTrack(0, 40)],
  ["red_primary_saturation", saturationTrack(0)],
  ["green_primary_hue", hueTrack(120, 50)],
  ["green_primary_saturation", saturationTrack(120)],
  ["blue_primary_hue", hueTrack(240, 40)],
  ["blue_primary_saturation", saturationTrack(240)],
  ...MIXER_PROPERTIES.flatMap((p) =>
    BANDS.map((band): [ParamKey, string] => [mixerKey(p.value, band.name), MIXER_TRACK[p.value](band.hue)]),
  ),
]);

/** The tinted track of a colour slider; none for the neutral ones. */
export const trackOf = (key: ParamKey): string | undefined => TRACKS.get(key);

/** The temperature slider spans where photos live; the field beside it
    takes Lightroom's full range. */
export const TEMPERATURE_SLIDER = { min: 2000, max: 12000, step: 50 } as const;
