/* Before/after examples on the Home page: real `raw_retouch.auto` runs (Claude
   on, model opus, effort medium, five photos at once on an RTX 4070 SUPER) on
   CC0 sample RAWs from raw.pixls.us, never anyone's own photos.

   Before: the full-resolution render of the as-shot settings (camera white
   balance, Lightroom's defaults, this photo's automatic Detail and Lens
   Corrections). After: the auto run's full-resolution JPEG. Both scaled to
   the same web size and JPEG quality (public/examples/<key>-before|after.jpg).

   `changes` are the five settings that moved furthest from as shot, by share
   of their slider's range; `scene` is one sentence of Claude's own scene read,
   verbatim. A fifth sample (ILCE-6700) is not shown as a pair because Claude
   ranked its as-shot render first and the app returned it unchanged. */

export type SliderChange = {
  /** `Params` key. */
  key: string;
  /** Lightroom's label. */
  label: string;
  /** As shot (Lightroom's default). */
  from: number;
  /** The chosen edit. */
  to: number;
};

export type Example = {
  key: string;
  camera: string;
  /** Null when the body has a built-in lens. */
  lens: string | null;
  focalMm: number;
  fNumber: number;
  exposureS: number;
  iso: number;
  /** Full-resolution size of the developed photo. */
  megapixels: number;
  /** What the photo shows, for alt text. */
  subject: string;
  /** Name of the look that won, as Claude or the app named it. */
  look: string;
  /** How that look came to be. */
  lookSource: "claude" | "builtin" | "refined";
  /** Claude's refinement, when it won its blind A/B and became the result. */
  refinement?: string;
  scene: string;
  changes: SliderChange[];
  /** Seconds: the whole auto run, the GPU search, and Claude's calls. */
  totalS: number;
  searchS: number;
  claudeS: number;
  claudeCalls: number;
  /** Web image size (both images of a pair). */
  width: number;
  height: number;
  /** The sample on raw.pixls.us: its name in the lab's sample set,
      the name it has there, the camera's folder, and the upload date. */
  source: { file: string; pixls: string; url: string; uploaded: string };
};

const pixlsFolder = (model: string) => `https://raw.pixls.us/data/Sony/${model}/`;

export const EXAMPLES: Example[] = [
  {
    key: "rx100",
    camera: "Sony DSC-RX100M3",
    lens: null,
    focalMm: 8.8,
    fNumber: 2.8,
    exposureS: 1 / 500,
    iso: 125,
    megapixels: 20,
    subject: "a red playground train under an electricity pylon and an overcast sky",
    look: "Faithful Overcast",
    lookSource: "claude",
    scene:
      "The sky is heavily clipped and the light is flat, so the edit needs strong highlight recovery to bring back cloud detail and a little midtone snap so the red train and the green grass stand out without looking garish.",
    changes: [
      { key: "highlights", label: "Highlights", from: 0, to: -75 },
      { key: "blacks", label: "Blacks", from: 0, to: -12 },
      { key: "vibrance", label: "Vibrance", from: 0, to: 12 },
      { key: "shadows", label: "Shadows", from: 0, to: 10 },
      { key: "luminance_blue", label: "Blue Luminance", from: 0, to: -10 },
    ],
    totalS: 36.5,
    searchS: 3.6,
    claudeS: 30.5,
    claudeCalls: 4,
    width: 1600,
    height: 1067,
    source: {
      file: "DSC-RX100M3__DSC00734.ARW",
      pixls: "Sony - DSC-RX100M3 - 12bit 12bit compressed (3:2).ARW",
      url: pixlsFolder("DSC-RX100M3"),
      uploaded: "2017-12-29",
    },
  },
  {
    key: "a6000",
    camera: "Sony ILCE-6000",
    lens: "E 18-55mm F3.5-5.6 OSS",
    focalMm: 39,
    fNumber: 5,
    exposureS: 1 / 160,
    iso: 100,
    megapixels: 24,
    subject: "a rocky creek bed with driftwood under autumn foliage",
    look: "Natural Overcast",
    lookSource: "claude",
    scene:
      "The Neutral render is a little dark and flat, with no clipping at either end. It needs about +0.4 EV, a gentle contrast lift and modest colour enrichment, while keeping the pale granite neutral and its highlights intact.",
    changes: [
      { key: "highlights", label: "Highlights", from: 0, to: -20 },
      { key: "blacks", label: "Blacks", from: 0, to: -18 },
      { key: "whites", label: "Whites", from: 0, to: 15 },
      { key: "vibrance", label: "Vibrance", from: 0, to: 15 },
      { key: "contrast", label: "Contrast", from: 0, to: 12 },
    ],
    totalS: 44.8,
    searchS: 3.7,
    claudeS: 38.3,
    claudeCalls: 6,
    width: 1440,
    height: 960,
    source: {
      file: "ILCE-6000__DSC01542.ARW",
      pixls: "Sony - ILCE-6000 - 12bit 12bit compressed (3:2).ARW",
      url: pixlsFolder("ILCE-6000"),
      uploaded: "2017-11-26",
    },
  },
  {
    key: "a6400",
    camera: "Sony ILCE-6400",
    lens: "E PZ 18-105mm F4 G OSS",
    focalMm: 105,
    fNumber: 4,
    exposureS: 1 / 250,
    iso: 250,
    megapixels: 24,
    subject: "a stone frog statue by a wooden fence, with a blurred lawn and a blue tarp behind",
    look: "Natural Garden",
    lookSource: "refined",
    refinement: "Blue Saturation from −12 to −20, so the tarp stops pulling the eye from the frog. Won 2 of 2 blind comparisons.",
    scene:
      "It's a bit dark and flat. It needs a small exposure lift, a gentle curve and a calmer tarp, so the frog stays the subject and the soft mood is kept.",
    changes: [
      { key: "saturation_blue", label: "Blue Saturation", from: 0, to: -20 },
      { key: "blacks", label: "Blacks", from: 0, to: -18 },
      { key: "highlights", label: "Highlights", from: 0, to: -15 },
      { key: "whites", label: "Whites", from: 0, to: 12 },
      { key: "contrast", label: "Contrast", from: 0, to: 10 },
    ],
    totalS: 41.5,
    searchS: 3.6,
    claudeS: 35.4,
    claudeCalls: 6,
    width: 1600,
    height: 1067,
    source: {
      file: "ILCE-6400__DSC00087.ARW",
      pixls: "Sony - ILCE-6400 - 12bit 12bit compressed (3:2).ARW",
      url: pixlsFolder("ILCE-6400"),
      uploaded: "2019-09-19",
    },
  },
  {
    key: "a7r5",
    camera: "Sony ILCE-7RM5",
    lens: "FE 50mm F1.2 GM",
    focalMm: 50,
    fNumber: 1.2,
    exposureS: 1 / 2000,
    iso: 100,
    megapixels: 60,
    subject: "a sunlit park lawn in front of slim trees and a white tent",
    look: "Natural Park",
    lookSource: "claude",
    scene:
      "The foreground grass is very bright and its yellow-green is close to neon, so the edit should mainly tame the lawn's brightness and saturation, keep detail in the tent, and leave the shadowed foliage readable.",
    changes: [
      { key: "highlights", label: "Highlights", from: 0, to: -30 },
      { key: "saturation_yellow", label: "Yellow Saturation", from: 0, to: -15 },
      { key: "blacks", label: "Blacks", from: 0, to: -12 },
      { key: "shadows", label: "Shadows", from: 0, to: 10 },
      { key: "saturation_green", label: "Green Saturation", from: 0, to: -10 },
    ],
    totalS: 47.8,
    searchS: 3.2,
    claudeS: 39.9,
    claudeCalls: 6,
    width: 1600,
    height: 1067,
    source: {
      file: "ILCE-7RM5__7RM5-LosslessCompressedLarge.ARW",
      pixls: "Sony - ILCE-7RM5 - 14bit (4:3).ARW",
      url: pixlsFolder("ILCE-7RM5"),
      uploaded: "2022-12-18",
    },
  },
];

export const exampleImage = (key: string, side: "before" | "after") =>
  `${import.meta.env.BASE_URL}examples/${key}-${side}.jpg`;

const signed = (n: number) => (n > 0 ? `+${n}` : n < 0 ? `−${Math.abs(n)}` : "0");

/** "Highlights −75", "Exposure +0.40 EV". */
export function formatChange(change: SliderChange) {
  if (change.key === "exposure") return `${change.label} ${signed(Number(change.to.toFixed(2)))} EV`;
  if (change.key === "temperature") return `${change.label} ${change.to} K`;
  return `${change.label} ${signed(change.to)}`;
}

/** "39 mm · f/5 · 1/160 s · ISO 100". */
export function formatExposure(e: Example) {
  const shutter = e.exposureS >= 1 ? `${e.exposureS} s` : `1/${Math.round(1 / e.exposureS)} s`;
  return `${e.focalMm} mm · f/${e.fNumber} · ${shutter} · ISO ${e.iso}`;
}

export const LOOK_SOURCE: Record<Example["lookSource"], string> = {
  claude: "Claude's look",
  builtin: "Built-in look",
  refined: "Claude's look, refined",
};
