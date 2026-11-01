# Design language

The UI reuses the ATSGrade web app's design system (`ats-checker/src/app/globals.css`
and `src/components/ui`), ported into `web/ui`. Same tokens, same primitives,
different brand mark.

## Principles

- **Neutral first.** Surfaces are grayscale. Color carries meaning only.
- **One action color.** `--ds-blue-700` for primary buttons, focus rings and the brand mark's top sheet. Inline text links (`text.link`) use the same hue one step darker, `--ds-blue-900`, because blue-700 text at 13px is below AA contrast in both themes.
- **Status lives in dots.** `StatusSignal` / `Badge` boxes stay neutral; the dot is green (ok), red (failed), amber (warning), blue (running, pulsing), gray (idle).
- **Mono for machine things.** Geist Mono for uppercase eyebrows, IDs, numbers, metadata. Geist Sans for everything else.
- **Quiet depth.** 1px hairline borders (`--ds-gray-alpha-400`), `depth-surface` for raised panels, a 72px grid page background.

## Tokens (`web/ui/src/styles/tokens.css`)

- Scales `gray`, `blue`, `red`, `amber`, `green`, `teal`, `purple`, `pink`, steps 100–1000, stored as HSL triplets (`--ds-blue-700-value`), exposed as colors (`--ds-blue-700`).
- `--ds-background-100` (cards) and `--ds-background-200` (page, sunken areas).
- `--ds-gray-alpha-*` for borders and overlays.
- Light by default; dark via `prefers-color-scheme`, or forced with `<html data-theme="light|dark">` (saved as `spl-theme`).

## Shape and type

| Element | Value |
|---|---|
| Buttons | 7px radius, 40px (md) / 32px (sm) tall |
| Cards, surfaces | 8px radius |
| Badges | 6px radius, 24px tall |
| Page title | 34px / 42px (sm+), semibold |
| Lead | 17px / 32px line |
| Body / small | 15px / 14px |
| Eyebrow / meta | 13px / 12px mono uppercase |
| Container | `--layout-max-w` 1360px |

## Components (`@spl/ui`)

`Button`, `ButtonLink`, `Card` (+ header/title/description/footer), `Surface`, `SectionHeader`, `DataPanel` (titled panel: Surface + SectionHeader + body), `ActionBar` (wrapping row of actions), `Skeleton` (shimmer placeholder), `IconTile` (40px icon or label tile), `Badge`, `StatusSignal`, `Gauge`, `BrandLogo`/`BrandMark`, `text` (type scale), `cn`.

Controls:

- `ProgressBar`: 4px neutral bar (gray-1000 fill, not a status colour); determinate with `value` 0–1, the shared indeterminate sweep without. Always give it a `label`.
- `Switch`: an on/off setting with label and optional description; on is the action blue. A real `<label>` for a `role="switch"` button.
- `SegmentedControl`: two to five exclusive choices (a model, a count, a mode) on native radios, so arrow keys and screen readers work as a radio group.
- `RangeInput`: native range in the system look (`.ds-range`): 4px track, 14px neutral thumb, focus ring on the thumb. The fill starts at `origin` (0 for a bipolar slider); `track` swaps in a token gradient, e.g. white balance blue→amber.

Pages with app-specific views (the Retouch page, `#/retouch`) compose these; photos and other user content may carry their own colours, and labels laid over a photo use a translucent black chip with white text so they read on any image. Its scopes are data, drawn with tokens: histogram channels in the red, green and blue 700 steps, their overlaps in amber, teal and pink 700, all three in gray-700; only the vectorscope's dots take the photo's own hues.

Its Develop panel (Lightroom's Basic, Tone Curve, Color Mixer, Detail, Lens Corrections and Calibration, laid out from the host's `PARAM_SPECS`) shows each section as a disclosure row with a mono summary, remembered per browser. The Color Mixer's and Calibration's slider tracks and band chips are data like the vectorscope: hues computed from Lightroom's band centres, not brand colours. A 6px blue-700 dot marks a value set away from Lightroom's default (as shot for white balance); a gray tick under a track marks the auto edit's value once a slider leaves it.

## Motion

Reveal and stagger on scroll, ping for live status, skeleton shimmer, indeterminate bar, check draw. All disabled under `prefers-reduced-motion`.

## Rules for new UI

- Compose `@spl/ui` primitives and `text.*` classes; don't hardcode font sizes or hex colors.
- New status colors map to existing scales.
- Fonts are bundled (`@fontsource-variable/geist*`); no external font or script loads, so the UI works on offline GPU machines.
