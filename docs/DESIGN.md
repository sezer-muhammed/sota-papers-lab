# Design language

The UI reuses the ATSGrade web app's design system (`ats-checker/src/app/globals.css`
and `src/components/ui`), ported into `web/ui`. Same tokens, same primitives,
different brand mark.

## Principles

- **Neutral first.** Surfaces are grayscale. Color carries meaning only.
- **One action color.** `--ds-blue-700` for primary buttons, focus rings and the brand mark's top sheet.
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

`Button`, `ButtonLink`, `Card` (+ header/title/description/footer), `Surface`, `SectionHeader`, `Badge`, `StatusSignal`, `Gauge`, `BrandLogo`/`BrandMark`, `text` (type scale), `cn`.

## Motion

Reveal and stagger on scroll, ping for live status, skeleton shimmer, indeterminate bar, check draw. All disabled under `prefers-reduced-motion`.

## Rules for new UI

- Compose `@spl/ui` primitives and `text.*` classes; don't hardcode font sizes or hex colors.
- New status colors map to existing scales.
- Fonts are bundled (`@fontsource-variable/geist*`); no external font or script loads, so the UI works on offline GPU machines.
