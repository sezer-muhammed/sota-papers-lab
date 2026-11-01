import { cn, text } from "@spl/ui";
import { SectionTitle } from "./section-title";

/* Numbers are the ranges over the example runs (five photos at once on one
   RTX 4070 SUPER, Claude opus at medium effort). */
const STEPS = [
  {
    title: "Decode",
    body: "rawler reads the ARW: sensor data, the camera's white balance and the lens's own correction profile.",
    number: "0.15–0.35 s",
  },
  {
    title: "Develop on the GPU",
    body: "CUDA kernels demosaic with RCD and apply Adobe's DNG colour and white-balance math. A CUDA renderer then draws 56 Lightroom-style settings, lens corrections included.",
    number: "≈0.35 s",
  },
  {
    title: "Plan looks",
    body: "Automatic settings and four built-in looks: natural, vivid, bright, moody. Claude reads the photo and its histogram and proposes up to three looks of its own.",
    number: "up to 7 looks",
  },
  {
    title: "Search the sliders",
    body: "CMA-ES tunes Basic, Tone Curve, Presence, Color Mixer and Calibration per look, under hard limits on clipping, crushed shadows and gamut, held near expert edits.",
    number: "1,500 renders a look · 3.2–3.7 s",
  },
  {
    title: "Judge blind, refine once",
    body: "Claude ranks the finalists under neutral labels, twice with the order swapped. Its one refinement is kept only if it wins a blind A/B. The as-shot render can win too.",
    number: "4–6 Claude calls in all · 30–40 s",
  },
  {
    title: "Export",
    body: "A full-resolution JPEG, a 16-bit TIFF if you want one, and an XMP sidecar with every setting under its Lightroom name.",
    number: "1.1–3.1 s",
  },
];

export function Pipeline() {
  return (
    <section id="how" tabIndex={-1} aria-labelledby="how-title" className="scroll-mt-4 outline-none md:scroll-mt-24">
      <SectionTitle id="how-title" eyebrow="How it works" title="Six steps from RAW file to edit">
        About 5 seconds a photo on the GPU alone, about 40 with Claude planning and judging.
      </SectionTitle>
      <ol className="mt-8 grid grid-cols-1 gap-x-8 gap-y-8 sm:grid-cols-2 lg:grid-cols-3">
        {STEPS.map((step, i) => (
          <li key={step.title} className="border-t border-[var(--ds-gray-alpha-400)] pt-4">
            <p className={text.meta}>{String(i + 1).padStart(2, "0")}</p>
            <h3 className="mt-2 text-[16px] font-medium leading-6 text-[var(--ds-gray-1000)]">{step.title}</h3>
            <p className={cn(text.small, "mt-1")}>{step.body}</p>
            <p className={cn(text.mono, "mt-3")}>{step.number}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
