import { ArrowRight, Workflow } from "lucide-react";
import { ActionBar, Button, ButtonLink, cn, text } from "@spl/ui";
import { jumpTo } from "../lib/scroll";
import { Compare } from "./compare";
import { ExampleFacts } from "./example-facts";
import type { Example } from "./examples";

const FACTS = [
  { value: "56", label: "Lightroom settings per edit" },
  { value: "≈5 s", label: "a photo on the GPU" },
  { value: "≈40 s", label: "with Claude judging" },
  { value: "XMP", label: "sidecar for Lightroom" },
];

/* The value proposition beside the best example, as a before/after the
   visitor can drag. */
export function Hero({ example }: { example: Example }) {
  return (
    <section
      aria-labelledby="home-title"
      className="grid grid-cols-1 gap-8 pt-2 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)] lg:gap-12"
    >
      <div className="lg:pt-4">
        <p className={text.eyebrow}>raw-retouch · Sony ARW</p>
        <h1 id="home-title" className={cn(text.pageTitle, "mt-3 text-balance text-[var(--ds-gray-1000)]")}>
          Sony RAW in, a finished edit out.
        </h1>
        <p className={cn(text.lead, "mt-4")}>
          The app develops each photo on the GPU, searches Lightroom's sliders for a natural edit, and lets Claude
          pick the best version in a blind test. You get a full-size JPEG and an XMP sidecar with every setting.
        </p>
        <ActionBar className="mt-6">
          <ButtonLink href="#/retouch" icon={ArrowRight} variant="primary">
            Retouch your photos
          </ButtonLink>
          <Button variant="ghost" icon={Workflow} onClick={() => jumpTo("how")}>
            How it works
          </Button>
        </ActionBar>
        <dl className="mt-8 grid grid-cols-2 gap-x-6 gap-y-4 border-t border-[var(--ds-gray-alpha-400)] pt-5">
          {FACTS.map((fact) => (
            // Label first for screen readers, value on top for the eye.
            <div key={fact.label} className="flex flex-col">
              <dt className={cn(text.caption, "order-2")}>{fact.label}</dt>
              <dd className="order-1 font-mono text-[20px] leading-7 text-[var(--ds-gray-1000)]">{fact.value}</dd>
            </div>
          ))}
        </dl>
      </div>

      <figure className="min-w-0">
        <Compare example={example} priority />
        <figcaption>
          <ExampleFacts example={example} className="mt-4" />
        </figcaption>
      </figure>
    </section>
  );
}
