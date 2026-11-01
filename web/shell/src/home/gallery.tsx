import { cn, text } from "@spl/ui";
import { Compare } from "./compare";
import { ExampleFacts } from "./example-facts";
import { EXAMPLES, type Example } from "./examples";
import { SectionTitle } from "./section-title";

/* The other examples, each a smaller before/after with its facts, then where
   the photos come from. */
export function Gallery({ examples }: { examples: Example[] }) {
  return (
    <section aria-labelledby="examples-title">
      <SectionTitle id="examples-title" eyebrow="Examples" title="More sample RAWs, edited automatically">
        Each result is the app's first answer, with no slider touched by hand. Drag a divider to compare.
      </SectionTitle>
      <ul className="mt-8 grid grid-cols-1 gap-x-8 gap-y-12 lg:grid-cols-3">
        {examples.map((example) => (
          <li key={example.key}>
            <figure>
              <Compare example={example} />
              <figcaption>
                <ExampleFacts example={example} className="mt-4" />
              </figcaption>
            </figure>
          </li>
        ))}
      </ul>
      <div className={cn(text.caption, "mt-10 max-w-3xl space-y-2")}>
        <p>
          A fifth sample, a sunny lane shot on an ILCE-6700, came back unchanged: Claude ranked the as-shot render
          above every edit, and the app keeps it in the running for exactly that case.
        </p>
        <p>
          Before is the as-shot render (camera white balance, Lightroom's defaults); after is the app's result. Times
          are for five photos run at once on one RTX 4070 SUPER. The sample RAWs come from raw.pixls.us (
          {EXAMPLES.map((e, i) => (
            <span key={e.key}>
              <a className={text.link} href={e.source.url} rel="noreferrer" target="_blank">
                {e.camera}
              </a>
              {i < EXAMPLES.length - 1 ? ", " : ""}
            </span>
          ))}
          ), released into the public domain under{" "}
          <a className={text.link} href="https://creativecommons.org/publicdomain/zero/1.0/" rel="noreferrer" target="_blank">
            CC0
          </a>
          .
        </p>
      </div>
    </section>
  );
}
