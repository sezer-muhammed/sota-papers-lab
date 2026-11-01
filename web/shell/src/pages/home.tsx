import type { Health } from "@spl/sdk";
import { cn, text } from "@spl/ui";
import type { Loaded } from "../api";
import { HostStrip } from "../components/host-strip";
import { EXAMPLES } from "../home/examples";
import { Gallery } from "../home/gallery";
import { Hero } from "../home/hero";
import { Pipeline } from "../home/pipeline";
import { Research } from "../home/research";
import { homeHostState } from "../lib/env";

const [HERO, ...MORE] = EXAMPLES;

/* The landing page shows the app it hosts: a before/after of a real
   raw-retouch run, more examples, the pipeline, the papers behind it, and
   last the GPU this page is connected to (#/host). Static apart from the
   shared health request: no other API call, never a POST. */
export function HomePage({ health }: { health: Loaded<Health> }) {
  const state = homeHostState(health);

  return (
    <div className="space-y-20 sm:space-y-24">
      {HERO ? <Hero example={HERO} /> : null}
      <Gallery examples={MORE} />
      <Pipeline />
      <Research />
      <div className="space-y-3">
        <p className={text.eyebrow}>This host</p>
        <HostStrip health={health} refreshAll={() => void health.reload()} state={state} />
        <p className={cn(text.caption, "max-w-3xl")}>
          The lab runs every app as typed ops on this GPU. See them all under{" "}
          <a className={text.link} href="#/ops">
            Ops
          </a>{" "}
          and every past run under{" "}
          <a className={text.link} href="#/runs">
            Runs
          </a>
          .
        </p>
      </div>
    </div>
  );
}
