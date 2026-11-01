import { useMemo } from "react";
import { ArrowRight, Play, Workflow } from "lucide-react";
import type { Health, RunDetail } from "@spl/sdk";
import { ActionBar, Button, ButtonLink, cn, text } from "@spl/ui";
import {
  client,
  hasToken,
  isUnauthorized,
  useLoad,
  useNow,
  useRefreshOnReturn,
  type Loaded,
} from "../api";
import { AppsPanel } from "../components/apps-panel";
import { CopyCommand } from "../components/copy-command";
import { CtaBand } from "../components/cta-band";
import { HostStrip } from "../components/host-strip";
import { LabSpecimen } from "../components/lab-specimen";
import { ModesPanel } from "../components/modes-panel";
import { RecentRuns } from "../components/recent-runs";
import { RUN_WINDOW, groupApps } from "../lib/catalog";
import { homeHostState } from "../lib/env";
import { jumpTo } from "../lib/scroll";

/* The landing page: static hero, then the host, its apps and recent runs, how
   every app is tested, and a try-it band. 4 requests on load (health is
   shared with the header and footer), no polling, never a POST. */
export function HomePage({ health }: { health: Loaded<Health> }) {
  const now = useNow(60_000);
  const ops = useLoad(() => client.ops());
  const runs = useLoad(() => client.runs({ limit: RUN_WINDOW }));
  const newest = runs.data?.[0];
  const latest = useLoad<RunDetail | undefined>(
    () => (newest ? client.runDetail(newest.id) : Promise.resolve(undefined)),
    [newest?.id, newest?.status],
  );
  useRefreshOnReturn(() => {
    void ops.reload();
    void runs.reload();
    // The detail reloads by itself when the newest run changes; after a failure, ask again.
    if (latest.error) void latest.reload();
  });

  const state = homeHostState(health);
  const hostDown = state === "offline";
  const locked = isUnauthorized(ops.error) || isUnauthorized(runs.error);
  const apps = useMemo(() => groupApps(ops.data ?? [], runs.data ?? []), [ops.data, runs.data]);
  const windowFull = runs.data?.length === RUN_WINDOW;
  const refreshAll = () => {
    void health.reload();
    void ops.reload();
    void runs.reload();
    if (latest.error) void latest.reload();
  };

  // Commands for this origin. The token itself is never printed.
  const origin = window.location.origin;
  const auth = hasToken() || locked ? ` -H "authorization: Bearer $SPL_AUTH_TOKEN"` : "";
  const op = ops.data?.[0]?.id;
  const listCommand = `curl -s ${origin}/v1/ops${auth}`;
  const runCommand = `curl -s ${origin}/v1/runs -H 'content-type: application/json' -d '{"op":"${op ?? "<app>.<verb>"}","input":{}}'${auth}`;

  return (
    <div className="space-y-8">
      <section
        aria-labelledby="home-title"
        className="grid grid-cols-1 gap-6 pt-4 xl:grid-cols-[minmax(0,1fr)_520px] xl:items-center xl:gap-8 xl:py-4"
      >
        <div className="max-w-3xl">
          <p className={text.eyebrow}>Paper reproduction lab</p>
          <h1 id="home-title" className={cn(text.display, "mt-3 text-balance text-[var(--ds-gray-1000)]")}>
            State-of-the-art papers, rebuilt in Rust and CUDA.
          </h1>
          <p className={cn(text.lead, "mt-5 max-w-2xl")}>
            Each paper becomes an app with typed ops. Run it on its own, chain it with other apps, and
            benchmark it against the numbers the paper claims.
          </p>
          <ActionBar className="mt-6">
            <ButtonLink href="#/ops" icon={Play} variant="primary">
              Run an op
            </ButtonLink>
            <ButtonLink href="#/runs" icon={ArrowRight}>
              Browse runs
            </ButtonLink>
            <Button variant="ghost" icon={Workflow} onClick={() => jumpTo("modes")}>
              How it works
            </Button>
          </ActionBar>
        </div>
        <LabSpecimen
          health={health}
          hostDown={hostDown}
          latest={latest}
          locked={locked}
          now={now}
          onRetry={refreshAll}
          runs={runs}
        />
      </section>

      <HostStrip health={health} refreshAll={refreshAll} state={state} />

      <AppsPanel
        apps={apps}
        hostDown={hostDown}
        locked={locked}
        now={now}
        ops={ops}
        runs={runs}
        windowFull={windowFull}
      />

      {locked ? null : <RecentRuns hostDown={hostDown} now={now} runs={runs} />}

      <ModesPanel health={health} ops={ops} state={state} />

      <CtaBand
        id="try"
        className="reveal"
        eyebrow="Try it"
        title={state === "online" ? "Run an op on this GPU." : "Run an op on this host."}
        description="Pick an op, send JSON input, and get its output back with metrics, timings and the environment, recorded as a run you can compare later."
        actions={
          <>
            <ButtonLink href="#/ops" icon={Play} variant="primary">
              Run an op
            </ButtonLink>
            <ButtonLink href="#/runs" icon={ArrowRight}>
              Browse runs
            </ButtonLink>
          </>
        }
        footer={
          <>
            <div>
              <p className={text.eyebrowSm}>List ops</p>
              <CopyCommand className="mt-1.5" command={listCommand} label="list ops command" />
            </div>
            <div>
              <p className={text.eyebrowSm}>{op ? `Run ${op}` : "Run an op"}</p>
              <CopyCommand className="mt-1.5" command={runCommand} label="run command" />
            </div>
          </>
        }
      />
    </div>
  );
}
