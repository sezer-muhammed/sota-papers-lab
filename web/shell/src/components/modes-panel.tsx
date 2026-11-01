import { Play, Target, Workflow, type LucideIcon } from "lucide-react";
import type { Health, OpDescriptor } from "@spl/sdk";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
  DataPanel,
  IconTile,
  Skeleton,
  StatusSignal,
  cn,
  text,
} from "@spl/ui";
import type { Loaded } from "../api";
import type { HomeHostState } from "../lib/env";
import { plural } from "../lib/format";

type Mode = {
  number: string;
  icon: LucideIcon;
  title: string;
  description: string;
  /** Fixed example; 01's comes from the host's first op. */
  command?: string;
  status: "live" | "roadmap";
  /** Status line while on the roadmap. */
  roadmap?: string;
};

// flip 02/03 when docs/ARCHITECTURE.md roadmap items 3 (pipelines) and 4 (benchmarks) land
const MODES: Mode[] = [
  {
    number: "01",
    icon: Play,
    title: "Run alone",
    description:
      "Call any op with JSON from the Ops page, the CLI or HTTP. Each run stores its input, output, metrics, timings and the GPU it ran on.",
    status: "live",
  },
  {
    number: "02",
    icon: Workflow,
    title: "Chain apps",
    description:
      "A pipeline passes one op's output artifacts into the next op by ID. Apps never import each other, so any app can feed any other.",
    command: "load → imgcodec.encode → imgcodec.decode → metrics.compare",
    status: "roadmap",
    roadmap: "On the roadmap · pipeline runner",
  },
  {
    number: "03",
    icon: Target,
    title: "Benchmark against the paper",
    description:
      "A suite runs a pipeline over a dataset and a parameter sweep, then puts the measured numbers next to the claims in the app's paper.toml.",
    command: "pipeline × dataset × sweep → vs. paper.toml",
    status: "roadmap",
    roadmap: "On the roadmap · benchmark suites",
  },
];

/* "How every app is tested": the three modes from docs/ARCHITECTURE.md. Only
   01 is live today; its status reads the host. */
export function ModesPanel({
  health,
  ops,
  state,
}: {
  health: Loaded<Health>;
  ops: Loaded<OpDescriptor[]>;
  state: HomeHostState;
}) {
  const firstOp = ops.data?.[0]?.id;

  return (
    <DataPanel
      id="modes"
      eyebrow="How every app is tested"
      title="Alone, chained, benchmarked."
      summary="Every op declares typed input and output, so the same op runs by itself, as a pipeline step or inside a benchmark suite, with no app-specific glue."
      padded
      className="reveal scroll-mt-4 outline-none md:scroll-mt-24"
    >
      <div className="stagger grid grid-cols-1 gap-3 lg:grid-cols-3">
        {MODES.map((mode) => {
          const command = mode.command ?? `spl run ${firstOp ?? "<app>.<verb>"}`;
          return (
            <Card key={mode.number} depth="base" className="flex flex-col">
              <CardHeader action={<span className={text.eyebrowSm}>{mode.number}</span>}>
                <IconTile icon={mode.icon} />
                <CardTitle className="mt-4">{mode.title}</CardTitle>
                <CardDescription>{mode.description}</CardDescription>
              </CardHeader>
              {/* Wraps instead of truncating: the whole example stays readable without a hover. */}
              <code
                className={cn(
                  text.meta,
                  "block break-words rounded-[7px] border border-[var(--ds-gray-alpha-300)] bg-[var(--ds-background-200)] px-3 py-2 text-[var(--ds-gray-1000)]",
                )}
              >
                {command}
              </code>
              <div className="mt-auto pt-4">
                {mode.status === "roadmap" ? (
                  <StatusSignal variant="inline" color="var(--ds-gray-700)">
                    {mode.roadmap}
                  </StatusSignal>
                ) : (
                  <LiveStatus state={state} opCount={health.data?.ops ?? 0} />
                )}
              </div>
            </Card>
          );
        })}
      </div>
    </DataPanel>
  );
}

/* Card 01's status: the op count comes from public /v1/health, so it still
   reads right when /v1/ops needs a token. */
function LiveStatus({ opCount, state }: { opCount: number; state: HomeHostState }) {
  if (state === "connecting") return <Skeleton className="h-4 w-40" />;
  if (state === "offline") {
    // Red, like the host strip and the header pill for the same state.
    return (
      <StatusSignal variant="inline" color="var(--ds-red-700)">
        Host unreachable
      </StatusSignal>
    );
  }
  if (opCount === 0) {
    return (
      <StatusSignal variant="inline" color="var(--ds-gray-700)">
        No ops linked yet
      </StatusSignal>
    );
  }
  return state === "online" ? (
    <StatusSignal variant="inline" color="var(--ds-green-700)">
      Live · {plural(opCount, "op")} on this host
    </StatusSignal>
  ) : (
    <StatusSignal variant="inline" color="var(--ds-amber-700)">
      {plural(opCount, "op")} · no GPU, CUDA ops will fail
    </StatusSignal>
  );
}
