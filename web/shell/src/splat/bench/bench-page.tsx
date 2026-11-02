import { useCallback, useState } from "react";
import { BarChart3 } from "lucide-react";
import { Badge, Button, Surface, SectionHeader, text } from "@spl/ui";
import { client, errorMessage, useLoad } from "../../api";
import type { RunDetail } from "@spl/sdk";
import type { BenchData, SceneResult } from "./types";
import { ComparisonTable, PaperClaimsTable } from "./comparison-table";

type BenchState =
  | { phase: "idle" }
  | { phase: "running"; progress: string }
  | { phase: "done"; data: BenchData }
  | { phase: "error"; message: string };

/** The Bench tab in #/3d: shows the Section 1 comparison table. */
export function BenchPage() {
  const [state, setState] = useState<BenchState>({ phase: "idle" });
  const storedRuns = useLoad(
    () => client.runs({ op: "image_to_3d.bench", limit: 10 }),
    [],
  );

  const latestRun = storedRuns.data?.find((r) => r.status === "succeeded");

  const loadFromRun = useCallback(async (runId: string) => {
    try {
      setState({ phase: "running", progress: "Loading results…" });
      const detail: RunDetail = await client.runDetail(runId);
      const output = detail.run.output as BenchData | null;
      if (!output) {
        setState({ phase: "error", message: "Run has no output" });
        return;
      }
      setState({ phase: "done", data: output });
    } catch (err) {
      setState({ phase: "error", message: errorMessage(err) });
    }
  }, []);

  const startBench = useCallback(async () => {
    try {
      setState({ phase: "running", progress: "Starting benchmark…" });
      const run = await client.startRun("image_to_3d.bench", {
        suite: "benchmarks/image-to-3d-nvs-colmap.toml",
        baselines_jsonl: "~/spl-work/bench/results.jsonl",
        paper_toml: "apps/image-to-3d/paper.toml",
      });
      const detail = await client.watchRun(run.id, {
        onDetail: (d) => {
          if (d.progress) {
            setState({ phase: "running", progress: d.progress.message });
          }
        },
      });
      const output = detail.run.output as BenchData | null;
      if (!output || detail.run.status !== "succeeded") {
        setState({
          phase: "error",
          message: detail.run.error_message ?? "Benchmark failed",
        });
        return;
      }
      setState({ phase: "done", data: output });
    } catch (err) {
      setState({ phase: "error", message: errorMessage(err) });
    }
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className={text.eyebrow}>benchmark</p>
          <h2 className={`${text.bandTitle} mt-1 text-[var(--ds-gray-1000)]`}>
            NVS Quality & Speed
          </h2>
          <p className={`${text.caption} mt-1`}>
            Our trainer vs baselines on COLMAP-posed scenes (Section 1 table).
          </p>
        </div>
        <div className="flex gap-2">
          {latestRun && state.phase === "idle" && (
            <Button
              variant="secondary"
              onClick={() => void loadFromRun(latestRun.id)}
            >
              Load latest
            </Button>
          )}
          <Button
            icon={BarChart3}
            disabled={state.phase === "running"}
            onClick={() => void startBench()}
          >
            {state.phase === "running" ? "Running…" : "Run benchmark"}
          </Button>
        </div>
      </div>

      {state.phase === "running" && (
        <Surface tone="sunken" className="p-4">
          <p className={text.caption}>{state.progress}</p>
        </Surface>
      )}

      {state.phase === "error" && (
        <Surface tone="sunken" className="p-4">
          <p className={`${text.caption} text-[var(--ds-red-900)]`}>{state.message}</p>
        </Surface>
      )}

      {state.phase === "done" && (
        <>
          <ComparisonTable data={state.data} />
          <PaperClaimsTable claims={state.data.paper_claims} />
          <SceneResultsDetail results={state.data.results} />
        </>
      )}

      {state.phase === "idle" && !latestRun && (
        <Surface tone="sunken" className="p-6 text-center">
          <p className={text.body}>
            No benchmark results yet. Run the benchmark on the GPU host, or load
            results from a previous run.
          </p>
        </Surface>
      )}
    </div>
  );
}

function SceneResultsDetail({ results }: { results: SceneResult[] }) {
  if (results.length === 0) return null;

  return (
    <Surface tone="flat" className="overflow-x-auto">
      <SectionHeader title="Per-Scene Detail" />
      <table className="w-full text-left text-[13px]">
        <thead>
          <tr className="border-b border-[var(--ds-gray-alpha-400)] text-[var(--ds-gray-900)]">
            <th className="px-3 py-2 font-medium">Scene</th>
            <th className="px-3 py-2 font-medium">Preset</th>
            <th className="px-3 py-2 font-medium text-right">PSNR</th>
            <th className="px-3 py-2 font-medium text-right">SSIM</th>
            <th className="px-3 py-2 font-medium text-right">LPIPS</th>
            <th className="px-3 py-2 font-medium text-right">Time</th>
            <th className="px-3 py-2 font-medium text-right">VRAM</th>
            <th className="px-3 py-2 font-medium text-right">Gaussians</th>
            <th className="px-3 py-2 font-medium">Status</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.scene} className="border-b border-[var(--ds-gray-alpha-200)]">
              <td className="px-3 py-2 font-medium text-[var(--ds-gray-1000)]">{r.scene}</td>
              <td className="px-3 py-2">{r.preset}</td>
              <td className="px-3 py-2 text-right tabular-nums">{r.psnr.toFixed(2)} dB</td>
              <td className="px-3 py-2 text-right tabular-nums">{r.ssim.toFixed(3)}</td>
              <td className="px-3 py-2 text-right tabular-nums">
                {r.lpips_vgg != null ? r.lpips_vgg.toFixed(3) : "—"}
              </td>
              <td className="px-3 py-2 text-right tabular-nums">
                {r.train_time_s < 60
                  ? `${r.train_time_s.toFixed(1)} s`
                  : `${(r.train_time_s / 60).toFixed(1)} min`}
              </td>
              <td className="px-3 py-2 text-right tabular-nums">{r.peak_vram_mib.toFixed(0)} MiB</td>
              <td className="px-3 py-2 text-right tabular-nums">
                {r.num_gaussians >= 1e6
                  ? `${(r.num_gaussians / 1e6).toFixed(2)}M`
                  : `${(r.num_gaussians / 1e3).toFixed(0)}k`}
              </td>
              <td className="px-3 py-2">
                <Badge tone={r.status === "ok" ? "green" : "red"}>{r.status}</Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Surface>
  );
}
