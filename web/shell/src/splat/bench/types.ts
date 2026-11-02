/** Benchmark data types — mirrors the Rust BenchOutput. */

export type SceneResult = {
  scene: string;
  dataset: string;
  preset: string;
  psnr: number;
  ssim: number;
  lpips_vgg: number | null;
  train_time_s: number;
  peak_vram_mib: number;
  num_gaussians: number;
  status: string;
};

export type BaselineRow = {
  tool: string;
  scene: string;
  dataset: string;
  tag: string | null;
  status: string;
  psnr: number | null;
  ssim: number | null;
  lpips: number | null;
  train_s: number | null;
  wall_s: number | null;
  peak_vram_mib: number | null;
  num_gaussians: number | null;
  gpu: string | null;
};

export type PaperClaim = {
  method: string;
  dataset: string;
  metric: string;
  value: number;
  unit: string;
  gpu: string | null;
};

export type BenchData = {
  results: SceneResult[];
  baselines: BaselineRow[];
  paper_claims: PaperClaim[];
};

/** A unified row for the comparison table. */
export type ComparisonRow = {
  scene: string;
  dataset: string;
  ours: {
    psnr: number;
    ssim: number;
    lpips: number | null;
    time_s: number;
    vram_mib: number;
    gaussians: number;
  } | null;
  baselines: Record<
    string,
    {
      psnr: number | null;
      ssim: number | null;
      lpips: number | null;
      time_s: number | null;
      vram_mib: number | null;
    }
  >;
};

/** Build unified rows from bench output for the comparison table. */
export function buildComparisonRows(data: BenchData): ComparisonRow[] {
  const scenes = new Set<string>();
  for (const r of data.results) scenes.add(r.scene);
  for (const b of data.baselines) scenes.add(b.scene);

  return [...scenes].sort().map((scene) => {
    const our = data.results.find((r) => r.scene === scene && r.status === "ok");
    const baselineEntries = data.baselines.filter((b) => b.scene === scene);
    const baselines: ComparisonRow["baselines"] = {};
    for (const b of baselineEntries) {
      const key = b.tag ? `${b.tool} (${b.tag})` : b.tool;
      baselines[key] = {
        psnr: b.psnr,
        ssim: b.ssim,
        lpips: b.lpips,
        time_s: b.train_s ?? b.wall_s,
        vram_mib: b.peak_vram_mib,
      };
    }
    return {
      scene,
      dataset: our?.dataset ?? baselineEntries[0]?.dataset ?? "",
      ours: our
        ? {
            psnr: our.psnr,
            ssim: our.ssim,
            lpips: our.lpips_vgg,
            time_s: our.train_time_s,
            vram_mib: our.peak_vram_mib,
            gaussians: our.num_gaussians,
          }
        : null,
      baselines,
    };
  });
}
