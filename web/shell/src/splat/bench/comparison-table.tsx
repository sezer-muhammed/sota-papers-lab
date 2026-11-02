import { Badge, Surface, SectionHeader, text } from "@spl/ui";
import type { BenchData, PaperClaim } from "./types";
import { buildComparisonRows } from "./types";

function fmt(v: number | null | undefined, decimals = 2): string {
  if (v == null) return "—";
  return v.toFixed(decimals);
}

function fmtTime(s: number | null | undefined): string {
  if (s == null) return "—";
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${(s / 60).toFixed(1)} min`;
}

function fmtGaussians(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`;
  return String(n);
}

function paperClaimFor(
  claims: PaperClaim[],
  method: string,
  dataset: string,
  metric: string,
): number | null {
  const c = claims.find(
    (p) =>
      p.method === method &&
      p.dataset === dataset &&
      p.metric === metric,
  );
  return c?.value ?? null;
}

/** The Section 1 comparison table: our numbers next to baselines and paper claims. */
export function ComparisonTable({ data }: { data: BenchData }) {
  const rows = buildComparisonRows(data);
  const baselineNames = [...new Set(data.baselines.map((b) => (b.tag ? `${b.tool} (${b.tag})` : b.tool)))].sort();

  if (rows.length === 0) {
    return <p className={text.caption}>No benchmark results yet.</p>;
  }

  return (
    <Surface tone="flat" className="overflow-x-auto">
      <SectionHeader title="NVS Quality Comparison" />
      <table className="w-full text-left text-[13px]">
        <thead>
          <tr className="border-b border-[var(--ds-gray-alpha-400)] text-[var(--ds-gray-900)]">
            <th className="px-3 py-2 font-medium">Scene</th>
            <th className="px-3 py-2 font-medium">Dataset</th>
            <th className="px-3 py-2 font-medium text-right">PSNR (dB)</th>
            <th className="px-3 py-2 font-medium text-right">SSIM</th>
            <th className="px-3 py-2 font-medium text-right">LPIPS</th>
            <th className="px-3 py-2 font-medium text-right">Time</th>
            <th className="px-3 py-2 font-medium text-right">VRAM</th>
            <th className="px-3 py-2 font-medium text-right">Gaussians</th>
            {baselineNames.map((name) => (
              <th key={name} className="px-3 py-2 font-medium text-right">
                {name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.scene} className="border-b border-[var(--ds-gray-alpha-200)]">
              <td className="px-3 py-2 font-medium text-[var(--ds-gray-1000)]">{row.scene}</td>
              <td className="px-3 py-2">{row.dataset}</td>
              <td className="px-3 py-2 text-right tabular-nums">
                {row.ours ? (
                  <MetricCell value={row.ours.psnr} decimals={2} suffix=" dB" />
                ) : (
                  "—"
                )}
              </td>
              <td className="px-3 py-2 text-right tabular-nums">{fmt(row.ours?.ssim, 3)}</td>
              <td className="px-3 py-2 text-right tabular-nums">{fmt(row.ours?.lpips, 3)}</td>
              <td className="px-3 py-2 text-right tabular-nums">{fmtTime(row.ours?.time_s)}</td>
              <td className="px-3 py-2 text-right tabular-nums">
                {row.ours ? `${row.ours.vram_mib.toFixed(0)} MiB` : "—"}
              </td>
              <td className="px-3 py-2 text-right tabular-nums">{fmtGaussians(row.ours?.gaussians)}</td>
              {baselineNames.map((name) => {
                const b = row.baselines[name];
                return (
                  <td key={name} className="px-3 py-2 text-right tabular-nums text-[var(--ds-gray-900)]">
                    {b ? `${fmt(b.psnr)} / ${fmtTime(b.time_s)}` : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </Surface>
  );
}

function MetricCell({
  value,
  decimals = 2,
  suffix = "",
}: {
  value: number;
  decimals?: number;
  suffix?: string;
}) {
  return (
    <span className="text-[var(--ds-gray-1000)]">
      {value.toFixed(decimals)}
      {suffix}
    </span>
  );
}

/** Paper claims table: published numbers from each method. */
export function PaperClaimsTable({ claims }: { claims: PaperClaim[] }) {
  if (claims.length === 0) return null;

  const methods = [...new Set(claims.map((c) => c.method))].sort();
  const datasets = [...new Set(claims.map((c) => c.dataset))].sort();
  const metrics = [...new Set(claims.map((c) => c.metric))].sort();

  return (
    <Surface tone="flat" className="overflow-x-auto">
      <SectionHeader title="Paper Claims" />
      <table className="w-full text-left text-[13px]">
        <thead>
          <tr className="border-b border-[var(--ds-gray-alpha-400)] text-[var(--ds-gray-900)]">
            <th className="px-3 py-2 font-medium">Method</th>
            <th className="px-3 py-2 font-medium">Dataset</th>
            <th className="px-3 py-2 font-medium">GPU</th>
            {metrics.map((m) => (
              <th key={m} className="px-3 py-2 font-medium text-right">{m}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {methods.flatMap((method) =>
            datasets.map((dataset) => {
              const row = claims.filter((c) => c.method === method && c.dataset === dataset);
              if (row.length === 0) return null;
              return (
                <tr key={`${method}-${dataset}`} className="border-b border-[var(--ds-gray-alpha-200)]">
                  <td className="px-3 py-2 font-medium text-[var(--ds-gray-1000)]">{method}</td>
                  <td className="px-3 py-2">{dataset}</td>
                  <td className="px-3 py-2">
                    {row[0]?.gpu ? <Badge tone="gray">{row[0].gpu}</Badge> : "—"}
                  </td>
                  {metrics.map((metric) => {
                    const val = paperClaimFor(claims, method, dataset, metric);
                    const claim = claims.find(
                      (c) => c.method === method && c.dataset === dataset && c.metric === metric,
                    );
                    return (
                      <td key={metric} className="px-3 py-2 text-right tabular-nums">
                        {val != null ? `${val} ${claim?.unit ?? ""}`.trim() : "—"}
                      </td>
                    );
                  })}
                </tr>
              );
            }),
          )}
        </tbody>
      </table>
    </Surface>
  );
}
