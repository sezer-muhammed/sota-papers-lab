import type { ArtifactRef } from "@spl/sdk";
import { SectionHeader, Surface, cn, text } from "@spl/ui";
import { ArtifactDownload } from "./panels";

/* The downloads of a finished scene: each file with its size and one line
   on what it is for. Reads the reconstruct and train outputs alike. */

type Item = { key: string; label: string; name: string; what: string };

export const DOWNLOADS: Item[] = [
  { key: "spz", label: "SPZ", name: "scene.spz", what: "Compressed splats, about 10x smaller than PLY; for web viewers (Spark, Niantic, SuperSplat)." },
  { key: "ply", label: "PLY", name: "scene.ply", what: "Full-precision 3D Gaussian splat; opens in most splat tools and editors." },
  { key: "glb", label: "GLB mesh", name: "scene.glb", what: "Textured triangle mesh for Blender, game engines and AR." },
  { key: "mesh", label: "GLB mesh", name: "scene.glb", what: "Textured triangle mesh for Blender, game engines and AR." },
  { key: "colmap", label: "COLMAP", name: "colmap.zip", what: "Cameras and points in COLMAP format, to train other splat or NeRF tools on these poses." },
  { key: "transforms", label: "transforms.json", name: "transforms.json", what: "Cameras in Nerfstudio / instant-ngp format." },
  { key: "cameras", label: "Cameras", name: "cameras.json", what: "Every camera's pose and intrinsics as plain JSON." },
];

const isRef = (v: unknown): v is ArtifactRef =>
  Boolean(v && typeof v === "object" && typeof (v as ArtifactRef).id === "string" && typeof (v as ArtifactRef).size_bytes === "number");

/** The output's files in display order, one GLB at most. */
export function downloadsOf(output: unknown): (Item & { file: ArtifactRef })[] {
  if (!output || typeof output !== "object") return [];
  const o = output as Record<string, unknown>;
  const seen = new Set<string>();
  return DOWNLOADS.flatMap((d) => {
    const file = o[d.key];
    if (!isRef(file) || seen.has(d.name)) return [];
    seen.add(d.name);
    return [{ ...d, file }];
  });
}

export function DownloadsPanel({ output, className }: { output: unknown; className?: string }) {
  const files = downloadsOf(output);
  if (!files.length) return null;
  return (
    <Surface aria-labelledby="splat-downloads-title" className={className}>
      <SectionHeader titleId="splat-downloads-title" eyebrow="Result" title="Downloads" />
      <ul data-testid="splat-downloads" className="divide-y divide-[var(--ds-gray-alpha-400)]">
        {files.map((f) => (
          <li key={f.name} className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-4 py-3">
            <ArtifactDownload label={f.label} name={f.name} file={f.file} />
            <p className={cn(text.caption, "min-w-0 flex-1 basis-48")}>{f.what}</p>
          </li>
        ))}
      </ul>
    </Surface>
  );
}
