import { useRef, useState, type DragEvent } from "react";
import { Check, FileBox, Images, Square, Upload } from "lucide-react";
import type { ArtifactRef } from "@spl/sdk";
import { Button, IconTile, ProgressBar, RangeInput, SectionHeader, SegmentedControl, Surface, Switch, cn, text } from "@spl/ui";
import { ArtifactDownload, mib } from "./panels";
import {
  STAGES,
  STAGE_LABEL,
  pickFiles,
  stageStates,
  type EngineSchedule,
  type LicenceProfile,
  type LiveNumbers,
  type Mode,
  type Picked,
  type Preset,
  type ReconstructOptions,
  type ReconstructOutput,
} from "./reconstruct";
import type { TrainOptions } from "./use-train";
import type { Sending } from "./use-reconstruct";

type Input = "media" | "colmap";

const PRESETS: { value: Preset; label: string }[] = [
  { value: "fast", label: "Fast" },
  { value: "quality", label: "Quality" },
  { value: "big", label: "Big" },
];

const PRESET_NOTE: Record<Preset, string> = {
  fast: "Fewer iterations and Gaussians: a preview in about a minute.",
  quality: "The full schedule: best quality, a few minutes on this GPU.",
  big: "Maximum Gaussians and iterations: highest fidelity, longest training.",
};

const PROFILES: { value: LicenceProfile; label: string }[] = [
  { value: "research", label: "Research" },
  { value: "commercial", label: "Commercial" },
];

const PROFILE_NOTE: Record<LicenceProfile, string> = {
  research: "VGGT-Ω pose engine (FAIR-NC licence, research only).",
  commercial: "DA3-BASE pose engine (Apache-2.0, commercial use OK).",
};

const SCHEDULES: { value: EngineSchedule; label: string }[] = [
  { value: "dense", label: "Dense" },
  { value: "ours+sage", label: "Fast" },
];

const SCHEDULE_NOTE: Record<EngineSchedule, string> = {
  dense: "Full dense attention: best accuracy, more VRAM.",
  "ours+sage": "Sparse + SAGE attention: faster, lower VRAM.",
};

const MODES: { value: Mode; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "scene", label: "Scene" },
  { value: "object", label: "Object" },
];

const MODE_NOTE: Record<Mode, string> = {
  auto: "Decides from the photos: walked-through scene or object on a table.",
  scene: "A room, a garden, a street: cameras move through the space.",
  object: "One thing seen from around it; the background is cropped away.",
};

const ACCEPT = ".jpg,.jpeg,.png,.webp,.mp4,.mov,.m4v,image/jpeg,image/png,image/webp,video/mp4,video/quicktime";

/* Where a scene comes in: unposed photos or one phone video (reconstructed
   with VGGT-Ω poses), or a COLMAP-posed zip (trained on its poses). Drop
   files anywhere on the zone or pick them; they go up in resumable chunks. */
export function NewScenePanel({
  busy,
  maxUploadBytes,
  onReconstruct,
  onTrainZip,
  onView,
  options,
  setOptions,
  sending,
  onStopSending,
  trainOptions,
  setTrainOptions,
  zipUpload,
}: {
  busy: boolean;
  maxUploadBytes?: number;
  onReconstruct: (picked: Exclude<Picked, { kind: "none" }>) => void;
  onTrainZip: (file: File) => void;
  onView: (file: File) => void;
  options: ReconstructOptions;
  setOptions: (options: ReconstructOptions) => void;
  sending?: Sending;
  onStopSending: () => void;
  trainOptions: TrainOptions;
  setTrainOptions: (options: TrainOptions) => void;
  zipUpload?: { name: string; fraction: number };
}) {
  const [input, setInput] = useState<Input>("media");
  const [note, setNote] = useState<string>();
  const [over, setOver] = useState(false);
  const media = useRef<HTMLInputElement>(null);
  const zip = useRef<HTMLInputElement>(null);
  const splat = useRef<HTMLInputElement>(null);

  const isObject = options.mode === "object";

  const take = (files: File[]) => {
    if (busy) return;
    const picked = pickFiles(files, isObject);
    if (picked.kind === "none") return setNote(picked.reason);
    const big = (picked.kind === "photos" ? picked.files : [picked.file]).find(
      (f) => maxUploadBytes && f.size > maxUploadBytes,
    );
    if (big && maxUploadBytes) {
      return setNote(`${big.name} is ${mib(big.size)}; this host takes files up to ${mib(maxUploadBytes)}.`);
    }
    setNote(picked.skipped.length ? `Skipped ${picked.skipped.length} other file(s): ${picked.skipped.slice(0, 3).join(", ")}.` : undefined);
    onReconstruct(picked);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setOver(false);
    take(Array.from(e.dataTransfer.files));
  };

  return (
    <Surface aria-labelledby="splat-source-title">
      <SectionHeader titleId="splat-source-title" eyebrow="Input" title="New scene" />
      <div className="space-y-5 p-4">
        <SegmentedControl<Input>
          label="Input"
          value={input}
          onChange={setInput}
          options={[
            { value: "media", label: "Photos or video" },
            { value: "colmap", label: "COLMAP zip" },
          ]}
        />
        {input === "media" ? (
          <div
            data-testid="splat-drop"
            onDragOver={(e) => {
              e.preventDefault();
              e.dataTransfer.dropEffect = busy ? "none" : "copy";
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={onDrop}
            className={cn(
              "flex flex-col items-start gap-3 rounded-[8px] border-2 border-dashed p-4 transition-colors",
              over ? "border-[var(--ds-blue-700)] bg-[var(--ds-background-200)]" : "border-[var(--ds-gray-alpha-500)]",
            )}
          >
            <IconTile icon={Images} />
            <p className={text.caption}>
              {isObject
                ? "Drop 1 to 4 photos of an object (JPEG, PNG, WebP). Background will be removed; a PBR mesh (.glb) is generated."
                : "Drop 20 to 200 photos (JPEG, PNG, WebP) walking around the scene, or one phone video (MP4, MOV). No poses needed."}
            </p>
            <Button variant="primary" icon={Upload} disabled={busy} onClick={() => media.current?.click()}>
              Choose photos or video
            </Button>
            <input
              ref={media}
              type="file"
              multiple
              accept={ACCEPT}
              tabIndex={-1}
              aria-hidden="true"
              className="hidden"
              data-testid="media-input"
              onChange={(event) => {
                const files = Array.from(event.target.files ?? []);
                event.target.value = "";
                if (files.length) take(files);
              }}
            />
          </div>
        ) : (
          <div className="flex flex-col items-start gap-3 rounded-[8px] border-2 border-dashed border-[var(--ds-gray-alpha-500)] p-4">
            <IconTile icon={FileBox} />
            <p className={text.caption}>
              A zip with <span className="font-mono">sparse/0/</span> (cameras, images, points3D, .bin or .txt) and
              an <span className="font-mono">images/</span> folder.
            </p>
            <Button variant="primary" icon={Upload} disabled={busy} onClick={() => zip.current?.click()}>
              Choose zip and train
            </Button>
            <input
              ref={zip}
              type="file"
              accept=".zip,application/zip"
              tabIndex={-1}
              aria-hidden="true"
              className="hidden"
              data-testid="zip-input"
              onChange={(event) => {
                const file = event.target.files?.[0];
                event.target.value = "";
                if (!file) return;
                if (maxUploadBytes && file.size > maxUploadBytes) {
                  return setNote(`${file.name} is ${mib(file.size)}; this host takes up to ${mib(maxUploadBytes)}.`);
                }
                setNote(undefined);
                onTrainZip(file);
              }}
            />
          </div>
        )}

        {sending ? (
          <div className="space-y-1.5">
            <ProgressBar label={`Uploading ${sending.what}`} value={sending.total ? sending.sent / sending.total : 0} />
            <div className="flex items-center justify-between gap-3">
              <p className={text.meta}>
                Uploading {sending.what} · {mib(sending.sent)} of {mib(sending.total)}
              </p>
              <Button size="sm" icon={Square} onClick={onStopSending}>
                Stop
              </Button>
            </div>
            <p className={text.meta}>Interrupted? Drop the same files again: only the missing parts go up.</p>
          </div>
        ) : null}
        {zipUpload ? (
          <div className="space-y-1.5">
            <ProgressBar label={`Uploading ${zipUpload.name}`} value={zipUpload.fraction} />
            <p className={text.meta}>
              Uploading {zipUpload.name} · {Math.round(zipUpload.fraction * 100)}%
            </p>
          </div>
        ) : null}
        {note ? (
          <p role="alert" className={text.caption}>
            {note}
          </p>
        ) : null}

        <div className="space-y-4 border-t border-[var(--ds-gray-alpha-400)] pt-4">
          <div className="space-y-2">
            <p className="text-[14px] font-medium text-[var(--ds-gray-1000)]">Preset</p>
            <SegmentedControl<Preset>
              label="Preset"
              value={input === "media" ? options.preset : trainOptions.preset}
              disabled={busy}
              onChange={(preset) =>
                input === "media" ? setOptions({ ...options, preset }) : setTrainOptions({ ...trainOptions, preset })
              }
              options={PRESETS}
            />
            <p className={text.caption}>{PRESET_NOTE[input === "media" ? options.preset : trainOptions.preset]}</p>
          </div>
          {input === "media" ? (
            <>
              <div className="space-y-2">
                <p className="text-[14px] font-medium text-[var(--ds-gray-1000)]">Mode</p>
                <SegmentedControl<Mode>
                  label="Mode"
                  value={options.mode}
                  disabled={busy}
                  onChange={(mode) => setOptions({ ...options, mode })}
                  options={MODES}
                />
                <p className={text.caption}>{MODE_NOTE[options.mode]}</p>
              </div>
              {!isObject && (
                <label className="flex flex-col gap-2">
                  <span className="flex items-baseline justify-between text-[14px] font-medium text-[var(--ds-gray-1000)]">
                    Max images <span className="font-mono text-[13px] font-normal">{options.maxImages}</span>
                  </span>
                  <RangeInput
                    min={10}
                    max={400}
                    step={10}
                    value={options.maxImages}
                    disabled={busy}
                    aria-label="Max images"
                    onValueChange={(maxImages) => setOptions({ ...options, maxImages })}
                  />
                  <span className={text.caption}>Frames taken from a video, or photos kept from a larger set.</span>
                </label>
              )}
              {!isObject && (
                <Switch
                  label="Mesh"
                  description="Also bake a textured .glb mesh (slower)."
                  checked={options.mesh}
                  disabled={busy}
                  onCheckedChange={(mesh) => setOptions({ ...options, mesh })}
                />
              )}
              <div className="space-y-2">
                <p className="text-[14px] font-medium text-[var(--ds-gray-1000)]">Licence</p>
                <SegmentedControl<LicenceProfile>
                  label="Licence profile"
                  value={options.profile}
                  disabled={busy}
                  onChange={(profile) => setOptions({ ...options, profile })}
                  options={PROFILES}
                />
                <p className={text.caption}>{PROFILE_NOTE[options.profile]}</p>
              </div>
              {options.profile === "research" && (
                <div className="space-y-2">
                  <p className="text-[14px] font-medium text-[var(--ds-gray-1000)]">Attention</p>
                  <SegmentedControl<EngineSchedule>
                    label="Engine schedule"
                    value={options.engineSchedule}
                    disabled={busy}
                    onChange={(engineSchedule) => setOptions({ ...options, engineSchedule })}
                    options={SCHEDULES}
                  />
                  <p className={text.caption}>{SCHEDULE_NOTE[options.engineSchedule]}</p>
                </div>
              )}
            </>
          ) : (
            <>
              <Switch
                label="Evaluate"
                description="Hold out every 8th photo and report PSNR and SSIM on them."
                checked={trainOptions.evaluate}
                disabled={busy}
                onCheckedChange={(evaluate) => setTrainOptions({ ...trainOptions, evaluate })}
              />
              <Switch
                label="Limit width to 1600 px"
                description="Downscale larger photos first; faster and within 12 GB."
                checked={trainOptions.limitWidth}
                disabled={busy}
                onCheckedChange={(limitWidth) => setTrainOptions({ ...trainOptions, limitWidth })}
              />
            </>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3 border-t border-[var(--ds-gray-alpha-400)] pt-4">
          <Button size="sm" icon={FileBox} onClick={() => splat.current?.click()}>
            Open .ply / .spz
          </Button>
          <span className={text.meta}>Views a file in this browser; nothing is uploaded.</span>
          <input
            ref={splat}
            type="file"
            accept=".ply,.spz"
            tabIndex={-1}
            aria-hidden="true"
            className="hidden"
            data-testid="splat-input"
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) onView(file);
            }}
          />
        </div>
      </div>
    </Surface>
  );
}

/** ingest → posing → init → refine → mesh → export, with the live numbers. */
export function StageBar({ finished, live, mesh }: { finished: boolean; live: LiveNumbers; mesh: boolean }) {
  const states = stageStates(live.stage, { mesh, finished });
  const shown = STAGES.filter((s) => states[s] !== "skipped");
  return (
    <div className="space-y-3">
      <ol className="grid gap-1" style={{ gridTemplateColumns: `repeat(${shown.length}, minmax(0, 1fr))` }} aria-label="Stages">
        {shown.map((s) => {
          const state = states[s];
          return (
            <li key={s} className="min-w-0" aria-current={state === "active" ? "step" : undefined}>
              <div
                className={cn(
                  "h-1 rounded-full",
                  state === "done"
                    ? "bg-[var(--ds-gray-1000)]"
                    : state === "active"
                      ? "ds-indeterminate bg-[var(--ds-gray-alpha-200)]"
                      : "bg-[var(--ds-gray-alpha-200)]",
                )}
              />
              <p
                className={cn(
                  "mt-1.5 flex items-center gap-1 truncate font-mono text-[11px] uppercase leading-4",
                  state === "todo" ? "text-[var(--ds-gray-700)]" : "text-[var(--ds-gray-1000)]",
                )}
              >
                {state === "done" ? <Check aria-hidden="true" className="h-3 w-3 shrink-0" /> : null}
                {STAGE_LABEL[s]}
                <span className="sr-only">: {state === "done" ? "done" : state === "active" ? "in progress" : "to do"}</span>
              </p>
            </li>
          );
        })}
      </ol>
      <dl className="grid grid-cols-3 gap-x-4 gap-y-1.5">
        <Fact
          k="iteration"
          v={
            live.iteration === undefined
              ? "–"
              : `${live.iteration.toLocaleString()}${live.iterations ? `/${live.iterations.toLocaleString()}` : ""}`
          }
        />
        <Fact k="held-out PSNR" v={live.psnr === undefined ? "–" : `${live.psnr.toFixed(2)} dB`} />
        <Fact k="gaussians" v={live.gaussians === undefined ? "–" : live.gaussians.toLocaleString()} />
      </dl>
    </div>
  );
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <div className="min-w-0">
      <dt className={text.meta}>{k}</dt>
      <dd className="break-words font-mono text-[14px] text-[var(--ds-gray-1000)]">{v}</dd>
    </div>
  );
}

/** Numbers and downloads of a finished reconstruction. */
export function ReconstructResult({ out }: { out: ReconstructOutput }) {
  const rows: [string, string][] = [];
  if (out.summary?.images) rows.push(["images", out.summary.images.toLocaleString()]);
  if (out.summary?.total_s) rows.push(["time", `${out.summary.total_s.toFixed(1)} s`]);
  if (out.summary?.final_gaussians) rows.push(["gaussians", out.summary.final_gaussians.toLocaleString()]);
  if (out.eval?.psnr !== undefined) rows.push(["PSNR", `${out.eval.psnr.toFixed(2)} dB`]);
  if (out.eval?.ssim !== undefined) rows.push(["SSIM", out.eval.ssim.toFixed(4)]);
  if (out.eval?.lpips !== undefined) rows.push(["LPIPS", out.eval.lpips.toFixed(3)]);
  if (out.engine) rows.push(["engine", out.engine]);
  if (out.licence) rows.push(["licence", out.licence]);
  const all: [string, string, ArtifactRef | undefined][] = [
    ["SPZ", "scene.spz", out.spz],
    ["PLY", "scene.ply", out.ply],
    ["GLB mesh", "scene.glb", out.glb],
    ["Cameras", "cameras.json", out.cameras],
    ["COLMAP", "colmap.zip", out.colmap],
    ["transforms.json", "transforms.json", out.transforms],
  ];
  const files = all.filter((f): f is [string, string, ArtifactRef] => Boolean(f[2]));
  return (
    <div className="space-y-4 border-t border-[var(--ds-gray-alpha-400)] pt-4">
      {rows.length ? (
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3">
          {rows.map(([k, v]) => (
            <Fact key={k} k={k} v={v} />
          ))}
        </dl>
      ) : null}
      {files.length ? (
        <div className="flex flex-wrap gap-2" role="group" aria-label="Downloads">
          {files.map(([label, name, ref]) => (
            <ArtifactDownload key={label} label={label} name={name} file={ref} />
          ))}
        </div>
      ) : null}
    </div>
  );
}
