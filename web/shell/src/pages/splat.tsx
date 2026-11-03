import { useState } from "react";
import { Plus } from "lucide-react";
import type { Health } from "@spl/sdk";
import { ButtonLink } from "@spl/ui";
import type { Loaded } from "../api";
import { PageIntro } from "../components/page-intro";
import { RunPanel } from "../splat/panels";
import { BenchPage } from "../splat/bench";
import { DEFAULT_RECONSTRUCT, type ReconstructOptions } from "../splat/reconstruct";
import { NewScenePanel } from "../splat/reconstruct-panels";
import { CaptureTips, SampleScenes } from "../splat/samples";
import { useReconstruct } from "../splat/use-reconstruct";
import { DEFAULT_OPTIONS, useTrain, type TrainOptions } from "../splat/use-train";
import { SplatViewer, type SplatSource } from "../splat/viewer";

/* #/3d: photos or a phone video in (image_to_3d.reconstruct, no poses
   needed), or a COLMAP dataset (image_to_3d.train); Gaussian splats out,
   viewed here while they form. #/3d/<run> follows (or re-opens) one run.
   #/3d/bench shows the benchmark comparison table. */
export function SplatPage({ health, runId }: { health: Loaded<Health>; runId?: string }) {
  if (runId === "bench") return <BenchPage />;
  const run = useReconstruct(runId);
  const zip = useTrain(undefined);
  const [options, setOptions] = useState<ReconstructOptions>(DEFAULT_RECONSTRUCT);
  const [trainOptions, setTrainOptions] = useState<TrainOptions>(DEFAULT_OPTIONS);
  // A .ply / .spz opened from disk; shown instead of the run's splats.
  const [local, setLocal] = useState<SplatSource>();
  const busy = Boolean(run.sending || zip.upload);
  const source = local ?? run.splat;
  const error = run.error ?? zip.error;

  return (
    <div className="space-y-6">
      <PageIntro
        eyebrow="image-to-3d"
        title="Image to 3D"
        lead="Photos or a phone video in, 3D Gaussian splats out, on this GPU. Cameras and a point cloud show within seconds, then the scene sharpens as it trains. Orbit it, compare any view with its photo, and download .spz, .ply or the cameras."
        action={
          <>
            <ButtonLink href="#/3d/bench">Bench</ButtonLink>
            {runId ? (
              <ButtonLink icon={Plus} href="#/3d">
                New scene
              </ButtonLink>
            ) : null}
          </>
        }
      />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <SplatViewer
          source={source}
          points={local ? undefined : run.points}
          cameras={local ? [] : run.cameras}
          photoFor={local ? undefined : run.photoFor}
        />
        <div className="space-y-6">
          {runId ? (
            <RunPanel
              detail={run.detail}
              error={run.error}
              snapshot={local ? undefined : run.snapshot}
              onCancel={() => void run.cancel()}
            />
          ) : (
            <NewScenePanel
              busy={busy}
              maxUploadBytes={health.data?.max_upload_bytes}
              options={options}
              setOptions={setOptions}
              trainOptions={trainOptions}
              setTrainOptions={setTrainOptions}
              sending={run.sending}
              onStopSending={run.stopSending}
              zipUpload={zip.upload}
              onReconstruct={(picked) => {
                setLocal(undefined);
                void run.start(picked, options);
              }}
              onTrainZip={(file) => {
                setLocal(undefined);
                void zip.start(file, trainOptions);
              }}
              onView={(file) => {
                const format = file.name.toLowerCase().endsWith(".spz") ? "spz" : "ply";
                void file.arrayBuffer().then((bytes) =>
                  setLocal({ key: `${file.name}:${file.size}:${file.lastModified}`, bytes, format }),
                );
              }}
            />
          )}
          {!runId && error ? (
            <p role="alert" className="text-[13px] text-[var(--ds-gray-900)]">
              {error}
            </p>
          ) : null}
        </div>
      </div>
      {runId ? null : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
          <SampleScenes
            busy={busy}
            onStart={(scene, preset) => {
              setLocal(undefined);
              void run.startSample(scene, preset, options);
            }}
          />
          <CaptureTips />
        </div>
      )}
    </div>
  );
}
