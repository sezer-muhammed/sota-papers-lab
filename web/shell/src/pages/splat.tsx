import { useState } from "react";
import { Plus } from "lucide-react";
import type { Health } from "@spl/sdk";
import { ButtonLink } from "@spl/ui";
import type { Loaded } from "../api";
import { PageIntro } from "../components/page-intro";
import { RunPanel, SourcePanel } from "../splat/panels";
import { BenchPage } from "../splat/bench";
import { DEFAULT_OPTIONS, useTrain, type TrainOptions } from "../splat/use-train";
import { SplatViewer, type SplatSource } from "../splat/viewer";

/* #/3d: a COLMAP dataset in, Gaussian splats out, viewed here while they
   train. #/3d/<run> follows (or re-opens) one run of image_to_3d.train.
   #/3d/bench shows the benchmark comparison table. */
export function SplatPage({ health, runId }: { health: Loaded<Health>; runId?: string }) {
  if (runId === "bench") return <BenchPage />;
  const train = useTrain(runId);
  const [options, setOptions] = useState<TrainOptions>(DEFAULT_OPTIONS);
  // A .ply / .spz opened from disk; shown instead of the run's splats.
  const [local, setLocal] = useState<SplatSource>();
  const busy = Boolean(train.upload);
  const source = local ?? train.splat;

  return (
    <div className="space-y-6">
      <PageIntro
        eyebrow="image-to-3d"
        title="Image to 3D"
        lead="Posed photos in, 3D Gaussian splats out, trained on this GPU. Watch the scene form while it trains, orbit it here, and download it as .spz or .ply."
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
        <SplatViewer source={source} cameras={local ? [] : train.cameras} />
        <div className="space-y-6">
          {runId ? (
            <RunPanel
              detail={train.detail}
              error={train.error}
              snapshot={local ? undefined : train.snapshot}
              onCancel={() => void train.cancel()}
            />
          ) : (
            <SourcePanel
              busy={busy}
              maxUploadBytes={health.data?.max_upload_bytes}
              options={options}
              setOptions={setOptions}
              upload={train.upload}
              onStart={(file) => {
                setLocal(undefined);
                void train.start(file, options);
              }}
              onView={(file) => {
                const format = file.name.toLowerCase().endsWith(".spz") ? "spz" : "ply";
                void file.arrayBuffer().then((bytes) =>
                  setLocal({ key: `${file.name}:${file.size}:${file.lastModified}`, bytes, format }),
                );
              }}
            />
          )}
          {!runId && train.error ? (
            <p role="alert" className="text-[13px] text-[var(--ds-gray-900)]">
              {train.error}
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}
