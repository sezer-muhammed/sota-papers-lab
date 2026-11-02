import { useCallback, useEffect, useRef, useState } from "react";
import { MEDIA, isFinished, type Announcement, type RunDetail } from "@spl/sdk";
import type { TrainInput } from "@spl/bindings/image-to-3d/TrainInput";
import type { TrainOutput } from "@spl/bindings/image-to-3d/TrainOutput";
import { client, describeError, fetchArtifact } from "../api";
import { parseCameras, type Frustum } from "./cameras";
import type { SplatSource } from "./viewer";

export const TRAIN_OP = "image_to_3d.train";

export type Preset = "fast" | "quality";

export type TrainOptions = {
  preset: Preset;
  /** Render the held-out views and report PSNR / SSIM. */
  evaluate: boolean;
  /** Downscale photos wider than 1600 px (the Inria rule for user data). */
  limitWidth: boolean;
};

export const DEFAULT_OPTIONS: TrainOptions = { preset: "quality", evaluate: true, limitWidth: true };

/** Snapshot cadence asked of the op for the live viewer. */
const SNAPSHOT_EVERY_S = 10;

export type Upload = { name: string; fraction: number };

export function trainInput(zip: string, options: TrainOptions): TrainInput {
  return {
    zip,
    load: {
      images_dir: "images",
      test_every: 8,
      principal_point: "colmap",
      ...(options.limitWidth ? { max_width: 1600 } : {}),
    },
    train: { preset: options.preset, seed: 0, overrides: {}, snapshot_every_s: SNAPSHOT_EVERY_S },
    eval: options.evaluate,
    export: ["ply", "spz"],
    dry_run: false,
  };
}

const formatOf = (a: { media_type: string }) => (a.media_type === MEDIA.spz ? "spz" : "ply");

/* One training run of the #/3d page, addressed by the route (#/3d/<run>), so
   a reload or a shared link re-attaches to it. Uploads the zip, starts the
   op, polls it, and fetches each newly announced splat and cameras artifact
   for the viewer; at the end, the final splat if no snapshot came. */
export function useTrain(runId: string | undefined) {
  const [upload, setUpload] = useState<Upload>();
  const [error, setError] = useState<string>();
  const [detail, setDetail] = useState<RunDetail>();
  const [splat, setSplat] = useState<SplatSource>();
  const [cameras, setCameras] = useState<Frustum[]>([]);
  const [snapshot, setSnapshot] = useState<{ revision: number; info: unknown }>();
  const uploadAbort = useRef<AbortController>(undefined);

  useEffect(() => {
    setDetail(undefined);
    setSplat(undefined);
    setCameras([]);
    setSnapshot(undefined);
    if (!runId) return;
    setError(undefined);
    const abort = new AbortController();
    const signal = abort.signal;
    let gotSplat = false;

    const loadSplat = (id: string, format: "ply" | "spz", live: boolean) =>
      fetchArtifact(id, signal)
        .then((blob) => blob.arrayBuffer())
        .then((bytes) => {
          if (!signal.aborted) setSplat({ key: runId, bytes, format, live });
        });

    const onAnnounce = (a: Announcement) => {
      if (a.slot === "splat") {
        gotSplat = true;
        setSnapshot({ revision: a.revision, info: a.info });
        loadSplat(a.artifact.id, formatOf(a.artifact), true).catch(() => undefined);
      } else if (a.slot === "cameras") {
        fetchArtifact(a.artifact.id, signal)
          .then((blob) => blob.text())
          .then((t) => !signal.aborted && setCameras(parseCameras(JSON.parse(t))))
          .catch(() => undefined);
      }
    };

    client
      .watchRun(runId, { signal, onDetail: setDetail, onAnnounce })
      .then((last) => {
        const out = last.run.output as TrainOutput | null;
        const final = out?.spz ?? out?.ply;
        if (last.run.status === "succeeded" && final && !gotSplat) {
          return loadSplat(final.id, formatOf(final), false);
        }
      })
      .catch((e: unknown) => {
        if (!signal.aborted) setError(describeError(e));
      });
    return () => abort.abort();
  }, [runId]);

  const start = useCallback(async (file: File, options: TrainOptions) => {
    setError(undefined);
    const abort = new AbortController();
    uploadAbort.current = abort;
    setUpload({ name: file.name, fraction: 0 });
    try {
      const zip = await client.uploadArtifact(file, {
        filename: file.name,
        mediaType: "application/zip",
        onProgress: (fraction) => setUpload({ name: file.name, fraction }),
        signal: abort.signal,
      });
      const run = await client.startRun(TRAIN_OP, trainInput(zip.id, options));
      window.location.hash = `#/3d/${encodeURIComponent(run.id)}`;
    } catch (e) {
      if (!abort.signal.aborted) setError(describeError(e));
    } finally {
      setUpload(undefined);
      uploadAbort.current = undefined;
    }
  }, []);

  const cancel = useCallback(async () => {
    uploadAbort.current?.abort();
    if (runId && detail && !isFinished(detail.run.status)) {
      try {
        setDetail({ ...detail, run: await client.cancelRun(runId) });
      } catch (e) {
        setError(describeError(e));
      }
    }
  }, [runId, detail]);

  return { upload, error, detail, splat, cameras, snapshot, start, cancel };
}
