import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MEDIA, isFinished, type Announcement, type RunDetail } from "@spl/sdk";
import { client, describeError, fetchArtifact } from "../api";
import { parseCameras, type Frustum } from "./cameras";
import { fileFingerprint, localResumeStore, uploadChunked } from "./chunked";
import {
  RECONSTRUCT_OP,
  matchPhotos,
  mediaTypeOf,
  reconstructInput,
  type Picked,
  type ReconstructInput,
  type ReconstructOptions,
  type ReconstructOutput,
} from "./reconstruct";
import type { PhotoFor, PointsSource, SplatSource } from "./viewer";

export type Sending = {
  /** "12 photos" or the video's name. */
  what: string;
  sent: number;
  total: number;
};

/** Photo files picked in this tab, by the run they went to: thumbnails and
    the wipe use them without downloading the photos back. */
const localPhotos = new Map<string, File[]>();

/** Photo uploads in flight at once (each one goes up in parts). */
const FILES_PARALLEL = 3;

const formatOf = (a: { media_type: string }) => (a.media_type === MEDIA.ply ? "ply" : "spz");

/* One run of the #/3d page, addressed by the route (#/3d/<run>), so a reload
   or a shared link re-attaches to it. Works for `image_to_3d.reconstruct`
   and the COLMAP `image_to_3d.train` alike: it fetches each newly announced
   artifact — `cameras` (frustums), `points` (the posing point cloud) and
   `splat` snapshots (hot-swapped) — and at the end the final splat if no
   snapshot came. Starting a reconstruction uploads the photos or video in
   resumable chunks first. */
export function useReconstruct(runId: string | undefined) {
  const [sending, setSending] = useState<Sending>();
  const [error, setError] = useState<string>();
  const [detail, setDetail] = useState<RunDetail>();
  const [splat, setSplat] = useState<SplatSource>();
  const [points, setPoints] = useState<PointsSource>();
  const [cameras, setCameras] = useState<Frustum[]>([]);
  const [snapshot, setSnapshot] = useState<{ revision: number; info: unknown }>();
  const sendAbort = useRef<AbortController>(undefined);

  useEffect(() => {
    setDetail(undefined);
    setSplat(undefined);
    setPoints(undefined);
    setCameras([]);
    setSnapshot(undefined);
    if (!runId) return;
    setError(undefined);
    const abort = new AbortController();
    const signal = abort.signal;
    let gotSplat = false;

    const bytesOf = (id: string) => fetchArtifact(id, signal).then((blob) => blob.arrayBuffer());
    const loadSplat = (id: string, format: "ply" | "spz", live: boolean) =>
      bytesOf(id).then((bytes) => {
        if (!signal.aborted) setSplat({ key: runId, bytes, format, live });
      });

    const onAnnounce = (a: Announcement) => {
      if (a.slot === "splat" || a.slot === "snapshot") {
        gotSplat = true;
        setSnapshot({ revision: a.revision, info: a.info });
        loadSplat(a.artifact.id, formatOf(a.artifact), true).catch(() => undefined);
      } else if (a.slot === "points") {
        bytesOf(a.artifact.id)
          .then((bytes) => !signal.aborted && setPoints({ key: `${runId}:${a.revision}`, bytes }))
          .catch(() => undefined);
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
        const out = last.run.output as ReconstructOutput | null;
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

  const start = useCallback(async (picked: Exclude<Picked, { kind: "none" }>, options: ReconstructOptions) => {
    setError(undefined);
    const abort = new AbortController();
    sendAbort.current = abort;
    const files = picked.kind === "photos" ? picked.files : [picked.file];
    const total = files.reduce((n, f) => n + f.size, 0);
    const what = picked.kind === "photos" ? `${files.length} photos` : picked.file.name;
    const sentBy = new Map<File, number>();
    const report = () => {
      let sent = 0;
      for (const n of sentBy.values()) sent += n;
      setSending({ what, sent, total });
    };
    report();
    try {
      const ids: string[] = new Array(files.length);
      const queue = files.map((_, i) => i);
      const worker = async () => {
        for (let i = queue.shift(); i !== undefined; i = queue.shift()) {
          const file = files[i]!;
          const artifact = await uploadChunked(client, file, {
            filename: file.name,
            mediaType: mediaTypeOf(file),
            kind: picked.kind === "photos" ? "image" : "blob",
            fingerprint: fileFingerprint(file),
            store: localResumeStore,
            signal: abort.signal,
            onProgress: (sent) => {
              sentBy.set(file, sent);
              report();
            },
          });
          ids[i] = artifact.id;
        }
      };
      await Promise.all(Array.from({ length: Math.min(FILES_PARALLEL, files.length) }, worker));
      const input = reconstructInput(picked.kind === "photos" ? { images: ids } : { video: ids[0] }, options);
      const run = await client.startRun(RECONSTRUCT_OP, input);
      if (picked.kind === "photos") localPhotos.set(run.id, picked.files);
      window.location.hash = `#/3d/${encodeURIComponent(run.id)}`;
    } catch (e) {
      if (!abort.signal.aborted) setError(describeError(e));
    } finally {
      setSending(undefined);
      sendAbort.current = undefined;
    }
  }, []);

  const stopSending = useCallback(() => sendAbort.current?.abort(), []);

  const cancel = useCallback(async () => {
    if (runId && detail && !isFinished(detail.run.status)) {
      try {
        setDetail({ ...detail, run: await client.cancelRun(runId) });
      } catch (e) {
        setError(describeError(e));
      }
    }
  }, [runId, detail]);

  // Photos for thumbnails and the wipe: the files picked in this tab, else
  // the run's photo artifacts, matched to cameras by name or order.
  const input = detail?.run.input as Partial<ReconstructInput> | undefined;
  const remoteImages = detail?.run.op_id === RECONSTRUCT_OP && Array.isArray(input?.images) ? input.images : undefined;
  const remoteKey = remoteImages?.join(",");
  const photoFor = useMemo<PhotoFor | undefined>(() => {
    const files = runId ? localPhotos.get(runId) : undefined;
    if (!cameras.length) return undefined;
    if (files) {
      const match = matchPhotos(
        cameras.map((c) => c.name),
        files.map((f) => f.name),
      );
      return async (i) => (match[i] === undefined ? undefined : files[match[i]!]);
    }
    const ids = remoteKey ? remoteKey.split(",") : undefined;
    if (!ids || ids.length !== cameras.length) return undefined;
    const cache = new Map<number, Promise<Blob>>();
    return (i) => {
      const id = ids[i];
      if (!id) return Promise.resolve(undefined);
      if (!cache.has(i)) cache.set(i, fetchArtifact(id));
      return cache.get(i)!;
    };
  }, [runId, cameras, remoteKey]);

  return { sending, error, detail, splat, points, cameras, snapshot, photoFor, start, stopSending, cancel };
}
