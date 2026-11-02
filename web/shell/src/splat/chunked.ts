/* Resumable chunked uploads (`/v1/uploads`). A file goes up in parts of
   16 MiB, a few at a time, so no request passes a proxy's body limit
   (Cloudflare stops at 100 MB) and a dropped connection costs one part.
   The session id is remembered per file (name, size, mtime), so dropping
   the same file again after a reload or a failure sends only the missing
   parts. No DOM here beyond what the injected api needs: unit tested. */
import type { ArtifactKind, ArtifactRef, UploadSession } from "@spl/sdk";

export const PART_BYTES = 16 * 2 ** 20;

/** The SplClient methods this uses (a fake in tests). */
export type UploadApi = {
  createUpload(request: {
    size: number;
    filename?: string;
    media_type?: string;
    kind?: ArtifactKind;
    part_bytes?: number;
  }): Promise<UploadSession>;
  uploadStatus(id: string): Promise<UploadSession>;
  putUploadPart(
    id: string,
    index: number,
    part: Blob,
    options: { onProgress?: (loaded: number) => void; signal?: AbortSignal },
  ): Promise<UploadSession>;
  completeUpload(id: string): Promise<ArtifactRef>;
};

/** Where session ids wait for a retry: localStorage in the page. */
export type ResumeStore = {
  get(key: string): string | undefined;
  set(key: string, id: string): void;
  delete(key: string): void;
};

export type ChunkedOptions = {
  filename: string;
  mediaType: string;
  kind?: ArtifactKind;
  /** Identifies the file across reloads; no resume without it. */
  fingerprint?: string;
  store?: ResumeStore;
  partBytes?: number;
  /** Parts in flight at once. */
  parallel?: number;
  /** Tries per part before the upload fails. */
  tries?: number;
  onProgress?: (sent: number, total: number) => void;
  signal?: AbortSignal;
  /** Waits between tries (tests pass a no-op). */
  sleep?: (ms: number) => Promise<void>;
};

const status = (e: unknown) => (e && typeof e === "object" && "status" in e ? Number(e.status) : 0);
const aborted = (e: unknown) => e instanceof DOMException && e.name === "AbortError";

/** Byte range of part `n`. */
export function partRange(session: Pick<UploadSession, "size" | "part_bytes">, n: number): [number, number] {
  const start = n * session.part_bytes;
  return [start, Math.min(start + session.part_bytes, session.size)];
}

/** Parts still to send, ascending. */
export function missingParts(session: Pick<UploadSession, "parts" | "received">): number[] {
  const have = new Set(session.received);
  return Array.from({ length: session.parts }, (_, n) => n).filter((n) => !have.has(n));
}

export function fileFingerprint(file: File): string {
  return `upload:${file.name}:${file.size}:${file.lastModified}`;
}

async function openSession(api: UploadApi, file: Blob, o: ChunkedOptions, partBytes: number) {
  const key = o.fingerprint;
  const known = key ? o.store?.get(key) : undefined;
  if (known) {
    try {
      const session = await api.uploadStatus(known);
      if (session.size === file.size) return session;
    } catch (e) {
      if (status(e) !== 404) throw e;
    }
    o.store?.delete(key!);
  }
  const session = await api.createUpload({
    size: file.size,
    filename: o.filename,
    media_type: o.mediaType,
    kind: o.kind,
    part_bytes: partBytes,
  });
  if (key) o.store?.set(key, session.id);
  return session;
}

/** Uploads `file` and resolves with its artifact. */
export async function uploadChunked(api: UploadApi, file: Blob, o: ChunkedOptions): Promise<ArtifactRef> {
  const partBytes = o.partBytes ?? PART_BYTES;
  const tries = o.tries ?? 5;
  const wait = o.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  let session = await openSession(api, file, o, partBytes);

  for (let round = 0; ; round += 1) {
    const todo = missingParts(session);
    const inFlight = new Map<number, number>();
    let done = session.size - todo.reduce((n, p) => n + (partRange(session, p)[1] - partRange(session, p)[0]), 0);
    const report = () => {
      let sent = done;
      for (const loaded of inFlight.values()) sent += loaded;
      o.onProgress?.(Math.min(sent, session.size), session.size);
    };
    report();

    const sendPart = async (n: number) => {
      const [a, b] = partRange(session, n);
      for (let attempt = 1; ; attempt += 1) {
        o.signal?.throwIfAborted();
        try {
          inFlight.set(n, 0);
          await api.putUploadPart(session.id, n, file.slice(a, b), {
            signal: o.signal,
            onProgress: (loaded) => {
              inFlight.set(n, Math.min(loaded, b - a));
              report();
            },
          });
          inFlight.delete(n);
          done += b - a;
          report();
          return;
        } catch (e) {
          inFlight.delete(n);
          // Client errors other than "being written" will not get better.
          const s = status(e);
          if (aborted(e) || attempt >= tries || (s >= 400 && s < 500 && s !== 409 && s !== 429)) throw e;
          await wait(Math.min(1000 * 2 ** (attempt - 1), 15_000));
        }
      }
    };

    const queue = [...todo];
    const worker = async () => {
      for (let n = queue.shift(); n !== undefined; n = queue.shift()) await sendPart(n);
    };
    await Promise.all(Array.from({ length: Math.min(o.parallel ?? 3, queue.length) }, worker));

    try {
      const artifact = await api.completeUpload(session.id);
      if (o.fingerprint) o.store?.delete(o.fingerprint);
      return artifact;
    } catch (e) {
      // 409: the host misses a part (a write it lost); send what it lacks.
      if (status(e) !== 409 || round >= 2) throw e;
      session = await api.uploadStatus(session.id);
    }
  }
}

/** The page's ResumeStore. */
export const localResumeStore: ResumeStore = {
  get: (key) => {
    try {
      return localStorage.getItem(key) ?? undefined;
    } catch {
      return undefined;
    }
  },
  set: (key, id) => {
    try {
      localStorage.setItem(key, id);
    } catch {
      /* private mode: no resume */
    }
  },
  delete: (key) => {
    try {
      localStorage.removeItem(key);
    } catch {
      /* nothing stored */
    }
  },
};
