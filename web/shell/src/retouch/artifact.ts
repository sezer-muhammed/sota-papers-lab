import { useEffect, useState } from "react";
import { client, fetchArtifact, hasToken } from "../api";

/* Artifacts sit behind SPL_AUTH_TOKEN like the rest of /v1, and an <img> or
   a download link cannot send a bearer header. Without a token (the tailnet
   setup) plain URLs work. With one, the bytes are fetched with the header and
   shown as blob URLs; artifacts are content-addressed, so a blob never goes
   stale and a small cache keeps thumbnails from refetching. */

const BLOB_CACHE_SIZE = 120;
const blobs = new Map<string, Promise<string>>();

function blobUrl(id: string): Promise<string> {
  const cached = blobs.get(id);
  if (cached) {
    blobs.delete(id);
    blobs.set(id, cached);
    return cached;
  }
  const url = fetchArtifact(id).then((blob) => URL.createObjectURL(blob));
  url.catch(() => {
    if (blobs.get(id) === url) blobs.delete(id);
  });
  blobs.set(id, url);
  for (const [oldest, promise] of blobs) {
    if (blobs.size <= BLOB_CACHE_SIZE) break;
    blobs.delete(oldest);
    void promise.then(
      (u) => URL.revokeObjectURL(u),
      () => undefined,
    );
  }
  return url;
}

/** A URL for <img src>; `failed` when a token-protected fetch failed. */
export function useArtifactSrc(id: string | undefined): { src?: string; failed: boolean } {
  const viaBlob = hasToken();
  const [loaded, setLoaded] = useState<{ id: string; src?: string }>();

  useEffect(() => {
    if (!id || !viaBlob) return;
    let live = true;
    blobUrl(id).then(
      (src) => live && setLoaded({ id, src }),
      () => live && setLoaded({ id }),
    );
    return () => {
      live = false;
    };
  }, [id, viaBlob]);

  if (!id) return { failed: false };
  if (!viaBlob) return { src: client.artifactUrl(id), failed: false };
  if (loaded?.id !== id) return { failed: false };
  return { src: loaded.src, failed: loaded.src === undefined };
}

/** An artifact decoded as an image, for drawing on a canvas. */
export async function loadArtifactImage(id: string): Promise<HTMLImageElement> {
  const image = new Image();
  image.decoding = "async";
  image.src = hasToken() ? await blobUrl(id) : client.artifactUrl(id);
  await image.decode();
  return image;
}

/** Saves an artifact as `filename`. The host sends it as an attachment under
    that name; the anchor's download attribute covers a host that does not. */
export async function downloadArtifact(id: string, filename: string): Promise<void> {
  if (!hasToken()) {
    clickDownload(client.artifactUrl(id, { download: filename }), filename);
    return;
  }
  const url = URL.createObjectURL(await fetchArtifact(id));
  clickDownload(url, filename);
  // Long enough for the browser to start reading it.
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

function clickDownload(href: string, filename: string) {
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  anchor.rel = "noopener";
  anchor.style.display = "none";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
}
