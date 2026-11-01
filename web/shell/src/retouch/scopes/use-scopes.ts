import { useEffect, useState } from "react";
import { useArtifactSrc } from "../artifact";
import { computeScopes, fitWithin, type Scopes } from "./compute";

/* Scopes of a stored preview JPEG, measured in the browser: the image is
   drawn into a small canvas (nearest neighbour, so the sample keeps the
   photo's real values instead of averaging the extremes away) and read
   back. At 512 px on the long side that is ~175k pixels, a few
   milliseconds of work, so it runs on the main thread after a short
   debounce. Artifacts are content-addressed, so results are cached by id. */

/** Long edge of the sample. */
export const SAMPLE_EDGE = 512;
/** Live slider previews arrive in bursts; measure the one that settles. */
const DEBOUNCE_MS = 120;
const CACHE_SIZE = 32;
const cache = new Map<string, Scopes>();

function remember(id: string, scopes: Scopes) {
  cache.delete(id);
  cache.set(id, scopes);
  for (const oldest of cache.keys()) {
    if (cache.size <= CACHE_SIZE) break;
    cache.delete(oldest);
  }
}

async function measure(src: string): Promise<Scopes> {
  const image = new Image();
  image.decoding = "async";
  image.src = src;
  await image.decode();
  const { width, height } = fitWithin(image.naturalWidth, image.naturalHeight, SAMPLE_EDGE);
  if (width === 0) throw new Error("empty image");
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("no 2D canvas");
  context.imageSmoothingEnabled = false;
  context.drawImage(image, 0, 0, width, height);
  return computeScopes(context.getImageData(0, 0, width, height).data);
}

export type ScopesState = {
  /** The newest measured scopes; while the next image is measured the
      previous ones stay up, so live previews never flash. */
  scopes?: Scopes;
  /** `scopes` are not (yet) of the current image. */
  pending: boolean;
  /** The current image could not be fetched, decoded or read back. */
  failed: boolean;
};

export function useScopes(id: string | undefined): ScopesState {
  const { src, failed: fetchFailed } = useArtifactSrc(id);
  const [measured, setMeasured] = useState<{ id: string; scopes: Scopes }>();
  const [broken, setBroken] = useState<string>();
  const cached = id ? cache.get(id) : undefined;

  useEffect(() => {
    if (!id || !src || cache.has(id)) return;
    let live = true;
    const timer = window.setTimeout(() => {
      measure(src).then(
        (scopes) => {
          remember(id, scopes);
          if (live) setMeasured({ id, scopes });
        },
        () => {
          if (live) setBroken(id);
        },
      );
    }, DEBOUNCE_MS);
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [id, src]);

  if (!id) return { pending: false, failed: false };
  const current = cached ?? (measured?.id === id ? measured.scopes : undefined);
  if (!current && (fetchFailed || broken === id)) return { pending: false, failed: true };
  return { scopes: current ?? measured?.scopes, pending: !current, failed: false };
}
