import { useEffect, useState, type ReactNode } from "react";
import { ImageOff } from "lucide-react";
import { Skeleton, cn } from "@spl/ui";
import { useArtifactSrc } from "./artifact";

/* A stored JPEG filling a box the caller sizes and positions (relative): a
   shimmer until it has loaded, a quiet placeholder when it can't. Without an
   id, `empty` (or the shimmer). Absolutely positioned, so a portrait image
   can never stretch a 3:2 box (an in-flow image counts toward a flex item's
   minimum height). */
export function ArtifactImage({
  alt,
  className,
  empty,
  fit = "contain",
  id,
  priority = false,
}: {
  alt: string;
  className?: string;
  empty?: ReactNode;
  fit?: "contain" | "cover";
  id?: string;
  /** Load at once instead of lazily (above the fold). */
  priority?: boolean;
}) {
  const { src, failed } = useArtifactSrc(id);
  const [loaded, setLoaded] = useState<string>();
  const [broken, setBroken] = useState<string>();
  const ready = src !== undefined && loaded === src;
  const error = failed || (src !== undefined && broken === src);

  return (
    <span className={cn("absolute inset-0 block overflow-hidden", className)}>
      {!id && empty ? (
        empty
      ) : error ? (
        <ImageUnavailable />
      ) : !ready ? (
        <Skeleton className="absolute inset-0 rounded-none" />
      ) : null}
      {src ? (
        <img
          key={src}
          src={src}
          alt={alt}
          loading={priority ? "eager" : "lazy"}
          decoding="async"
          draggable={false}
          onLoad={() => setLoaded(src)}
          onError={() => setBroken(src)}
          className={cn(
            "absolute inset-0 h-full w-full transition-opacity duration-200 motion-reduce:transition-none",
            fit === "cover" ? "object-cover" : "object-contain",
            ready ? "opacity-100" : "opacity-0",
          )}
        />
      ) : null}
    </span>
  );
}

export function ImageUnavailable({ label = "Image unavailable" }: { label?: string }) {
  return (
    <span className="absolute inset-0 grid place-items-center bg-[var(--ds-background-200)] p-2 text-center">
      <span className="flex flex-col items-center gap-1 text-[12px] text-[var(--ds-gray-700)]">
        <ImageOff aria-hidden="true" className="h-4 w-4" />
        {label}
      </span>
    </span>
  );
}

/** The newest image for `id` that has finished decoding. While a new one
    loads, the previous stays up, so stepping through renders never flashes. */
export function useDecodedSrc(id: string | undefined): {
  src?: string;
  pending: boolean;
  failed: boolean;
} {
  const { src: wanted, failed: fetchFailed } = useArtifactSrc(id);
  const [shown, setShown] = useState<string>();
  const [broken, setBroken] = useState<string>();

  useEffect(() => {
    if (!wanted) return;
    let live = true;
    const image = new Image();
    image.decoding = "async";
    image.src = wanted;
    image.decode().then(
      () => live && setShown(wanted),
      () => live && setBroken(wanted),
    );
    return () => {
      live = false;
    };
  }, [wanted]);

  const failed = fetchFailed || (wanted !== undefined && broken === wanted);
  return {
    src: id ? shown : undefined,
    pending: id !== undefined && !failed && shown !== wanted,
    failed,
  };
}
