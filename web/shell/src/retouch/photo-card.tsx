import { memo } from "react";
import { ImageOff } from "lucide-react";
import { ActionBar, Card, cn, text } from "@spl/ui";
import { ArtifactImage } from "./images";
import { cameraLine, exposureLine, formatBytes } from "./format";
import { photoInfo, photoPhase, thumbnailId, type Photo } from "./model";
import type { RetouchSession } from "./session";
import { PhaseSignal, PhotoActions, PhotoProgress } from "./status";

/* One photo in the grid: its image (the edit once there is one), name and
   status, camera, lens and exposure, what it is doing, and its actions. */
export const PhotoCard = memo(function PhotoCard({
  canRetryUpload,
  photo,
  position,
  session,
}: {
  canRetryUpload: boolean;
  photo: Photo;
  position?: number;
  session: RetouchSession;
}) {
  const phase = photoPhase(photo);
  const info = photoInfo(photo);
  const href = `#/retouch/${encodeURIComponent(photo.key)}`;
  const failedEarly = phase === "failed" && !info;

  return (
    <Card depth="base" className="flex w-full flex-col p-0">
      {/* The title link is the keyboard stop; the image is a larger mouse target for the same link. */}
      <a
        href={href}
        tabIndex={-1}
        aria-hidden="true"
        className="relative block aspect-[3/2] shrink-0 overflow-hidden border-b border-[var(--ds-gray-alpha-300)] bg-[var(--ds-background-200)]"
      >
        <ArtifactImage
          id={thumbnailId(photo)}
          alt=""
          empty={
            failedEarly ? (
              <span className="absolute inset-0 grid place-items-center text-[var(--ds-gray-700)]">
                <ImageOff aria-hidden="true" className="h-5 w-5" />
              </span>
            ) : undefined
          }
        />
      </a>
      <div className="flex flex-1 flex-col gap-2 p-3">
        <div className="flex items-start justify-between gap-3">
          <h3 className="min-w-0">
            <a
              href={href}
              className="block truncate rounded-[4px] font-mono text-[13px] font-semibold text-[var(--ds-gray-1000)] outline-none hover:underline focus-visible:shadow-[var(--ds-focus-ring)]"
              title={photo.filename}
            >
              {photo.filename}
            </a>
          </h3>
          <PhaseSignal photo={photo} />
        </div>
        {info ? (
          <div className="min-w-0">
            <p className={cn(text.caption, "truncate")} title={cameraLine(info)}>
              {cameraLine(info)}
            </p>
            <p className={cn(text.meta, "truncate")}>{exposureLine(info) || formatBytes(photo.size)}</p>
          </div>
        ) : (
          <p className={text.meta}>{formatBytes(photo.size)}</p>
        )}
        <PhotoProgress photo={photo} position={position} />
        <ActionBar className="mt-auto pt-1">
          <PhotoActions photo={photo} session={session} canRetryUpload={canRetryUpload} />
        </ActionBar>
      </div>
    </Card>
  );
});
