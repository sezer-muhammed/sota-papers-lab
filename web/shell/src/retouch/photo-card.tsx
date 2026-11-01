import { memo } from "react";
import { ImageOff, MapPin } from "lucide-react";
import { ActionBar, Card, cn, text } from "@spl/ui";
import { ArtifactImage } from "./images";
import { cameraLine, exposureLine, formatBytes, formatCaptured, placeLine } from "./format";
import { INSTRUCTION_EXAMPLES, InstructionField, UsedInstruction } from "./instruction";
import { photoInfo, photoPhase, thumbnailId, usedInstruction, type Photo } from "./model";
import { cleanInstruction } from "./options";
import type { RetouchSession } from "./session";
import { PhaseSignal, PhotoActions, PhotoProgress } from "./status";

/* One photo in the grid: its image (the edit once there is one), when and
   where it was taken, its status, camera, lens and exposure, what it is
   doing, its actions, and the box for the edit wanted. The file name stays
   in the link's title and accessible name. */
export const PhotoCard = memo(function PhotoCard({
  canRetryUpload,
  defaultInstruction,
  photo,
  position,
  session,
}: {
  canRetryUpload: boolean;
  /** The default box, shown in the placeholder of an empty box. */
  defaultInstruction: string;
  photo: Photo;
  position?: number;
  session: RetouchSession;
}) {
  const phase = photoPhase(photo);
  const info = photoInfo(photo);
  const href = `#/retouch/${encodeURIComponent(photo.key)}`;
  const failedEarly = phase === "failed" && !info;
  const captured = info ? formatCaptured(info.captured_at) : undefined;
  const place = info ? placeLine(info) : undefined;
  const own = photo.instruction ?? "";
  const fallback = cleanInstruction(defaultInstruction);
  const used = usedInstruction(photo.result);
  const showUsed = used !== undefined && used !== cleanInstruction(own);

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
          <div className="min-w-0">
            <h3 className="min-w-0">
              <a
                href={href}
                className="block truncate rounded-[4px] text-[14px] font-semibold leading-5 text-[var(--ds-gray-1000)] outline-none hover:underline focus-visible:shadow-[var(--ds-focus-ring)]"
                title={photo.filename}
                aria-label={captured ? `${captured}, ${photo.filename}` : photo.filename}
              >
                {/* The file name only until the metadata is read. */}
                {captured ?? (info ? "Date unknown" : photo.filename)}
              </a>
            </h3>
            {place ? (
              <p className={cn(text.caption, "mt-0.5 flex min-w-0 items-center gap-1")} title={place}>
                <MapPin aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-[var(--ds-gray-700)]" />
                <span className="truncate">{place}</span>
              </p>
            ) : null}
          </div>
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
        {showUsed ? <UsedInstruction instruction={used} /> : null}
        <ActionBar className="mt-auto pt-1">
          <PhotoActions photo={photo} session={session} canRetryUpload={canRetryUpload} />
        </ActionBar>
      </div>
      <div className="border-t border-[var(--ds-gray-alpha-300)] p-3">
        <InstructionField
          label="Edit wanted (optional)"
          value={own}
          onChange={(value) => session.setInstruction(photo.key, value)}
          placeholder={fallback ? `Default: ${fallback}` : INSTRUCTION_EXAMPLES.photo}
        />
      </div>
    </Card>
  );
});
