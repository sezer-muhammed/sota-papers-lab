import { useState } from "react";
import { Download } from "lucide-react";
import { ButtonLink, cn, text } from "@spl/ui";
import { client, hasToken } from "../api";
import { downloadArtifact } from "./artifact";
import { formatBytes } from "./format";
import type { FileRef, FinalFiles } from "./model";

/* A real link to the file, so it works from the keyboard, middle-click and
   "Save link as". On a token-protected host the click fetches it with the
   header instead (see artifact.ts). */
export function DownloadButton({ file }: { file: FileRef }) {
  const [state, setState] = useState<"idle" | "busy" | "failed">("idle");

  return (
    <ButtonLink
      size="sm"
      icon={Download}
      href={client.artifactUrl(file.id, { download: file.name })}
      download={file.name}
      aria-disabled={state === "busy" || undefined}
      onClick={(event) => {
        if (!hasToken()) return;
        event.preventDefault();
        if (state === "busy") return;
        setState("busy");
        downloadArtifact(file.id, file.name).then(
          () => setState("idle"),
          () => setState("failed"),
        );
      }}
    >
      {state === "failed" ? "Retry" : "Download"}
      <span className="sr-only"> {file.name}</span>
    </ButtonLink>
  );
}

const KINDS = [
  { key: "jpeg", label: "JPEG" },
  { key: "png", label: "PNG" },
  { key: "xmp", label: "XMP" },
  { key: "tiff", label: "16-bit TIFF" },
] as const;

export function FileList({ className, files }: { className?: string; files: FinalFiles }) {
  const rows = KINDS.flatMap((kind) => {
    const file = files[kind.key];
    return file ? [{ ...kind, file }] : [];
  });
  if (rows.length === 0) return null;
  return (
    <ul className={cn("divide-y divide-[var(--ds-gray-alpha-300)]", className)}>
      {rows.map(({ file, key, label }) => (
        <li key={key} className="flex items-center justify-between gap-3 py-2">
          <div className="min-w-0">
            <p className="truncate font-mono text-[13px] text-[var(--ds-gray-1000)]" title={file.name}>
              {file.name}
            </p>
            <p className={text.meta}>
              {label} · {formatBytes(file.size)}
            </p>
          </div>
          <DownloadButton file={file} />
        </li>
      ))}
    </ul>
  );
}

/** Lightroom renders the XMP's sliders with its own engine, so the look is
    close to ours, not identical. */
export function XmpNote({ className }: { className?: string }) {
  return (
    <p className={cn(text.caption, className)}>
      The XMP sets Lightroom's develop settings (Basic, Tone Curve, Color Mixer, Detail, Lens Corrections,
      Calibration) to these values. Lightroom renders them with its own engine, so
      the result approximates this JPEG; put the .xmp next to the RAW with the same name before importing.
    </p>
  );
}
