import { useEffect, useRef, useState } from "react";
import { ImagePlus, Upload } from "lucide-react";
import { Button, IconTile, cn, text } from "@spl/ui";
import { RAW_ACCEPT, filesFromEntries, takeDrop } from "./files";

/* Where photos come in: a file picker, and a drop target that lights up while
   files are dragged anywhere over the page (see useFileDrop). Compact once
   the session has photos. */
export function DropZone({
  compact,
  disabled = false,
  dragging,
  onFiles,
}: {
  compact: boolean;
  disabled?: boolean;
  dragging: boolean;
  onFiles: (files: File[]) => void;
}) {
  const input = useRef<HTMLInputElement>(null);

  const picker = (
    <>
      <Button
        variant={compact ? "secondary" : "primary"}
        icon={Upload}
        disabled={disabled}
        onClick={() => input.current?.click()}
      >
        Choose files
      </Button>
      <input
        ref={input}
        type="file"
        multiple
        accept={RAW_ACCEPT}
        tabIndex={-1}
        aria-hidden="true"
        className="hidden"
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          // Cleared, so choosing the same files again still fires a change.
          event.target.value = "";
          if (files.length) onFiles(files);
        }}
      />
    </>
  );

  return (
    <section
      aria-labelledby="retouch-drop-title"
      className={cn(
        "rounded-[8px] border-2 border-dashed transition-colors",
        dragging
          ? "border-[var(--ds-blue-700)] bg-[var(--ds-blue-100)]"
          : "border-[var(--ds-gray-alpha-500)] bg-[var(--ds-background-100)]",
        compact
          ? "flex flex-wrap items-center gap-x-4 gap-y-3 px-4 py-3"
          : "flex flex-col items-center gap-3 px-4 py-10 text-center sm:py-14",
      )}
    >
      <IconTile icon={ImagePlus} />
      {/* Compact: the text keeps a readable width; the button wraps below it on phones. */}
      <div className={cn("min-w-0", compact && "min-w-[12rem] flex-1")}>
        <h2 id="retouch-drop-title" className="text-[17px] font-semibold leading-6 text-[var(--ds-gray-1000)]">
          {dragging ? "Drop to add these photos" : compact ? "Add more photos" : "Drop RAW photos here"}
        </h2>
        <p className={cn(text.caption, "mt-0.5")}>
          Sony .ARW, also DNG, NEF, CR2, CR3, RAF, ORF and RW2. Folders work too.
        </p>
      </div>
      {picker}
      {compact ? null : <p className={text.meta}>Uploads two at a time to this host</p>}
    </section>
  );
}

/** Files dragged over any part of the page: `dragging` while they hover, and
    `onFiles` with what was dropped (folders walked). Also keeps a drop that
    misses the zone from opening the file in the tab. */
export function useFileDrop(onFiles: (files: File[]) => void, enabled: boolean): boolean {
  const [dragging, setDragging] = useState(false);
  const latest = useRef({ onFiles, enabled });

  useEffect(() => {
    latest.current = { onFiles, enabled };
  });

  useEffect(() => {
    // dragenter/dragleave fire for every element crossed; count to know when
    // the drag has really left the window.
    let depth = 0;
    const carriesFiles = (event: DragEvent) => Array.from(event.dataTransfer?.types ?? []).includes("Files");
    const enter = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      depth += 1;
      if (latest.current.enabled) setDragging(true);
    };
    const over = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = latest.current.enabled ? "copy" : "none";
    };
    const leave = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setDragging(false);
    };
    const drop = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      depth = 0;
      setDragging(false);
      if (!event.dataTransfer || !latest.current.enabled) return;
      const { entries, files } = takeDrop(event.dataTransfer);
      void filesFromEntries(entries).then((found) => {
        const all = [...files, ...found];
        if (all.length) latest.current.onFiles(all);
      });
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragover", over);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragover", over);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
      setDragging(false);
    };
  }, []);

  return dragging && enabled;
}
