/* RAW files from a picker or a drop, folders included. */

/** Sony first: the app is tuned on ARW; rawler reads the others too. */
export const RAW_EXTENSIONS = ["arw", "dng", "nef", "cr2", "cr3", "raf", "orf", "rw2"];

/** For <input accept>: some pickers match case-sensitively, so both cases. */
export const RAW_ACCEPT = RAW_EXTENSIONS.flatMap((ext) => [`.${ext}`, `.${ext.toUpperCase()}`]).join(",");

export function isRawFileName(name: string): boolean {
  const dot = name.lastIndexOf(".");
  return dot > 0 && RAW_EXTENSIONS.includes(name.slice(dot + 1).toLowerCase());
}

/** A drop's entries (files and folders). Read them inside the drop handler:
    the browser empties the DataTransfer once the handler returns. */
export function takeDrop(data: DataTransfer): { entries: FileSystemEntry[]; files: File[] } {
  const entries: FileSystemEntry[] = [];
  const files: File[] = [];
  for (const item of Array.from(data.items ?? [])) {
    if (item.kind !== "file") continue;
    const entry = typeof item.webkitGetAsEntry === "function" ? item.webkitGetAsEntry() : null;
    if (entry) {
      entries.push(entry);
    } else {
      const file = item.getAsFile();
      if (file) files.push(file);
    }
  }
  if (entries.length === 0 && files.length === 0) files.push(...Array.from(data.files ?? []));
  return { entries, files };
}

/** Every file under the dropped entries, folders walked a few levels deep.
    An entry the browser refuses to read is skipped. */
export async function filesFromEntries(entries: FileSystemEntry[]): Promise<File[]> {
  const found: File[] = [];

  const walk = async (entry: FileSystemEntry, depth: number): Promise<void> => {
    if (entry.isFile) {
      found.push(
        await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject)),
      );
      return;
    }
    if (!entry.isDirectory || depth > 6) return;
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    // readEntries answers in batches (100 in Chromium) until an empty one.
    for (;;) {
      const batch = await new Promise<FileSystemEntry[]>((resolve, reject) =>
        reader.readEntries(resolve, reject),
      );
      if (batch.length === 0) break;
      for (const child of batch) await walk(child, depth + 1).catch(() => undefined);
    }
  };

  for (const entry of entries) await walk(entry, 0).catch(() => undefined);
  return found;
}
