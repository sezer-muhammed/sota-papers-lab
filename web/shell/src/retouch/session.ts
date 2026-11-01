import { SplError, type RunDetail, type RunRecord, type SplClient } from "@spl/sdk";
import type { InspectInput } from "@spl/bindings/raw-retouch/InspectInput";
import type { InspectOutput } from "@spl/bindings/raw-retouch/InspectOutput";
import type { Params } from "@spl/bindings/raw-retouch/Params";
import type { Region } from "@spl/bindings/raw-retouch/Region";
import type { RenderInput } from "@spl/bindings/raw-retouch/RenderInput";
import { describeError, isUnauthorized } from "../api";
import { isRawFileName } from "./files";
import { completeParams, sameParams } from "./format";
import {
  asAutoInput,
  asAutoOutput,
  asInspectOutput,
  asRenderOutput,
  canRetouch,
  finalFiles,
  isActive,
  isFinished,
  isParams,
  needsRetouch,
  queueOrder,
  type AutoState,
  type DownloadKind,
  type ExportState,
  type FileRef,
  type ManualState,
  type Photo,
  type SessionState,
} from "./model";
import {
  DEFAULT_OPTIONS,
  HOST_SLOTS,
  INSTRUCTION_MAX,
  autoInput,
  exportOptions,
  loadOptions,
  normalizeOptions,
  resolveInstruction,
  saveOptions,
  type OptionsStorage,
  type RetouchOptions,
} from "./options";

export const OPS = {
  inspect: "raw_retouch.inspect",
  auto: "raw_retouch.auto",
  render: "raw_retouch.render",
} as const;

const UPLOAD_SLOTS = 2;
const POLL_MS = { inspect: 1000, auto: 1500, preview: 200, export: 1000 } as const;
const PREVIEW_DEBOUNCE_MS = 350;
/** Between "Download all" clicks, so browsers take each as its own download. */
const DOWNLOAD_GAP_MS = 600;
/** 404s a poll tolerates before a run counts as lost: a store that lags the
    run it just started should not fail the photo. */
const MISSING_LIMIT = 5;
const SESSION_KEY = "spl-retouch-session";
/** The Web Lock whose holder drives the retouch queue: one tab per browser. */
const QUEUE_LOCK = "spl-retouch-queue";

/** No export files: what a preview render asks for. */
const NO_EXPORT = { jpeg: false, jpeg_quality: 95, tiff16: false, xmp: false, png: false, png_bits: 8 };

const DOWNLOAD_LABEL: Record<DownloadKind, string> = { jpeg: "JPEG", png: "PNG", xmp: "XMP" };

/** A 1:1 crop rendered for the zoomed viewer. */
export type RegionRender = { id: string; width: number; height: number };

export type RetouchClient = Pick<SplClient, "uploadArtifact" | "startRun" | "runDetail" | "cancelRun">;

export type SessionDeps = {
  /** Called for every request, so a token saved later is picked up. */
  client: () => RetouchClient;
  storage?: OptionsStorage;
  /** Saves one artifact under a file name (see artifact.ts). */
  download: (id: string, filename: string) => Promise<void>;
  /** Wake polls when the tab returns; warn before closing during an upload;
      share the queue with other tabs (one drives it). */
  browser?: boolean;
};

export type AddResult = { added: number; retried: number; duplicates: number; rejected: string[] };

/* What survives a reload: ids and small summaries. Results come back from
   the runs themselves (GET /v1/runs/{id}). */
type SavedPhoto = {
  key: string;
  filename: string;
  size: number;
  artifactId?: string;
  inspect?: InspectOutput;
  inspectRunId?: string;
  inspectError?: string;
  autoRunId?: string;
  /** The last successful auto run, when a later run is the current one. */
  resultRunId?: string;
  queued?: number;
  instruction?: string;
  manual?: {
    base: Params;
    params: Params;
    previewId?: string;
    previewParams?: Params;
    exportRunId?: string;
    exportParams?: Params;
  };
};

type PreviewSlot = { timer?: ReturnType<typeof setTimeout>; busy: boolean; again: boolean };

/**
 * The Retouch page's working state and every request behind it: uploads two
 * at a time, inspect after each upload, the auto queue at the chosen
 * parallelism, manual previews and exports, polling, cancellation and
 * persistence. Components read it with useSyncExternalStore and call its
 * methods; nothing here touches React.
 */
export class RetouchSession {
  private state: SessionState;
  private readonly deps: SessionDeps;
  private readonly listeners = new Set<() => void>();
  /** Files picked but not uploaded yet, kept until their upload succeeds (for Retry). */
  private readonly files = new Map<string, File>();
  private readonly uploads = new Map<string, AbortController>();
  /** One per photo. Aborted when the photo leaves the session, which ends its polls. */
  private readonly lifetimes = new Map<string, AbortController>();
  private readonly sleepers = new Map<string, Set<() => void>>();
  private readonly previews = new Map<string, PreviewSlot>();
  /** Cancel pressed while the run's startRun was still in flight. */
  private readonly cancelOnStart = new Set<string>();
  private nextOrder = 1;
  private pumpScheduled = false;
  private savedText = "";
  private announcements = 0;
  /** This tab drives the retouch queue. Two tabs restoring one session would
      otherwise each start the queued photos, retouching them twice. */
  private leader = true;
  /** Auto runs this tab has followed, so a follower never mistakes an old
      run in the leader's saved session for one it started from this queue. */
  private readonly seenRuns = new Set<string>();

  constructor(deps: SessionDeps) {
    this.deps = deps;
    const saved = parseSaved(readSaved(deps.storage));
    this.state = {
      photos: saved.photos.map(restorePhoto),
      options: loadOptions(deps.storage),
      instruction: saved.instruction,
      follower: false,
      locked: false,
    };
    this.nextOrder = 1 + Math.max(0, ...saved.photos.map((s) => s.queued ?? 0));
    this.savedText = serialize(this.state);
    if (deps.browser) {
      this.watchBrowser();
      this.claimQueue();
    }
    this.resume(saved.photos);
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = () => this.state;

  /* ---- Photos ---- */

  /** Adds RAW files; other files are rejected and files already in the
      session (same name and size) skipped, unless their upload had failed. */
  addFiles(files: File[]): AddResult {
    const result: AddResult = { added: 0, retried: 0, duplicates: 0, rejected: [] };
    const photos = this.state.photos.slice();
    for (const file of files) {
      if (!isRawFileName(file.name) || file.size === 0) {
        result.rejected.push(file.name);
        continue;
      }
      const index = photos.findIndex((p) => p.filename === file.name && p.size === file.size);
      const existing = photos[index];
      if (existing) {
        if (existing.upload.status === "failed") {
          this.files.set(existing.key, file);
          photos[index] = { ...existing, upload: { status: "waiting" } };
          result.retried += 1;
        } else {
          result.duplicates += 1;
        }
        continue;
      }
      const key = newKey(photos);
      this.files.set(key, file);
      photos.push({
        key,
        filename: file.name,
        size: file.size,
        upload: { status: "waiting" },
        inspect: { status: "idle" },
        auto: { status: "idle" },
      });
      result.added += 1;
    }
    if (result.added || result.retried) this.commit({ ...this.state, photos });
    return result;
  }

  remove(key: string) {
    const photo = this.photo(key);
    if (!photo) return;
    this.teardown(photo);
    this.commit({ ...this.state, photos: this.state.photos.filter((p) => p.key !== key) });
  }

  /** Forgets every photo; running retouches and exports are cancelled. The
      uploaded files and finished runs stay on the host. */
  clear() {
    for (const photo of this.state.photos) this.teardown(photo);
    this.commit({ ...this.state, photos: [], trouble: undefined, downloading: undefined });
  }

  /** Whether a failed upload can be retried: the browser still holds the
      file (not after a reload). Changes only together with a state commit. */
  hasFile(key: string): boolean {
    return this.files.has(key);
  }

  retryUpload(key: string) {
    if (!this.files.has(key)) return;
    this.patch(key, (p) => (p.upload.status === "failed" ? { ...p, upload: { status: "waiting" } } : p));
  }

  retryInspect(key: string) {
    const photo = this.photo(key);
    if (photo?.artifactId && photo.inspect.status === "failed") void this.inspect(key);
  }

  /* ---- Retouch queue ---- */

  /** Queues one photo (Retouch, Re-run). It starts when a slot is free. */
  retouch(key: string) {
    this.patch(key, (p) => (canRetouch(p) ? { ...p, auto: { status: "queued", order: this.nextOrder++ } } : p));
  }

  /** Queues every photo that is not done yet; returns how many. Photos still
      uploading or being read start once they are ready. */
  retouchAll(): number {
    let count = 0;
    const photos = this.state.photos.map((p): Photo => {
      if (!needsRetouch(p)) return p;
      count += 1;
      return { ...p, auto: { status: "queued", order: this.nextOrder++ } };
    });
    if (count) this.commit({ ...this.state, photos });
    return count;
  }

  /** A queued photo leaves the queue; a running one is asked to stop. */
  cancel(key: string) {
    const photo = this.photo(key);
    if (!photo) return;
    const auto = photo.auto;
    if (auto.status === "queued") {
      this.patch(key, (p) => ({
        ...p,
        auto: p.result ? { status: "succeeded", runId: p.result.runId } : { status: "idle" },
      }));
    } else if (auto.status === "restoring") {
      void this.requestCancel(key, auto.runId);
    } else if (auto.status === "running") {
      if (auto.runId) {
        void this.requestCancel(key, auto.runId);
      } else {
        this.cancelOnStart.add(key);
        this.patch(key, (p) => (p.auto.status === "running" ? { ...p, auto: { ...p.auto, cancelling: true } } : p));
      }
    }
  }

  cancelAll() {
    for (const photo of this.state.photos) if (isActive(photo)) this.cancel(photo.key);
  }

  /* ---- Instructions ---- */

  /** The photo's own box. Read when its retouch starts, so a queued photo
      follows what the box says by then. */
  setInstruction(key: string, text: string) {
    const instruction = text.slice(0, INSTRUCTION_MAX);
    this.patch(key, (p) => ((p.instruction ?? "") === instruction ? p : { ...p, instruction: instruction || undefined }));
  }

  /** The default box, for photos whose own box is empty. */
  setDefaultInstruction(text: string) {
    const instruction = text.slice(0, INSTRUCTION_MAX);
    if (instruction !== this.state.instruction) this.commit({ ...this.state, instruction });
  }

  /** What a retouch of this photo would follow now. */
  instructionFor(photo: Photo): string | undefined {
    return resolveInstruction(photo.instruction, this.state.instruction);
  }

  /* ---- Options ---- */

  setOptions(patch: Partial<RetouchOptions>) {
    const options = normalizeOptions({ ...this.state.options, ...patch });
    saveOptions(this.deps.storage, options);
    this.commit({ ...this.state, options });
  }

  resetOptions() {
    this.setOptions(DEFAULT_OPTIONS);
  }

  /* ---- Manual adjust ---- */

  /** Opens manual editing from the chosen edit (or neutral before one exists). */
  startAdjust(key: string) {
    this.patch(key, (p) => {
      if (p.manual) return p;
      const base =
        p.result?.output.params ?? (p.inspect.status === "done" ? p.inspect.output.neutral : undefined);
      if (!base) return p;
      return { ...p, manual: { base, params: base, preview: knownPreview(p, base) } };
    });
  }

  /** New slider values. The preview renders once they rest for 350 ms, one
      render at a time; a render that finishes for values no longer current
      is dropped. */
  setManual(key: string, params: Params) {
    const photo = this.photo(key);
    if (!photo?.manual) return;
    const known = knownPreview(photo, params);
    this.patch(key, (p) =>
      p.manual ? { ...p, manual: { ...p.manual, params, error: undefined, preview: known ?? p.manual.preview } } : p,
    );
    const slot = this.previewSlot(key);
    if (slot.timer !== undefined) clearTimeout(slot.timer);
    slot.timer = undefined;
    if (known) return;
    slot.timer = setTimeout(() => {
      slot.timer = undefined;
      void this.renderPreview(key);
    }, PREVIEW_DEBOUNCE_MS);
  }

  resetManual(key: string) {
    const manual = this.photo(key)?.manual;
    if (manual) this.setManual(key, manual.base);
  }

  /** Renders the current sliders at full size with the export files. */
  async exportFull(key: string) {
    const photo = this.photo(key);
    const manual = photo?.manual;
    if (!photo?.artifactId || !manual || manual.export?.status === "running") return;
    const params = manual.params;
    const input: RenderInput = {
      raw: photo.artifactId,
      params,
      size: "full",
      filename: photo.filename,
      // A full-size export is for the JPEG, whatever the auto options say.
      export: exportOptions(this.state.options, true),
    };
    const life = this.lifetime(key);
    this.setExport(key, () => ({ status: "running", params, progress: null, cancelling: false }));
    let record: RunRecord;
    try {
      record = await this.deps.client().startRun(OPS.render, input);
    } catch (error) {
      if (life.aborted) return;
      this.noteAuth(error);
      this.setExport(key, () => ({
        status: "failed",
        params,
        error: `Couldn't start the export: ${describeError(error)}`,
      }));
      return;
    }
    if (life.aborted) {
      this.cancelQuietly(record.id);
      return;
    }
    this.setExport(key, (e) => (e?.status === "running" ? { ...e, runId: record.id } : e));
    await this.followExport(key, record, params, false);
  }

  cancelExport(key: string) {
    const exported = this.photo(key)?.manual?.export;
    if (exported?.status !== "running" || !exported.runId) return;
    const runId = exported.runId;
    this.setExport(key, (e) => (e?.status === "running" ? { ...e, cancelling: true } : e));
    this.deps
      .client()
      .cancelRun(runId)
      .catch((error: unknown) => {
        this.noteAuth(error);
        this.setExport(key, (e) => (e?.status === "running" ? { ...e, cancelling: false } : e));
      })
      .finally(() => this.wake(key));
  }

  /* ---- Zoom ---- */

  /**
   * Renders a crop of the photo at full resolution (1:1) with `params`, for
   * the zoomed viewer. Resolves with the crop's JPEG; rejects when `signal`
   * aborts (which also cancels the run on the host) or the render fails.
   */
  async renderRegion(key: string, params: Params, region: Region, signal: AbortSignal): Promise<RegionRender> {
    const photo = this.photo(key);
    if (!photo?.artifactId) throw new Error("The photo is not uploaded yet.");
    const input: RenderInput = {
      raw: photo.artifactId,
      params,
      size: "preview",
      filename: photo.filename,
      export: NO_EXPORT,
      region,
    };
    const client = this.deps.client();
    let run: RunRecord;
    try {
      run = await client.startRun(OPS.render, input);
    } catch (error) {
      this.noteAuth(error);
      throw error;
    }
    const id = run.id;
    const cancel = () => this.cancelQuietly(id);
    if (signal.aborted) {
      cancel();
      throw abortError();
    }
    signal.addEventListener("abort", cancel, { once: true });
    try {
      let failures = 0;
      while (!isFinished(run.status)) {
        await pause(failures ? Math.min(POLL_MS.preview * 2 ** failures, 5000) : POLL_MS.preview, signal);
        try {
          run = (await client.runDetail(id)).run;
          failures = 0;
        } catch (error) {
          if (signal.aborted) throw abortError();
          this.noteAuth(error);
          failures += 1;
          if (failures > 6) throw error;
        }
      }
      const output = run.status === "succeeded" ? asRenderOutput(run.output) : undefined;
      if (!output) throw new Error(runFailure(run, "The 1:1 render"));
      return { id: output.preview.id, width: output.width, height: output.height };
    } finally {
      signal.removeEventListener("abort", cancel);
    }
  }

  /* ---- Downloads ---- */

  /** Every photo's final JPEG (or PNG, or XMP), one browser download each. */
  async downloadAll(kind: DownloadKind) {
    if (this.state.downloading) return;
    const items = this.state.photos
      .map((p) => finalFiles(p)[kind])
      .filter((file): file is FileRef => file !== undefined);
    if (items.length === 0) return;
    let failed = 0;
    for (const [index, item] of items.entries()) {
      this.commit({ ...this.state, downloading: { kind, done: index, total: items.length } });
      try {
        await this.deps.download(item.id, item.name);
      } catch {
        failed += 1;
      }
      await new Promise((resolve) => setTimeout(resolve, DOWNLOAD_GAP_MS));
    }
    this.commit({ ...this.state, downloading: undefined });
    const label = DOWNLOAD_LABEL[kind];
    this.announce(
      failed
        ? `${failed} of ${items.length} ${label} downloads failed.`
        : `Started ${items.length} ${label} ${items.length === 1 ? "download" : "downloads"}.`,
    );
  }

  /* ---- Internals: state ---- */

  private photo(key: string) {
    return this.state.photos.find((p) => p.key === key);
  }

  private patch(key: string, update: (photo: Photo) => Photo) {
    const index = this.state.photos.findIndex((p) => p.key === key);
    const current = this.state.photos[index];
    if (!current) return;
    const next = update(current);
    if (next === current) return;
    const photos = this.state.photos.slice();
    photos[index] = next;
    this.commit({ ...this.state, photos });
  }

  private setExport(key: string, update: (current: ExportState | undefined) => ExportState | undefined) {
    this.patch(key, (p) => {
      if (!p.manual) return p;
      const next = update(p.manual.export);
      return next === p.manual.export ? p : { ...p, manual: { ...p.manual, export: next } };
    });
  }

  private commit(next: SessionState) {
    this.state = next;
    this.persist();
    for (const listener of [...this.listeners]) listener();
    this.schedulePump();
  }

  private persist() {
    const text = serialize(this.state);
    if (text === this.savedText) return;
    this.savedText = text;
    try {
      if (this.state.photos.length > 0 || this.state.instruction) this.deps.storage?.setItem(SESSION_KEY, text);
      else this.deps.storage?.removeItem(SESSION_KEY);
    } catch {
      /* quota or private mode: this visit still works, a reload forgets it */
    }
  }

  private announce(text: string) {
    this.announcements += 1;
    this.commit({ ...this.state, announcement: { id: this.announcements, text } });
  }

  private noteAuth(error: unknown) {
    if (isUnauthorized(error) && !this.state.locked) this.commit({ ...this.state, locked: true });
  }

  private setTrouble(message: string | undefined) {
    if (this.state.trouble !== message) this.commit({ ...this.state, trouble: message });
  }

  /* ---- Internals: scheduling ---- */

  private schedulePump() {
    if (this.pumpScheduled) return;
    this.pumpScheduled = true;
    queueMicrotask(() => {
      this.pumpScheduled = false;
      this.pump();
    });
  }

  /** Starts what the slots allow: uploads two at a time, retouches at the
      chosen parallelism in queue order, each after its photo has been read.
      Retouches leave one of the host's run slots free for previews, zoom
      tiles and exports, and only the tab that holds the queue starts them. */
  private pump() {
    const { photos, options } = this.state;
    let uploading = photos.filter((p) => p.upload.status === "uploading").length;
    for (const photo of photos) {
      if (uploading >= UPLOAD_SLOTS) break;
      if (photo.upload.status !== "waiting") continue;
      uploading += 1;
      void this.upload(photo.key);
    }

    if (!this.leader) return;
    const limit = Math.max(1, Math.min(options.parallel, HOST_SLOTS - 1));
    let active = photos.filter((p) => p.auto.status === "running" || p.auto.status === "restoring").length;
    const ready = photos
      .filter(
        (p) =>
          p.auto.status === "queued" &&
          p.artifactId !== undefined &&
          (p.inspect.status === "done" || p.inspect.status === "failed"),
      )
      .sort((a, b) => queueOrder(a) - queueOrder(b));
    for (const photo of ready) {
      if (active >= limit) break;
      active += 1;
      void this.runAuto(photo.key);
    }
  }

  /** The photo's lifetime signal; already aborted for a photo that is gone,
      so late callbacks never start a poll nobody can stop. */
  private lifetime(key: string): AbortSignal {
    let controller = this.lifetimes.get(key);
    if (!controller) {
      controller = new AbortController();
      if (this.photo(key)) this.lifetimes.set(key, controller);
      else controller.abort();
    }
    return controller.signal;
  }

  private teardown(photo: Photo) {
    const key = photo.key;
    this.uploads.get(key)?.abort();
    this.uploads.delete(key);
    this.files.delete(key);
    this.cancelOnStart.delete(key);
    const slot = this.previews.get(key);
    if (slot?.timer !== undefined) clearTimeout(slot.timer);
    this.previews.delete(key);
    const auto = photo.auto;
    if ((auto.status === "running" || auto.status === "restoring") && auto.runId) this.cancelQuietly(auto.runId);
    const exported = photo.manual?.export;
    if (exported?.status === "running" && exported.runId) this.cancelQuietly(exported.runId);
    this.lifetimes.get(key)?.abort();
    this.lifetimes.delete(key);
    this.sleepers.delete(key);
  }

  /** Waits `ms`, or less when woken (tab visible again, cancel sent) or when
      the photo leaves the session. */
  private sleep(key: string, ms: number, life: AbortSignal): Promise<void> {
    if (life.aborted) return Promise.resolve();
    let waiting = this.sleepers.get(key);
    if (!waiting) {
      waiting = new Set();
      this.sleepers.set(key, waiting);
    }
    const set = waiting;
    return new Promise((resolve) => {
      const done = () => {
        clearTimeout(timer);
        set.delete(done);
        life.removeEventListener("abort", done);
        resolve();
      };
      const timer = setTimeout(done, ms);
      set.add(done);
      life.addEventListener("abort", done, { once: true });
    });
  }

  private wake(key?: string) {
    const sets = key === undefined ? [...this.sleepers.values()] : [this.sleepers.get(key)];
    for (const set of sets) if (set) for (const done of [...set]) done();
  }

  private watchBrowser() {
    document.addEventListener("visibilitychange", () => {
      // Hidden tabs throttle timers; poll at once when the user comes back.
      if (document.visibilityState === "visible") this.wake();
    });
    window.addEventListener("online", () => this.wake());
    window.addEventListener("beforeunload", (event) => {
      if (!this.state.photos.some((p) => p.upload.status === "waiting" || p.upload.status === "uploading")) return;
      event.preventDefault();
      event.returnValue = "";
    });
    // While another tab drives the queue, follow the runs it starts.
    window.addEventListener("storage", (event) => {
      if (event.key === SESSION_KEY && !this.leader) this.adoptRuns(event.newValue);
    });
  }

  /** Takes the queue lock: at once when no other tab of this session holds
      it, else when that tab closes. Without Web Locks every tab drives its
      own queue, as before. */
  private claimQueue() {
    const locks = typeof navigator !== "undefined" ? navigator.locks : undefined;
    if (!locks?.request) return;
    this.leader = false;
    const lead = () => {
      this.leader = true;
      if (this.state.follower) this.commit({ ...this.state, follower: false });
      else this.schedulePump();
      // Held for the tab's life.
      return new Promise<void>(() => undefined);
    };
    locks
      .request(QUEUE_LOCK, { ifAvailable: true }, (lock) => {
        if (lock) return lead();
        this.commit({ ...this.state, follower: true });
        void locks.request(QUEUE_LOCK, lead).catch(() => undefined);
        return undefined;
      })
      .catch(() => {
        // Locks refused (e.g. an opaque origin): drive the queue here.
        this.leader = true;
        this.schedulePump();
      });
  }

  /** The leading tab saved the session: photos queued here that it has
      started are followed instead of started a second time. */
  private adoptRuns(text: string | null) {
    for (const saved of parseSaved(text).photos) {
      const runId = str(saved.autoRunId);
      const photo = this.photo(saved.key);
      if (!runId || this.seenRuns.has(runId) || !photo || photo.auto.status !== "queued") continue;
      if (photo.result?.runId === runId) continue;
      this.patch(photo.key, (p) => (p.auto.status === "queued" ? { ...p, auto: { status: "restoring", runId } } : p));
      void this.followAuto(photo.key, runId, true);
    }
  }

  /** Picks up what a reload interrupted: reads, retouches and exports that
      were running, results to show, and the queue. */
  private resume(saved: SavedPhoto[]) {
    for (const s of saved) {
      const photo = this.photo(s.key);
      if (!photo) continue;
      if (photo.inspect.status === "running") {
        if (photo.inspect.runId) void this.followInspect(photo.key, photo.inspect.runId, true);
        else void this.inspect(photo.key);
      }
      if (photo.auto.status === "restoring") void this.followAuto(photo.key, photo.auto.runId, true);
      if (s.resultRunId) void this.restoreResult(photo.key, s.resultRunId);
      const exported = photo.manual?.export;
      if (exported?.status === "running" && exported.runId) {
        void this.followExport(photo.key, exported.runId, exported.params, true);
      }
    }
    this.schedulePump();
  }

  /* ---- Internals: runs ---- */

  /**
   * Polls a run until it ends. Resolves with the final record, or undefined
   * once the photo leaves the session. Network errors back off and retry
   * forever (the host may be restarting); a run the host keeps answering 404
   * for is reported as failed.
   */
  private async follow(
    key: string,
    first: RunRecord | string,
    interval: number,
    immediate: boolean,
    onDetail?: (detail: RunDetail) => void,
  ): Promise<RunRecord | undefined> {
    if (typeof first !== "string" && isFinished(first.status)) return first;
    const id = typeof first === "string" ? first : first.id;
    const life = this.lifetime(key);
    // A run restored after a reload that the host does not know is gone for
    // good (a wiped data dir); a run started just now may still be landing.
    const missingLimit = immediate ? 2 : MISSING_LIMIT;
    let failures = 0;
    let missing = 0;
    let wait = immediate ? 0 : interval;
    for (;;) {
      if (wait > 0) await this.sleep(key, wait, life);
      if (life.aborted) return undefined;
      try {
        const detail = await this.deps.client().runDetail(id);
        if (life.aborted) return undefined;
        failures = 0;
        missing = 0;
        this.setTrouble(undefined);
        onDetail?.(detail);
        if (isFinished(detail.run.status)) return detail.run;
        wait = interval;
      } catch (error) {
        if (life.aborted) return undefined;
        if (error instanceof SplError && error.status === 404) {
          missing += 1;
          if (missing >= missingLimit) return lostRun(id);
        } else {
          failures += 1;
          this.noteAuth(error);
          this.setTrouble(describeError(error));
        }
        wait = Math.min(interval * 2 ** Math.min(failures + missing, 4), 15_000);
      }
    }
  }

  private cancelQuietly(runId: string) {
    this.deps
      .client()
      .cancelRun(runId)
      .catch(() => undefined);
  }

  private async upload(key: string) {
    const file = this.files.get(key);
    if (!file) {
      this.patch(key, (p) => ({
        ...p,
        upload: { status: "failed", error: "The file is no longer available. Add it again." },
      }));
      return;
    }
    const controller = new AbortController();
    this.uploads.set(key, controller);
    this.patch(key, (p) => ({ ...p, upload: { status: "uploading", fraction: 0 } }));
    try {
      const ref = await this.deps.client().uploadArtifact(file, {
        filename: file.name,
        signal: controller.signal,
        onProgress: (fraction) =>
          this.patch(key, (p) =>
            p.upload.status === "uploading" && Math.floor(fraction * 100) !== Math.floor(p.upload.fraction * 100)
              ? { ...p, upload: { status: "uploading", fraction } }
              : p,
          ),
      });
      if (typeof ref?.id !== "string") throw new Error("The host answered without an artifact id.");
      this.files.delete(key);
      this.patch(key, (p) => ({ ...p, artifactId: ref.id, upload: { status: "done" } }));
      void this.inspect(key);
    } catch (error) {
      if (controller.signal.aborted) return;
      this.noteAuth(error);
      this.patch(key, (p) => ({
        ...p,
        upload: { status: "failed", error: uploadError(error) },
        // A queued retouch needs the upload; it re-queues with Retry.
        auto: p.auto.status === "queued" ? { status: "idle" } : p.auto,
      }));
    } finally {
      if (this.uploads.get(key) === controller) this.uploads.delete(key);
    }
  }

  private async inspect(key: string) {
    const raw = this.photo(key)?.artifactId;
    if (!raw) return;
    const life = this.lifetime(key);
    this.patch(key, (p) => ({ ...p, inspect: { status: "running" } }));
    let record: RunRecord;
    try {
      const input: InspectInput = { raw };
      record = await this.deps.client().startRun(OPS.inspect, input);
    } catch (error) {
      if (life.aborted) return;
      this.noteAuth(error);
      this.patch(key, (p) => ({
        ...p,
        inspect: { status: "failed", error: `Couldn't read the file: ${describeError(error)}` },
      }));
      return;
    }
    if (life.aborted) return;
    this.patch(key, (p) => (p.inspect.status === "running" ? { ...p, inspect: { status: "running", runId: record.id } } : p));
    await this.followInspect(key, record, false);
  }

  private async followInspect(key: string, first: RunRecord | string, immediate: boolean) {
    const run = await this.follow(key, first, POLL_MS.inspect, immediate);
    if (!run) return;
    const output = run.status === "succeeded" ? asInspectOutput(run.output) : undefined;
    this.patch(key, (p) => ({
      ...p,
      inspect: output
        ? { status: "done", output }
        : { status: "failed", runId: run.id, error: runFailure(run, "Reading the file") },
    }));
  }

  private async runAuto(key: string) {
    const photo = this.photo(key);
    if (!photo?.artifactId) return;
    const life = this.lifetime(key);
    const input = autoInput(this.state.options, photo.artifactId, photo.filename, this.instructionFor(photo));
    this.patch(key, (p) => ({
      ...p,
      auto: { status: "running", serverQueued: false, progress: null, logs: [], startedAtMs: null, cancelling: false },
    }));
    let record: RunRecord;
    try {
      record = await this.deps.client().startRun(OPS.auto, input);
    } catch (error) {
      this.cancelOnStart.delete(key);
      if (life.aborted) return;
      this.noteAuth(error);
      this.patch(key, (p) => ({
        ...p,
        auto: { status: "failed", error: `Couldn't start the retouch: ${describeError(error)}` },
      }));
      return;
    }
    if (life.aborted) {
      this.cancelQuietly(record.id);
      return;
    }
    this.patch(key, (p) =>
      p.auto.status === "running"
        ? {
            ...p,
            auto: {
              ...p.auto,
              runId: record.id,
              serverQueued: record.status === "queued",
              startedAtMs: record.started_at_ms,
            },
          }
        : p,
    );
    if (this.cancelOnStart.delete(key)) void this.requestCancel(key, record.id);
    await this.followAuto(key, record, false);
  }

  private async followAuto(key: string, first: RunRecord | string, immediate: boolean) {
    // A run started or seen running in this visit is announced and replaces
    // manual edits; one that had ended before a reload just restores.
    let live = typeof first !== "string";
    this.seenRuns.add(typeof first === "string" ? first : first.id);
    const run = await this.follow(key, first, POLL_MS.auto, immediate, (detail) => {
      if (isFinished(detail.run.status)) return;
      live = true;
      this.patch(key, (p) => {
        const auto = p.auto;
        if (auto.status !== "running" && auto.status !== "restoring") return p;
        const running = auto.status === "running" ? auto : undefined;
        const next: AutoState = {
          status: "running",
          runId: detail.run.id,
          serverQueued: detail.run.status === "queued",
          progress: detail.progress,
          logs: detail.logs,
          startedAtMs: detail.run.started_at_ms,
          cancelling: running?.cancelling ?? false,
          notice: running?.notice,
        };
        return running && sameRunning(running, next) ? p : { ...p, auto: next };
      });
    });
    if (run) this.finishAuto(key, run, live);
  }

  private finishAuto(key: string, run: RunRecord, live: boolean) {
    const photo = this.photo(key);
    if (!photo) return;
    const name = photo.filename;
    if (run.status === "succeeded") {
      const output = asAutoOutput(run.output);
      if (!output) {
        this.patch(key, (p) => ({
          ...p,
          auto: {
            status: "failed",
            runId: run.id,
            error: "The host returned a result this page can't read. Reload the page after updating the host.",
          },
        }));
        return;
      }
      this.patch(key, (p) => ({
        ...p,
        auto: { status: "succeeded", runId: run.id },
        result: { output, runId: run.id, input: asAutoInput(run.input) },
        // Manual edits were made against the result this run replaces.
        manual: live ? undefined : p.manual,
      }));
      if (live) {
        this.previews.delete(key);
        this.announce(`${name} is retouched.`);
      }
      return;
    }
    if (run.status === "cancelled") {
      this.patch(key, (p) => ({ ...p, auto: { status: "cancelled", runId: run.id } }));
      if (live) this.announce(`${name}: retouch cancelled.`);
      return;
    }
    const error = runFailure(run, "The retouch");
    this.patch(key, (p) => ({ ...p, auto: { status: "failed", runId: run.id, error } }));
    if (live) this.announce(`${name} failed: ${error}`);
  }

  private async restoreResult(key: string, runId: string) {
    this.seenRuns.add(runId);
    const run = await this.follow(key, runId, POLL_MS.auto, true);
    const output = run?.status === "succeeded" ? asAutoOutput(run.output) : undefined;
    if (!run || !output) return;
    this.patch(key, (p) => (p.result ? p : { ...p, result: { output, runId: run.id, input: asAutoInput(run.input) } }));
  }

  private async requestCancel(key: string, runId: string) {
    this.patch(key, (p) =>
      p.auto.status === "running" && p.auto.runId === runId
        ? { ...p, auto: { ...p.auto, cancelling: true, notice: undefined } }
        : p,
    );
    try {
      await this.deps.client().cancelRun(runId);
    } catch (error) {
      this.noteAuth(error);
      // 409: the run ended first; the next poll shows how.
      if (!(error instanceof SplError && error.status === 409)) {
        this.patch(key, (p) =>
          p.auto.status === "running" && p.auto.runId === runId
            ? { ...p, auto: { ...p.auto, cancelling: false, notice: `Couldn't cancel: ${describeError(error)}` } }
            : p,
        );
      }
    }
    this.wake(key);
  }

  private previewSlot(key: string): PreviewSlot {
    let slot = this.previews.get(key);
    if (!slot) {
      slot = { busy: false, again: false };
      this.previews.set(key, slot);
    }
    return slot;
  }

  private async renderPreview(key: string) {
    const photo = this.photo(key);
    const manual = photo?.manual;
    if (!photo?.artifactId || !manual) return;
    const params = manual.params;
    if (manual.preview && sameParams(manual.preview.params, params)) return;
    const slot = this.previewSlot(key);
    if (slot.busy) {
      slot.again = true;
      return;
    }
    slot.busy = true;
    const life = this.lifetime(key);
    const current = (p: Photo) => p.manual !== undefined && sameParams(p.manual.params, params);
    try {
      const input: RenderInput = {
        raw: photo.artifactId,
        params,
        size: "preview",
        filename: photo.filename,
        export: NO_EXPORT,
      };
      const record = await this.deps.client().startRun(OPS.render, input);
      const run = await this.follow(key, record, POLL_MS.preview, false);
      if (!run) return;
      const output = run.status === "succeeded" ? asRenderOutput(run.output) : undefined;
      this.patch(key, (p) => {
        if (!current(p) || !p.manual) return p;
        const manualNext: ManualState = output
          ? { ...p.manual, preview: { id: output.preview.id, params }, error: undefined }
          : { ...p.manual, error: runFailure(run, "The preview") };
        return { ...p, manual: manualNext };
      });
    } catch (error) {
      if (life.aborted) return;
      this.noteAuth(error);
      this.patch(key, (p) =>
        current(p) && p.manual ? { ...p, manual: { ...p.manual, error: `Preview failed: ${describeError(error)}` } } : p,
      );
    } finally {
      slot.busy = false;
      if (slot.again && !life.aborted) {
        slot.again = false;
        void this.renderPreview(key);
      }
    }
  }

  private async followExport(key: string, first: RunRecord | string, params: Params, immediate: boolean) {
    const run = await this.follow(key, first, POLL_MS.export, immediate, (detail) => {
      if (isFinished(detail.run.status)) return;
      this.setExport(key, (e) =>
        e?.status === "running" &&
        (e.runId !== detail.run.id ||
          e.progress?.fraction !== detail.progress?.fraction ||
          e.progress?.message !== detail.progress?.message)
          ? { ...e, runId: detail.run.id, progress: detail.progress }
          : e,
      );
    });
    if (!run) return;
    const output = run.status === "succeeded" ? asRenderOutput(run.output) : undefined;
    this.setExport(key, (): ExportState => {
      if (output) return { status: "succeeded", runId: run.id, params, output };
      if (run.status === "cancelled") return { status: "cancelled", runId: run.id, params };
      return { status: "failed", runId: run.id, params, error: runFailure(run, "The export") };
    });
  }
}

/* ---- Helpers ---- */

/** A preview that already exists for these sliders: the chosen edit, a
    candidate, the neutral render, or the last manual preview. */
function knownPreview(photo: Photo, params: Params): { id: string; params: Params } | undefined {
  const output = photo.result?.output;
  if (output) {
    if (sameParams(output.params, params)) return { id: output.after.id, params };
    for (const candidate of output.candidates) {
      if (isParams(candidate.params) && sameParams(candidate.params, params) && candidate.preview?.id) {
        return { id: candidate.preview.id, params };
      }
    }
  }
  if (photo.inspect.status === "done" && sameParams(photo.inspect.output.neutral, params)) {
    return { id: photo.inspect.output.preview.id, params };
  }
  const last = photo.manual?.preview;
  return last && sameParams(last.params, params) ? last : undefined;
}

function sameRunning(a: Extract<AutoState, { status: "running" }>, b: Extract<AutoState, { status: "running" }>) {
  return (
    a.runId === b.runId &&
    a.serverQueued === b.serverQueued &&
    a.startedAtMs === b.startedAtMs &&
    a.progress?.fraction === b.progress?.fraction &&
    a.progress?.message === b.progress?.message &&
    a.logs.length === b.logs.length &&
    a.logs[a.logs.length - 1] === b.logs[b.logs.length - 1]
  );
}

function runFailure(run: RunRecord, what: string): string {
  if (run.status === "cancelled") return `${what} was cancelled.`;
  if (run.status === "succeeded") return "The host returned a result this page can't read.";
  return run.error_message?.trim() || `${what} failed without a message.`;
}

function uploadError(error: unknown): string {
  if (error instanceof SplError && error.status === 413) return "The host refused the file as too large (HTTP 413).";
  return `Upload failed: ${describeError(error)}`;
}

/** A run the host no longer knows, as a failed record. */
function lostRun(id: string): RunRecord {
  return {
    id,
    op_id: "",
    status: "failed",
    input: null,
    output: null,
    error_message: "The host no longer has this run.",
    host_name: null,
    git_sha: null,
    env: null,
    created_at_ms: 0,
    started_at_ms: null,
    finished_at_ms: null,
  };
}

/** Eight base-36 characters, unique in the session; also the photo's URL. */
function newKey(photos: Photo[]): string {
  for (;;) {
    const key = Math.random().toString(36).slice(2, 10).padEnd(8, "0");
    if (!photos.some((p) => p.key === key)) return key;
  }
}

function abortError(): Error {
  return new DOMException("The request was aborted.", "AbortError");
}

/** Waits `ms`; rejects at once when `signal` aborts. */
function pause(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortError());
      return;
    }
    const stop = () => {
      clearTimeout(timer);
      reject(abortError());
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", stop);
      resolve();
    }, ms);
    signal.addEventListener("abort", stop, { once: true });
  });
}

function serialize(state: Pick<SessionState, "photos" | "instruction">): string {
  return JSON.stringify({
    v: 1,
    photos: state.photos.map(toSaved),
    ...(state.instruction ? { instruction: state.instruction } : {}),
  });
}

function toSaved(photo: Photo): SavedPhoto {
  const { auto, inspect, manual } = photo;
  const autoRunId = "runId" in auto ? auto.runId : undefined;
  return {
    key: photo.key,
    filename: photo.filename,
    size: photo.size,
    artifactId: photo.artifactId,
    inspect: inspect.status === "done" ? inspect.output : undefined,
    inspectRunId: inspect.status === "running" ? inspect.runId : undefined,
    inspectError: inspect.status === "failed" ? inspect.error : undefined,
    autoRunId,
    resultRunId: photo.result && photo.result.runId !== autoRunId ? photo.result.runId : undefined,
    queued: auto.status === "queued" ? auto.order : undefined,
    instruction: photo.instruction || undefined,
    manual: manual && {
      base: manual.base,
      params: manual.params,
      previewId: manual.preview?.id,
      previewParams: manual.preview?.params,
      exportRunId: manual.export?.runId,
      exportParams: manual.export?.params,
    },
  };
}

function readSaved(storage: OptionsStorage | undefined): string | null {
  try {
    return storage?.getItem(SESSION_KEY) ?? null;
  } catch {
    return null;
  }
}

type Saved = { photos: SavedPhoto[]; instruction: string };

function parseSaved(text: string | null | undefined): Saved {
  const none: Saved = { photos: [], instruction: "" };
  try {
    if (!text) return none;
    const data: unknown = JSON.parse(text);
    if (typeof data !== "object" || data === null) return none;
    const { v, photos, instruction } = data as { v?: unknown; photos?: unknown; instruction?: unknown };
    if (v !== 1 || !Array.isArray(photos)) return none;
    const seen = new Set<string>();
    return {
      instruction: typeof instruction === "string" ? instruction.slice(0, INSTRUCTION_MAX) : "",
      photos: photos.filter((s: unknown): s is SavedPhoto => {
        if (typeof s !== "object" || s === null) return false;
        const { key, filename, size } = s as Partial<SavedPhoto>;
        if (typeof key !== "string" || !key || seen.has(key)) return false;
        if (typeof filename !== "string" || typeof size !== "number") return false;
        seen.add(key);
        return true;
      }),
    };
  } catch {
    return none;
  }
}

const str = (value: unknown) => (typeof value === "string" && value ? value : undefined);

function restorePhoto(saved: SavedPhoto): Photo {
  const artifactId = str(saved.artifactId);
  const inspectOutput = asInspectOutput(saved.inspect);
  const inspectError = str(saved.inspectError);
  const autoRunId = str(saved.autoRunId);
  const m = saved.manual;
  const exportRunId = str(m?.exportRunId);
  const manual: ManualState | undefined =
    m && isParams(m.base) && isParams(m.params)
      ? {
          base: completeParams(m.base),
          params: completeParams(m.params),
          preview: str(m.previewId)
            ? { id: m.previewId!, params: completeParams(isParams(m.previewParams) ? m.previewParams : m.params) }
            : undefined,
          export: exportRunId
            ? {
                status: "running",
                runId: exportRunId,
                params: completeParams(isParams(m.exportParams) ? m.exportParams : m.params),
                progress: null,
                cancelling: false,
              }
            : undefined,
        }
      : undefined;

  return {
    key: saved.key,
    filename: saved.filename,
    size: saved.size,
    artifactId,
    upload: artifactId
      ? { status: "done" }
      : { status: "failed", error: "The upload was interrupted. Add the file again to continue." },
    inspect: inspectOutput
      ? { status: "done", output: inspectOutput }
      : inspectError
        ? { status: "failed", error: inspectError }
        : artifactId
          ? { status: "running", runId: str(saved.inspectRunId) }
          : { status: "idle" },
    auto: autoRunId
      ? { status: "restoring", runId: autoRunId }
      : typeof saved.queued === "number" && artifactId
        ? { status: "queued", order: saved.queued }
        : { status: "idle" },
    manual,
    instruction: str(saved.instruction)?.slice(0, INSTRUCTION_MAX),
  };
}
