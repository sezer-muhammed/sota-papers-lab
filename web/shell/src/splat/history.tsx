import { useCallback, useEffect, useState } from "react";
import { Box, Film, Images, Pencil, RefreshCw, RotateCw, Trash2 } from "lucide-react";
import { isFinished, type RunRecord } from "@spl/sdk";
import { Button, SectionHeader, SegmentedControl, Surface, Switch, cn, text } from "@spl/ui";
import { client, describeError } from "../api";
import { RunStatusSignal } from "../components/run-status";
import { useArtifactSrc } from "../retouch/artifact";
import { framesOf, presetOf } from "./progress";
import { RECONSTRUCT_OP, SAMPLE_SCENES, type Preset } from "./reconstruct";

/* The runs gallery on #/3d: past reconstructions with a thumbnail, date,
   input, preset, frames, held-out PSNR and status. Open, rename (names
   live in this browser), delete (the host drops the run and the files only
   it used) and re-run the same input with other settings. */

const NAMES_KEY = "spl.3d.names";
const LIMIT = 60;

function readNames(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(NAMES_KEY) ?? "{}") as Record<string, string>;
  } catch {
    return {};
  }
}

/** The host's image_to_3d.reconstruct runs, newest first, refreshed while any is going. */
export function useRuns(op = RECONSTRUCT_OP) {
  const [runs, setRuns] = useState<RunRecord[]>();
  const [error, setError] = useState<string>();
  const reload = useCallback(() => {
    client.runs({ op, limit: LIMIT }).then(
      (r) => {
        setRuns(r);
        setError(undefined);
      },
      (e: unknown) => setError(describeError(e)),
    );
  }, [op]);
  useEffect(reload, [reload]);
  const going = runs?.some((r) => !isFinished(r.status));
  useEffect(() => {
    if (!going) return;
    const t = setInterval(reload, 3000);
    return () => clearInterval(t);
  }, [going, reload]);
  return { runs, error, reload, setRuns };
}

type Input = { images?: string[]; video?: string; dataset?: string; preset?: string; mesh?: boolean; max_frames?: number };
const inputOf = (r: RunRecord) => (r.input ?? {}) as Input;

export function inputKind(r: RunRecord): { label: string; icon: typeof Box } {
  const i = inputOf(r);
  if (i.video) return { label: "Video", icon: Film };
  if (i.dataset) return { label: "Sample", icon: Box };
  return { label: `${i.images?.length ?? 0} photos`, icon: Images };
}

/** Held-out PSNR from the output, when the op measured one. */
export function psnrOf(r: RunRecord): number | undefined {
  const o = (r.output ?? {}) as { eval?: { psnr?: number }; held_out_psnr?: number };
  const v = o.eval?.psnr ?? o.held_out_psnr;
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

export function defaultName(r: RunRecord) {
  const i = inputOf(r);
  const sample = i.dataset ? SAMPLE_SCENES.find((s) => s.dataset === i.dataset) : undefined;
  if (sample) return sample.title;
  return `${inputKind(r).label} scene`;
}

const when = (ms: number) =>
  new Date(ms).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

function Thumb({ run }: { run: RunRecord }) {
  const i = inputOf(run);
  const sample = i.dataset ? SAMPLE_SCENES.find((s) => s.dataset === i.dataset) : undefined;
  const { src } = useArtifactSrc(sample ? undefined : i.images?.[0]);
  const url = sample ? `${import.meta.env.BASE_URL}samples/${sample.key}.webp` : src;
  const Icon = inputKind(run).icon;
  return url ? (
    <img src={url} alt="" loading="lazy" draggable={false} className="aspect-[8/5] w-full bg-[var(--ds-gray-100)] object-cover" />
  ) : (
    <div className="grid aspect-[8/5] w-full place-items-center bg-[var(--ds-gray-100)] text-[var(--ds-gray-700)]">
      <Icon aria-hidden="true" className="h-6 w-6" />
    </div>
  );
}

type Busy = { id: string; what: "rename" | "delete" | "rerun" };

export function RunsGallery({ current }: { current?: string }) {
  const { runs, error, reload, setRuns } = useRuns();
  const [names, setNames] = useState(readNames);
  const [open, setOpen] = useState<Busy>();
  const [problem, setProblem] = useState<string>();

  const saveName = (id: string, name: string) => {
    const next = { ...names };
    if (name.trim()) next[id] = name.trim();
    else delete next[id];
    setNames(next);
    localStorage.setItem(NAMES_KEY, JSON.stringify(next));
  };

  const remove = async (id: string) => {
    setProblem(undefined);
    try {
      await client.deleteRun(id);
      setRuns((r) => r?.filter((x) => x.id !== id));
      saveName(id, "");
      setOpen(undefined);
      if (current === id) location.hash = "#/3d";
    } catch (e) {
      setProblem(describeError(e));
    }
  };

  const rerun = async (run: RunRecord, o: { preset: Preset; mesh: boolean; maxFrames: number }) => {
    setProblem(undefined);
    try {
      const input = { ...inputOf(run), preset: o.preset, mesh: o.mesh, max_frames: o.maxFrames };
      const made = await client.startRun(RECONSTRUCT_OP, input);
      if (names[run.id]) saveName(made.id, `${names[run.id]} (${o.preset})`);
      setOpen(undefined);
      reload();
      location.hash = `#/3d/${encodeURIComponent(made.id)}`;
    } catch (e) {
      setProblem(describeError(e));
    }
  };

  return (
    <Surface aria-labelledby="splat-history-title">
      <SectionHeader
        titleId="splat-history-title"
        eyebrow="History"
        title="Your scenes"
        action={
          <Button size="sm" icon={RefreshCw} onClick={reload}>
            <span className="sr-only">Refresh</span>
          </Button>
        }
      />
      {problem || error ? (
        <p role="alert" className={cn(text.caption, "px-4 pt-3")}>
          {problem ?? error}
        </p>
      ) : null}
      {runs && !runs.length ? <p className={cn(text.caption, "p-4")}>No scenes yet. Drop photos or a video above, or try a sample.</p> : null}
      <ul data-testid="splat-history" className="grid gap-4 p-4 sm:grid-cols-2 xl:grid-cols-4">
        {(runs ?? []).map((run) => {
          const psnr = psnrOf(run);
          const frames = framesOf(run);
          const name = names[run.id] ?? defaultName(run);
          const mine = open?.id === run.id ? open.what : undefined;
          return (
            <li
              key={run.id}
              data-run={run.id}
              className={cn(
                "min-w-0 overflow-hidden rounded-[8px] border",
                current === run.id ? "border-[var(--ds-blue-700)]" : "border-[var(--ds-gray-alpha-400)]",
              )}
            >
              <a href={`#/3d/${encodeURIComponent(run.id)}`} className="block outline-none focus-visible:shadow-[var(--ds-focus-ring)]" aria-label={`Open ${name}`}>
                <Thumb run={run} />
              </a>
              <div className="space-y-2 p-3">
                {mine === "rename" ? (
                  <RenameForm initial={name} onCancel={() => setOpen(undefined)} onSave={(v) => (saveName(run.id, v), setOpen(undefined))} />
                ) : (
                  <a href={`#/3d/${encodeURIComponent(run.id)}`} className="block truncate text-[14px] font-medium text-[var(--ds-gray-1000)] hover:underline">
                    {name}
                  </a>
                )}
                <p className={cn(text.caption, "truncate")}>
                  {when(run.created_at_ms)} · {inputKind(run).label}
                  {presetOf(run) ? ` · ${presetOf(run)}` : ""}
                  {frames ? ` · ${frames} frames` : ""}
                </p>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <RunStatusSignal status={run.status} />
                  {psnr !== undefined ? <span className="font-mono text-[12px] text-[var(--ds-gray-900)]">{psnr.toFixed(2)} dB held-out</span> : null}
                </div>
                {mine === "delete" ? (
                  <div role="group" aria-label="Confirm delete" className="space-y-2 rounded-[6px] bg-[var(--ds-gray-100)] p-2">
                    <p className={text.caption}>Delete this scene and its files on the host? This cannot be undone.</p>
                    <div className="flex gap-2">
                      <Button size="sm" variant="primary" icon={Trash2} onClick={() => void remove(run.id)}>
                        Delete
                      </Button>
                      <Button size="sm" onClick={() => setOpen(undefined)}>
                        Keep
                      </Button>
                    </div>
                  </div>
                ) : null}
                {mine === "rerun" ? <RerunForm run={run} onCancel={() => setOpen(undefined)} onStart={(o) => void rerun(run, o)} /> : null}
                {mine ? null : (
                  <div className="flex flex-wrap gap-1">
                    <Button size="sm" variant="ghost" icon={Pencil} onClick={() => setOpen({ id: run.id, what: "rename" })}>
                      Rename
                    </Button>
                    <Button size="sm" variant="ghost" icon={RotateCw} onClick={() => setOpen({ id: run.id, what: "rerun" })}>
                      Re-run
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      icon={Trash2}
                      disabled={!isFinished(run.status)}
                      title={isFinished(run.status) ? undefined : "Cancel the run first"}
                      onClick={() => setOpen({ id: run.id, what: "delete" })}
                    >
                      Delete
                    </Button>
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </Surface>
  );
}

function RenameForm({ initial, onCancel, onSave }: { initial: string; onCancel: () => void; onSave: (v: string) => void }) {
  const [v, setV] = useState(initial);
  return (
    <form
      className="flex gap-1"
      onSubmit={(e) => {
        e.preventDefault();
        onSave(v);
      }}
    >
      <input
        autoFocus
        aria-label="Scene name"
        value={v}
        maxLength={80}
        onChange={(e) => setV(e.target.value)}
        onKeyDown={(e) => e.key === "Escape" && onCancel()}
        className="h-8 min-w-0 flex-1 rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2 text-[13px] text-[var(--ds-gray-1000)] outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
      />
      <Button size="sm" type="submit">
        Save
      </Button>
    </form>
  );
}

const PRESETS: { value: Preset; label: string }[] = [
  { value: "fast", label: "Fast" },
  { value: "quality", label: "Quality" },
  { value: "big", label: "Big" },
];

function RerunForm({
  onCancel,
  onStart,
  run,
}: {
  run: RunRecord;
  onCancel: () => void;
  onStart: (o: { preset: Preset; mesh: boolean; maxFrames: number }) => void;
}) {
  const i = inputOf(run);
  const [preset, setPreset] = useState<Preset>(((i.preset as Preset) ?? "quality") === "quality" ? "fast" : "quality");
  const [mesh, setMesh] = useState(Boolean(i.mesh));
  const [maxFrames, setMaxFrames] = useState(i.max_frames ?? 200);
  return (
    <div role="group" aria-label="Re-run settings" className="space-y-2 rounded-[6px] bg-[var(--ds-gray-100)] p-2">
      <SegmentedControl label="Preset" options={PRESETS} value={preset} onChange={setPreset} />
      <Switch className="justify-start gap-3" label="Textured mesh" checked={mesh} onCheckedChange={setMesh} />
      <label className="flex items-center gap-2">
        <span className={text.caption}>Max frames</span>
        <input
          type="number"
          min={2}
          max={1000}
          value={maxFrames}
          onChange={(e) => setMaxFrames(Math.max(2, Number(e.target.value) || 2))}
          className="h-8 w-20 rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2 font-mono text-[13px] text-[var(--ds-gray-1000)]"
        />
      </label>
      <div className="flex gap-2">
        <Button size="sm" variant="primary" icon={RotateCw} onClick={() => onStart({ preset, mesh, maxFrames })}>
          Start
        </Button>
        <Button size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
