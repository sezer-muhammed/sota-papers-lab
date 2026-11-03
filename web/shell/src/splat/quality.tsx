import { useEffect, useState } from "react";
import { Download, Eraser, Repeat, Square, TriangleAlert } from "lucide-react";
import { Badge, Button, ButtonLink, cn, text } from "@spl/ui";
import { client, fetchArtifact, hasToken } from "../api";
import { downloadArtifact } from "../retouch/artifact";
import {
  RATING_NOTE,
  captureFacts,
  floaterSummary,
  longerPreset,
  verdictHeadline,
  type CaptureReport,
  type Rating,
  type ResultReport,
} from "./reconstruct";

/* The #/3d run panel's report cards: how the capture went (shown as soon as
   the cameras are in, before training) and how good the result is on frames
   it never saw, with the two follow-up actions. */

const VERDICT_TONE = { good: "green", ok: "amber", poor: "red" } as const;
const RATING_TONE = { good: "green", ok: "amber", weak: "red" } as const;

function Facts({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="grid grid-cols-3 gap-x-4 gap-y-1.5">
      {rows.map(([k, v]) => (
        <div key={k} className="min-w-0">
          <dt className={text.meta}>{k}</dt>
          <dd className="truncate font-mono text-[13px] text-[var(--ds-gray-1000)]">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Capture report: verdict, numbers, problems and tips. While the run is
    still going and the capture is poor, it says so and offers to stop. */
export function CaptureCard({ capture, going, onStop }: { capture: CaptureReport; going: boolean; onStop?: () => void }) {
  const verdict = (capture.verdict in VERDICT_TONE ? capture.verdict : "ok") as keyof typeof VERDICT_TONE;
  const warn = verdict === "poor" && going;
  return (
    <section
      aria-label="Capture report"
      role={warn ? "alert" : undefined}
      className={cn(
        "space-y-3 rounded-[8px] border p-3",
        warn ? "border-[var(--ds-red-400)] bg-[var(--ds-red-100)]" : "border-[var(--ds-gray-alpha-400)]",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="flex items-center gap-1.5 text-[14px] font-medium text-[var(--ds-gray-1000)]">
          {warn ? <TriangleAlert aria-hidden="true" className="h-4 w-4 shrink-0 text-[var(--ds-red-900)]" /> : null}
          Capture
        </p>
        <Badge tone={VERDICT_TONE[verdict]}>{verdict}</Badge>
      </div>
      <p className={text.caption}>
        {verdictHeadline(verdict)}
        {warn ? " Training continues; you can stop now and record again with the tips below." : ""}
      </p>
      <Facts rows={captureFacts(capture)} />
      {capture.problems.length ? (
        <ul className="list-disc space-y-1 pl-4 text-[13px] text-[var(--ds-gray-1000)]">
          {capture.problems.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      ) : null}
      {capture.tips.length ? (
        <div className="space-y-1">
          <p className={text.meta}>Next time</p>
          <ul className="list-disc space-y-1 pl-4 text-[13px] text-[var(--ds-gray-900)]">
            {capture.tips.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {warn && onStop ? (
        <Button size="sm" icon={Square} onClick={onStop}>
          Stop the run
        </Button>
      ) : null}
    </section>
  );
}

/** An artifact as an object URL (revoked on change). */
function useArtifactUrl(id: string | undefined) {
  const [url, setUrl] = useState<string>();
  useEffect(() => {
    setUrl(undefined);
    if (!id) return;
    const abort = new AbortController();
    let made: string | undefined;
    fetchArtifact(id, abort.signal)
      .then((blob) => {
        made = URL.createObjectURL(blob);
        setUrl(made);
      })
      .catch(() => undefined);
    return () => {
      abort.abort();
      if (made) URL.revokeObjectURL(made);
    };
  }, [id]);
  return url;
}

/** "clip.mp4#120" → "frame 120"; a photo path → its file name. */
const viewLabel = (name: string) => (name.includes("#") ? `frame ${name.split("#").pop()}` : name.split(/[\\/]/).pop());

function WorstView({ name, psnr, thumb }: { name: string; psnr: number; thumb?: string }) {
  const url = useArtifactUrl(thumb);
  return (
    <figure className="min-w-0 space-y-1">
      <div className="aspect-[3/1] overflow-hidden rounded-[6px] bg-[var(--ds-background-200)]">
        {url ? <img src={url} alt={`Render (left) and photo (right) of ${name}`} className="h-full w-full object-cover" /> : null}
      </div>
      <figcaption className={cn(text.meta, "flex justify-between gap-2")}>
        <span className="truncate">{viewLabel(name)}</span>
        <span className="shrink-0 font-mono">{psnr.toFixed(1)} dB</span>
      </figcaption>
    </figure>
  );
}

function CleanDownload({ id }: { id: string }) {
  const name = "scene-clean.spz";
  return (
    <ButtonLink
      size="sm"
      icon={Download}
      href={client.artifactUrl(id, { download: name })}
      download={name}
      onClick={(event) => {
        if (!hasToken()) return;
        event.preventDefault();
        void downloadArtifact(id, name);
      }}
    >
      Clean .spz
    </ButtonLink>
  );
}

/** Result report: rating from held-out PSNR / SSIM, the worst views, and
    the actions (train longer, show without floaters). */
export function ResultCard({
  quality,
  clean,
  onShowClean,
  onTrainLonger,
  error,
}: {
  quality: ResultReport;
  clean: boolean;
  onShowClean?: (on: boolean) => void;
  onTrainLonger?: () => void;
  error?: string;
}) {
  const rating = quality.rating as Rating | undefined;
  const next = longerPreset(quality.preset);
  const f = quality.floaters;
  const [starting, setStarting] = useState(false);
  return (
    <section aria-label="Result report" className="space-y-3 rounded-[8px] border border-[var(--ds-gray-alpha-400)] p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-[14px] font-medium text-[var(--ds-gray-1000)]">Result</p>
        {rating ? <Badge tone={RATING_TONE[rating] ?? "gray"}>{rating}</Badge> : null}
      </div>
      {rating && quality.psnr !== undefined && quality.ssim !== undefined ? (
        <>
          <p className={text.caption}>{RATING_NOTE[rating]}</p>
          <Facts
            rows={[
              ["held-out PSNR", `${quality.psnr.toFixed(2)} dB`],
              ["SSIM", quality.ssim.toFixed(3)],
              ["frames", `${quality.held_out} held out`],
            ]}
          />
        </>
      ) : (
        <p className={text.caption}>Too few frames to hold some out, so there is no quality score.</p>
      )}
      {quality.worst.length ? (
        <div className="space-y-1.5">
          <p className={text.meta}>Weakest views: render (left), photo (right)</p>
          <div className="grid gap-2">
            {quality.worst.map((v) => (
              <WorstView key={v.name} name={v.name} psnr={v.psnr} thumb={v.thumb} />
            ))}
          </div>
        </div>
      ) : null}
      {f && f.after < f.before ? (
        <p className={text.caption}>
          {clean ? "Showing without floaters: removed " : "Floaters found: "}
          {floaterSummary(f)}.
        </p>
      ) : null}
      <div className="flex flex-wrap gap-2">
        {onTrainLonger && next ? (
          <Button
            size="sm"
            icon={Repeat}
            disabled={starting}
            onClick={() => {
              setStarting(true);
              Promise.resolve(onTrainLonger()).finally(() => setStarting(false));
            }}
          >
            Train longer ({next})
          </Button>
        ) : null}
        {onShowClean && f?.spz ? (
          <Button size="sm" icon={Eraser} onClick={() => onShowClean(!clean)}>
            {clean ? "Show with floaters" : "Remove floaters"}
          </Button>
        ) : null}
        {clean && f?.spz ? <CleanDownload id={f.spz} /> : null}
      </div>
      {onTrainLonger && next ? (
        <p className={text.meta}>Train longer reruns this input at the {next} preset; this run stays as it is.</p>
      ) : null}
      {error ? (
        <p role="alert" className={text.caption}>
          {error}
        </p>
      ) : null}
    </section>
  );
}
