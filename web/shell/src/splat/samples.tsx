import { Button, SectionHeader, Surface, text } from "@spl/ui";
import { SAMPLE_SCENES, type Preset, type SampleScene } from "./reconstruct";

const thumb = (key: string) => `${import.meta.env.BASE_URL}samples/${key}.webp`;

/* "Try a sample scene": photo folders the GPU host already has. One click
   starts image_to_3d.reconstruct on the folder, Fast or Quality. */
export function SampleScenes({ busy, onStart }: { busy: boolean; onStart: (scene: SampleScene, preset: Preset) => void }) {
  return (
    <Surface aria-labelledby="splat-samples-title">
      <SectionHeader titleId="splat-samples-title" eyebrow="No photos yet?" title="Try a sample scene" />
      <ul className="grid gap-4 p-4 sm:grid-cols-2 xl:grid-cols-3">
        {SAMPLE_SCENES.map((s) => (
          <li
            key={s.key}
            className="overflow-hidden rounded-[8px] border border-[var(--ds-gray-alpha-400)]"
          >
            <img
              src={thumb(s.key)}
              alt=""
              width={320}
              height={200}
              loading="lazy"
              className="aspect-[8/5] w-full bg-[var(--ds-gray-100)] object-cover"
            />
            <div className="space-y-3 p-3">
              <div>
                <p className="text-[14px] font-medium text-[var(--ds-gray-1000)]">{s.title}</p>
                <p className={text.caption}>
                  {s.source} · {s.photos} photos
                </p>
              </div>
              <div className="flex gap-2">
                <Button size="sm" disabled={busy} aria-label={`${s.title}, Fast`} onClick={() => onStart(s, "fast")}>
                  Fast
                </Button>
                <Button
                  size="sm"
                  variant="primary"
                  disabled={busy}
                  aria-label={`${s.title}, Quality`}
                  onClick={() => onStart(s, "quality")}
                >
                  Quality
                </Button>
              </div>
            </div>
          </li>
        ))}
      </ul>
      <p className={`${text.caption} px-4 pb-4`}>
        Public research datasets, read from this host's dataset folder (research use only).
      </p>
    </Surface>
  );
}

const TIPS = [
  "Walk slowly around the subject, or through the space, at a steady pace.",
  "Keep lots of overlap: each part of the scene should appear in many frames.",
  "Avoid motion blur and fast turns; pan gently and keep the phone level.",
  "Lock or keep the exposure steady; avoid moving in and out of bright light.",
  "Film 30 to 60 seconds; for photos, 50 to 200 shots.",
  "Include texture. Mirrors, glass, shiny surfaces and plain walls reconstruct poorly.",
];

export function CaptureTips() {
  return (
    <Surface aria-labelledby="splat-tips-title">
      <SectionHeader titleId="splat-tips-title" eyebrow="Help" title="How to film a good capture" />
      <ul className="list-disc space-y-2 p-4 pl-8 text-[13px] text-[var(--ds-gray-900)]">
        {TIPS.map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>
    </Surface>
  );
}
