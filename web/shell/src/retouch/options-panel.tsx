import { useId, type ReactNode } from "react";
import { RotateCcw } from "lucide-react";
import { Button, SegmentedControl, Surface, Switch, cn, text } from "@spl/ui";
import { NumberField } from "./number-field";
import { LIMITS, type RetouchOptions } from "./options";

const models = [
  { value: "opus" as const, label: "Opus" },
  { value: "sonnet" as const, label: "Sonnet" },
];

const parallelChoices = Array.from({ length: LIMITS.parallel.max - LIMITS.parallel.min + 1 }, (_, i) => {
  const n = String(LIMITS.parallel.min + i);
  return { value: n, label: n };
});

/* The settings for the next retouches, saved in this browser. */
export function OptionsPanel({
  className,
  onChange,
  onReset,
  options,
}: {
  className?: string;
  onChange: (patch: Partial<RetouchOptions>) => void;
  onReset: () => void;
  options: RetouchOptions;
}) {
  const claude = options.useClaude;
  const qualityId = useId();
  const evalsId = useId();

  return (
    <Surface tone="flat" aria-labelledby="retouch-options-title" className={cn("overflow-hidden", className)}>
      <div className="border-b border-[var(--ds-gray-alpha-400)] px-4 py-3">
        <p className={text.eyebrowSm}>Options</p>
        <h2 id="retouch-options-title" className="mt-0.5 text-[17px] font-semibold leading-6 text-[var(--ds-gray-1000)]">
          How to retouch
        </h2>
        <p className={cn(text.caption, "mt-1")}>Saved in this browser. Changes apply to retouches you start next.</p>
      </div>

      <div className="divide-y divide-[var(--ds-gray-alpha-300)]">
        <Group title="Claude">
          <Switch
            checked={claude}
            onCheckedChange={(useClaude) => onChange({ useClaude })}
            label="Use Claude"
            description="Plans looks, judges the finalists side by side and proposes one refinement. Off: the objective score picks."
          />
          <Row label="Model" disabled={!claude}>
            <SegmentedControl
              label="Claude model"
              value={options.model}
              options={models}
              disabled={!claude}
              onChange={(model) => onChange({ model })}
            />
          </Row>
          <Switch
            checked={options.plan}
            disabled={!claude}
            onCheckedChange={(plan) => onChange({ plan })}
            label="Plan looks"
            description="Up to three looks for the search to start from."
          />
          <Switch
            checked={options.refine}
            disabled={!claude}
            onCheckedChange={(refine) => onChange({ refine })}
            label="Refine"
            description="One tweak of the winner, kept only if it wins a blind A/B."
          />
        </Group>

        <Group title="Export">
          <Switch
            checked={options.jpeg}
            onCheckedChange={(jpeg) => onChange({ jpeg })}
            label="Full-resolution JPEG"
          />
          <Row label="JPEG quality" htmlFor={qualityId} disabled={!options.jpeg}>
            <NumberField
              id={qualityId}
              value={options.jpegQuality}
              min={LIMITS.jpegQuality.min}
              max={LIMITS.jpegQuality.max}
              disabled={!options.jpeg}
              onCommit={(jpegQuality) => onChange({ jpegQuality })}
              className="w-20"
            />
          </Row>
          <Switch
            checked={options.xmp}
            onCheckedChange={(xmp) => onChange({ xmp })}
            label="XMP sidecar"
            description="The ten sliders for Lightroom or Camera Raw, next to the RAW."
          />
          <Switch
            checked={options.tiff16}
            onCheckedChange={(tiff16) => onChange({ tiff16 })}
            label="16-bit TIFF"
            description="Full resolution, about 6 bytes a pixel."
          />
        </Group>

        <Group title="Queue">
          <Row label="Photos in parallel">
            <SegmentedControl
              label="Photos in parallel"
              value={String(options.parallel)}
              options={parallelChoices}
              onChange={(value) => onChange({ parallel: Number(value) })}
            />
          </Row>
        </Group>

        <details className="group px-4 py-3">
          <summary className="cursor-pointer list-none rounded-[4px] outline-none focus-visible:shadow-[var(--ds-focus-ring)] [&::-webkit-details-marker]:hidden">
            <span className={cn(text.eyebrowSm, "inline-flex items-center gap-1.5")}>
              <span aria-hidden="true" className="inline-block transition-transform group-open:rotate-90">
                ›
              </span>
              Advanced
            </span>
          </summary>
          <div className="mt-3 space-y-2">
            <Row label="Evaluations per look" htmlFor={evalsId}>
              <NumberField
                id={evalsId}
                value={options.evalsPerLook}
                min={LIMITS.evalsPerLook.min}
                max={LIMITS.evalsPerLook.max}
                step={LIMITS.evalsPerLook.step}
                onCommit={(evalsPerLook) => onChange({ evalsPerLook })}
              />
            </Row>
            <p className={text.caption}>The search budget for each look. More finds slightly better edits, slower.</p>
          </div>
        </details>

        <div className="px-4 py-3">
          <Button size="sm" variant="ghost" icon={RotateCcw} onClick={onReset}>
            Reset to defaults
          </Button>
        </div>
      </div>
    </Surface>
  );
}

function Group({ children, title }: { children: ReactNode; title: string }) {
  const id = useId();
  return (
    <div role="group" aria-labelledby={id} className="space-y-3 px-4 py-3">
      <p id={id} className={text.eyebrowSm}>
        {title}
      </p>
      {children}
    </div>
  );
}

/* A label on the left, a control on the right. `htmlFor` for inputs; radio
   groups carry their own accessible name, so their row label is plain text. */
function Row({
  children,
  disabled = false,
  htmlFor,
  label,
}: {
  children: ReactNode;
  disabled?: boolean;
  htmlFor?: string;
  label: string;
}) {
  const labelClass = cn("text-[14px] font-medium leading-6 text-[var(--ds-gray-1000)]", disabled && "opacity-50");
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
      {htmlFor ? (
        <label htmlFor={htmlFor} className={labelClass}>
          {label}
        </label>
      ) : (
        <span aria-hidden="true" className={labelClass}>
          {label}
        </span>
      )}
      {children}
    </div>
  );
}
