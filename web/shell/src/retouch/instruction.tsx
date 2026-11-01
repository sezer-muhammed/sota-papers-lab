import { useId } from "react";
import { Quote } from "lucide-react";
import { cn, text } from "@spl/ui";
import { INSTRUCTION_MAX } from "./options";

/** Examples of what the boxes take, for their placeholders. */
export const INSTRUCTION_EXAMPLES = {
  photo: "e.g. black and white, the best B&W for this scene",
  fallback: "e.g. high contrast and saturated · optimum colour distribution · natural skin tones · cinematic",
} as const;

/* A few words on the edit wanted, sent to the retouch as its instruction.
   Optional and empty by default; saved with the session. */
export function InstructionField({
  className,
  hint,
  label,
  labelHidden = false,
  onChange,
  placeholder,
  rows = 2,
  value,
}: {
  className?: string;
  /** A line under the box (what an empty box does). */
  hint?: string;
  label: string;
  /** The label for assistive tech only (a card where the context is clear). */
  labelHidden?: boolean;
  onChange: (value: string) => void;
  placeholder: string;
  rows?: number;
  value: string;
}) {
  const id = useId();
  const hintId = useId();
  const near = value.length > INSTRUCTION_MAX - 60;

  return (
    <div className={cn("min-w-0 space-y-1", className)}>
      <label htmlFor={id} className={labelHidden ? "sr-only" : cn(text.eyebrowSm, "block")}>
        {label}
      </label>
      <textarea
        id={id}
        value={value}
        rows={rows}
        maxLength={INSTRUCTION_MAX}
        placeholder={placeholder}
        spellCheck
        aria-describedby={hint || near ? hintId : undefined}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          // The photo page binds the arrow keys to the previous and next photo.
          event.stopPropagation();
        }}
        className="block w-full resize-y rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 py-1.5 text-[13px] leading-5 text-[var(--ds-gray-1000)] outline-none placeholder:text-[var(--ds-gray-700)] focus-visible:shadow-[var(--ds-focus-ring)]"
      />
      {hint || near ? (
        <p id={hintId} className={cn(text.meta, "flex justify-between gap-3 normal-case")}>
          <span>{hint}</span>
          {near ? (
            <span className="shrink-0 tabular-nums">
              {value.length}/{INSTRUCTION_MAX}
            </span>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}

/** The instruction a finished run followed. */
export function UsedInstruction({ className, instruction }: { className?: string; instruction?: string }) {
  if (!instruction) return null;
  return (
    <p className={cn(text.caption, "flex min-w-0 items-start gap-1.5", className)} title={instruction}>
      <Quote aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[var(--ds-gray-700)]" />
      <span className="min-w-0">
        <span className="sr-only">Edit asked for: </span>
        <span className="line-clamp-3 break-words text-[var(--ds-gray-1000)]">{instruction}</span>
      </span>
    </p>
  );
}
