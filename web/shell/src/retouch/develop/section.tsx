import { useId, type ReactNode } from "react";
import { ChevronRight, Undo2 } from "lucide-react";
import { Button, cn, text } from "@spl/ui";
import { setSectionOpen, useDevelopPrefs } from "./prefs";

/* One Lightroom panel inside Develop: a disclosure header (title, and a
   summary that stays visible when it is closed), an optional Reset for the
   section, and its body. Open or closed is remembered per browser. */
export function DevelopSection({
  children,
  id,
  onReset,
  summary,
  title,
}: {
  children: ReactNode;
  id: string;
  /** Shown while the section differs from the auto edit. */
  onReset?: () => void;
  summary?: string;
  title: string;
}) {
  const open = useDevelopPrefs().open.includes(id);
  const headerId = useId();
  const bodyId = useId();

  return (
    <section aria-labelledby={headerId} className="border-t border-[var(--ds-gray-alpha-400)] first:border-t-0">
      <div className="flex items-center gap-1">
        <h3 className="min-w-0 flex-1">
          <button
            type="button"
            id={headerId}
            aria-expanded={open}
            aria-controls={bodyId}
            onClick={() => setSectionOpen(id, !open)}
            className="-mx-1.5 flex min-h-11 w-[calc(100%+0.75rem)] items-center gap-2 rounded-[6px] px-1.5 text-left outline-none hover:bg-[var(--ds-gray-alpha-100)] focus-visible:shadow-[var(--ds-focus-ring)]"
          >
            <ChevronRight
              aria-hidden="true"
              className={cn(
                "h-4 w-4 shrink-0 text-[var(--ds-gray-700)] transition-transform motion-reduce:transition-none",
                open && "rotate-90",
              )}
            />
            <span className="shrink-0 whitespace-nowrap text-[14px] font-semibold text-[var(--ds-gray-1000)]">
              {title}
            </span>
            {summary ? <span className={cn(text.meta, "ml-auto truncate pl-2")}>{summary}</span> : null}
          </button>
        </h3>
        {onReset ? (
          <Button size="sm" variant="ghost" icon={Undo2} onClick={onReset} title={`Reset ${title} to the auto edit`}>
            Reset<span className="sr-only"> {title}</span>
          </Button>
        ) : null}
      </div>
      <div id={bodyId} hidden={!open} className="pb-4 pt-1">
        {open ? children : null}
      </div>
    </section>
  );
}

/** The mark of a value set away from Lightroom's default; `id` is on its
    text, for a control's aria-describedby. */
export function SetMark({ id, title }: { id?: string; title?: string }) {
  return (
    <span title={title} className="inline-flex shrink-0">
      <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-[var(--ds-blue-700)]" />
      <span id={id} className="sr-only">
        , {title ?? "set away from the default"}
      </span>
    </span>
  );
}

/** A block's heading inside a section ("Tone", "Sharpening"). */
export function BlockTitle({ children, id }: { children: ReactNode; id?: string }) {
  return (
    <p id={id} className={text.eyebrowSm}>
      {children}
    </p>
  );
}
