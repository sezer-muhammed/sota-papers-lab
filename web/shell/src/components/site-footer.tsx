import { cn, text } from "@spl/ui";
import { sameHashScroll } from "../lib/scroll";

const linkClass =
  "rounded-[4px] outline-none hover:text-[var(--ds-gray-1000)] focus-visible:shadow-[var(--ds-focus-ring)]";

/* Minimal sibling of ATSGrade's site footer, on every route. `version` is
   left out until /v1/health answers, and while it fails. */
export function SiteFooter({ version }: { version?: string }) {
  return (
    <footer
      className={cn(
        text.caption,
        "mx-auto w-full max-w-[var(--layout-max-w)] border-t border-[var(--ds-gray-alpha-300)] py-6 text-[var(--ds-gray-700)]",
      )}
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p>
          SOTA Papers Lab
          {version ? <span className="font-mono"> · spl v{version}</span> : null}
        </p>
        <nav aria-label="Footer" className="flex flex-wrap gap-3 font-mono">
          <a className={linkClass} href="#/">
            home
          </a>
          <a className={linkClass} href="#/retouch">
            retouch
          </a>
          <a className={linkClass} href="#/ops">
            ops
          </a>
          <a className={linkClass} href="#/runs">
            runs
          </a>
          <a className={linkClass} href="#/host" onClick={sameHashScroll}>
            host
          </a>
          <a className={linkClass} href="/v1/health" target="_blank" rel="noreferrer">
            /v1/health
          </a>
        </nav>
      </div>
    </footer>
  );
}
