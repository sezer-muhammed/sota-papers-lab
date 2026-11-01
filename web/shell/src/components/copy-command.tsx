import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";
import { Button, cn, text } from "@spl/ui";

type CopyState = "idle" | "copied" | "manual";

function isApple() {
  return /Mac|iPhone|iPad|iPod/i.test(navigator.platform || navigator.userAgent);
}

/* A one-line command with a Copy button. The tailnet URL is plain http, which
   is not a secure context, so the async clipboard API is usually missing there:
   fall back to execCommand on a hidden textarea, then to selecting the text
   and asking for the shortcut. */
export function CopyCommand({
  className,
  command,
  label,
}: {
  className?: string;
  command: string;
  /** What is copied, for screen readers ("Copy {label}"). */
  label: string;
}) {
  const [state, setState] = useState<CopyState>("idle");
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (state === "idle") return;
    const timer = window.setTimeout(() => setState("idle"), state === "copied" ? 1500 : 4000);
    return () => window.clearTimeout(timer);
  }, [state]);

  async function copy(button: HTMLButtonElement) {
    if (window.isSecureContext && navigator.clipboard) {
      try {
        await navigator.clipboard.writeText(command);
        setState("copied");
        return;
      } catch {
        /* permission denied: try the fallback below */
      }
    }
    if (copyWithTextarea(command, button)) {
      setState("copied");
      return;
    }
    const code = codeRef.current;
    const selection = window.getSelection();
    if (code && selection) {
      const range = document.createRange();
      range.selectNodeContents(code);
      selection.removeAllRanges();
      selection.addRange(range);
    }
    setState("manual");
  }

  const apple = isApple();
  const buttonText =
    state === "copied" ? "Copied" : state === "manual" ? (apple ? "Press ⌘C" : "Press Ctrl+C") : "Copy";
  const announcement =
    state === "copied"
      ? "Copied to clipboard"
      : state === "manual"
        ? `Command selected. Press ${apple ? "Command" : "Control"} C to copy.`
        : "";

  return (
    <div
      className={cn(
        "flex min-w-0 items-center gap-2 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] py-1 pl-3 pr-1",
        className,
      )}
    >
      <code
        ref={codeRef}
        className={cn(
          text.meta,
          "min-w-0 flex-1 select-all overflow-x-auto whitespace-nowrap leading-6 text-[var(--ds-gray-1000)]",
        )}
      >
        {command}
      </code>
      <Button
        size="sm"
        variant="ghost"
        icon={state === "copied" ? Check : Copy}
        onClick={(event) => void copy(event.currentTarget)}
      >
        {buttonText}
        <span className="sr-only"> {label}</span>
      </Button>
      <span aria-live="polite" className="sr-only">
        {announcement}
      </span>
    </div>
  );
}

/* execCommand("copy") on a hidden readonly textarea: works without a secure
   context. Returns whether the browser reported a copy. */
function copyWithTextarea(value: string, restoreFocus: HTMLElement) {
  const area = document.createElement("textarea");
  area.value = value;
  area.readOnly = true;
  area.setAttribute("aria-hidden", "true");
  area.style.position = "fixed";
  area.style.top = "0";
  area.style.left = "0";
  area.style.opacity = "0";
  document.body.append(area);
  area.select();
  area.setSelectionRange(0, value.length);
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch {
    copied = false;
  }
  area.remove();
  restoreFocus.focus();
  return copied;
}
