import { useState } from "react";
import { Monitor, Moon, Sun } from "lucide-react";
import { BrandLogo, StatusSignal, applyTheme, cn, readTheme, type ThemePreference } from "@spl/ui";
import type { Route } from "../app";

const nav: { route: Route; label: string }[] = [
  { route: "overview", label: "Overview" },
  { route: "ops", label: "Ops" },
  { route: "runs", label: "Runs" },
];

const nextTheme: Record<ThemePreference, ThemePreference> = {
  system: "light",
  light: "dark",
  dark: "system",
};

const themeIcon = { system: Monitor, light: Sun, dark: Moon };

export function SiteHeader({
  checking,
  online,
  route,
}: {
  checking: boolean;
  online: boolean;
  route: Route;
}) {
  const [theme, setTheme] = useState(readTheme);
  const ThemeIcon = themeIcon[theme];

  return (
    <header className="depth-surface sticky top-3 z-30 mx-auto flex min-h-14 w-full max-w-[var(--layout-max-w)] flex-wrap items-center gap-x-3 gap-y-2 rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-3 py-2">
      <a className="flex min-w-0 items-center gap-3" href="#/">
        <BrandLogo />
        <span className="hidden truncate border-l border-[var(--ds-gray-alpha-400)] pl-3 text-[13px] text-[var(--ds-gray-700)] sm:block">
          GPU research host
        </span>
      </a>
      <nav aria-label="Main" className="ml-auto flex items-center gap-1">
        {nav.map((item) => (
          <a
            key={item.route}
            href={`#/${item.route === "overview" ? "" : item.route}`}
            aria-current={route === item.route ? "page" : undefined}
            className={cn(
              "rounded-[6px] px-2.5 py-1.5 text-[13px] font-medium text-[var(--ds-gray-900)] transition-colors hover:bg-[var(--ds-gray-100)] hover:text-[var(--ds-gray-1000)]",
              route === item.route && "bg-[var(--ds-gray-100)] text-[var(--ds-gray-1000)]",
            )}
          >
            {item.label}
          </a>
        ))}
      </nav>
      <StatusSignal
        variant="pill"
        pulse={online}
        color={
          checking ? "var(--ds-gray-700)" : online ? "var(--ds-green-700)" : "var(--ds-red-700)"
        }
      >
        {checking ? "Connecting" : online ? "Host online" : "Host offline"}
      </StatusSignal>
      <button
        type="button"
        aria-label={`Theme: ${theme}. Switch to ${nextTheme[theme]}`}
        title={`Theme: ${theme}`}
        className="grid h-8 w-8 place-items-center rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] text-[var(--ds-gray-900)] outline-none transition-colors hover:bg-[var(--ds-gray-100)] focus-visible:shadow-[var(--ds-focus-ring)]"
        onClick={() => {
          const next = nextTheme[theme];
          applyTheme(next);
          setTheme(next);
        }}
      >
        <ThemeIcon aria-hidden="true" className="h-4 w-4" />
      </button>
    </header>
  );
}
