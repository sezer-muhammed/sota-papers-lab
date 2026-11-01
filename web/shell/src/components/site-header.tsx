import { useState } from "react";
import { Monitor, Moon, Sun } from "lucide-react";
import type { Health } from "@spl/sdk";
import { BrandLogo, StatusSignal, applyTheme, cn, readTheme, type ThemePreference } from "@spl/ui";
import type { Loaded } from "../api";
import type { Route } from "../app";
import { hostState, type HostState } from "../lib/env";
import { sameHashScroll } from "../lib/scroll";

const nav: { page: Route["page"]; href: string; label: string }[] = [
  { page: "home", href: "#/", label: "Home" },
  { page: "retouch", href: "#/retouch", label: "Retouch" },
  { page: "ops", href: "#/ops", label: "Ops" },
  { page: "runs", href: "#/runs", label: "Runs" },
];

// Phones get the short word; the dot never carries the state alone.
const signals: Record<HostState, { long: string; short: string; color: string; pulse: boolean }> = {
  connecting: { long: "Connecting", short: "Checking", color: "var(--ds-gray-700)", pulse: false },
  online: { long: "Host online", short: "Online", color: "var(--ds-green-700)", pulse: true },
  "no-gpu": { long: "No GPU", short: "No GPU", color: "var(--ds-amber-700)", pulse: false },
  offline: { long: "Host offline", short: "Offline", color: "var(--ds-red-700)", pulse: false },
  locked: { long: "Token required", short: "Locked", color: "var(--ds-amber-700)", pulse: false },
};

const nextTheme: Record<ThemePreference, ThemePreference> = {
  system: "light",
  light: "dark",
  dark: "system",
};

const themeIcon = { system: Monitor, light: Sun, dark: Moon };

export function SiteHeader({ health, route }: { health: Loaded<Health>; route: Route }) {
  const [theme, setTheme] = useState(readTheme);
  const ThemeIcon = themeIcon[theme];
  const signal = signals[hostState(health)];

  return (
    <header className="depth-surface mobile-header-scrolls sticky z-30 mx-auto flex min-h-14 w-full max-w-[var(--layout-max-w)] flex-wrap items-center gap-x-3 gap-y-2 rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-3 py-2 md:top-3">
      <a
        className="flex min-w-0 items-center gap-3 rounded-[6px] outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
        href="#/"
      >
        <BrandLogo />
        <span className="hidden truncate border-l border-[var(--ds-gray-alpha-400)] pl-3 text-[13px] text-[var(--ds-gray-700)] md:block">
          GPU research host
        </span>
      </a>
      <nav aria-label="Main" className="ml-auto flex items-center gap-1">
        {nav.map((item) => (
          <a
            key={item.page}
            href={item.href}
            aria-current={route.page === item.page ? "page" : undefined}
            className={cn(
              "rounded-[6px] px-2.5 py-1.5 text-[13px] font-medium text-[var(--ds-gray-900)] outline-none transition-colors hover:bg-[var(--ds-gray-100)] hover:text-[var(--ds-gray-1000)] focus-visible:shadow-[var(--ds-focus-ring)]",
              route.page === item.page && "bg-[var(--ds-gray-100)] text-[var(--ds-gray-1000)]",
            )}
          >
            {item.label}
          </a>
        ))}
      </nav>
      <a
        href="#/host"
        onClick={sameHashScroll}
        title={signal.long}
        className="rounded-[5px] outline-none focus-visible:shadow-[var(--ds-focus-ring)]"
      >
        <StatusSignal variant="pill" color={signal.color} pulse={signal.pulse}>
          <span className="sm:hidden">{signal.short}</span>
          <span className="hidden sm:inline">{signal.long}</span>
        </StatusSignal>
        {/* No aria-label: the name must start with the word shown at each width
            (the hidden one is display:none, so it is left out of the name). */}
        <span className="sr-only">. Show host details</span>
      </a>
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
