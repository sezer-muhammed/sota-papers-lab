import { useSyncExternalStore } from "react";
import { MIXER_BANDS, MIXER_KINDS, type MixerBand, type MixerKind } from "./sections";

/* The Develop panel's view habits, kept per browser: which sections are
   open and how the Color Mixer is shown. One store, so the Auto and Adjust
   views (and any two panels on screen) agree. */

export type DevelopPrefs = {
  open: string[];
  /** HSL: one property for all eight bands. Color: all three for one band. */
  mixer: "hsl" | "color";
  kind: MixerKind;
  band: MixerBand;
};

const KEY = "spl-retouch-develop";

const DEFAULTS: DevelopPrefs = {
  open: ["basic", "toneCurve", "colorMixer"],
  mixer: "hsl",
  kind: "hue",
  band: "red",
};

function load(): DevelopPrefs {
  try {
    const saved = JSON.parse(window.localStorage.getItem(KEY) ?? "null") as Partial<DevelopPrefs> | null;
    if (!saved || typeof saved !== "object") return DEFAULTS;
    return {
      open: Array.isArray(saved.open) ? saved.open.filter((id): id is string => typeof id === "string") : DEFAULTS.open,
      mixer: saved.mixer === "color" ? "color" : "hsl",
      kind: MIXER_KINDS.find((k) => k.kind === saved.kind)?.kind ?? DEFAULTS.kind,
      band: MIXER_BANDS.find((b) => b.band === saved.band)?.band ?? DEFAULTS.band,
    };
  } catch {
    return DEFAULTS;
  }
}

let current: DevelopPrefs | undefined;
const listeners = new Set<() => void>();

const snapshot = () => (current ??= load());

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function updatePrefs(patch: Partial<DevelopPrefs>) {
  current = { ...snapshot(), ...patch };
  try {
    window.localStorage.setItem(KEY, JSON.stringify(current));
  } catch {
    /* private mode: kept for this page only */
  }
  for (const listener of listeners) listener();
}

export function useDevelopPrefs(): DevelopPrefs {
  return useSyncExternalStore(subscribe, snapshot, () => DEFAULTS);
}

/** Opens or closes one section. */
export function setSectionOpen(id: string, open: boolean) {
  const ids = snapshot().open.filter((other) => other !== id);
  updatePrefs({ open: open ? [...ids, id] : ids });
}
