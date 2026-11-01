import { useSyncExternalStore } from "react";
import type { HueBand } from "../scopes/compute";
import { BANDS, MIXER_PROPERTIES, SECTIONS, type MixerProperty, type SectionId } from "./layout";

/* How the Develop panels are laid out is a viewer's habit, kept per browser:
   which sections are open, and the Color Mixer's mode, property and band.
   One store, so the read-only and the editing views agree. */

export type MixerMode = "hsl" | "color";

export type DevelopPrefs = {
  open: Record<SectionId, boolean>;
  mixer: MixerMode;
  property: MixerProperty;
  band: HueBand;
};

const PREFS_KEY = "spl-retouch-develop";

const DEFAULTS: DevelopPrefs = {
  open: { basic: true, toneCurve: true, colorMixer: false, blackWhite: true, detail: false, lens: false, calibration: false },
  mixer: "hsl",
  property: "hue",
  band: "red",
};

function load(): DevelopPrefs {
  try {
    const saved = JSON.parse(window.localStorage.getItem(PREFS_KEY) ?? "null") as Partial<DevelopPrefs> | null;
    if (!saved || typeof saved !== "object") return DEFAULTS;
    const open = { ...DEFAULTS.open };
    for (const section of SECTIONS) {
      const value = saved.open?.[section.id];
      if (typeof value === "boolean") open[section.id] = value;
    }
    return {
      open,
      mixer: saved.mixer === "color" ? "color" : "hsl",
      property: MIXER_PROPERTIES.some((p) => p.value === saved.property) ? saved.property! : DEFAULTS.property,
      band: BANDS.some((b) => b.name === saved.band) ? saved.band! : DEFAULTS.band,
    };
  } catch {
    return DEFAULTS;
  }
}

let current: DevelopPrefs | undefined;
const listeners = new Set<() => void>();

function snapshot(): DevelopPrefs {
  current ??= load();
  return current;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function setDevelopPrefs(change: (prefs: DevelopPrefs) => DevelopPrefs) {
  current = change(snapshot());
  try {
    window.localStorage.setItem(PREFS_KEY, JSON.stringify(current));
  } catch {
    /* private mode: kept for this page only */
  }
  for (const listener of listeners) listener();
}

export function useDevelopPrefs(): DevelopPrefs {
  return useSyncExternalStore(subscribe, snapshot, () => DEFAULTS);
}

export function toggleSection(id: SectionId) {
  setDevelopPrefs((prefs) => ({ ...prefs, open: { ...prefs.open, [id]: !prefs.open[id] } }));
}
