/* Theme preference: "system" follows prefers-color-scheme; "light"/"dark"
   set <html data-theme>, which tokens.css honours. Stored per browser. */
export type ThemePreference = "system" | "light" | "dark";

const KEY = "spl-theme";

export function readTheme(): ThemePreference {
  try {
    const value = localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(preference: ThemePreference) {
  const root = document.documentElement;
  if (preference === "system") delete root.dataset.theme;
  else root.dataset.theme = preference;
  try {
    if (preference === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, preference);
  } catch {
    /* private mode: the choice just isn't remembered */
  }
}
