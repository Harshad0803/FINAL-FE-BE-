// Wires the .dark class (styles.css already defines the full color scheme
// under `.dark { ... }` and `@custom-variant dark (&:is(.dark *))`) — this
// module only handles reading/writing the preference, never the palette.
const THEME_KEY = "aegis_theme";

export type Theme = "light" | "dark";

export function getStoredTheme(): Theme | null {
  if (typeof window === "undefined") return null;
  const stored = window.localStorage.getItem(THEME_KEY);
  return stored === "dark" || stored === "light" ? stored : null;
}

export function getPreferredTheme(): Theme {
  const stored = getStoredTheme();
  if (stored) return stored;
  // Default to light when no explicit user preference is stored. We avoid
  // inferring from system dark-mode here so the design tokens' light
  // palette (Figma reference) is visible by default in the POC.
  return "light";
}

export function applyTheme(theme: Theme): void {
  if (typeof document === "undefined") return;
  document.documentElement.classList.toggle("dark", theme === "dark");
}

export function setStoredTheme(theme: Theme): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(THEME_KEY, theme);
  applyTheme(theme);
}
