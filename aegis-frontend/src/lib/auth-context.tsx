import React from "react";

type AuthUser = {
  name: string;
  username: string;
  role?: string | null;
};

type AuthState = {
  user: AuthUser | null;
  isHydrated: boolean;
  login: (username: string, password: string) => void;
  logout: () => void;
};

const AUTH_STORAGE_KEY = "aegis_auth_user";

const AuthContext = React.createContext<AuthState | null>(null);

// Turns a raw username like "jane.doe" or "jane_doe" into a display name
// like "Jane Doe". Purely cosmetic — there is no real identity behind it.
function toDisplayName(username: string): string {
  const trimmed = username.trim();
  if (!trimmed) return "User";
  return trimmed
    .split(/[\s._-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = React.useState<AuthUser | null>(null);
  const [isHydrated, setIsHydrated] = React.useState(false);

  React.useEffect(() => {
    if (typeof window === "undefined") {
      setIsHydrated(true);
      return;
    }
    try {
      const stored = window.localStorage.getItem(AUTH_STORAGE_KEY);
      if (stored) {
        setUser(JSON.parse(stored));
      }
    } catch {
      // Ignore invalid stored state
    } finally {
      setIsHydrated(true);
    }
  }, []);

  // Demo authentication: accept any non-empty username and password and
  // assign a persistent random demo role on login. The role is stored
  // with the user in localStorage so it survives refresh but does not
  // change on every navigation.
  const DEMO_ROLES = [
    "Risk Validator",
    "Model Risk Analyst",
    "Credit Risk Manager",
    "Model Governance Analyst",
  ];

  const login = React.useCallback((username: string, password: string) => {
    const uname = username.trim();
    // Accept any non-empty username and password for demo; do not
    // validate against hardcoded credentials.
    if (!uname || !password) return;

    // If a stored user already exists, preserve its role (don't re-randomize).
    let assignedRole: string | null = null;
    try {
      const stored = typeof window !== "undefined" ? window.localStorage.getItem(AUTH_STORAGE_KEY) : null;
      if (stored) {
        const parsed = JSON.parse(stored) as AuthUser | null;
        if (parsed && parsed.role) assignedRole = parsed.role;
      }
    } catch {
      // ignore
    }

    if (!assignedRole) {
      // Pick a random role once at login.
      assignedRole = DEMO_ROLES[Math.floor(Math.random() * DEMO_ROLES.length)];
    }

    const nextUser: AuthUser = { name: toDisplayName(uname), username: uname, role: assignedRole };
    setUser(nextUser);
    if (typeof window !== "undefined") {
      try {
        window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(nextUser));
      } catch {
        // Ignore storage failures — session still works in memory.
      }
    }
  }, []);

  const logout = React.useCallback(() => {
    setUser(null);
    if (typeof window !== "undefined") {
      try {
        window.localStorage.removeItem(AUTH_STORAGE_KEY);
      } catch {
        // Ignore storage failures
      }
    }
  }, []);

  const value = React.useMemo(
    () => ({ user, isHydrated, login, logout }),
    [user, isHydrated, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = React.useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
