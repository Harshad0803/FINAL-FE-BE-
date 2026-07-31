import React from "react";

type AuthUser = {
  name: string;
  username: string;
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

  // Dummy authentication — there is no backend check here. Any non-empty
  // username/password combination "logs in" the person and stores the
  // display name derived from their username so it can be shown in the
  // header. This exists purely to give the app a login screen, not to
  // secure anything.
  const login = React.useCallback((username: string, _password: string) => {
    const nextUser: AuthUser = { name: toDisplayName(username), username: username.trim() };
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
