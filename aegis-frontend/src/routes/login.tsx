import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";
import { ShieldCheck, User, Lock } from "lucide-react";
import { useAuth } from "@/lib/auth-context";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [{ title: "Sign in — Aegis Credit" }],
  }),
  component: LoginPage,
});

function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setError("Enter a username and password to continue.");
      return;
    }
    // Accept any non-empty username and password for demo; do not
    // validate against fixed credentials.

    login(username, password);
    navigate({ to: "/" });
  }

  return (
    <div className="relative min-h-screen w-full flex items-center justify-center px-4 py-12 overflow-hidden">
      {/* Subtle decorative background SVG — low contrast, behind the card */}
      <div className="pointer-events-none fixed inset-0 z-0 flex items-center justify-center">
        <style>{`
          .login-anim .line-flow { stroke-dasharray: 220; stroke-dashoffset: 220; animation: flow 14s linear infinite; }
          .login-anim .node-float { transform-origin: center; animation: floatY 9s ease-in-out infinite; }
          .login-anim .pulse { transform-origin: center; animation: pulse 6.5s ease-in-out infinite; }
          .login-anim .blink { animation: blink 3.8s linear infinite; }

          @keyframes flow { from { stroke-dashoffset: 220; } to { stroke-dashoffset: 0; } }
          @keyframes floatY { 0%{transform:translateY(0px)} 50%{transform:translateY(-6px)} 100%{transform:translateY(0px)} }
          @keyframes pulse { 0%{transform:scale(1);opacity:0.12}50%{transform:scale(1.06);opacity:0.22}100%{transform:scale(1);opacity:0.12} }
          @keyframes blink { 0%{opacity:0.08}20%{opacity:0.28}60%{opacity:0.08}100%{opacity:0.08} }

          /* Slight stagger overrides */
          .login-anim .blink.s1 { animation-duration: 3.2s; }
          .login-anim .blink.s2 { animation-duration: 4.6s; animation-delay: 1s; }
          .login-anim .node-float.slow { animation-duration: 12s; }
          .login-anim .node-float.fast { animation-duration: 7.5s; }

          /* Respect reduced motion preference */
          @media (prefers-reduced-motion: reduce) {
            .login-anim .line-flow, .login-anim .node-float, .login-anim .pulse, .login-anim .blink { animation: none !important; }
          }
        `}</style>

        <svg className="w-full h-full opacity-100 login-anim" viewBox="0 0 1440 900" preserveAspectRatio="none" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" aria-hidden>
          <defs>
            <linearGradient id="bgGrad" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0%" stopColor="#fbfffb" />
              <stop offset="50%" stopColor="#f3fff6" />
              <stop offset="100%" stopColor="#f6fbff" />
            </linearGradient>
            <filter id="softBlur" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="20" />
            </filter>
          </defs>

          <rect width="100%" height="100%" fill="url(#bgGrad)" />

          {/* Large flowing financial graph curves (visible) */}
          <g strokeLinecap="round" fill="none" opacity="0.18">
            <path d="M -40 700 C 120 520, 320 480, 520 560 C 720 640, 920 580, 1120 460 C 1320 340, 1500 300, 1580 260" stroke="#4ba34b" strokeWidth="4" className="line-flow" />
            <path d="M -20 760 C 160 600, 360 540, 560 600 C 760 660, 960 620, 1160 520 C 1360 420, 1560 380, 1580 360" stroke="#2f9db3" strokeWidth="4" className="line-flow" />
            <path d="M 0 820 C 180 700, 380 660, 580 700 C 780 740, 980 720, 1180 640 C 1380 560, 1580 520, 1600 500" stroke="#8fbf6d" strokeWidth="3" className="line-flow" />
          </g>

          {/* Connected circular data nodes along curves */}
          <g opacity="0.95">
            <circle className="node-float" cx="520" cy="560" r="8" fill="#86BC25" />
            <circle className="node-float slow" cx="1120" cy="460" r="6" fill="#4EA214" />
            <circle className="node-float fast" cx="320" cy="480" r="5" fill="#2f9db3" />
            <circle className="node-float" cx="760" cy="640" r="6" fill="#6FAF1C" />
          </g>

          {/* Network connection thin lines */}
          <g stroke="#6FAF1C" strokeWidth="1" opacity="0.28">
            <path d="M520 560 L760 640 L1120 460" strokeOpacity="0.36" />
            <path d="M320 480 L520 560" strokeOpacity="0.3" />
          </g>

          {/* Small data points / particles */}
          <g opacity="0.9">
            <circle className="blink s1" cx="640" cy="520" r="2.5" fill="#4EA214" />
            <circle className="blink s2" cx="980" cy="480" r="2.5" fill="#2f9db3" />
            <circle className="blink" cx="420" cy="600" r="2" fill="#86BC25" />
          </g>

          {/* Abstract model/chart panels */}
          <g opacity="0.38" fill="none" stroke="#9fcf9a" strokeWidth="1.2">
            <rect x="40" y="80" width="220" height="120" rx="10" strokeOpacity="0.22" />
            <rect x="1180" y="120" width="220" height="140" rx="10" strokeOpacity="0.22" />
            <path d="M62 170 L92 130 L122 150 L152 120 L182 140" stroke="#6FAF1C" strokeWidth="2" strokeOpacity="0.6" />
          </g>

          {/* Shield/security outline (subtle) */}
          <g transform="translate(220,140) scale(1.4)" opacity="0.36" className="pulse">
            <path d="M80 12 L116 34 L116 82 C96 102 64 132 44 142 C24 132 -16 102 -36 82 L-36 34 Z" stroke="#4ba34b" strokeWidth="2" fill="none" />
          </g>

          {/* Soft radial glow behind card center */}
          <circle cx="720" cy="360" r="260" fill="#e6fbec" opacity="0.5" filter="url(#softBlur)" />

          {/* faint grid / data pattern (static, low contrast) */}
          <g stroke="#cfeee0" strokeWidth="0.8" opacity="0.12">
            <line x1="60" y1="60" x2="1380" y2="60" />
            <line x1="60" y1="120" x2="1380" y2="120" />
            <line x1="60" y1="180" x2="1380" y2="180" />
          </g>
        </svg>
      </div>

      <div className="relative z-10 w-full max-w-sm rounded-2xl border border-border bg-card p-8 shadow-elegant">
        <div className="flex flex-col items-center text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl gradient-primary shadow-elegant">
            <ShieldCheck className="h-6 w-6 text-primary-foreground" />
          </div>
          <h1 className="mt-4 text-xl font-semibold tracking-tight">Sign in to Aegis Credit</h1>
          <p className="mt-1 text-sm text-muted-foreground">Demo mode — enter any username and any password to sign in.</p>
        </div>
        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          <div>
            <label htmlFor="username" className="mb-1.5 block text-xs font-medium text-muted-foreground">
              Username
            </label>
            <div className="relative">
              <User className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                id="username"
                autoFocus
                autoComplete="username"
                value={username}
                onChange={(e) => {
                  setUsername(e.target.value);
                  setError(null);
                }}
                placeholder="e.g. jane.doe"
                className="h-10 w-full rounded-lg border border-input bg-background pl-9 pr-3 text-sm outline-none focus:border-primary/60"
              />
            </div>
          </div>

          <div>
            <label htmlFor="password" className="mb-1.5 block text-xs font-medium text-muted-foreground">
              Password
            </label>
            <div className="relative">
              <Lock className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError(null);
                }}
                placeholder="••••••••"
                className="h-10 w-full rounded-lg border border-input bg-background pl-9 pr-3 text-sm outline-none focus:border-primary/60"
              />
            </div>
          </div>

          {error && <p className="text-xs text-destructive">{error}</p>}

          <button
            type="submit"
            className="mt-2 inline-flex w-full items-center justify-center rounded-lg gradient-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground shadow-elegant transition-transform hover:translate-x-0.5"
          >
            Sign in
          </button>
        </form>
      </div>
    </div>
  );
}
