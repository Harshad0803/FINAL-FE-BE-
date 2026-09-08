import { Link, useRouterState, useNavigate } from "@tanstack/react-router";
import {
  UploadCloud,
  ShieldCheck,
  Layers,
  Cpu,
  Sparkles,
  Settings,
  Bell,
  Search,
  Home,
  FileText,
  GitCompareArrows,
  BarChart3,
  Activity,
  ClipboardCheck,
  History,
  ChevronDown,
  Database,
} from "lucide-react";
import { type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth-context";

type NavItem = { to: string; label: string; icon: typeof Home; exact?: boolean };

const developmentNav: NavItem[] = [
  { to: "/dashboard", label: "Dashboard", icon: Home },
  { to: "/data-upload", label: "Data Upload", icon: UploadCloud },
  { to: "/data-preparation", label: "Data Preparation & Feature Engineering", icon: Layers },
  { to: "/model-training-evaluation", label: "Model Training & Evaluation", icon: Cpu },
  { to: "/explainability", label: "Explainability", icon: Sparkles },
  { to: "/activity-log", label: "Activity Log", icon: History },
];

const validationNav: NavItem[] = [
  { to: "/validation/intake", label: "Intake & Governance", icon: FileText, exact: true },
  { to: "/validation/data-quality", label: "Data & Model Soundness", icon: Database },
  { to: "/validation/challenger", label: "Model Replication", icon: GitCompareArrows },
  { to: "/validation/performance", label: "Benchmarking", icon: BarChart3 },
  { to: "/validation/stress", label: "Stress & Backtesting", icon: Activity },
  { to: "/validation/regulatory", label: "Explainability and Fairness", icon: ShieldCheck },
  { to: "/validation/findings", label: "Findings & Final Report", icon: ClipboardCheck },
];

const enterpriseNav: NavItem[] = [{ to: "/model-inventory", label: "Model Inventory", icon: Database }];
const platformNav: NavItem[] = [{ to: "/settings", label: "Settings", icon: Settings }];

function NavLinkItem({ item, pathname }: { item: NavItem; pathname: string }) {
  const Icon = item.icon;
  const active = item.exact ? pathname === item.to : pathname === item.to || pathname.startsWith(item.to + "/");

  return (
    <li>
      <Link
        to={item.to}
        className={cn(
          "group flex items-center rounded-lg px-3 py-2 text-[14px] font-medium leading-[1.25rem] tracking-[-0.01em] transition-colors",
          active ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-sidebar-foreground/70 hover:bg-sidebar-accent/70 hover:text-sidebar-accent-foreground",
        )}
      >
        <Icon
          className={cn(
            "h-[16px] w-[16px] shrink-0 stroke-[2]",
            active ? "text-sidebar-accent-foreground" : "text-sidebar-foreground/60 group-hover:text-sidebar-accent-foreground",
          )}
        />
        <span className="ml-3 truncate">{item.label}</span>
      </Link>
    </li>
  );
}

function initialsFromName(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isLoginPage = pathname === "/login";
  const isLanding = pathname === "/" || isLoginPage;
  const { user } = useAuth();

  return (
    <div className="flex min-h-screen w-full bg-background text-foreground">
      {!isLanding && (
        <aside className="sticky top-0 z-30 hidden h-screen w-[264px] shrink-0 flex-col bg-sidebar text-sidebar-foreground md:flex">
          <div className="flex h-[76px] items-center gap-3 border-b border-sidebar-border px-4">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary shadow-[0_8px_18px_rgba(37,99,235,0.32)]">
              <ShieldCheck className="h-4 w-4 text-primary-foreground" />
            </div>
            <div className="flex flex-col leading-[1.15]">
              <span className="text-[15px] font-semibold tracking-[-0.02em] text-sidebar-foreground">Credit Risk POC</span>
              <span className="mt-1 text-[9px] font-semibold uppercase tracking-[0.18em] text-sidebar-foreground/60">
                Model Development
              </span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-3 py-4">
            <div className="space-y-5">
              <section>
                <div className="mb-2 flex items-center justify-between px-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-sidebar-foreground/60">
                  <span>Development</span>
                  <ChevronDown className="h-3.5 w-3.5 text-sidebar-foreground/60" />
                </div>
                <ul className="space-y-1">
                  {developmentNav.map((item) => (
                    <NavLinkItem key={item.to} item={item} pathname={pathname} />
                  ))}
                </ul>
              </section>

              <section>
                <div className="mb-2 flex items-center justify-between px-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-sidebar-foreground/60">
                  <span>Validation</span>
                  <ChevronDown className="h-3.5 w-3.5 text-sidebar-foreground/60" />
                </div>
                <ul className="space-y-1">
                  {validationNav.map((item) => (
                    <NavLinkItem key={item.to} item={item} pathname={pathname} />
                  ))}
                </ul>
              </section>

              <section>
                <div className="mb-2 flex items-center justify-between px-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-sidebar-foreground/60">
                  <span>Enterprise</span>
                  <ChevronDown className="h-3.5 w-3.5 text-sidebar-foreground/60" />
                </div>
                <ul className="space-y-1">
                  {enterpriseNav.map((item) => (
                    <NavLinkItem key={item.to} item={item} pathname={pathname} />
                  ))}
                </ul>
              </section>

              <section className="pt-2">
                <div className="mb-2 flex items-center justify-between px-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-sidebar-foreground/60">
                  <span>Platform</span>
                  <ChevronDown className="h-3.5 w-3.5 text-sidebar-foreground/60" />
                </div>
                <ul className="space-y-1">
                  {platformNav.map((item) => (
                    <NavLinkItem key={item.to} item={item} pathname={pathname} />
                  ))}
                </ul>
              </section>
            </div>
          </div>

          <div className="border-t border-sidebar-border p-3">
            <div className="flex items-center gap-3 rounded-lg border border-sidebar-border bg-[#111827] px-3 py-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-[11px] font-semibold text-primary-foreground">
                {initialsFromName(user?.name ?? "Harshad")}
              </div>
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13px] font-medium text-sidebar-foreground">{user?.name ?? "Harshad"}</div>
                <div className="mt-0.5 text-[9px] font-semibold uppercase tracking-[0.16em] text-sidebar-foreground/60">
                  {user?.role ?? "MODEL GOVERNANCE ANALYST"}
                </div>
              </div>
            </div>
          </div>
        </aside>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header
          className={cn(
            "sticky top-0 z-20 flex items-center gap-4 px-4 backdrop-blur-md md:px-8",
            isLanding ? "h-[60px] border-b border-sidebar-border bg-sidebar" : "h-16 border-b border-border/70 bg-background/80",
          )}
        >
          {isLanding ? (
            <Link to="/" className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg gradient-primary shadow-elegant">
                <ShieldCheck className="h-5 w-5 text-primary-foreground" />
              </div>
              <div className="leading-tight">
                <div className="text-sm font-semibold tracking-tight text-white">Credit Risk POC</div>
                <div className="text-[10px] uppercase tracking-[0.18em] text-sidebar-foreground/60">AI Model Platform</div>
              </div>
            </Link>
          ) : (
            <div className="hidden flex-1 items-center gap-2 md:flex">
              <div className="relative w-full max-w-md">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  placeholder="Search models, rules, datasets…"
                  className="h-10 w-full rounded-lg border border-input bg-card pl-9 pr-3 text-sm outline-none ring-0 placeholder:text-muted-foreground/70 focus:border-primary/60 focus:bg-card"
                />
              </div>
            </div>
          )}
          {!isLanding && <div className="md:hidden text-sm font-semibold">Aegis Credit</div>}
          {!isLoginPage && (
            <div className="ml-auto flex items-center gap-2">
              {isLanding && (
                <Link
                  to="/development"
                  className="hidden items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-sidebar-foreground/70 hover:bg-white/5 hover:text-white sm:inline-flex"
                >
                  Develop
                </Link>
              )}
              {isLanding && (
                <Link
                  to="/validation"
                  className="hidden items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground sm:inline-flex"
                >
                  Validate
                </Link>
              )}
              {isLanding && <div className="hidden h-5 w-px bg-sidebar-border sm:block" />}
              <Link
                to="/settings"
                aria-label="Settings"
                className={cn(
                  "inline-flex items-center justify-center rounded-lg border text-muted-foreground hover:text-foreground",
                  isLanding
                    ? "h-8 w-8 border-transparent text-sidebar-foreground/60 hover:bg-white/10 hover:text-white"
                    : "h-10 w-10 border-border bg-card",
                  pathname === "/settings" && !isLanding && "border-primary/40 text-primary",
                )}
              >
                <Settings className="h-4 w-4" />
              </Link>
              <button
                className={cn(
                  "relative inline-flex items-center justify-center rounded-lg border text-muted-foreground hover:text-foreground",
                  isLanding ? "h-8 w-8 border-transparent text-sidebar-foreground/60 hover:bg-white/10 hover:text-white" : "h-10 w-10 border-border bg-card",
                )}
              >
                <Bell className="h-4 w-4" />
                <span className={cn("absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-destructive", isLanding && "ring-2 ring-sidebar")} />
              </button>
              {user && (
                <>
                  {isLanding && <div className="hidden h-5 w-px bg-sidebar-border sm:block" />}
                  <button
                    onClick={() => {
                      logout();
                      navigate({ to: "/login" });
                    }}
                    className={cn(
                      "hidden items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium sm:inline-flex",
                      isLanding ? "border-sidebar-border text-sidebar-foreground/70 hover:bg-white/5" : "border-border bg-card hover:bg-accent",
                    )}
                  >
                    Sign out
                  </button>
                  <div
                    className={cn(
                      "flex items-center gap-2.5",
                      isLanding ? "h-9" : "h-10 rounded-lg border border-border bg-card px-2 pr-3",
                    )}
                  >
                    <div
                      className={cn(
                        "flex items-center justify-center font-semibold text-primary-foreground",
                        isLanding ? "h-8 w-8 rounded-full bg-primary text-xs" : "h-7 w-7 rounded-md gradient-primary text-[11px]",
                      )}
                    >
                      {initialsFromName(user.name)}
                    </div>
                    <div className="hidden text-left leading-tight sm:block">
                      <div className={cn("text-xs font-semibold", isLanding && "text-white")}>{user.name}</div>
                      <div className={cn("text-[10px]", isLanding ? "text-sidebar-foreground/60" : "text-muted-foreground")}>{user.role ?? "Risk Validator"}</div>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </header>

        <main className="flex-1 px-4 py-6 md:px-8 md:py-8">{children}</main>
      </div>
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-col gap-3 md:mb-8 md:flex-row md:items-end md:justify-between">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">{title}</h1>
        {description && (
          <p className="mt-1.5 max-w-2xl text-sm text-muted-foreground">{description}</p>
        )}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
