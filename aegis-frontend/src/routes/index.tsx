import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { ArrowRight, Boxes, ShieldCheck, Sparkles, Database, Clock3, CheckCircle2, ChevronRight } from "lucide-react";
import { useDataset } from "@/lib/app-context";
import { api } from "@/lib/api";
import { useResumeState } from "@/hooks/use-resume-state";
import AnimatedNumber from "@/components/animated-number";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Aegis Credit — Model Development & Validation" },
      { name: "description", content: "Choose a workspace: build credit risk models or independently validate them for regulatory compliance and governance." },
    ],
  }),
  component: Landing,
});

// ─── Real-data shapes from the existing /model-inventory and /history/*
// endpoints (same ones @/routes/dashboard.tsx already reads) — kept local
// since only a handful of fields are needed here. ──
type InventoryModel = {
  model_id?: string;
  model_name?: string;
  model_type?: string;
  status?: string;
  model_risk_rating?: string;
  business_unit?: string;
  model_owner?: string;
  model_version?: string;
  last_validation_date?: string;
  next_validation_due?: string;
};

type InventoryHistoryEvent = { timestamp?: string; event?: string; description?: string };

type InventoryPayload = {
  models?: InventoryModel[];
  development?: InventoryModel[];
  validation?: InventoryModel[];
  history?: InventoryHistoryEvent[];
};

type HistoryRun = { timestamp?: string; stage?: string; run_id?: string };

function mostRecent<T extends { timestamp?: string }>(rows: T[]): T | null {
  const withDates = rows
    .map((r) => ({ row: r, t: r.timestamp ? Date.parse(r.timestamp) : NaN }))
    .filter((r) => Number.isFinite(r.t));
  if (!withDates.length) return null;
  withDates.sort((a, b) => b.t - a.t);
  return withDates[0].row;
}

function activityLabel(iso?: string | null): string {
  if (!iso) return "No activity yet";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "No activity yet";
  return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function isPastDue(iso?: string | null): boolean {
  if (!iso) return false;
  const d = new Date(iso);
  return !Number.isNaN(d.getTime()) && d.getTime() < Date.now();
}

// ─── Status pill — same tone language (emerald/amber/slate) used across
// Development & Validation, never implies real-time system health beyond
// what's actually known from application state. ──
type Tone = "emerald" | "amber" | "slate";
const TONE_CLASSES: Record<Tone, { bg: string; text: string; dot: string }> = {
  emerald: { bg: "bg-emerald-50", text: "text-emerald-700", dot: "bg-emerald-500" },
  amber: { bg: "bg-amber-50", text: "text-amber-700", dot: "bg-amber-500" },
  slate: { bg: "bg-slate-100", text: "text-slate-500", dot: "bg-slate-400" },
};

function StatusPill({ label, tone }: { label: string; tone: Tone }) {
  const t = TONE_CLASSES[tone];
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold", t.bg, t.text)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", t.dot)} />
      {label}
    </span>
  );
}

function PlatformStat({ value, label }: { value: ReactNode; label: string }) {
  return (
    <div className="flex min-w-[110px] flex-col items-center justify-center border-r border-[#1E2D47] bg-[#0F1D35] px-6 py-5 last:border-r-0">
      <span className="text-xl font-extrabold tracking-tight text-white">{value}</span>
      <span className="mt-1 text-center text-[9px] uppercase tracking-[0.06em] text-slate-500">{label}</span>
    </div>
  );
}

// Hex + 2-digit alpha suffix (e.g. "#1D4ED8" + "12" -> "#1D4ED812") — the
// same technique the Figma reference uses, applied via inline style so the
// per-workspace accent color renders correctly regardless of Tailwind's
// arbitrary-value/opacity-modifier support for CSS custom properties.
function withAlpha(hex: string, alpha: string): string {
  return `${hex}${alpha}`;
}

function StatPill({ value, label, accent }: { value: ReactNode; label: string; accent: string }) {
  return (
    <div className="flex flex-col items-center rounded-lg px-4 py-2.5" style={{ backgroundColor: withAlpha(accent, "12") }}>
      <span className="text-base font-extrabold leading-none" style={{ color: accent }}>{value}</span>
      <span className="mt-1 text-[10px] tracking-wide text-slate-400">{label}</span>
    </div>
  );
}

interface Workspace {
  number: string;
  icon: ReactNode;
  title: string;
  description: string;
  capabilities: string[];
  cta: string;
  to: string;
  accent: string;
  status: { label: string; tone: Tone };
  lastActivity: string;
  stats: { value: ReactNode; label: string }[];
}

function WorkspaceCard({ ws }: { ws: Workspace }) {
  const [hovered, setHovered] = useState(false);

  return (
    <Link
      to={ws.to}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="relative flex flex-col overflow-hidden rounded-2xl bg-white transition-all duration-200"
      style={{
        border: `1px solid ${hovered ? withAlpha(ws.accent, "55") : "#E2E8F0"}`,
        boxShadow: hovered ? "0 20px 48px -8px rgba(15,29,53,0.16)" : "0 2px 8px -2px rgba(15,29,53,0.07)",
        transform: hovered ? "translateY(-4px)" : "translateY(0)",
      }}
    >
      <div className="h-1 w-full transition-colors duration-300" style={{ backgroundColor: hovered ? ws.accent : "#E2E8F0" }} />

      <div className="flex items-center justify-between border-b border-slate-100 px-6 pb-4 pt-5">
        <span className="text-xs font-bold tracking-[0.16em]" style={{ color: ws.accent }}>{ws.number}</span>
        <StatusPill label={ws.status.label} tone={ws.status.tone} />
      </div>

      <div className="flex flex-1 flex-col px-6 pb-6 pt-5">
        <div className="mb-4 flex items-start gap-4">
          <div
            className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl transition-colors duration-200"
            style={{ backgroundColor: hovered ? ws.accent : withAlpha(ws.accent, "14"), color: hovered ? "#FFFFFF" : ws.accent }}
          >
            {ws.icon}
          </div>
          <div>
            <h3 className="text-lg font-extrabold tracking-tight text-slate-900">{ws.title}</h3>
            <div className="mt-1 flex items-center gap-1 text-xs text-slate-400">
              <Clock3 className="h-3 w-3" />
              {ws.lastActivity}
            </div>
          </div>
        </div>

        <p className="mb-5 text-sm leading-relaxed text-slate-500">{ws.description}</p>

        <div className="mb-5 flex gap-2">
          {ws.stats.map((s) => (
            <StatPill key={s.label} value={s.value} label={s.label} accent={ws.accent} />
          ))}
        </div>

        <div className="mb-6 flex flex-col gap-2">
          {ws.capabilities.map((cap) => (
            <div key={cap} className="flex items-center gap-2.5">
              <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded" style={{ backgroundColor: withAlpha(ws.accent, "26"), color: ws.accent }}>
                <CheckCircle2 className="h-2.5 w-2.5" />
              </span>
              <span className="text-xs font-medium text-slate-600">{cap}</span>
            </div>
          ))}
        </div>

        <div
          className="mt-auto flex items-center justify-between rounded-xl px-5 py-3.5 text-sm font-bold transition-all duration-200"
          style={{
            border: `1.5px solid ${hovered ? ws.accent : withAlpha(ws.accent, "40")}`,
            backgroundColor: hovered ? ws.accent : "#F8FAFC",
            color: hovered ? "#FFFFFF" : ws.accent,
          }}
        >
          <span>{ws.cta}</span>
          <ArrowRight className="h-4 w-4 transition-transform duration-200" style={{ transform: hovered ? "translateX(2px)" : "translateX(0)" }} />
        </div>
      </div>
    </Link>
  );
}

// 7 stages actually routed under /validation (matches validation.index.tsx's
// own dashboard) — duplicated here in the same simple, real-data-only form
// rather than importing a shared hook, so this page has no dependency on
// how the Validation workspace's own dashboard chooses to compute progress.
function useValidationProgress() {
  const {
    validationIntakeData, validationStage3Result, validationStage4Result,
    validationStage5Result, validationStage7Result, validationStage8Result,
  } = useDataset();
  const { data: resumedStress } = useResumeState<{ stage: string; report: any }>("validation_pipeline_log.csv", "stress_testing");

  const flags = [
    Boolean(validationIntakeData),
    Boolean(validationStage3Result),
    Boolean((validationStage4Result as any)?.replication),
    Boolean(validationStage5Result),
    Boolean((resumedStress as any)?.report),
    Boolean(validationStage7Result),
    Boolean(validationStage8Result),
  ];
  return { completedCount: flags.filter(Boolean).length, total: flags.length };
}

function Landing() {
  const ds = useDataset();
  const { completedCount: validationCompletedCount, total: validationTotal } = useValidationProgress();

  const [inventory, setInventory] = useState<InventoryPayload | null>(null);
  const [devHistory, setDevHistory] = useState<HistoryRun[]>([]);
  const [validationHistory, setValidationHistory] = useState<HistoryRun[]>([]);
  const [connection, setConnection] = useState<"connecting" | "connected" | "unreachable">("connecting");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [inventoryRes, devRes, valRes] = await Promise.all([
          api<InventoryPayload>("/model-inventory"),
          api<{ runs?: HistoryRun[] }>("/history/dev"),
          api<{ runs?: HistoryRun[] }>("/history/validation"),
        ]);
        if (cancelled) return;
        setInventory(inventoryRes);
        setDevHistory(devRes.runs ?? []);
        setValidationHistory(valRes.runs ?? []);
        setConnection("connected");
      } catch {
        if (!cancelled) setConnection("unreachable");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // ── Development workspace: real progress across the 4 stages that persist
  // into shared context (Explainability doesn't write back, so it isn't
  // counted — this only claims what it can verify). ──
  const devStageFlags = [Boolean(ds.profile), Boolean(ds.preprocessingResult), Boolean(ds.featureEngineeringResult), Boolean(ds.trainingResult)];
  const devStagesDone = devStageFlags.filter(Boolean).length;
  const devStageLabels = ["Data Upload", "Preprocessing", "Feature Engineering", "Training"];
  const devCurrentStage = devStagesDone === 0 ? "Not started" : devStageLabels[devStagesDone - 1];
  const devAccuracyMetric = (ds.trainingResult as any)?.evaluation_metrics?.accuracy;
  const devAccuracy = typeof devAccuracyMetric === "number" ? `${(devAccuracyMetric * 100).toFixed(1)}%` : "—";
  const devModelsRegistered = inventory?.development?.length ?? 0;
  const lastDevRun = mostRecent(devHistory);

  const devStatus: { label: string; tone: Tone } =
    devStagesDone === 0
      ? { label: "Not Started", tone: "slate" }
      : devStagesDone === devStageFlags.length
        ? { label: "Model Trained", tone: "emerald" }
        : { label: "In Progress", tone: "amber" };

  // ── Validation workspace: reuse the exact same real pipeline hook the
  // Validation dashboard uses, so the two pages never disagree. ──
  const findings = ds.validationStage8Result as { total_count?: number; high_count?: number; verdict?: string } | null | undefined;
  const lastValidationRun = mostRecent(validationHistory);

  const validationStatus: { label: string; tone: Tone } =
    validationCompletedCount === 0
      ? { label: "Not Started", tone: "slate" }
      : validationCompletedCount === validationTotal
        ? { label: "Complete", tone: "emerald" }
        : { label: "In Progress", tone: "amber" };

  // ── Inventory workspace: same aggregation dashboard.tsx uses for
  // "Total Models", so the number is identical wherever it's shown. ──
  const inventoryModels = inventory?.models ?? inventory?.development ?? [];
  const inventoryValidationModels = inventory?.validation ?? [];
  const totalModels = Math.max(
    inventoryModels.length || 0,
    (inventory?.development?.length ?? 0) + (inventory?.validation?.length ?? 0),
  );
  const allInventoryRows = [...inventoryModels, ...inventoryValidationModels];
  const validatedCount = allInventoryRows.filter((m) => (m.status ?? "").toLowerCase().includes("validated")).length;
  const reviewDueCount = allInventoryRows.filter(
    (m) => isPastDue(m.next_validation_due) || (m.status ?? "").toLowerCase().includes("review"),
  ).length;
  const lastInventoryEvent = mostRecent(inventory?.history ?? []);

  const inventoryStatus: { label: string; tone: Tone } =
    totalModels === 0 ? { label: "Empty", tone: "slate" } : { label: "Active", tone: "emerald" };

  const workspaces: Workspace[] = useMemo(
    () => [
      {
        number: "WORKSPACE 01",
        icon: <Boxes className="h-5 w-5" />,
        title: "Model Development",
        description: "Build, train, evaluate, and explain credit risk models with an end-to-end ML workflow covering the full model lifecycle.",
        capabilities: ["Data → Features → Training → Explainability", "Live model metrics & SHAP attribution"],
        cta: "Open Model Development",
        to: "/development",
        accent: "#1D4ED8",
        status: devStatus,
        lastActivity: lastDevRun?.timestamp ? `Last run ${activityLabel(lastDevRun.timestamp)}` : "No activity yet",
        stats: [
          { value: <AnimatedNumber value={devModelsRegistered} />, label: "MODELS" },
          { value: devAccuracy, label: "ACCURACY" },
          { value: devCurrentStage, label: "STAGE" },
        ],
      },
      {
        number: "WORKSPACE 02",
        icon: <ShieldCheck className="h-5 w-5" />,
        title: "Model Validation",
        description: "Independently validate existing models for performance, conceptual soundness, regulatory compliance, and governance readiness.",
        capabilities: ["Champion vs challenger benchmarking", "IFRS 9 / IFRS 7 / SS1/23 evidence pack"],
        cta: "Open Model Validation",
        to: "/validation",
        accent: "#0369A1",
        status: validationStatus,
        lastActivity: lastValidationRun?.timestamp ? `Last run ${activityLabel(lastValidationRun.timestamp)}` : "No activity yet",
        stats: [
          { value: <AnimatedNumber value={validationCompletedCount} />, label: `OF ${validationTotal} STAGES` },
          { value: findings?.total_count ?? "—", label: "FINDINGS" },
          { value: findings?.verdict ?? "Pending", label: "VERDICT" },
        ],
      },
      {
        number: "WORKSPACE 03",
        icon: <Database className="h-5 w-5" />,
        title: "Model Inventory",
        description: "Register models, attach data sources, track validation status, and export a governance-ready inventory for auditors and regulators.",
        capabilities: ["Persistent model registry", "Excel export for consultant reviews"],
        cta: "Open Model Inventory",
        to: "/model-inventory",
        accent: "#047857",
        status: inventoryStatus,
        lastActivity: lastInventoryEvent?.timestamp ? `Last update ${activityLabel(lastInventoryEvent.timestamp)}` : "No activity yet",
        stats: [
          { value: <AnimatedNumber value={totalModels} />, label: "REGISTERED" },
          { value: <AnimatedNumber value={validatedCount} />, label: "VALIDATED" },
          { value: <AnimatedNumber value={reviewDueCount} />, label: "REVIEW DUE" },
        ],
      },
    ],
    [
      devStatus, lastDevRun, devModelsRegistered, devAccuracy, devCurrentStage,
      validationStatus, lastValidationRun, validationCompletedCount, validationTotal, findings,
      inventoryStatus, lastInventoryEvent, totalModels, validatedCount, reviewDueCount,
    ],
  );

  return (
    <div className="-mx-4 -mt-6 flex flex-col md:-mx-8 md:-mt-8">
      {/* ── Hero band — restrained dark-navy enterprise treatment, matching
          the Figma reference exactly (diagonal gradient, faint grid, subtle
          right glow) rather than a bright/teal dashboard hero. ──────────── */}
      <div
        className="relative overflow-hidden border-b border-[#1E2D47]"
        style={{ background: "linear-gradient(135deg, #0F1D35 0%, #0B1629 60%, #111E38 100%)" }}
      >
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)",
            backgroundSize: "64px 64px",
          }}
        />
        <div
          className="pointer-events-none absolute inset-y-0 right-0 w-1/3"
          style={{ background: "radial-gradient(ellipse at 100% 50%, rgba(29,78,216,0.12) 0%, transparent 70%)" }}
        />

        <div className="relative mx-auto max-w-[1400px] px-4 py-12 md:px-8">
          <div className="flex flex-col items-start justify-between gap-8 md:flex-row md:items-center">
            <div className="max-w-xl">
              <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-blue-700/40 bg-blue-700/20 px-3 py-1.5 text-xs font-bold uppercase tracking-[0.09em] text-blue-300">
                <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />
                Enterprise AI Platform
              </div>
              <h1 className="text-[2rem] font-extrabold leading-[1.1] tracking-[-0.04em] text-white md:text-[2.75rem]">
                Choose your <span className="text-blue-400">workspace</span>
              </h1>
              <p className="mt-4 max-w-[480px] text-[15px] leading-[1.7] text-slate-400">
                Aegis Credit unifies model development and independent validation in a single, regulator-ready platform.
                Select a workspace to begin.
              </p>
            </div>

            <div className="hidden shrink-0 items-stretch overflow-hidden rounded-2xl border border-[#1E2D47] lg:flex">
              <PlatformStat value={<AnimatedNumber value={totalModels} />} label="MODELS REGISTERED" />
              <PlatformStat value={`${validationCompletedCount}/${validationTotal}`} label="VALIDATION STAGES" />
              <PlatformStat value={devCurrentStage} label="DEV STAGE" />
              <PlatformStat value={findings?.high_count ?? "—"} label="HIGH FINDINGS" />
            </div>
          </div>
        </div>
      </div>

      {/* ── Status bar ─────────────────────────────────────────────────── */}
      <div className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1400px] flex-col gap-2 px-4 py-2.5 text-xs sm:flex-row sm:items-center sm:justify-between md:px-8">
          <div className="flex items-center gap-2 text-slate-400">
            <span className="font-medium text-slate-500">Aegis Credit</span>
            <ChevronRight className="h-3 w-3" />
            <span className="font-medium text-slate-500">Validate</span>
            <ChevronRight className="h-3 w-3" />
            <span className="font-semibold text-slate-900">Workspace Selection</span>
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <span
              className={cn(
                "h-2 w-2 rounded-full",
                connection === "connected" ? "bg-emerald-500" : connection === "unreachable" ? "bg-red-500" : "bg-slate-300",
              )}
            />
            {connection === "connected" ? (
              <span>Connected to Aegis backend</span>
            ) : connection === "unreachable" ? (
              <span>Backend unreachable — workspace stats may be incomplete</span>
            ) : (
              <span>Connecting…</span>
            )}
          </div>
        </div>
      </div>

      {/* ── Workspace cards ────────────────────────────────────────────── */}
      <div className="mx-auto w-full max-w-[1400px] px-4 pb-14 pt-10 md:px-8">
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-bold tracking-tight text-slate-900">Available Workspaces</h2>
            <span className="rounded-md bg-indigo-100 px-2 py-0.5 text-xs font-bold text-indigo-800">
              {workspaces.length} ACTIVE
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
          {workspaces.map((ws) => (
            <WorkspaceCard key={ws.number} ws={ws} />
          ))}
        </div>

        <div className="mt-8 flex flex-col gap-1 text-xs text-slate-400 sm:flex-row sm:items-center sm:justify-between">
          <span className="flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            Aegis Credit · Model Risk Management Platform
          </span>
          <span>All workspace activity shown above is read from live application state.</span>
        </div>
      </div>
    </div>
  );
}
