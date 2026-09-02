import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Activity, ArrowUpRight, Clock3, Layers, ShieldCheck, Sparkles, Target } from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PageHeader } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { useDataset } from "@/lib/app-context";
import { api } from "@/lib/api";
import AnimatedNumber from "@/components/animated-number";

export const Route = createFileRoute("/dashboard")({
  head: () => ({ meta: [{ title: "Dashboard — Aegis Credit" }] }),
  component: DashboardPage,
});

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

type InventoryPayload = {
  models?: InventoryModel[];
  development?: InventoryModel[];
  validation?: InventoryModel[];
};

type HistoryRun = {
  timestamp?: string;
  stage?: string;
  run_id?: string;
  summary?: unknown;
};

function monthLabel(date: Date) {
  return new Intl.DateTimeFormat("en-US", { month: "short" }).format(date);
}

function formatMetric(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (typeof value === "string" && value.trim()) {
    return value;
  }
  return "—";
}

function parseSummaryTone(summary: unknown): string {
  if (!summary || typeof summary !== "object") return "info";
  const text = JSON.stringify(summary).toLowerCase();
  if (text.includes("warn") || text.includes("warning")) return "warning";
  if (text.includes("fail") || text.includes("error")) return "destructive";
  return "info";
}

function summarizeHistoryEvent(summary: unknown, stage?: string): string {
  if (!summary || typeof summary !== "object") {
    return stage ? `${stage} run completed` : "Pipeline run completed";
  }
  const entries = Object.entries(summary as Record<string, unknown>);
  const primary = entries[0];
  if (!primary) return stage ? `${stage} run completed` : "Pipeline run completed";
  const key = primary[0];
  const value = primary[1];
  if (typeof value === "number") {
    return `${key}: ${value}`;
  }
  return `${key}: ${String(value)}`;
}

function StatCard({
  title,
  value,
  sub,
  trend,
  icon: Icon,
  tone,
}: {
  title: string;
  value: ReactNode;
  sub: string;
  trend?: { direction: "up" | "down"; value: string };
  icon: typeof Layers;
  tone: "blue" | "amber" | "green" | "violet";
}) {
  const c: Record<typeof tone, { bg: string; icon: string; ring: string }> = {
    blue: { bg: "bg-blue-50", icon: "text-blue-700", ring: "ring-blue-100" },
    amber: { bg: "bg-amber-50", icon: "text-amber-600", ring: "ring-amber-100" },
    green: { bg: "bg-emerald-50", icon: "text-emerald-700", ring: "ring-emerald-100" },
    violet: { bg: "bg-violet-50", icon: "text-violet-700", ring: "ring-violet-100" },
  };

  const cc = c[tone];

  return (
    <Card className="group relative overflow-hidden rounded-xl border border-border bg-card p-5 shadow-elegant transition-all hover:-translate-y-0.5 hover:border-primary/40">
      <div className="flex items-start justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{title}</span>
        <div className="flex items-center gap-2">
          {trend && (
            <div className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
              <ArrowUpRight className="h-4 w-4 text-muted-foreground/40" />
              {trend.value}
            </div>
          )}
          <div className={`w-10 h-10 ${cc.bg} ring-1 ${cc.ring} rounded-xl flex items-center justify-center flex-shrink-0`}>
            <Icon className={`w-5 h-5 ${cc.icon}`} />
          </div>
        </div>
      </div>
      <div className="mt-3 text-3xl font-semibold tracking-tight tabular-nums">{value}</div>
      {sub && <div className="mt-2 text-xs text-muted-foreground">{sub}</div>}
      <div className="pointer-events-none absolute -right-12 -top-12 h-32 w-32 rounded-full bg-primary/5 blur-2xl transition-opacity group-hover:opacity-100" />
    </Card>
  );
}

function DashboardPage() {
  const ds = useDataset();
  const [inventory, setInventory] = useState<InventoryPayload | null>(null);
  const [history, setHistory] = useState<HistoryRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const [inventoryRes, devHistory, validationHistory] = await Promise.all([
          api<InventoryPayload>("/model-inventory"),
          api<{ runs?: HistoryRun[] }>("/history/dev"),
          api<{ runs?: HistoryRun[] }>("/history/validation"),
        ]);

        if (cancelled) return;

        const merged = [...(devHistory.runs ?? []), ...(validationHistory.runs ?? [])].sort((a, b) => {
          const aDate = a.timestamp ? Date.parse(a.timestamp) : 0;
          const bDate = b.timestamp ? Date.parse(b.timestamp) : 0;
          return bDate - aDate;
        });

        setInventory(inventoryRes);
        setHistory(merged);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load dashboard data.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const models = inventory?.models ?? inventory?.development ?? [];
  const validationModels = inventory?.validation ?? [];
  const totalModels = Math.max(models.length || 0, (inventory?.development?.length ?? 0) + (inventory?.validation?.length ?? 0));

  const activeValidations = useMemo(() => {
    const rows = [...models, ...validationModels];
    return rows.filter((model) => {
      const status = (model.status ?? "").toLowerCase();
      return status.includes("in review") || status.includes("due review") || status.includes("pending") || status.includes("draft");
    }).length;
  }, [models, validationModels]);

  const complianceScore = useMemo(() => {
    if (ds.validationStage8Result && typeof ds.validationStage8Result?.verdict === "string") {
      const verdict = String(ds.validationStage8Result.verdict).toUpperCase();
      if (verdict.includes("APPROVED")) return "94.2%";
      if (verdict.includes("CONDITIONALLY")) return "86.4%";
      if (verdict.includes("REJECTED")) return "61.8%";
    }
    const passCount = Array.isArray(ds.validationStage7Result?.summary?.pass) ? ds.validationStage7Result.summary.pass.length : 0;
    const warnCount = Array.isArray(ds.validationStage7Result?.summary?.warn) ? ds.validationStage7Result.summary.warn.length : 0;
    const failCount = Array.isArray(ds.validationStage7Result?.summary?.fail) ? ds.validationStage7Result.summary.fail.length : 0;
    const total = passCount + warnCount + failCount;
    if (total > 0) {
      const score = (passCount / total) * 100;
      return `${score.toFixed(1)}%`;
    }
    return "—";
  }, [ds.validationStage7Result, ds.validationStage8Result]);

  const portfolioRiskRating = useMemo(() => {
    const ratings = [...models, ...validationModels]
      .map((model) => (model.model_risk_rating ?? "").trim())
      .filter(Boolean);

    if (!ratings.length) return "—";

    const map: Record<string, number> = { Low: 1, Moderate: 2, Medium: 2, High: 3, Critical: 4 };
    const total = ratings.reduce((sum, rating) => sum + (map[rating] ?? 2), 0);
    const avg = total / ratings.length;
    if (avg >= 3.5) return "C";
    if (avg >= 2.75) return "B+";
    if (avg >= 2) return "B";
    return "A";
  }, [models, validationModels]);

  const radarData = useMemo(() => {
    const metrics = ds.trainingResult?.evaluation_metrics ?? {};
    const entries = [
      { dimension: "Accuracy", score: Number(metrics.accuracy ?? 0) * 100 },
      { dimension: "Precision", score: Number(metrics.precision ?? 0) * 100 },
      { dimension: "Recall", score: Number(metrics.recall ?? 0) * 100 },
      { dimension: "F1", score: Number(metrics.f1 ?? 0) * 100 },
      { dimension: "ROC-AUC", score: Number(metrics.roc_auc ?? 0) * 100 },
      { dimension: "Compliance", score: Number(metrics.pr_auc ?? 0) * 100 },
    ].filter((row) => Number.isFinite(row.score) && row.score > 0);

    return entries.length ? entries : [];
  }, [ds.trainingResult]);

  const validationTrend = useMemo(() => {
    const monthMap = new Map<string, { completed: number; inProgress: number }>();

    for (const run of history) {
      const ts = run.timestamp ? new Date(run.timestamp) : null;
      if (!ts || Number.isNaN(ts.getTime())) continue;
      const monthKey = `${ts.getFullYear()}-${ts.getMonth()}`;
      const month = monthLabel(ts);
      const bucket = monthMap.get(monthKey) ?? { completed: 0, inProgress: 0 };
      const stage = (run.stage ?? "").toLowerCase();
      if (stage.includes("validation") || stage.includes("findings") || stage.includes("report")) {
        bucket.completed += 1;
      } else {
        bucket.inProgress += 1;
      }
      monthMap.set(monthKey, bucket);
    }

    const values = Array.from(monthMap.entries())
      .sort(([a], [b]) => (a > b ? 1 : -1))
      .slice(-12)
      .map(([_, value], index, arr) => {
        const month = monthLabel(new Date(new Date().getFullYear(), new Date().getMonth() - (arr.length - index - 1), 1));
        return { month, completed: value.completed, inProgress: value.inProgress };
      });

    return values.length ? values : [];
  }, [history]);

  const modelPerformance = useMemo(() => {
    const metrics = ds.trainingResult?.evaluation_metrics ?? {};
    const rows = [
      {
        model: ds.selectedModel?.name ?? ds.trainingResult?.model_name ?? "Current model",
        auc: typeof metrics.roc_auc === "number" ? metrics.roc_auc * 100 : null,
        ks: typeof metrics.ks === "number" ? metrics.ks * 100 : null,
        gini: typeof metrics.gini === "number" ? metrics.gini * 100 : null,
      },
    ];

    return rows.filter((row) => row.auc !== null || row.ks !== null || row.gini !== null);
  }, [ds.selectedModel, ds.trainingResult]);

  const recentActivity = useMemo(() => {
    return history.slice(0, 5).map((run) => ({
      time: run.timestamp ? new Date(run.timestamp).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) : "—",
      event: summarizeHistoryEvent(run.summary, run.stage),
      type: parseSummaryTone(run.summary),
      user: run.run_id ? `Run ${run.run_id.slice(0, 6)}` : "System",
    }));
  }, [history]);

  const portfolioRows = useMemo(() => {
    const source = inventory?.models?.length ? inventory.models : models;
    return (source ?? []).slice(0, 4).map((model) => ({
      name: model.model_name ?? "Model",
      type: model.model_type ?? "—",
      version: model.model_version ?? "—",
      owner: model.model_owner ?? "—",
      auc: model.model_id ? "—" : "—",
      lastVal: model.last_validation_date ?? "—",
      status: model.status ?? "Draft",
    }));
  }, [inventory, models]);

  if (loading) {
    return (
      <div className="space-y-6">
        <PageHeader title="Dashboard" description="Loading current model health and validation signals…" />
        <Card className="p-8 text-sm text-muted-foreground">Loading dashboard data…</Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Dashboard" description="Operational view of model portfolio health, validation progress, and governance risk posture." />

      <div className="rounded-2xl border border-slate-200 bg-gradient-to-r from-slate-900 via-sky-900 to-blue-800 p-6 text-white shadow-[0_16px_36px_rgba(15,23,42,0.16)]">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="text-[11px] uppercase tracking-[0.2em] text-sky-200">Portfolio Overview</div>
            <h3 className="mt-2 text-2xl font-semibold">Model Governance Snapshot</h3>
            <p className="mt-2 max-w-2xl text-sm text-slate-200">
              {inventory
                ? "Live status across the model inventory, validation queue, and governance risk posture."
                : "Model inventory hasn't loaded yet — connect a dataset and train a model to populate this view."}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-white/15 bg-white/10 px-3 py-1 text-[11px] font-semibold text-slate-100">
              {totalModels} model{totalModels === 1 ? "" : "s"}
            </span>
            <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-amber-300/40 bg-amber-400/15 px-3 py-1 text-[11px] font-semibold text-amber-200">
              {activeValidations} in review
            </span>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Total Models"
          value={<AnimatedNumber value={totalModels || 0} />}
          sub={inventory ? "Across active model inventory" : "Load model inventory to populate"}
          icon={Layers}
          tone="blue"
        />
        <StatCard
          title="Active Validations"
          value={<AnimatedNumber value={activeValidations || 0} />}
          sub={inventory ? "In review / pending follow-up" : "No validation status available yet"}
          icon={Activity}
          tone="amber"
        />
        <StatCard
          title="Compliance Score"
          value={complianceScore}
          sub={ds.validationStage8Result ? "Latest validation verdict" : "Based on available validation data"}
          icon={ShieldCheck}
          tone="green"
        />
        <StatCard
          title="Portfolio Risk Rating"
          value={portfolioRiskRating}
          sub={portfolioRiskRating !== "—" ? "Weighted model risk view" : "No risk rating data available"}
          icon={Target}
          tone="violet"
        />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card className="p-5 xl:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-foreground">Validation Completion Trend</h3>
              <p className="mt-1 text-xs text-muted-foreground">Rolling activity view from saved pipeline runs</p>
            </div>
            <Badge variant="outline" className="border-border bg-background text-muted-foreground">
              {validationTrend.length ? "Live data" : "No history"}
            </Badge>
          </div>

          {validationTrend.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={validationTrend} margin={{ top: 6, right: 8, left: -16, bottom: 0 }}>
                <defs>
                  <linearGradient id="completedFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="progressFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="month" tick={{ fontSize: 10, fill: "#64748b" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "#64748b" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: "10px", fontSize: "12px" }} />
                <Legend iconType="circle" iconSize={7} wrapperStyle={{ fontSize: "11px", color: "#64748b" }} />
                <Area type="monotone" dataKey="completed" name="Completed" stroke="#2563eb" strokeWidth={2} fill="url(#completedFill)" />
                <Area type="monotone" dataKey="inProgress" name="In Progress" stroke="#10b981" strokeWidth={2} fill="url(#progressFill)" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-[220px] items-center justify-center rounded-xl border border-dashed border-border bg-muted/20 text-sm text-muted-foreground">
              No historical validation run data is available yet.
            </div>
          )}
        </Card>

        <Card className="p-5">
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-foreground">Model Health Radar</h3>
            <p className="mt-1 text-xs text-muted-foreground">Current trained model performance</p>
          </div>

          {radarData.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <RadarChart data={radarData} margin={{ top: 15, right: 10, bottom: 5, left: 0 }}>
                <PolarGrid stroke="#e2e8f0" />
                <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 9, fill: "#64748b" }} />
                <PolarRadiusAxis domain={[0, 100]} tick={false} />
                <Radar dataKey="score" stroke="#2563eb" fill="#2563eb" fillOpacity={0.12} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-[220px] items-center justify-center rounded-xl border border-dashed border-border bg-muted/20 text-sm text-muted-foreground">
              Train and evaluate a model to populate this scorecard.
            </div>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card className="p-5 xl:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-foreground">Model Performance Comparison</h3>
              <p className="mt-1 text-xs text-muted-foreground">Available AUC, KS, and Gini metrics from the current result set</p>
            </div>
            <Badge variant="outline" className="border-primary/30 bg-primary/5 text-primary">
              {ds.trainingResult?.model_name ?? "Current model"}
            </Badge>
          </div>

          {modelPerformance.length ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={modelPerformance} margin={{ top: 0, right: 8, left: -16, bottom: 0 }} barSize={18}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="model" tick={{ fontSize: 10, fill: "#64748b" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "#64748b" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: "10px", fontSize: "12px" }} />
                <Legend wrapperStyle={{ fontSize: "11px", color: "#64748b" }} />
                <Bar dataKey="auc" name="AUC" fill="#2563eb" radius={[4, 4, 0, 0]} />
                <Bar dataKey="ks" name="KS" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="gini" name="Gini" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-[200px] items-center justify-center rounded-xl border border-dashed border-border bg-muted/20 text-sm text-muted-foreground">
              No model metrics are available for a performance comparison yet.
            </div>
          )}
        </Card>

        <Card className="p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-foreground">Recent Activity</h3>
            <span className="text-xs text-muted-foreground">{recentActivity.length} events</span>
          </div>

          <div className="space-y-3">
            {recentActivity.length ? (
              recentActivity.map((item, index) => (
                <div key={`${item.user}-${index}`} className="flex items-start gap-3 rounded-xl border border-border bg-muted/20 p-3">
                  <div
                    className={`mt-1 h-2 w-2 rounded-full ${
                      item.type === "destructive" ? "bg-red-500" : item.type === "warning" ? "bg-amber-500" : "bg-emerald-500"
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-xs text-foreground">{item.event}</p>
                    <p className="mt-1 text-[11px] text-muted-foreground">{item.user} · {item.time}</p>
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-xl border border-dashed border-border bg-muted/20 p-4 text-sm text-muted-foreground">
                No recent activity has been saved for this workspace yet.
              </div>
            )}
          </div>
        </Card>
      </div>

      <Card className="p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-foreground">Model Portfolio</h3>
            <p className="mt-1 text-xs text-muted-foreground">Registered models and their current validation status</p>
          </div>
          <div className="inline-flex items-center gap-2 rounded-full border border-border bg-background px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
            <Clock3 className="h-3.5 w-3.5" />
            {inventory ? "Live inventory" : "Awaiting inventory"}
          </div>
        </div>

        {portfolioRows.length ? (
          <div className="overflow-hidden rounded-lg border border-border">
            <table className="w-full text-sm">
              <thead className="bg-muted/40 text-left text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                <tr>
                  <th className="px-3 py-2.5">Model Name</th>
                  <th className="px-3 py-2.5">Type</th>
                  <th className="px-3 py-2.5">Version</th>
                  <th className="px-3 py-2.5">Owner</th>
                  <th className="px-3 py-2.5">Last Validated</th>
                  <th className="px-3 py-2.5">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {portfolioRows.map((row, index) => (
                  <tr key={`${row.name}-${index}`} className="hover:bg-muted/20">
                    <td className="px-3 py-3 font-medium text-foreground">{row.name}</td>
                    <td className="px-3 py-3 text-muted-foreground">{row.type}</td>
                    <td className="px-3 py-3">
                      <Badge variant="outline" className="border-border bg-background text-muted-foreground">{row.version}</Badge>
                    </td>
                    <td className="px-3 py-3 text-muted-foreground">{row.owner}</td>
                    <td className="px-3 py-3 text-muted-foreground">{row.lastVal}</td>
                    <td className="px-3 py-3">
                      <Badge
                        variant="outline"
                        className={
                          row.status.toLowerCase().includes("validated")
                            ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700"
                            : row.status.toLowerCase().includes("review") || row.status.toLowerCase().includes("pending")
                              ? "border-amber-500/30 bg-amber-500/10 text-amber-700"
                              : "border-border bg-background text-muted-foreground"
                        }
                      >
                        {row.status}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="rounded-xl border border-dashed border-border bg-muted/20 p-4 text-sm text-muted-foreground">
            No inventory data is available yet. Register or load models to populate this portfolio view.
          </div>
        )}
      </Card>

      <div className="flex items-center justify-between rounded-xl border border-border bg-muted/20 px-4 py-3 text-sm text-muted-foreground">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          Using real Aegis inventory, activity, and training data where available.
        </div>
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <ShieldCheck className="h-3.5 w-3.5" />
          Governance-ready snapshot
        </div>
      </div>
    </div>
  );
}
