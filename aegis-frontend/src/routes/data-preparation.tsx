import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { PageHeader } from "@/components/app-shell";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useDataset } from "@/lib/app-context";
import { formUpload } from "@/lib/api";
import PlotlyChart from "@/components/plotly-chart";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import {
  AlertCircle, AlertTriangle, ArrowLeft, ArrowRight, BarChart3, CheckCircle2, ChevronDown, Download, Info,
  BarChart as BarChartIcon, Table as TableIcon, Brain, Loader2, Loader, RefreshCw, Trash2, Hash, Tag,
  Search, TrendingUp, Workflow, PieChart, Percent, Database, Layers,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { computeFeatureRemovalProposal } from "@/lib/feature-removal";
import { useResumeState } from "@/hooks/use-resume-state";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/data-preparation")({
  head: () => ({ meta: [{ title: "Data Preparation & Feature Engineering — Aegis Credit" }] }),
  component: DataPreparation,
  validateSearch: (search: Record<string, unknown>) => ({
    tab: search.tab === "preprocessing" ? "preprocessing" : "profiling",
  }),
});

function DataPreparation() {
  const { profile } = useDataset();
  const search = Route.useSearch();
  const [tab, setTab] = useState<string>(search.tab);

  // Gate: the reviewer must have a completed profile (populated once the
  // dataset is uploaded and /data/profile has run) before Preprocessing &
  // Feature Engineering — which both depend on that profile — can open.
  const profilingComplete = Boolean(profile);

  return (
    <div className="space-y-8">
      <PageHeader
        title="Data Preparation & Feature Engineering"
        description="Profile the dataset, then clean, split, and engineer features for modeling."
      />

      <Tabs value={tab} onValueChange={setTab} className="w-full">
        <TabsList>
          <TabsTrigger value="profiling">Data Profiling</TabsTrigger>
          <TabsTrigger
            value="preprocessing"
            disabled={!profilingComplete}
            className={!profilingComplete ? "cursor-not-allowed opacity-50" : ""}
          >
            Preprocessing & Feature Engineering
          </TabsTrigger>
        </TabsList>

        <TabsContent value="profiling" className="space-y-8 pt-4">
          <ProfilingTab onProceed={() => setTab("preprocessing")} />
        </TabsContent>

        <TabsContent value="preprocessing" className="space-y-8 pt-4">
          <PreprocessingFeaturesTab onBackToProfiling={() => setTab("profiling")} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Sub-tab 1 — Data Profiling (moved from profiling.tsx, unchanged logic)
// ═══════════════════════════════════════════════════════════════════════

const CLASS_DISTRIBUTION_COLORS = ["#065f46", "#10b981", "#6ee7b7", "#a7f3d0"];

// Validated 2-slot categorical pair (blue/emerald) — ties the split chart's
// class colors to the app's primary accent instead of the unrelated lime green.
const SPLIT_CLASS_COLORS: Record<string, string> = { "0": "#059669", "1": "#2563EB" };
const SPLIT_CLASS_FALLBACK = "#94A3B8";

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-[11px] text-muted-foreground">{sub}</div>}
    </div>
  );
}

function formatCsvRow(row: Record<string, any>) {
  return Object.values(row)
    .map((value) => {
      if (value === undefined || value === null) return "";
      const text = String(value).replace(/"/g, '""');
      return `"${text}"`;
    })
    .join(",");
}

// Diverging green-family scale for correlation cells: teal = negative, emerald = positive,
// intensity scales with |value| so weak correlations read as near-neutral. Both hues stay
// within the app's green palette while remaining distinguishable by their blue undertone.
function correlationCellStyle(value: number): { backgroundColor: string; color: string } {
  const clamped = Math.max(-1, Math.min(1, value ?? 0));
  const intensity = Math.abs(clamped);
  const [r, g, b] = clamped >= 0 ? [5, 150, 105] : [13, 148, 136];
  return {
    backgroundColor: `rgba(${r}, ${g}, ${b}, ${(0.1 + intensity * 0.8).toFixed(2)})`,
    color: intensity > 0.5 ? "#ffffff" : "inherit",
  };
}

function severityRank(severity?: string): number {
  if (severity === "high") return 0;
  if (severity === "medium") return 1;
  if (severity === "low") return 2;
  return 3;
}

function severityClasses(severity?: string): string {
  if (severity === "high") return "border-red-500 bg-red-500/5 text-red-900";
  if (severity === "medium") return "border-amber-500 bg-amber-500/5 text-amber-900";
  if (severity === "low") return "border-emerald-500 bg-emerald-500/5 text-emerald-900";
  return "border-border bg-muted text-muted-foreground";
}

function severityBadgeClasses(severity?: string): string {
  if (severity === "high") return "bg-red-100 text-red-700";
  if (severity === "medium") return "bg-amber-100 text-amber-700";
  if (severity === "low") return "bg-emerald-100 text-emerald-700";
  return "bg-muted text-muted-foreground";
}

function SeverityIcon({ severity }: { severity?: string }) {
  if (severity === "high") return <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-red-600" />;
  if (severity === "medium") return <AlertCircle className="h-3.5 w-3.5 shrink-0 text-amber-600" />;
  if (severity === "low") return <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-600" />;
  return <Info className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />;
}

// Compliance rule text can run to full regulatory paragraphs (e.g. macro
// variable / leakage rules cite chapter and verse). Show only the first
// sentence as a summary; the full text is available on expand.
function summarizeFlag(text: string, maxLength = 110): string {
  if (!text) return "";
  const firstSentenceMatch = text.match(/^.*?[.!?](?=\s|$)/);
  const firstSentence = firstSentenceMatch ? firstSentenceMatch[0].trim() : text;
  if (firstSentence.length <= maxLength) return firstSentence;
  return `${firstSentence.slice(0, maxLength).trim()}…`;
}

function isFlagSummarized(text: string): boolean {
  return summarizeFlag(text) !== text.trim();
}

function ProfilingTab({ onProceed }: { onProceed: () => void }) {
  const { file, profile, setProfile } = useDataset();
  const navigate = useNavigate();
  const [selectedTarget, setSelectedTarget] = useState<string | null>(profile?.target_col ?? null);
  const [activeProfile, setActiveProfile] = useState(profile);
  const [isLoadingTarget, setIsLoadingTarget] = useState(false);
  const [targetError, setTargetError] = useState<string | null>(null);
  const [expandedFlags, setExpandedFlags] = useState<Set<number>>(new Set());

  const toggleFlagExpanded = (idx: number) => {
    setExpandedFlags((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  useEffect(() => {
    setActiveProfile(profile);
  }, [profile]);

  const availableTargets = profile?.target_candidates ?? [];
  const availableColumns = profile?.columns ?? [];
  const candidateDefault = availableColumns.includes("loan_status")
    ? "loan_status"
    : availableTargets.length > 0
    ? availableTargets[0]
    : availableColumns[0] ?? null;

  useEffect(() => {
    if (!selectedTarget && candidateDefault) {
      setSelectedTarget(candidateDefault);
    }
  }, [candidateDefault, selectedTarget]);

  useEffect(() => {
    if (!file || !selectedTarget || !profile) return;
    if (profile.target_col === selectedTarget && profile.class_distribution) {
      return;
    }

    const fetchTargetProfile = async () => {
      setIsLoadingTarget(true);
      setTargetError(null);
      try {
        const form = new FormData();
        form.append("file", file);
        form.append("target_col", selectedTarget);
        const result = await formUpload("/data/profile", form);
        setActiveProfile(result as any);
        setProfile(result as any);
      } catch (err: any) {
        setTargetError(err?.message ?? "Failed to update profile for selected target.");
      } finally {
        setIsLoadingTarget(false);
      }
    };

    fetchTargetProfile();
  }, [file, profile, selectedTarget, setProfile]);

  if (!profile) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 text-center">
        <h3 className="text-lg font-semibold">No dataset available</h3>
        <p className="mt-2 text-sm text-muted-foreground">Upload a dataset on the Data Upload page to run profiling and populate these diagnostics.</p>
      </div>
    );
  }

  const active = activeProfile ?? profile;
  const rows = active.shape?.[0] ?? null;
  const cols = active.shape?.[1] ?? null;
  const numericCount = active.numeric_feature_count ?? null;
  const categoricalCount = active.categorical_feature_count ?? null;
  const missingCells = active.missing_cells ?? null;
  const missingPct = active.missing_percentage ?? null;
  const duplicateRows = active.duplicate_rows ?? null;
  const duplicateRate = active.duplicate_rate ?? null;
  const outlierAnalysis = active.outlier_analysis ?? {};
  const outlierEntriesAll = Object.entries(outlierAnalysis as Record<string, any>);
  const outlierEntries = outlierEntriesAll.filter(([, info]) => ((info as any)?.outlier_fraction ?? 0) > 0);
  const classDistribution = active.class_distribution ?? null;
  const targetSummary = active.target_summary ?? null;
  const correlationColumns: string[] = active.correlation_matrix?.columns ?? [];
  const correlationValues: number[][] = active.correlation_matrix?.values ?? [];
  const dataDictionary = active.data_dictionary ?? [];
  const leakageRiskCols = active.leakage_risk_cols ?? [];
  const dateIntegrity = active.date_integrity ?? {};
  const dateIntegrityEntries = Object.entries(dateIntegrity);
  const agent2Flags = active.agent2_flags_data ?? [];
  const agent2Error = active.agent2_error ?? null;

  const classChartData = useMemo(() => {
    if (!classDistribution) return [];
    return Object.entries(classDistribution).map(([name, value]) => ({ name, value: Number(value) }));
  }, [classDistribution]);

  const classDistributionFigure = useMemo(() => {
    if (!classChartData || classChartData.length === 0) return null;
    return {
      data: [
        {
          type: "pie",
          labels: classChartData.map((entry) => entry.name),
          values: classChartData.map((entry) => entry.value),
          hole: 0.45,
          marker: {
            colors: classChartData.map((_, index) => CLASS_DISTRIBUTION_COLORS[index % CLASS_DISTRIBUTION_COLORS.length]),
          },
          textinfo: "percent",
          hovertemplate: "%{label}: %{value:,}<br>%{percent}<extra></extra>",
        },
      ],
      layout: {
        margin: { t: 10, r: 10, b: 10, l: 10 },
        legend: { orientation: "h", y: -0.15 },
      },
    };
  }, [classChartData]);

  const sortedFlags = useMemo(
    () => [...agent2Flags].sort((a: any, b: any) => severityRank(a.severity) - severityRank(b.severity)),
    [agent2Flags]
  );
  const flagSeverityCounts = useMemo(() => {
    const counts = { high: 0, medium: 0, low: 0 };
    for (const flag of agent2Flags as any[]) {
      if (flag.severity === "high") counts.high += 1;
      else if (flag.severity === "medium") counts.medium += 1;
      else if (flag.severity === "low") counts.low += 1;
    }
    return counts;
  }, [agent2Flags]);

  const downloadDataSummary = () => {
    const headers = dataDictionary.length > 0 ? Object.keys(dataDictionary[0]) : [];
    const csv = [headers.join(","), ...dataDictionary.map(formatCsvRow)].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "data_summary.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const taskTypeLabel = active.task_type === "binary"
    ? "Binary Classification"
    : active.task_type === "multiclass"
    ? "Multiclass Classification"
    : active.task_type === "regression"
    ? "Regression"
    : "Unspecified";

  const taskBadgeVariant = active.task_type === "binary" ? "default" : active.task_type === "multiclass" ? "secondary" : active.task_type === "regression" ? "outline" : "secondary";

  return (
    <div className="space-y-8">
      <section className="grid grid-cols-1 gap-4 xl:grid-cols-[2fr_1fr]">
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <Stat
            label="Total Rows"
            value={rows !== null ? rows.toLocaleString() : "—"}
            sub={active.dataset_name ?? "Number of records in the dataset"}
          />
          <Stat
            label="Total Columns"
            value={cols !== null ? String(cols) : "—"}
            sub={numericCount !== null && categoricalCount !== null ? `${numericCount} numeric · ${categoricalCount} categorical` : "Number of fields in the dataset"}
          />
          <Stat
            label="Missing Values"
            value={missingCells !== null ? missingCells.toLocaleString() : missingPct !== null ? `${missingPct}%` : "—"}
            sub={missingPct !== null ? `${missingPct}% of all data cells are empty` : undefined}
          />
          <Stat
            label="Duplicate Rows"
            value={duplicateRows !== null ? String(duplicateRows) : "—"}
            sub={duplicateRate !== null ? `${duplicateRate}% of rows are exact copies of another row` : "Rows that are exact copies of another row"}
          />
        </div>

        <div className="rounded-xl border border-border bg-card p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Target Task</div>
              <div className="mt-1 text-lg font-semibold text-foreground">{taskTypeLabel}</div>
            </div>
            <Badge variant={taskBadgeVariant}>{taskTypeLabel}</Badge>
          </div>
          <div className="mt-4 space-y-2 text-sm text-muted-foreground">
            <div>Detected target candidates: {availableTargets.length > 0 ? availableTargets.join(", ") : "None"}</div>
            <div>Preferred target: {candidateDefault ?? "Not detected"}</div>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-4">
        <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="text-base font-semibold">Target variable</h2>
              <p className="text-xs text-muted-foreground">Choose the target column to compute distribution, imbalance, and task diagnostics.</p>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-full lg:w-64">
                <Select value={selectedTarget ?? ""} onValueChange={(value) => setSelectedTarget(value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select target" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableColumns.map((column) => (
                      <SelectItem key={column} value={column}>{column}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Button variant="outline" size="sm" onClick={downloadDataSummary} className="shrink-0 gap-2">
                <Download className="h-4 w-4" />
                Download data summary
              </Button>
            </div>
          </div>
          {isLoadingTarget && (
            <div className="mt-4 rounded-xl border border-border bg-muted p-3 text-sm text-muted-foreground">Updating target diagnostics…</div>
          )}
          {targetError && (
            <div className="mt-4 rounded-xl border border-destructive bg-destructive/10 p-3 text-sm text-destructive">{targetError}</div>
          )}
          {agent2Error && (
            <div className="mt-4 rounded-xl border border-destructive bg-destructive/10 p-3 text-sm text-destructive">
              <div className="font-medium">Data compliance check could not be completed.</div>
              <div className="mt-1 text-xs">{agent2Error}</div>
            </div>
          )}
          {agent2Flags.length > 0 && (
            <div className="mt-4 rounded-xl border border-border bg-card">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
                <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                  <AlertTriangle className="h-4 w-4 text-amber-600" />
                  <span>{agent2Flags.length} compliance flag{agent2Flags.length === 1 ? "" : "s"}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  {flagSeverityCounts.high > 0 && (
                    <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${severityBadgeClasses("high")}`}>
                      {flagSeverityCounts.high} high
                    </span>
                  )}
                  {flagSeverityCounts.medium > 0 && (
                    <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${severityBadgeClasses("medium")}`}>
                      {flagSeverityCounts.medium} medium
                    </span>
                  )}
                  {flagSeverityCounts.low > 0 && (
                    <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${severityBadgeClasses("low")}`}>
                      {flagSeverityCounts.low} low
                    </span>
                  )}
                </div>
              </div>
              <div className="divide-y divide-border">
                {sortedFlags.map((flag: any, idx: number) => {
                  const isOpen = expandedFlags.has(idx);
                  const summary = summarizeFlag(flag.flag ?? "");
                  const hasMore = isFlagSummarized(flag.flag ?? "") || flag.observed_value != null || flag.suggestion || flag.source || flag.principle;
                  return (
                    <div key={`${flag.rule_id ?? "flag"}-${idx}`} className={`border-l-4 px-4 py-2.5 ${severityClasses(flag.severity)}`}>
                      <button
                        type="button"
                        onClick={() => hasMore && toggleFlagExpanded(idx)}
                        className={`flex w-full items-start gap-2 text-left ${hasMore ? "cursor-pointer" : "cursor-default"}`}
                      >
                        <SeverityIcon severity={flag.severity} />
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                            <span>{flag.rule_id ?? "?"}</span>
                            {flag.not_verifiable && <span className="italic normal-case">· not verifiable</span>}
                          </div>
                          <div className="mt-0.5 text-xs text-foreground">{summary}</div>
                        </div>
                        {hasMore && (
                          <ChevronDown className={`mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform ${isOpen ? "rotate-180" : ""}`} />
                        )}
                      </button>
                      {isOpen && hasMore && (
                        <div className="mt-2 space-y-1.5 pl-5 text-xs text-muted-foreground">
                          {isFlagSummarized(flag.flag ?? "") && <div>{flag.flag}</div>}
                          {flag.observed_value !== undefined && flag.observed_value !== null && (
                            <div>Observed: <code className="text-foreground">{String(flag.observed_value)}</code></div>
                          )}
                          {flag.suggestion && <div>💡 {flag.suggestion}</div>}
                          {(flag.source || flag.principle) && (
                            <div className="text-[11px] text-muted-foreground/70">
                              {flag.source}{flag.source && flag.principle ? " — " : ""}{flag.principle}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <div className="rounded-lg border border-border bg-background p-3">
              <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                <Info className="h-4 w-4" />
                <span>Quality checks</span>
              </div>
              <div className="mt-2 text-sm text-foreground">
                {targetSummary?.is_imbalanced ? "Target class imbalance detected." : "Target distribution appears balanced for the current profile."}
              </div>
              {targetSummary?.imbalance_ratio ? (
                <div className="mt-2 text-xs text-muted-foreground">Imbalance ratio: {targetSummary.imbalance_ratio}:1</div>
              ) : null}
            </div>
            <div className="rounded-lg border border-border bg-background p-3">
              <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                <AlertTriangle className="h-4 w-4" />
                <span>Leakage & dates</span>
              </div>
              <div className="mt-2 text-sm text-foreground">
                {leakageRiskCols.length > 0
                  ? `${leakageRiskCols.length} potential leakage column${leakageRiskCols.length === 1 ? "" : "s"}`
                  : "No strong leakage signals detected."}
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                {dateIntegrityEntries.length > 0
                  ? `${dateIntegrityEntries.length} date field${dateIntegrityEntries.length === 1 ? "" : "s"} checked for future/ancient values`
                  : "No date fields detected."}
              </div>
            </div>
          </div>

          <div className="mt-4 border-t border-border pt-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
              <h3 className="text-sm font-semibold">Class Distribution</h3>
            </div>
            <p className="text-xs text-muted-foreground">
              {selectedTarget ? (
                <>How the selected target column (<code className="text-foreground">{selectedTarget}</code>) is split across the dataset</>
              ) : (
                "Select a target column above to see its value breakdown."
              )}
            </p>

            {classDistribution ? (
              <div className="mt-4 grid items-center gap-6 lg:grid-cols-[1fr_1.2fr]">
                <div className="grid gap-3">
                  {classChartData.map((entry, index) => {
                    const total = classChartData.reduce((sum, e) => sum + e.value, 0);
                    const pct = total > 0 ? (entry.value / total) * 100 : 0;
                    return (
                      <div key={entry.name} className="rounded-lg border border-border bg-background p-4">
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-2 text-sm text-muted-foreground">
                            <span
                              className="h-2.5 w-2.5 shrink-0 rounded-full"
                              style={{ backgroundColor: CLASS_DISTRIBUTION_COLORS[index % CLASS_DISTRIBUTION_COLORS.length] }}
                            />
                            {selectedTarget} = {entry.name}
                          </span>
                          <span className="text-xl font-semibold tabular-nums text-foreground">{entry.value.toLocaleString()}</span>
                        </div>
                        <div className="mt-1 pl-[18px] text-xs text-muted-foreground">{pct.toFixed(1)}% of records</div>
                      </div>
                    );
                  })}
                </div>
                <div className="h-72">
                  <PlotlyChart
                    figure={classDistributionFigure}
                    style={{ height: "100%", minHeight: "100%" }}
                    config={{ displayModeBar: false }}
                  />
                </div>
              </div>
            ) : (
              <div className="mt-4 rounded-xl border border-border bg-muted p-4 text-sm text-muted-foreground">Class distribution is not available until a valid target is selected.</div>
            )}
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-4 lg:grid-cols-[0.55fr_1.45fr]">
        <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-base font-semibold">Duplicate and outlier signals</h2>
          </div>
          <div className="mt-4 space-y-3 text-sm text-muted-foreground">
            <div className="rounded-lg border border-border bg-background p-3">
              <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Duplicate rate</div>
              <div className="mt-1 text-lg font-semibold text-foreground">{duplicateRate !== null ? `${duplicateRate}%` : "—"}</div>
              <div className="mt-1 text-xs">{duplicateRows !== null ? `${duplicateRows.toLocaleString()} duplicate row${duplicateRows === 1 ? "" : "s"}` : "No duplicate count available"}</div>
            </div>
            <div className="rounded-lg border border-border bg-background p-3">
              <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Outlier checks</div>
              <div className="mt-1 text-sm text-foreground">
                {outlierEntries.length > 0 ? (
                  <>
                    {[...outlierEntries]
                      .sort(([, a], [, b]) => ((b as any).outlier_fraction ?? 0) - ((a as any).outlier_fraction ?? 0))
                      .slice(0, 4)
                      .map(([column, info]) => (
                        <div key={column} className="mt-2 flex items-center justify-between gap-2">
                          <span className="truncate">{column}</span>
                          <span className="shrink-0 font-medium tabular-nums">{(((info as any).outlier_fraction ?? 0) * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    {outlierEntries.length > 4 && (
                      <div className="mt-2 text-xs text-muted-foreground">+{outlierEntries.length - 4} more column{outlierEntries.length - 4 === 1 ? "" : "s"} with outliers</div>
                    )}
                  </>
                ) : outlierEntriesAll.length > 0 ? (
                  "No numeric columns have flagged outliers."
                ) : (
                  "No numeric outlier analysis available."
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
          <div className="flex items-center gap-2">
            <Info className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-base font-semibold">Correlation snapshot</h2>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Pearson correlation across numeric features (up to 10 columns).</p>
          <div className="mt-4">
            {correlationColumns.length > 0 && correlationValues.length > 0 ? (
              <>
                <div className="flex justify-center overflow-x-auto">
                  <table style={{ borderCollapse: "separate", borderSpacing: "4px", width: "100%", maxWidth: "480px" }}>
                    <thead>
                      <tr>
                        <th className="p-0.5" />
                        {correlationColumns.map((column) => (
                          <th
                            key={column}
                            title={column}
                            className="max-w-[56px] truncate p-0.5 text-[10px] font-medium text-muted-foreground"
                          >
                            {column.length > 7 ? `${column.slice(0, 6)}…` : column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {correlationColumns.map((rowColumn, rowIndex) => (
                        <tr key={rowColumn}>
                          <th
                            title={rowColumn}
                            className="whitespace-nowrap p-0.5 pr-2 text-right text-[10px] font-medium text-muted-foreground"
                          >
                            {rowColumn.length > 10 ? `${rowColumn.slice(0, 9)}…` : rowColumn}
                          </th>
                          {(correlationValues[rowIndex] ?? []).map((value, colIndex) => (
                            <td
                              key={`${rowColumn}-${correlationColumns[colIndex]}`}
                              title={`${rowColumn} × ${correlationColumns[colIndex]}: ${value.toFixed(2)}`}
                              className="h-9 w-9 rounded-md text-center align-middle text-xs font-medium tabular-nums"
                              style={correlationCellStyle(value)}
                            >
                              {value.toFixed(2)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mx-auto mt-3 flex max-w-[480px] items-center gap-2 text-xs text-muted-foreground">
                  <span>-1</span>
                  <div
                    className="h-2 flex-1 rounded-full"
                    style={{ background: "linear-gradient(to right, rgba(13,148,136,0.9), rgba(148,163,184,0.15), rgba(5,150,105,0.9))" }}
                  />
                  <span>+1</span>
                </div>
              </>
            ) : (
              <div className="rounded-lg border border-border bg-background p-3 text-sm text-muted-foreground">No numeric correlation matrix available for this dataset.</div>
            )}
          </div>
        </div>
      </section>

      <div className="flex gap-3 pt-4">
        <Button variant="outline" onClick={() => navigate({ to: "/data-upload" })} className="gap-2">
          <ArrowLeft className="h-4 w-4" />
          Back to Data Upload
        </Button>
        <Button onClick={onProceed} className="gap-2 ml-auto">
          Proceed to Preprocessing
          <ArrowRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Sub-tab 2 — Preprocessing & Feature Engineering
// (moved from preprocessing.tsx + features.tsx, unchanged logic)
// ═══════════════════════════════════════════════════════════════════════

// ── Types for the interactive missing-value / transform workflow ───────────
type TreatmentEvidence = { missing_pct?: number; unique_values?: unknown[]; skewness?: number };
type TreatmentInfo = { treatment: string; reason: string; evidence: TreatmentEvidence };
type TransformRecommendation = {
  transform: "none" | "log1p" | "yeo_johnson";
  skew: number;
  post_transform_skew: number | null;
  reason: string;
  default_on: boolean;
};

const TREATMENT_LABELS: Record<string, string> = {
  unknown_category: "Unknown category",
  zero_fill: "Zero-fill",
  statistical: "Statistical",
  review_flag: "Review (sparse)",
};
const TREATMENT_OPTIONS = ["unknown_category", "zero_fill", "statistical", "review_flag"];

const TRANSFORM_LABELS: Record<string, string> = {
  none: "None",
  log1p: "Log",
  yeo_johnson: "Yeo-Johnson",
};
const TRANSFORM_OPTIONS = ["none", "log1p", "yeo_johnson"];

interface FeatureEngineeringResponse {
  // Type -> column-list map (e.g. { numeric: [...], categorical: [...] }),
  // not a per-column lookup — see resolveFeatureType() below for inversion.
  col_types?: Record<string, string[]>;
  target_col?: string;
  task_type?: string;
  feature_engineering_plan?: any;
  feature_engineering_summary?: any;
  engineered_feature_names?: string[];
  selected_features?: string[];
  dropped_features?: string[];
  encoding_summary?: Record<string, any>;
  feature_engineering_report?: Record<string, any>;
  feature_importance_summary?: Record<string, any>;
  x_engineered_shape?: number[];
  x_engineered_preview?: any[];
  final_engineered_dataset_preview?: any[];
  x_engineered_csv?: string;
  gini_scores?: Record<string, number>;
  ead_configuration?: {
    mode?: string;
    source_col?: string;
    method?: string;
    available?: boolean;
    missing_columns?: string[];
    selected?: Record<string, any>;
    summary?: Record<string, number | null>;
  };
  available_numeric_columns?: string[];
  interaction_features?: Array<{
    name?: string;
    feature_a?: string;
    feature_b?: string;
    type?: string;
    interaction_type?: string;
    score?: number;
    gini?: number;
    source?: string;
  }>;
}

// ═══════════════════════════════════════════════════════════════════════
// Dashboard visual primitives — presentation only. Every value they render
// is passed in from real preprocessing/feature-engineering state; none of
// them fetch data or hold data-shaping logic of their own.
// ═══════════════════════════════════════════════════════════════════════

// Inverts feature_engineering_plan.col_types ({numeric:[...], categorical:[...]}
// etc — a type→column-list map, not a per-column lookup) to find one column's
// type. Names outside every bucket are engineered/derived columns (e.g. a
// `_log`/`_woe`/`_bin` suffix or an interaction name) that never existed as a
// raw column.
function resolveFeatureType(colTypes: Record<string, string[]> | undefined, feature: string): string {
  if (!colTypes) return "Unknown";
  for (const [type, cols] of Object.entries(colTypes)) {
    if (Array.isArray(cols) && cols.includes(feature)) {
      return type.charAt(0).toUpperCase() + type.slice(1);
    }
  }
  return "Engineered";
}

// missing_pct from the backend is a fraction (0–1), matching how the existing
// Missing Value Treatment card already reads it.
function missingSeverityColor(fraction: number): { bar: string; text: string } {
  const pct = fraction * 100;
  if (pct >= 8) return { bar: "bg-red-500", text: "text-red-700" };
  if (pct >= 3) return { bar: "bg-amber-500", text: "text-amber-700" };
  if (pct > 0) return { bar: "bg-emerald-500", text: "text-emerald-700" };
  return { bar: "bg-slate-200", text: "text-slate-400" };
}

// Mirrors the thresholds severityBadge() already uses for the skew badge, so
// the bar and the badge next to it always agree on severity.
function skewBarColor(skew: number): string {
  const abs = Math.abs(skew);
  if (abs >= 2.0) return "bg-red-500";
  if (abs >= 1.5) return "bg-amber-500";
  return "bg-yellow-500";
}

function MiniBar({ fraction, colorClass }: { fraction: number; colorClass: string }) {
  const width = Number.isFinite(fraction) ? Math.max(0, Math.min(100, fraction * 100)) : 0;
  return (
    <div className="h-1.5 min-w-[48px] flex-1 overflow-hidden rounded-full bg-slate-100">
      <div className={cn("h-full rounded-full transition-all", colorClass)} style={{ width: `${width}%` }} />
    </div>
  );
}

function KpiTile({
  icon: Icon,
  label,
  value,
  sub,
  tone = "primary",
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  sub?: string;
  tone?: "primary" | "amber" | "emerald" | "rose" | "violet";
}) {
  const toneClasses: Record<string, string> = {
    primary: "bg-blue-500/10 text-blue-600",
    amber: "bg-amber-500/10 text-amber-600",
    emerald: "bg-emerald-500/10 text-emerald-600",
    rose: "bg-rose-500/10 text-rose-600",
    violet: "bg-violet-500/10 text-violet-600",
  };
  return (
    <div className="bg-white p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-500">{label}</span>
        <span className={cn("flex h-6 w-6 shrink-0 items-center justify-center rounded-md", toneClasses[tone])}>
          <Icon className="h-3.5 w-3.5" />
        </span>
      </div>
      <div className="mt-2 text-2xl font-bold tabular-nums text-slate-900">{value}</div>
      {sub && <div className="mt-0.5 truncate text-[11px] text-slate-500">{sub}</div>}
    </div>
  );
}

function KpiStrip({ tiles }: { tiles: Array<{ icon: React.ComponentType<{ className?: string }>; label: string; value: string; sub?: string; tone?: "primary" | "amber" | "emerald" | "rose" | "violet" }> }) {
  return (
    <div
      className="grid grid-cols-2 divide-x divide-y divide-slate-100 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm sm:grid-cols-3 lg:grid-cols-5 lg:divide-y-0"
    >
      {tiles.map((tile) => (
        <KpiTile key={tile.label} {...tile} />
      ))}
    </div>
  );
}

const SPLIT_COLORS = [
  { bg: "bg-blue-600", border: "border-blue-600", text: "text-blue-600" },
  { bg: "bg-violet-600", border: "border-violet-600", text: "text-violet-600" },
  { bg: "bg-emerald-600", border: "border-emerald-600", text: "text-emerald-600" },
] as const;

function SplitProportionBar({
  trainPct, valPct, testPct, trainN, valN, testN,
}: {
  trainPct: number | null; valPct: number | null; testPct: number | null;
  trainN: number | null; valN: number | null; testN: number | null;
}) {
  const segments = [
    { label: "Train", pct: trainPct, n: trainN, role: "Model fitting", c: SPLIT_COLORS[0] },
    { label: "Validation", pct: valPct, n: valN, role: "Hyperparameter tuning", c: SPLIT_COLORS[1] },
    { label: "Test", pct: testPct, n: testN, role: "Final evaluation", c: SPLIT_COLORS[2] },
  ];
  const hasData = segments.every((s) => s.pct !== null);

  return (
    <div>
      <div className="flex h-8 gap-0.5 overflow-hidden rounded-lg">
        {hasData ? (
          segments.map((s) => (
            <div
              key={s.label}
              className={cn("flex items-center justify-center text-[10.5px] font-bold text-white", s.c.bg)}
              style={{ width: `${s.pct}%` }}
              title={`${s.label}: ${s.pct?.toFixed(1)}%`}
            >
              {(s.pct ?? 0) > 12 ? s.label.slice(0, 5).toUpperCase() : ""}
            </div>
          ))
        ) : (
          <div className="flex-1 rounded-lg bg-slate-100" />
        )}
      </div>
      <div className="mt-4 grid grid-cols-3 gap-3">
        {segments.map((s) => (
          <div key={s.label} className={cn("rounded-lg border-l-4 bg-slate-50 p-3", s.c.border)}>
            <div className="flex items-baseline gap-1.5">
              <span className="text-lg font-bold tabular-nums text-slate-900">{s.n !== null ? s.n.toLocaleString() : "—"}</span>
              {s.pct !== null && <span className={cn("text-xs font-bold", s.c.text)}>{s.pct.toFixed(1)}%</span>}
            </div>
            <div className="text-xs font-semibold text-slate-900">{s.label}</div>
            <div className="text-[11px] text-slate-500">{s.role}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DataQualityDonut({ completeness }: { completeness: number | null }) {
  const radius = 34;
  const circumference = 2 * Math.PI * radius;
  const pct = completeness !== null ? Math.max(0, Math.min(100, completeness)) : 0;
  const offset = circumference - (pct / 100) * circumference;
  return (
    <div className="relative flex h-20 w-20 shrink-0 items-center justify-center">
      <svg width="80" height="80" className="-rotate-90">
        <circle cx="40" cy="40" r={radius} fill="none" stroke="#f1f5f9" strokeWidth="7" />
        {completeness !== null && (
          <circle
            cx="40" cy="40" r={radius} fill="none" stroke="#10b981" strokeWidth="7"
            strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round"
          />
        )}
      </svg>
      <div className="absolute text-center">
        <div className="text-base font-bold text-slate-900">{completeness !== null ? `${completeness.toFixed(0)}%` : "—"}</div>
        <div className="text-[8.5px] font-semibold uppercase tracking-wide text-slate-500">Complete</div>
      </div>
    </div>
  );
}

function DataQualitySnapshot({
  completeness, duplicateRate, outlierColumnCount, isImbalanced, imbalanceRatio, leakageColumnCount, dateFieldCount,
}: {
  completeness: number | null;
  duplicateRate: number | null;
  outlierColumnCount: number | null;
  isImbalanced: boolean | null;
  imbalanceRatio: number | null;
  leakageColumnCount: number | null;
  dateFieldCount: number | null;
}) {
  const rows: Array<{ label: string; display: string; status: "ok" | "warn" | "neutral" }> = [
    { label: "Duplicate rows", display: duplicateRate !== null ? `${duplicateRate}%` : "—", status: duplicateRate === null ? "neutral" : duplicateRate > 1 ? "warn" : "ok" },
    { label: "Outlier columns", display: outlierColumnCount !== null ? String(outlierColumnCount) : "—", status: outlierColumnCount === null ? "neutral" : outlierColumnCount > 0 ? "warn" : "ok" },
    { label: "Target balance", display: isImbalanced === null ? "—" : isImbalanced ? `${imbalanceRatio ?? "?"}:1` : "Balanced", status: isImbalanced === null ? "neutral" : isImbalanced ? "warn" : "ok" },
    { label: "Leakage risk columns", display: leakageColumnCount !== null ? String(leakageColumnCount) : "—", status: leakageColumnCount === null ? "neutral" : leakageColumnCount > 0 ? "warn" : "ok" },
    { label: "Date fields checked", display: dateFieldCount !== null ? (dateFieldCount > 0 ? String(dateFieldCount) : "None") : "—", status: "neutral" },
  ];
  const dotClass: Record<string, string> = { ok: "bg-emerald-500", warn: "bg-amber-500", neutral: "bg-slate-300" };

  return (
    <div className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-1.5 text-sm font-semibold text-slate-900">
            <PieChart className="h-3.5 w-3.5 text-slate-400" />
            Data Quality Snapshot
          </div>
          <div className="text-xs text-slate-500">From profiling diagnostics</div>
        </div>
        <DataQualityDonut completeness={completeness} />
      </div>
      <div className="mt-5 space-y-2.5">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-2.5">
            <span className={cn("h-2 w-2 shrink-0 rounded-full", dotClass[r.status])} />
            <span className="flex-1 text-xs font-medium text-slate-600">{r.label}</span>
            <span className="text-xs font-bold tabular-nums text-slate-900">{r.display}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const PIPELINE_CHIP_COLORS = ["bg-slate-600", "bg-sky-600", "bg-violet-600", "bg-blue-600", "bg-emerald-600"];

function PipelineFlow({ stages }: { stages: Array<{ label: string; value: number | null; sub?: string }> }) {
  const anyData = stages.some((s) => s.value !== null);
  if (!anyData) return null;
  return (
    <div className="rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 to-slate-800 p-5 shadow-sm">
      <div className="flex items-center gap-1.5 text-sm font-semibold text-slate-100">
        <Workflow className="h-4 w-4 text-slate-400" />
        Feature Engineering Pipeline
      </div>
      <div className="mt-1 text-xs text-slate-400">End-to-end preprocessing workflow, in order</div>
      <div className="mt-5 flex items-center gap-1 overflow-x-auto pb-1">
        {stages.map((s, i) => (
          <div key={s.label} className="flex items-center">
            <div className="flex flex-col items-center gap-1.5">
              <div className={cn("flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-white/10 text-sm font-bold text-white", PIPELINE_CHIP_COLORS[i % PIPELINE_CHIP_COLORS.length])}>
                {s.value !== null ? s.value : "—"}
              </div>
              <div className="text-center">
                <div className="whitespace-nowrap text-[10.5px] font-semibold text-slate-200">{s.label}</div>
                {s.sub && <div className="whitespace-nowrap text-[9.5px] text-slate-500">{s.sub}</div>}
              </div>
            </div>
            {i < stages.length - 1 && <ArrowRight className="mx-2 mb-5 h-3.5 w-3.5 shrink-0 text-slate-600" />}
          </div>
        ))}
      </div>
    </div>
  );
}

function TransformationsAppliedCard({ rows }: { rows: Array<{ transform?: string }> }) {
  if (!rows || rows.length === 0) return null;
  const labelFor = (t?: string) => {
    if (!t || t === "-" || t === "none") return "No transform";
    if (t === "log1p") return "Log transform";
    if (t === "yeo_johnson") return "Yeo-Johnson";
    return t;
  };
  const counts = new Map<string, number>();
  rows.forEach((r) => {
    const label = labelFor(r.transform);
    counts.set(label, (counts.get(label) ?? 0) + 1);
  });
  const palette = ["bg-blue-600", "bg-violet-600", "bg-emerald-600", "bg-slate-300"];
  const entries = Array.from(counts.entries());
  const total = rows.length;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="text-sm font-semibold text-slate-900">Transformations Applied</div>
      <div className="text-xs text-slate-500">
        {total} feature{total === 1 ? "" : "s"} across {entries.length} strateg{entries.length === 1 ? "y" : "ies"}
      </div>
      <div className="mt-4 flex h-3 gap-0.5 overflow-hidden rounded-full">
        {entries.map(([label, count], i) => (
          <div key={label} className={palette[i % palette.length]} style={{ width: `${(count / total) * 100}%` }} title={`${label}: ${count}`} />
        ))}
      </div>
      <div className="mt-4 space-y-2">
        {entries.map(([label, count], i) => (
          <div key={label} className="flex items-center gap-2.5">
            <span className={cn("h-2.5 w-2.5 shrink-0 rounded-sm", palette[i % palette.length])} />
            <span className="flex-1 text-xs font-medium text-slate-600">{label}</span>
            <span className="text-xs font-bold tabular-nums text-slate-900">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

type FeatureOverviewRow = {
  feature: string;
  type: string;
  iv: number | null;
  gini: number | null;
  missingPct: number | null;
  status: "selected" | "review" | "removed";
};

const FEATURE_STATUS_BADGE: Record<string, { label: string; className: string }> = {
  selected: { label: "Selected", className: "bg-emerald-100 text-emerald-700" },
  review: { label: "Review", className: "bg-amber-100 text-amber-700" },
  removed: { label: "Removed", className: "bg-rose-100 text-rose-700" },
};

function FeatureOverviewTable({ rows }: { rows: FeatureOverviewRow[] }) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "selected" | "review" | "removed">("all");

  const filtered = rows.filter((r) => {
    const matchesSearch = r.feature.toLowerCase().includes(search.toLowerCase());
    const matchesFilter = filter === "all" || r.status === filter;
    return matchesSearch && matchesFilter;
  });

  const counts = {
    selected: rows.filter((r) => r.status === "selected").length,
    review: rows.filter((r) => r.status === "review").length,
    removed: rows.filter((r) => r.status === "removed").length,
  };

  const ivMax = Math.max(0.1, ...rows.map((r) => r.iv ?? 0));
  const giniMax = Math.max(0.1, ...rows.map((r) => Math.abs(r.gini ?? 0)));

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-5 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-slate-900">Feature Overview</div>
            <div className="mt-1 flex flex-wrap gap-3 text-[11.5px] font-semibold">
              <span className="text-emerald-700">{counts.selected} selected</span>
              <span className="text-amber-700">{counts.review} review</span>
              <span className="text-rose-700">{counts.removed} removed</span>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search features…"
                className="h-8 w-44 rounded-lg border border-slate-200 bg-slate-50 pl-8 pr-3 text-xs text-slate-700 outline-none focus:border-blue-400"
              />
            </div>
            {(["all", "selected", "review", "removed"] as const).map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setFilter(v)}
                className={cn(
                  "rounded-md border px-2.5 py-1 text-[11px] font-semibold capitalize transition-colors",
                  filter === v ? "border-blue-600 bg-blue-50 text-blue-700" : "border-slate-200 bg-white text-slate-500 hover:border-slate-300",
                )}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="p-6 text-center text-sm text-slate-500">No feature-level data available for this dataset yet.</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-sm">
            <thead>
              <tr className="bg-slate-50">
                {["Feature", "Type", "IV", "Gini", "Missing %", "Status"].map((col) => (
                  <th key={col} className="whitespace-nowrap px-4 py-2.5 text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-500">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((r, i) => (
                <tr key={r.feature} className={cn("border-t border-slate-100", i % 2 === 1 && "bg-slate-50/40")}>
                  <td className="px-4 py-2.5 font-mono text-xs font-semibold text-slate-900">{r.feature}</td>
                  <td className="px-4 py-2.5">
                    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10.5px] font-semibold text-slate-600">{r.type}</span>
                  </td>
                  <td className="px-4 py-2.5">
                    {r.iv !== null ? (
                      <div className="flex items-center gap-2">
                        <span className="w-9 shrink-0 font-bold tabular-nums text-slate-900">{r.iv.toFixed(2)}</span>
                        <MiniBar fraction={r.iv / ivMax} colorClass="bg-blue-600" />
                      </div>
                    ) : <span className="text-slate-300">—</span>}
                  </td>
                  <td className="px-4 py-2.5">
                    {r.gini !== null ? (
                      <div className="flex items-center gap-2">
                        <span className="w-9 shrink-0 font-bold tabular-nums text-slate-900">{r.gini.toFixed(2)}</span>
                        <MiniBar fraction={Math.abs(r.gini) / giniMax} colorClass="bg-violet-600" />
                      </div>
                    ) : <span className="text-slate-300">—</span>}
                  </td>
                  <td className="px-4 py-2.5">
                    {r.missingPct !== null ? (
                      <span className={cn("font-semibold tabular-nums", missingSeverityColor(r.missingPct).text)}>{(r.missingPct * 100).toFixed(1)}%</span>
                    ) : <span className="text-slate-300">—</span>}
                  </td>
                  <td className="px-4 py-2.5">
                    <span className={cn("rounded-full px-2 py-0.5 text-[10.5px] font-semibold", FEATURE_STATUS_BADGE[r.status].className)}>{FEATURE_STATUS_BADGE[r.status].label}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && (
            <div className="p-6 text-center text-sm text-slate-500">No features match this search/filter.</div>
          )}
        </div>
      )}
    </div>
  );
}

function FeatureImportanceBars({ items, metricLabel }: { items: Array<{ name: string; score: number }>; metricLabel: string }) {
  if (items.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center gap-1.5 text-sm font-semibold text-slate-900">
          <TrendingUp className="h-3.5 w-3.5 text-slate-400" />
          Feature Importance
        </div>
        <div className="mt-4 text-sm text-slate-500">No importance scores are available for this dataset or task type yet.</div>
      </div>
    );
  }
  const max = Math.max(...items.map((i) => Math.abs(i.score)), 0.01);
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-1.5 text-sm font-semibold text-slate-900">
        <TrendingUp className="h-3.5 w-3.5 text-slate-400" />
        Feature Importance
      </div>
      <div className="text-xs text-slate-500">{metricLabel}</div>
      <div className="mt-5 space-y-2.5">
        {items.map((item, i) => (
          <div key={item.name} className="flex items-center gap-3">
            <span className="w-4 shrink-0 text-right text-[10px] font-bold text-slate-400">{i + 1}</span>
            <span className="w-40 shrink-0 truncate font-mono text-[11.5px] font-semibold text-slate-700" title={item.name}>{item.name}</span>
            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full bg-gradient-to-r from-blue-700 to-blue-500"
                style={{ width: `${(Math.abs(item.score) / max) * 100}%` }}
              />
            </div>
            <span className="w-12 shrink-0 text-right text-xs font-bold tabular-nums text-slate-900">{item.score.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function PreprocessingFeaturesTab({ onBackToProfiling }: { onBackToProfiling: () => void }) {
  const { profile } = useDataset();

  if (!profile) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 text-center">
        <h3 className="text-lg font-semibold">No dataset available</h3>
        <p className="mt-2 text-sm text-muted-foreground">Upload a dataset on the Data Upload page before preprocessing can run.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <PreprocessingSection onBackToProfiling={onBackToProfiling} />
      <FeaturesSection />
    </div>
  );
}

function PreprocessingSection({ onBackToProfiling }: { onBackToProfiling: () => void }) {
  const { profile, file, preprocessingResult, setPreprocessingResult } = useDataset();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Seed local preview state from context so returning to this page (e.g. via
  // Back from a later step) doesn't lose the split/preprocessing already run.
  const [preprocess, setPreprocess] = useState<any>(preprocessingResult ?? null);
  const [testSize, setTestSize] = useState(preprocessingResult?.split_config?.test_size ?? 0.15);
  const [valSize, setValSize] = useState(preprocessingResult?.split_config?.val_size ?? 0.15);

  // Resume where the reviewer left off: if this session has no preprocessing
  // result yet, pull the last saved /data/preprocess run from the backend.
  const { data: resumedPreprocess } = useResumeState<Record<string, any>>("dev_pipeline_log.csv", "preprocessing");
  useEffect(() => {
    if (!preprocessingResult && resumedPreprocess) {
      setPreprocessingResult(resumedPreprocess);
      setPreprocess(resumedPreprocess);
      setTestSize(resumedPreprocess?.split_config?.test_size ?? 0.15);
      setValSize(resumedPreprocess?.split_config?.val_size ?? 0.15);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumedPreprocess]);
  // Not user-configurable in the UI — hardcoded to match the backend default.
  // Stage 4's seed-stability check (R4.6) varies the seed programmatically via
  // its own control on the Model Validation screen, independent of this value.
  const randomSeed = 42;

  // ── Reviewer's confirmed choices — sent back to the API on every call ──
  const [treatmentOverrides, setTreatmentOverrides] = useState<Record<string, string>>({});
  const [dropCols, setDropCols] = useState<Record<string, boolean>>({});
  const [transformChoices, setTransformChoices] = useState<Record<string, string>>({});
  const [strategyOverride, setStrategyOverride] = useState<string | null>(null);
  const initializedDefaults = useRef(false);

  // ── On-demand "impact of dropping this feature" analysis (review_flag
  //    columns only) — fetched lazily per column when the reviewer expands
  //    it, not for every column on every /data/preprocess call. ──
  const [dropImpactOpen, setDropImpactOpen] = useState<Record<string, boolean>>({});
  const [dropImpactLoading, setDropImpactLoading] = useState<Record<string, boolean>>({});
  const [dropImpactError, setDropImpactError] = useState<Record<string, string>>({});
  const [dropImpact, setDropImpact] = useState<Record<string, any>>({});

  const fetchDropImpact = async (col: string) => {
    if (!file || !preprocess?.target_col) return;
    setDropImpactLoading((prev) => ({ ...prev, [col]: true }));
    setDropImpactError((prev) => ({ ...prev, [col]: "" }));
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("target_col", preprocess.target_col);
      form.append("test_size", String(testSize));
      form.append("val_size", String(valSize));
      form.append("random_seed", String(randomSeed));
      form.append("columns", JSON.stringify([col]));
      const result = await formUpload("/data/drop-impact", form);
      const impact = (result as any)?.drop_impact?.[col];
      if (impact?.error) {
        setDropImpactError((prev) => ({ ...prev, [col]: impact.error }));
      } else {
        setDropImpact((prev) => ({ ...prev, [col]: impact }));
      }
    } catch (err: any) {
      setDropImpactError((prev) => ({
        ...prev,
        [col]: err?.body?.detail ?? err?.message ?? "Impact analysis failed.",
      }));
    } finally {
      setDropImpactLoading((prev) => ({ ...prev, [col]: false }));
    }
  };

  const toggleDropImpact = (col: string) => {
    const nextOpen = !dropImpactOpen[col];
    setDropImpactOpen((prev) => ({ ...prev, [col]: nextOpen }));
    if (nextOpen && !dropImpact[col] && !dropImpactLoading[col]) {
      fetchDropImpact(col);
    }
  };

  // Consumed once: if we mounted with a cached result already in context (e.g.
  // navigating back from Feature Engineering/Training and forward again), skip
  // the very next auto-run and reuse it instead of silently re-POSTing and
  // potentially producing a fresh (if non-deterministic) split. Any later
  // change the reviewer makes to test size, seed, or treatments still runs.
  const skipInitialAutoRun = useRef(preprocessingResult !== null);

  useEffect(() => {
    const runPreprocess = async () => {
      if (!profile) return;

      if (skipInitialAutoRun.current) {
        skipInitialAutoRun.current = false;
        return;
      }

      const allColumns = Array.isArray(profile.columns) ? profile.columns : [];
      let targetCol: string | null = null;

      if (allColumns.includes("loan_status")) {
        targetCol = "loan_status";
      } else if (Array.isArray(profile.target_candidates) && profile.target_candidates.length > 0) {
        targetCol = profile.target_candidates[0];
      } else if (typeof profile.target_col === "string" && profile.target_col.trim() !== "") {
        targetCol = profile.target_col;
      }

      if (!targetCol || targetCol === "string" || targetCol.trim() === "") {
        setError("No valid target column found. Please upload a dataset with a recognized target variable.");
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const form = new FormData();
        if (file) {
          form.append("file", file);
        }
        form.append("target_col", targetCol);
        form.append("test_size", String(testSize));
        form.append("val_size", String(valSize));
        form.append("random_seed", String(randomSeed));
        form.append("treatment_overrides", JSON.stringify(treatmentOverrides));
        form.append(
          "drop_cols",
          JSON.stringify(Object.entries(dropCols).filter(([, v]) => v).map(([k]) => k)),
        );
        form.append("transform_choices", JSON.stringify(transformChoices));
        if (strategyOverride) {
          form.append("strategy_override", strategyOverride);
        }

        const result = await formUpload("/data/preprocess", form);
        setPreprocess(result);
        // Publish to shared context so Training (which no longer re-splits)
        // can read split_stats / split_config directly.
        setPreprocessingResult(result);

        // Seed local selection state from the platform's proposal, but only
        // ONCE — after that, the reviewer's own edits are what's sent back,
        // never silently overwritten by a fresh proposal on a later call.
        if (!initializedDefaults.current) {
          const proposal = (result as any)?.missing_treatment_proposal ?? {};
          const recommendations = (result as any)?.transform_recommendations ?? {};

          const seededDrop: Record<string, boolean> = {};
          Object.entries(proposal).forEach(([col, info]: [string, any]) => {
            if (info?.treatment === "review_flag") seededDrop[col] = true;
          });

          const seededTransforms: Record<string, string> = {};
          Object.entries(recommendations).forEach(([col, rec]: [string, any]) => {
            if (rec?.transform && rec.transform !== "none") {
              seededTransforms[col] = rec.transform;
            }
          });

          if (Object.keys(seededDrop).length > 0) setDropCols(seededDrop);
          if (Object.keys(seededTransforms).length > 0) setTransformChoices(seededTransforms);
          initializedDefaults.current = true;
        }
      } catch (err: any) {
        setError(err?.body?.detail ?? err?.message ?? "Preprocessing failed.");
        setPreprocess(null);
      } finally {
        setLoading(false);
      }
    };

    runPreprocess();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    profile, file, testSize, valSize, randomSeed,
    treatmentOverrides, dropCols, transformChoices, strategyOverride,
  ]);

  useEffect(() => {
    if (!preprocess?.split_config) return;
    setTestSize(preprocess.split_config.test_size ?? 0.15);
    setValSize(preprocess.split_config.val_size ?? 0.15);
  }, [preprocess?.split_config]);

  const strategySummary = Array.isArray(preprocess?.preprocessing_strategy_summary)
    ? preprocess.preprocessing_strategy_summary
    : [];

  const splitStats = preprocess?.split_stats ?? {};
  const classDistributionData = useMemo(() => {
    if (!Array.isArray(preprocess?.class_distribution_chart)) return [];
    const grouped: Record<string, Record<string, number>> = {};
    preprocess.class_distribution_chart.forEach((item: any) => {
      const split = item.split ?? "";
      const klass = item.class ?? "";
      const proportion = Number(item.proportion) ?? 0;
      if (!grouped[split]) grouped[split] = { split } as Record<string, number>;
      grouped[split][klass] = proportion;
    });
    return Object.values(grouped);
  }, [preprocess?.class_distribution_chart]);

  const classKeys = useMemo(() => {
    if (!Array.isArray(preprocess?.class_distribution_chart)) return [];
    return Array.from(new Set(preprocess.class_distribution_chart.map((item: any) => String(item.class))));
  }, [preprocess?.class_distribution_chart]);

  const classDistributionFigure = useMemo(() => {
    if (!classDistributionData || classDistributionData.length === 0) return null;
    const x = classDistributionData.map((d: any) => d.split ?? "");
    const labelText = (value: number) => (value >= 0.08 ? `${Math.round(value * 100)}%` : "");

    const traces = [
      {
        type: "bar",
        name: "Class 0",
        x,
        y: classDistributionData.map((row: any) => Number(row["0"] ?? 0)),
        text: classDistributionData.map((row: any) => labelText(Number(row["0"] ?? 0))),
        textposition: "inside",
        insidetextanchor: "middle",
        textfont: { color: "#ffffff", size: 12, family: "Inter, ui-sans-serif, system-ui, sans-serif" },
        marker: { color: SPLIT_CLASS_COLORS["0"], line: { color: "#ffffff", width: 2 } },
        hovertemplate: "Class 0 · %{y:.1%}<extra></extra>",
      },
      {
        type: "bar",
        name: "Class 1",
        x,
        y: classDistributionData.map((row: any) => Number(row["1"] ?? 0)),
        text: classDistributionData.map((row: any) => labelText(Number(row["1"] ?? 0))),
        textposition: "inside",
        insidetextanchor: "middle",
        textfont: { color: "#ffffff", size: 12, family: "Inter, ui-sans-serif, system-ui, sans-serif" },
        marker: { color: SPLIT_CLASS_COLORS["1"], line: { color: "#ffffff", width: 2 }, cornerradius: 4 },
        hovertemplate: "Class 1 · %{y:.1%}<extra></extra>",
      },
    ];

    const layout: any = {
      barmode: "stack",
      bargap: 0.55,
      showlegend: false,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { family: "Inter, ui-sans-serif, system-ui, sans-serif", size: 12, color: "#64748B" },
      margin: { t: 10, r: 12, l: 44, b: 32 },
      xaxis: { title: "", automargin: true, showgrid: false, linecolor: "#E2E8F0", tickfont: { color: "#64748B" } },
      yaxis: {
        title: "",
        tickformat: ".0%",
        dtick: 0.25,
        automargin: true,
        range: [0, 1],
        gridcolor: "#E2E8F0",
        zeroline: false,
        tickfont: { color: "#64748B" },
      },
    };

    return { data: traces, layout };
  }, [classDistributionData]);


  // ── Missing-value treatment proposal (every column classify_missing_treatment
  //    found — i.e. every column that actually has missing values) ──
  const missingProposal: Record<string, TreatmentInfo> = preprocess?.missing_treatment_proposal ?? {};
  const missingProposalEntries = Object.entries(missingProposal);
  const imputationStrategy = preprocess?.imputation_strategy;
  const recalibratedColumns: Array<{ column: string; treatment: string }> = preprocess?.recalibrated_columns ?? [];
  const reviewMissingThreshold: number = preprocess?.review_missing_threshold ?? 0.4;

  // ── Skew-driven transform recommendations — only columns that need a
  //    real decision are ever shown; symmetric/mild-skew columns are silently
  //    left alone (recommend_transform already resolved "none" for them). ──
  const transformRecommendations: Record<string, TransformRecommendation> = preprocess?.transform_recommendations ?? {};
  const transformDecisions = Object.entries(transformRecommendations)
    .filter(([, rec]) => rec.transform !== "none")
    .sort((a, b) => Math.abs(b[1].skew) - Math.abs(a[1].skew));

  const downloadCsv = (csv: string | undefined, filename: string) => {
    if (!csv) return;
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const severityBadge = (skew: number) => {
    const abs = Math.abs(skew);
    if (abs >= 2.0) return { label: "High skew", className: "bg-red-500/15 text-red-700 border-red-500/30" };
    if (abs >= 1.5) return { label: "Moderate skew", className: "bg-amber-500/15 text-amber-700 border-amber-500/30" };
    return { label: "Mild skew", className: "bg-yellow-500/15 text-yellow-700 border-yellow-500/30" };
  };

  const totalRows = splitStats.total ?? (
    splitStats.train_n != null && splitStats.val_n != null && splitStats.test_n != null
      ? splitStats.train_n + splitStats.val_n + splitStats.test_n
      : null
  );

  // ── Data Quality Snapshot — reuses the same profiling diagnostics already
  //    shown on the Profiling tab, so this recap needs no new data. ──
  const missingPctValue: number | null = profile?.missing_percentage ?? null;
  const completeness = missingPctValue !== null ? 100 - missingPctValue : null;
  const duplicateRateValue: number | null = profile?.duplicate_rate ?? null;
  const outlierColumnCount = profile?.outlier_analysis
    ? Object.values(profile.outlier_analysis as Record<string, any>).filter((info: any) => (info?.outlier_fraction ?? 0) > 0).length
    : null;
  const isImbalanced: boolean | null = profile?.target_summary?.is_imbalanced ?? null;
  const imbalanceRatio: number | null = profile?.target_summary?.imbalance_ratio ?? null;
  const leakageColumnCount = Array.isArray(profile?.leakage_risk_cols) ? profile.leakage_risk_cols.length : null;
  const dateFieldCount = profile?.date_integrity ? Object.keys(profile.date_integrity).length : null;

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-200 bg-gradient-to-r from-slate-900 via-sky-900 to-blue-800 p-6 text-white shadow-[0_16px_36px_rgba(15,23,42,0.16)]">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-[11px] uppercase tracking-[0.2em] text-sky-200">Step 3</div>
            <h3 className="mt-2 text-2xl font-semibold">Preprocessing Config &amp; Split Strategy</h3>
            <p className="mt-2 max-w-2xl text-sm text-slate-200">
              Finalize X/y, then split before any feature engineering so every transformation learns only from train data.
            </p>
          </div>
          <Button variant="secondary" onClick={onBackToProfiling} className="bg-white/10 text-white hover:bg-white/15 border border-white/20">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Profiling
          </Button>
        </div>
      </div>

      <KpiStrip
        tiles={[
          { icon: Database, label: "Total Records", value: totalRows !== null ? totalRows.toLocaleString() : "—", sub: "Rows across train/val/test", tone: "primary" },
          { icon: Percent, label: "Missing Values", value: missingPctValue !== null ? `${missingPctValue}%` : "—", sub: profile?.missing_cells != null ? `${Number(profile.missing_cells).toLocaleString()} cells` : undefined, tone: "amber" },
          { icon: Trash2, label: "Duplicate Rows", value: profile?.duplicate_rows != null ? String(profile.duplicate_rows) : "—", sub: duplicateRateValue !== null ? `${duplicateRateValue}% of rows` : undefined, tone: "rose" },
          { icon: AlertTriangle, label: "Columns Flagged", value: String(missingProposalEntries.length), sub: "Need a missing-value decision", tone: "amber" },
          { icon: BarChartIcon, label: "Skew Transforms", value: String(transformDecisions.length), sub: "Recommended on numeric columns", tone: "violet" },
        ]}
      />

      <div className="rounded-2xl border border-blue-200 bg-blue-50 p-4">
        <div className="text-sm font-semibold text-blue-900">Leakage control</div>
        <p className="mt-2 text-sm text-blue-900/90">
          The dataset is split before any feature engineering. Missing-value treatment, imputation strategy,
          skew/transform recommendations, IV/WOE, correlation/VIF, and feature-selection decisions are all
          learned on the training split only and applied unchanged to validation/test.
        </p>
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="text-base font-semibold text-slate-900">Split configuration</div>
          <div className="mt-4">
            <SplitProportionBar
              trainPct={splitStats.train_pct ?? null}
              valPct={splitStats.val_pct ?? null}
              testPct={splitStats.test_pct ?? null}
              trainN={splitStats.train_n ?? null}
              valN={splitStats.val_n ?? null}
              testN={splitStats.test_n ?? null}
            />
          </div>
          <div className="mt-6 grid gap-5">
            <div>
              <div className="flex items-center justify-between text-sm font-medium text-slate-700">
                <span>Test Size</span>
                <span className="font-mono text-slate-900">{Math.round(testSize * 100)}%</span>
              </div>
              <input
                type="range"
                min={0.05}
                max={0.45}
                step={0.05}
                value={testSize}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  const maxVal = Math.min(value, 0.95 - valSize);
                  setTestSize(maxVal);
                }}
                className="mt-3 w-full accent-blue-600"
              />
              <div className="mt-2 text-xs text-slate-500">{splitStats.test_n ? `${splitStats.test_n.toLocaleString()} samples` : "Test split count"}</div>
            </div>

            <div>
              <div className="flex items-center justify-between text-sm font-medium text-slate-700">
                <span>Validation Size</span>
                <span className="font-mono text-slate-900">{Math.round(valSize * 100)}%</span>
              </div>
              <input
                type="range"
                min={0.05}
                max={0.45}
                step={0.05}
                value={valSize}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  const maxVal = Math.min(value, 0.95 - testSize);
                  setValSize(maxVal);
                }}
                className="mt-3 w-full accent-blue-600"
              />
              <div className="mt-2 text-xs text-slate-500">{splitStats.val_n ? `${splitStats.val_n.toLocaleString()} samples` : "Validation split count"}</div>
            </div>
          </div>
        </div>

        <DataQualitySnapshot
          completeness={completeness}
          duplicateRate={duplicateRateValue}
          outlierColumnCount={outlierColumnCount}
          isImbalanced={isImbalanced}
          imbalanceRatio={imbalanceRatio}
          leakageColumnCount={leakageColumnCount}
          dateFieldCount={dateFieldCount}
        />
      </div>

      {loading && (
        <div className="flex items-center justify-center gap-2 rounded-2xl border border-slate-200 bg-white p-5 text-sm text-slate-600 shadow-sm">
          <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
          Building adaptive preprocessing pipeline...
        </div>
      )}

      {error && (
        <div className="rounded-2xl border border-red-200 bg-red-50 p-5 text-sm text-red-700 shadow-sm">
          {error}
        </div>
      )}

      {preprocess ? (
        <>
          {classDistributionData.length > 0 && (
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="text-base font-semibold text-slate-900">Class Distribution per Split</div>
                  <div className="mt-1 text-sm text-slate-500">Train, validation and test split proportions by class.</div>
                </div>
                <div className="flex flex-wrap gap-2 text-xs">
                  {classKeys.map((label) => (
                    <div key={label} className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 font-medium text-slate-600">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: SPLIT_CLASS_COLORS[String(label)] ?? SPLIT_CLASS_FALLBACK }} />
                      Class {label}
                    </div>
                  ))}
                </div>
              </div>
              <div className="mt-5 h-72">
                {classDistributionFigure ? (
                  <PlotlyChart figure={classDistributionFigure} style={{ height: "100%", minHeight: "100%" }} />
                ) : null}
              </div>
            </div>
          )}

          <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
            <Card className="shadow-sm border-slate-200 bg-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm text-slate-900">
                  <Brain className="h-4 w-4 text-blue-600" />
                  Missing Value Treatment
                </CardTitle>
                <CardDescription>
                  Each column is classified by its data shape alone — no column-name guessing. Review the
                  proposal and override anything before it&apos;s applied.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {missingProposalEntries.length === 0 ? (
                  <div className="flex items-center gap-2 text-sm text-slate-600">
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                    No missing values in the training features — imputation not required.
                  </div>
                ) : (
                  <>
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600 leading-relaxed">
                      <span className="font-medium text-slate-900">Unknown category</span> — categorical column, filled with an explicit &apos;Unknown&apos; value. <span className="font-medium text-slate-900">Zero-fill</span> — binary or structural-zero numeric column. <span className="font-medium text-slate-900">Statistical</span> — genuinely missing numeric values, filled jointly via MICE, KNN, or median. <span className="font-medium text-slate-900">Review</span> — over {Math.round(reviewMissingThreshold * 100)}% missing, too sparse to impute reliably.
                    </div>

                    {missingProposalEntries
                      .sort((a, b) => (b[1].evidence?.missing_pct ?? 0) - (a[1].evidence?.missing_pct ?? 0))
                      .map(([col, info]) => {
                        const isDropped = Boolean(dropCols[col]);
                        const currentTreatment = treatmentOverrides[col] ?? info.treatment;
                        const missingPct = info.evidence?.missing_pct ?? 0;
                        const isReviewFlag = info.treatment === "review_flag";

                        return (
                          <div
                            key={col}
                            className={`rounded-xl border p-3 ${isReviewFlag ? "border-amber-300 bg-amber-50" : "border-slate-200 bg-slate-50"}`}
                          >
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="font-medium text-sm text-slate-900">{col}</span>
                              <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[11px] text-slate-600">
                                {(missingPct * 100).toFixed(1)}% missing
                              </span>
                              {isReviewFlag && (
                                <span className="inline-flex items-center gap-1 rounded-full border border-amber-300 bg-amber-100 px-2 py-0.5 text-[11px] text-amber-700">
                                  <AlertTriangle className="h-3 w-3" />
                                  Sparse
                                </span>
                              )}
                            </div>
                            <div className="mt-2 flex items-center gap-2">
                              <MiniBar fraction={missingPct} colorClass={missingSeverityColor(missingPct).bar} />
                            </div>
                            <p className="mt-1.5 text-xs text-slate-600">{info.reason}</p>

                            {isReviewFlag && (
                              <div className="mt-2">
                                <button
                                  type="button"
                                  onClick={() => toggleDropImpact(col)}
                                  className="text-xs font-medium text-amber-700 underline decoration-dotted underline-offset-2 hover:text-amber-800"
                                >
                                  {dropImpactOpen[col] ? "Hide" : "Show"} impact of dropping this feature
                                </button>

                                {dropImpactOpen[col] && (
                                  <div className="mt-2 rounded-xl border border-slate-200 bg-white p-3">
                                    {dropImpactLoading[col] ? (
                                      <div className="flex items-center gap-2 text-xs text-slate-600">
                                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                        Analyzing impact of dropping {col}...
                                      </div>
                                    ) : dropImpactError[col] ? (
                                      <div className="text-xs text-red-600">{dropImpactError[col]}</div>
                                    ) : dropImpact[col] ? (
                                      <>
                                        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                          <div>
                                            <div className="text-xs text-slate-500">Predictive importance (IV)</div>
                                            <div className="text-sm font-semibold tabular-nums text-slate-900">
                                              {dropImpact[col].iv !== null && dropImpact[col].iv !== undefined
                                                ? dropImpact[col].iv.toFixed(3)
                                                : "n/a"}
                                            </div>
                                            {dropImpact[col].iv_label && (
                                              <div className="text-xs text-slate-500">{dropImpact[col].iv_label}</div>
                                            )}
                                          </div>
                                          <div>
                                            <div className="text-xs text-slate-500">Most correlated feature</div>
                                            {dropImpact[col].redundant_col ? (
                                              <>
                                                <div className="text-sm font-semibold text-slate-900">{dropImpact[col].redundant_col}</div>
                                                <div className="text-xs text-slate-500">{"|corr|="}{Math.abs(dropImpact[col].redundant_corr).toFixed(2)}</div>
                                              </>
                                            ) : (
                                              <>
                                                <div className="text-sm font-semibold text-slate-900">None found</div>
                                                <div className="text-xs text-slate-500">no redundancy ≥ 0.60</div>
                                              </>
                                            )}
                                          </div>
                                        </div>

                                        <div className={`mt-3 rounded-md border-l-4 p-2.5 text-xs ${dropImpact[col].verdict_tone === "safe" ? "border-emerald-500 bg-emerald-50 text-emerald-900" : dropImpact[col].verdict_tone === "caution" ? "border-amber-500 bg-amber-50 text-amber-900" : dropImpact[col].verdict_tone === "risk" ? "border-red-500 bg-red-50 text-red-900" : "border-slate-200 bg-slate-50 text-slate-600"}`}>
                                          <span className="font-medium">Verdict: </span>
                                          {dropImpact[col].verdict}
                                        </div>
                                      </>
                                    ) : null}
                                  </div>
                                )}
                              </div>
                            )}

                            <div className="mt-3 flex flex-wrap items-center gap-3">
                              <div className="flex items-center gap-2">
                                <span className="text-xs text-slate-500">Treatment</span>
                                <Select
                                  value={currentTreatment}
                                  disabled={isDropped}
                                  onValueChange={(value) => setTreatmentOverrides((prev) => ({ ...prev, [col]: value }))}
                                >
                                  <SelectTrigger className="h-8 w-[180px] border-primary bg-primary text-xs text-primary-foreground hover:bg-primary/90 focus:ring-primary data-[placeholder]:text-primary-foreground [&>span]:text-primary-foreground [&_svg]:text-primary-foreground [&_svg]:opacity-80 disabled:border-primary/40 disabled:bg-primary/40 disabled:text-primary-foreground/70">
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {TREATMENT_OPTIONS.map((opt) => (
                                      <SelectItem key={opt} value={opt}>{TREATMENT_LABELS[opt]}</SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              </div>

                              <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600">
                                <Checkbox
                                  checked={isDropped}
                                  onCheckedChange={(checked) => setDropCols((prev) => ({ ...prev, [col]: Boolean(checked) }))}
                                />
                                Drop variable
                              </label>
                            </div>
                          </div>
                        );
                      })}

                    {recalibratedColumns.length > 0 && (
                      <div className="flex items-start gap-2 rounded-xl border border-sky-200 bg-sky-50 p-3 text-xs text-sky-900">
                        <Info className="mt-0.5 h-4 w-4 shrink-0" />
                        <div>
                          <span className="font-medium">Recalibrated</span> — kept despite being flagged for
                          review, so a real imputation method was found instead of leaving it untreated: {recalibratedColumns.map((r) => `${r.column} → ${TREATMENT_LABELS[r.treatment] ?? r.treatment}`).join(", ")}
                        </div>
                      </div>
                    )}

                    {imputationStrategy && (
                      <div className="rounded-xl border border-slate-200 border-l-4 border-l-blue-600 bg-white p-3">
                        <div className="text-sm font-medium text-slate-900">
                          Statistical imputation method: <span className="font-mono">{imputationStrategy.method?.toUpperCase()}</span>
                        </div>
                        <p className="mt-1 text-xs text-slate-600">{imputationStrategy.reason}</p>
                        <div className="mt-2 flex items-center gap-2">
                          <span className="text-xs text-slate-500">Override</span>
                          <Select
                            value={strategyOverride ?? "auto"}
                            onValueChange={(value) => setStrategyOverride(value === "auto" ? null : value)}
                          >
                            <SelectTrigger className="h-8 w-[160px] border-primary bg-primary text-xs text-primary-foreground hover:bg-primary/90 focus:ring-primary data-[placeholder]:text-primary-foreground [&>span]:text-primary-foreground [&_svg]:text-primary-foreground [&_svg]:opacity-80">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="auto">Auto (recommended)</SelectItem>
                              <SelectItem value="mice">MICE</SelectItem>
                              <SelectItem value="knn">KNN</SelectItem>
                              <SelectItem value="median">Median</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </CardContent>
            </Card>

            <Card className="shadow-sm border-slate-200 bg-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm text-slate-900">
                  <BarChartIcon className="h-4 w-4 text-violet-600" />
                  Skew-Driven Transforms
                </CardTitle>
                <CardDescription>
                  {transformDecisions.length > 0
                    ? `${transformDecisions.length} of ${Object.keys(transformRecommendations).length} numeric column(s) are skewed enough to matter — everything else is left alone.`
                    : "No numeric columns are skewed enough to need a transform."}
                </CardDescription>
              </CardHeader>
              {transformDecisions.length > 0 && (
                <CardContent className="space-y-3">
                  {transformDecisions.map(([col, rec]) => {
                    const badge = severityBadge(rec.skew);
                    const current = transformChoices[col] ?? "none";
                    return (
                      <div key={col} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-medium text-sm text-slate-900">{col}</span>
                          <span className={`rounded-full border px-2 py-0.5 text-[11px] ${badge.className}`}>
                            {badge.label} · {rec.skew.toFixed(2)}
                          </span>
                          <span className="text-[11px] text-slate-500">
                            recommended: <span className="font-medium text-slate-800">{TRANSFORM_LABELS[rec.transform]}</span>
                          </span>
                        </div>
                        <div className="mt-2 flex items-center gap-2">
                          <MiniBar fraction={Math.min(1, Math.abs(rec.skew) / 3)} colorClass={skewBarColor(rec.skew)} />
                        </div>
                        <p className="mt-1.5 text-xs text-slate-600">{rec.reason}</p>
                        <div className="mt-3 flex items-center gap-2">
                          <span className="text-xs text-slate-500">Apply</span>
                          <Select
                            value={current}
                            onValueChange={(value) => setTransformChoices((prev) => ({ ...prev, [col]: value }))}
                          >
                            <SelectTrigger className="h-8 w-[160px] border-primary bg-primary text-xs text-primary-foreground hover:bg-primary/90 focus:ring-primary data-[placeholder]:text-primary-foreground [&>span]:text-primary-foreground [&_svg]:text-primary-foreground [&_svg]:opacity-80">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {TRANSFORM_OPTIONS.map((opt) => (
                                <SelectItem key={opt} value={opt}>{TRANSFORM_LABELS[opt]}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    );
                  })}
                </CardContent>
              )}
            </Card>
          </div>

          <TransformationsAppliedCard rows={strategySummary} />

          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2 text-base font-semibold text-slate-900">
              <TableIcon className="h-4 w-4 text-slate-600" />
              Preprocessing Strategy Summary
            </div>
            <div className="mt-4 overflow-x-auto">
              {strategySummary.length > 0 ? (
                <table className="min-w-full border-collapse text-sm">
                  <thead>
                    <tr>
                      <th className="border-b border-slate-200 px-3 py-2 text-left font-medium text-slate-500">#</th>
                      <th className="border-b border-slate-200 px-3 py-2 text-left font-medium text-slate-500">Column</th>
                      <th className="border-b border-slate-200 px-3 py-2 text-left font-medium text-slate-500">Type</th>
                      <th className="border-b border-slate-200 px-3 py-2 text-left font-medium text-slate-500">Scaler</th>
                      <th className="border-b border-slate-200 px-3 py-2 text-left font-medium text-slate-500">Imputer</th>
                      <th className="border-b border-slate-200 px-3 py-2 text-left font-medium text-slate-500">Encoding</th>
                      <th className="border-b border-slate-200 px-3 py-2 text-left font-medium text-slate-500">Transform</th>
                    </tr>
                  </thead>
                  <tbody>
                    {strategySummary.map((row: any, index: number) => (
                      <tr key={index} className={index % 2 === 0 ? "bg-white" : "bg-slate-50/50"}>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-600">{index + 1}</td>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-700">{row.feature}</td>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-600">{row.type}</td>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-600">{row.scaler}</td>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-600">{row.imputer}</td>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-600">{row.encoding}</td>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-600">{row.transform ?? "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="p-6 text-center text-sm text-slate-500">No preprocessing strategy summary available.</div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <div className="text-sm font-semibold text-slate-900">Original Dataset</div>
                  <p className="text-xs text-slate-500">The dataset exactly as uploaded, before any processing.</p>
                </div>
                <Button
                  variant="outline"
                  onClick={() => downloadCsv(preprocess?.original_dataset_csv, "original_dataset.csv")}
                  className="gap-2 self-start sm:self-auto"
                >
                  <Download className="h-4 w-4" />
                  Download
                </Button>
              </div>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <div className="text-sm font-semibold text-slate-900">Transformed Dataset</div>
                  <p className="text-xs text-slate-500">Training split after imputation, scaling and encoding.</p>
                </div>
                <Button
                  variant="outline"
                  onClick={() => downloadCsv(preprocess?.processed_dataset_csv, "transformed_dataset.csv")}
                  className="gap-2 self-start sm:self-auto"
                >
                  <Download className="h-4 w-4" />
                  Download
                </Button>
              </div>
            </div>
          </div>

          <Separator />

          <div className="flex gap-3 pt-2">
            <Button variant="outline" onClick={onBackToProfiling} className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to Profiling
            </Button>
          </div>
        </>
      ) : !loading && !error ? (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 text-center text-sm text-slate-500 shadow-sm">
          Preparing preprocessing results...
        </div>
      ) : null}
    </div>
  );
}

function FeaturesSection() {
  const navigate = useNavigate();
  const { file, profile, featureEngineeringResult, setFeatureEngineeringResult, preprocessingResult } = useDataset();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Seed from the shared context so returning to this page (e.g. via Back from
  // Training) shows the already-computed result instead of looking reset.
  const [engineeringResult, setEngineeringResult] = useState<FeatureEngineeringResponse | null>(
    (featureEngineeringResult as FeatureEngineeringResponse | null) ?? null,
  );

  // Resume where the reviewer left off: if this session has no feature
  // engineering result yet, pull the last saved /data/feature-engineering
  // run from the backend.
  const { data: resumedFeatures } = useResumeState<FeatureEngineeringResponse>(
    "dev_pipeline_log.csv",
    "feature_engineering",
  );
  useEffect(() => {
    if (!featureEngineeringResult && resumedFeatures) {
      setFeatureEngineeringResult(resumedFeatures as unknown as Record<string, any>);
      setEngineeringResult(resumedFeatures);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumedFeatures]);

  const feRequestIdRef = useRef(0);

  // ── Feature Removal — propose-confirm ──────────────────────────────────────
  const [removeChecked, setRemoveChecked] = useState<Record<string, boolean>>({});
  const [confirmedRemoveCols, setConfirmedRemoveCols] = useState<string[] | null>(null);
  const [applyingRemoval, setApplyingRemoval] = useState(false);

  const targetCol = useMemo(() => {
    if (!profile) return "";
    if (profile.columns && Array.isArray(profile.columns) && profile.columns.includes("loan_status")) {
      return "loan_status";
    }
    if (profile.target_candidates && Array.isArray(profile.target_candidates) && profile.target_candidates.length > 0) {
      return profile.target_candidates[0];
    }
    return "";
  }, [profile]);

  const runFeatureEngineering = async (overrideConfirmedRemove?: string[]) => {
    if (!file || !targetCol || targetCol === "string") {
      setError("Could not determine target column. Please check the uploaded dataset.");
      return;
    }
    const requestId = ++feRequestIdRef.current;
    try {
      setLoading(true);
      setError(null);

      const form = new FormData();
      form.append("file", file);
      form.append("target_col", targetCol);
      const remove = overrideConfirmedRemove ?? confirmedRemoveCols;
      if (remove !== null && remove !== undefined) {
        form.append("confirmed_remove_cols", JSON.stringify(remove));
      }

      const result = await formUpload<FeatureEngineeringResponse>("/data/feature-engineering", form);
      if (requestId !== feRequestIdRef.current) {
        // A newer request (e.g. triggered by a removal re-run) was issued
        // while this one was in flight — drop this stale response instead of
        // overwriting the newer result.
        return;
      }
      setEngineeringResult(result);
      // Publish to shared context so navigating away and back (or forward to
      // Training) reuses this result instead of forcing a recompute.
      setFeatureEngineeringResult(result as unknown as Record<string, any>);
    } catch (err) {
      if (requestId !== feRequestIdRef.current) return;
      const message = err instanceof Error ? err.message : "Failed to run feature engineering";
      setError(message);
    } finally {
      if (requestId === feRequestIdRef.current) {
        setLoading(false);
        setApplyingRemoval(false);
      }
    }
  };

  // Consumed once: if we mounted with a cached result already in context (e.g.
  // navigating back from Training and forward again), skip the very next
  // auto-run and reuse it instead of silently recomputing feature engineering
  // from scratch. Any later change to file/profile still triggers a real
  // recompute.
  const skipInitialAutoRun = useRef(engineeringResult !== null);

  useEffect(() => {
    if (!file || !profile) {
      setError("No dataset uploaded. Please upload a dataset first.");
      return;
    }
    if (skipInitialAutoRun.current) {
      skipInitialAutoRun.current = false;
      return;
    }
    runFeatureEngineering();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, profile]);

  const plan = engineeringResult?.feature_engineering_plan ?? {};
  const summary = engineeringResult?.feature_engineering_summary ?? {};

  const addedFeatures = Array.isArray(summary.added) ? summary.added : [];
  const removedFeatures = Array.isArray(summary.removed) ? summary.removed : [];

  const regulatoryAlerts = Array.isArray(summary.regulatory_alerts)
    ? summary.regulatory_alerts
    : Array.isArray(plan.regulatory_alerts)
    ? plan.regulatory_alerts
    : [];

  const interactionFeatures = Array.isArray(engineeringResult?.interaction_features)
    ? engineeringResult.interaction_features
    : [];

  // ── Feature Removal — propose-confirm, computed client-side ────────────────
  // Shared with the Explainability > Summary full report — see
  // lib/feature-removal.ts for the cascade-rescue logic itself.
  const removalProposal = useMemo(() => computeFeatureRemovalProposal(plan), [plan]);

  // Fill in default checkbox state for any newly-proposed feature, preserving
  // whatever the reviewer already toggled for features seen before.
  useEffect(() => {
    setRemoveChecked((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const row of removalProposal.rows) {
        if (!(row.feature in next)) {
          next[row.feature] = row.defaultRemove;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [removalProposal.rows.map((r) => r.feature).join("|")]);

  const downloadDecisionLog = async () => {
    if (!file) return;
    try {
      setLoading(true);
      const form = new FormData();
      form.append("file", file);
      const target_col = profile.target_candidates && profile.target_candidates.length ? profile.target_candidates[0] : "";
      form.append("target_col", target_col);
      const res = await formUpload<any>("/data/feature-decision-log", form);
      if (res && res.content_base64) {
        const blob = new Blob([Uint8Array.from(atob(res.content_base64), c => c.charCodeAt(0))], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = res.file_name || 'feature_decision_log.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
      }
    } catch (err) {
      console.error(err);
      setError('Failed to download decision log.');
    } finally {
      setLoading(false);
    }
  };

  const downloadEngineeredDataset = () => {
    if (!engineeringResult?.x_engineered_csv) return;
    const blob = new Blob([engineeringResult.x_engineered_csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "engineered_dataset.csv";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  const applyRemovalChoices = () => {
    const confirmed = removalProposal.rows.filter((row) => removeChecked[row.feature]).map((row) => row.feature);
    setApplyingRemoval(true);
    setConfirmedRemoveCols(confirmed);
    runFeatureEngineering(confirmed);
  };

  const canProceed = !!engineeringResult && !loading && !error;

  if (!file || !profile) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-6">
        <div className="flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-amber-600" />
          <div>
            <div className="font-semibold text-amber-900">No Dataset</div>
            <div className="text-sm text-amber-800">Upload a dataset on the Data Upload page to see feature engineering results.</div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6">
        <div className="flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-red-600" />
          <div>
            <div className="font-semibold text-red-900">Error</div>
            <div className="text-sm text-red-800">{error}</div>
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-12">
        <Loader className="h-8 w-8 animate-spin text-primary" />
        <div className="text-sm text-muted-foreground">Running feature engineering...</div>
      </div>
    );
  }

  if (!engineeringResult) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="text-center text-sm text-muted-foreground">Feature engineering did not return a result.</div>
      </div>
    );
  }

  const originalFeatures = Array.isArray(summary.original_shape) ? summary.original_shape[1] ?? null : null;
  const finalFeatures = Array.isArray(summary.final_shape) ? summary.final_shape[1] ?? null : null;

  // Recap of what preprocessing produced — the starting point feature
  // engineering builds on top of.
  const preprocessSummary = {
    feature_count: preprocessingResult?.feature_count ?? preprocessingResult?.summary_metrics?.features_basic,
    duplicates_removed:
      preprocessingResult?.duplicates_removed ?? preprocessingResult?.summary_metrics?.duplicates_removed ?? 0,
    numeric_feature_count:
      preprocessingResult?.numeric_feature_count ?? preprocessingResult?.summary_metrics?.numeric_columns,
    categorical_feature_count:
      preprocessingResult?.categorical_feature_count ?? preprocessingResult?.summary_metrics?.categorical_columns,
    // Boolean/datetime columns are real modeled features (present in
    // feature_count) but aren't numeric or categorical — surfaced separately
    // so the four counts always reconcile instead of silently undercounting.
    other_feature_count:
      (preprocessingResult?.boolean_feature_count ?? preprocessingResult?.summary_metrics?.boolean_columns ?? 0) +
      (preprocessingResult?.datetime_feature_count ?? preprocessingResult?.summary_metrics?.datetime_columns ?? 0),
  };

  // ── Feature Overview table — built entirely from real, already-fetched
  //    values: original-column universe from feature_engineering_plan.col_types
  //    (a type→column-list map, not per-column — see resolveFeatureType),
  //    IV/Gini from the same plan, missing% joined from the preprocessing
  //    result (only columns with any missing values are present there, so a
  //    missing entry means 0%), and status derived from dropped_features /
  //    the same cascade-rescue removal proposal already driving the
  //    interactive table below. Target and id-typed columns are excluded —
  //    they aren't candidate features. ──
  const colTypes = engineeringResult?.col_types;
  const ivScores: Record<string, number> = plan.iv_scores ?? {};
  const giniScores: Record<string, number> = engineeringResult?.gini_scores ?? {};
  const miScores: Record<string, number> = engineeringResult?.feature_importance_summary?.mi_scores ?? {};
  const missingTreatmentMap: Record<string, TreatmentInfo> = preprocessingResult?.missing_treatment_proposal ?? {};
  const droppedFeatureSet = new Set(Array.isArray(engineeringResult?.dropped_features) ? engineeringResult.dropped_features : []);
  const reviewFeatureSet = new Set(removalProposal.rows.map((r) => r.feature).filter((f) => !droppedFeatureSet.has(f)));

  const featureUniverse = colTypes
    ? Array.from(new Set(Object.entries(colTypes).filter(([type]) => type !== "id").flatMap(([, cols]) => (Array.isArray(cols) ? cols : []))))
    : [];

  const featureOverviewRows: FeatureOverviewRow[] = featureUniverse
    .filter((f) => f !== engineeringResult?.target_col)
    .map((feature) => ({
      feature,
      type: resolveFeatureType(colTypes, feature),
      iv: ivScores[feature] ?? null,
      gini: giniScores[feature] ?? null,
      missingPct: missingTreatmentMap[feature]?.evidence?.missing_pct ?? (preprocessingResult ? 0 : null),
      status: (droppedFeatureSet.has(feature) ? "removed" : reviewFeatureSet.has(feature) ? "review" : "selected") as FeatureOverviewRow["status"],
    }))
    .sort((a, b) => (b.iv ?? -1) - (a.iv ?? -1));

  // Rank by whichever real per-feature score the backend actually populated
  // for this task/dataset — never claim a metric (e.g. "XGBoost importance")
  // Aegis doesn't compute at this stage.
  const importanceSource: [string, Record<string, number>][] = [
    ["Univariate Gini coefficient (numeric features, training split)", giniScores],
    ["Information Value — IV (numeric & categorical candidates)", ivScores],
    ["Mutual Information score (numeric features)", miScores],
  ];
  const [importanceLabel, importanceMap] = importanceSource.find(([, scores]) => Object.keys(scores).length > 0) ?? [
    "No importance scores available",
    {} as Record<string, number>,
  ];
  const importanceItems = Object.entries(importanceMap)
    .map(([name, score]) => ({ name, score: Number(score) }))
    .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
    .slice(0, 10);

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-200 bg-gradient-to-r from-slate-900 via-indigo-900 to-blue-800 p-6 text-white shadow-[0_16px_36px_rgba(15,23,42,0.16)]">
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="text-[11px] uppercase tracking-[0.2em] text-indigo-200">Step 4</div>
            <h3 className="mt-2 text-2xl font-semibold">Feature Engineering</h3>
            <p className="mt-2 text-sm text-slate-200">Engineered features, multicollinearity diagnostics, and importance preview.</p>
          </div>
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-lg border border-white/20 bg-white/10 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/15"
            onClick={downloadEngineeredDataset}
          >
            <Download className="h-4 w-4" />
            Download engineered dataset
          </button>
        </div>
      </div>

      {preprocessingResult && (
        <KpiStrip
          tiles={[
            { icon: TableIcon, label: "Feature Count After Cleanup", value: preprocessSummary.feature_count != null ? String(preprocessSummary.feature_count) : "—", sub: "After removing sparse/ID columns", tone: "primary" },
            { icon: Trash2, label: "Duplicate Rows Removed", value: String(preprocessSummary.duplicates_removed ?? 0), sub: "Exact-copy rows dropped pre-split", tone: "rose" },
            { icon: Hash, label: "Numeric Columns", value: preprocessSummary.numeric_feature_count != null ? String(preprocessSummary.numeric_feature_count) : "—", sub: "Continuous fields for modeling", tone: "emerald" },
            { icon: Tag, label: "Categorical Columns", value: preprocessSummary.categorical_feature_count != null ? String(preprocessSummary.categorical_feature_count) : "—", sub: "Non-numeric, need encoding", tone: "violet" },
            ...(preprocessSummary.other_feature_count > 0
              ? [{ icon: TableIcon, label: "Other Columns", value: String(preprocessSummary.other_feature_count), sub: "Boolean/datetime, engineered separately", tone: "primary" as const }]
              : []),
          ]}
        />
      )}

      {(originalFeatures !== null || finalFeatures !== null || addedFeatures.length > 0 || removedFeatures.length > 0) && (
        <KpiStrip
          tiles={[
            ...(originalFeatures !== null ? [{ icon: Layers, label: "Original Features", value: String(originalFeatures), sub: "Before feature engineering", tone: "primary" as const }] : []),
            ...(finalFeatures !== null ? [{ icon: Layers, label: "Final Features", value: String(finalFeatures), sub: "After feature engineering", tone: "emerald" as const }] : []),
            ...(addedFeatures.length > 0 ? [{ icon: TrendingUp, label: "Features Added", value: String(addedFeatures.length), sub: "New engineered columns", tone: "violet" as const }] : []),
            ...(removedFeatures.length > 0 ? [{ icon: Trash2, label: "Features Removed", value: String(removedFeatures.length), sub: "Dropped during engineering", tone: "rose" as const }] : []),
          ]}
        />
      )}

      <PipelineFlow
        stages={[
          { label: "Raw Features", value: profile?.shape?.[1] ?? null, sub: "input" },
          { label: "Cleaned", value: preprocessSummary.feature_count ?? null, sub: "deduplicated" },
          { label: "Missing-Treated", value: preprocessingResult ? Object.keys(preprocessingResult.missing_treatment_proposal ?? {}).length : null, sub: "columns treated" },
          { label: "Transformed", value: preprocessingResult ? Object.values(preprocessingResult.transform_recommendations ?? {}).filter((r: any) => r?.transform && r.transform !== "none").length : null, sub: "skew-corrected" },
          { label: "Model-Ready", value: finalFeatures, sub: "final features" },
        ]}
      />

      <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-base font-semibold text-slate-900">Feature Removal Proposal</h2>
          <p className="mt-1 text-xs text-slate-500">
            Features proposed for removal by automated analysis. Untick any row to retain that feature. Click Apply
            to re-run feature engineering with your confirmed choices.
          </p>

          {removalProposal.rescueSet.size > 0 && (
            <div className="mt-4 flex items-start gap-2 rounded-xl border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900">
              <RefreshCw className="mt-0.5 h-4 w-4 flex-shrink-0" />
              <div>
                <strong>Cascade rescue</strong> — {Array.from(removalProposal.rescueSet).map((f) => `\`${f}\``).join(", ")} pre-retained: both members of a correlated pair were proposed for removal; the higher-IV member was kept so the information family doesn't vanish entirely.
              </div>
            </div>
          )}

          {removalProposal.rows.length > 0 ? (
            <>
              <div className="mt-4 overflow-x-auto">
                <table className="min-w-full border-collapse text-sm">
                  <thead>
                    <tr className="text-left text-[11px] uppercase tracking-wider text-slate-500">
                      <th className="border-b border-slate-200 px-3 py-2">#</th>
                      <th className="border-b border-slate-200 px-3 py-2">Feature</th>
                      <th className="border-b border-slate-200 px-3 py-2">IV</th>
                      <th className="border-b border-slate-200 px-3 py-2">Reason</th>
                      <th className="border-b border-slate-200 px-3 py-2">Remove?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {removalProposal.rows.map((row, rowIndex) => (
                      <tr key={row.feature} className={rowIndex % 2 === 0 ? "bg-white" : "bg-slate-50/50"}>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-500">{rowIndex + 1}</td>
                        <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-800">{row.feature}</td>
                        <td className="border-b border-slate-200 px-3 py-2 text-xs text-slate-700">{row.iv !== null ? row.iv.toFixed(4) : "—"}</td>
                        <td className="border-b border-slate-200 px-3 py-2 text-xs text-slate-600">{row.reason}</td>
                        <td className="border-b border-slate-200 px-3 py-2 text-xs">
                          <input
                            type="checkbox"
                            checked={!!removeChecked[row.feature]}
                            onChange={(e) => setRemoveChecked((prev) => ({ ...prev, [row.feature]: e.target.checked }))}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <button
                type="button"
                disabled={applyingRemoval || loading}
                className="mt-4 inline-flex items-center gap-2 rounded-lg border border-blue-600 bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                onClick={applyRemovalChoices}
              >
                {applyingRemoval ? <Loader className="h-4 w-4 animate-spin" /> : null}
                Apply removal choices
              </button>
            </>
          ) : (
            <p className="mt-4 text-sm text-slate-500">No features proposed for removal on this dataset.</p>
          )}
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-base font-semibold text-slate-900">Generated interactions</h2>
          {interactionFeatures.length > 0 ? (
            <>
              <p className="mt-1 text-xs text-slate-500">
                IV and Gini are each interaction's own predictive power — the metrics that let it pass evaluation
                (min IV, redundancy filtering) — not a lift over the source features alone.
              </p>
              <div className="mt-4 overflow-x-auto">
                <table className="min-w-full border-collapse text-sm">
                  <thead>
                    <tr className="text-left text-[11px] uppercase tracking-wider text-slate-500">
                      <th className="border-b border-slate-200 px-3 py-2">#</th>
                      <th className="border-b border-slate-200 px-3 py-2">Feature A</th>
                      <th className="border-b border-slate-200 px-3 py-2">Feature B</th>
                      <th className="border-b border-slate-200 px-3 py-2">Type</th>
                      <th className="border-b border-slate-200 px-3 py-2">IV</th>
                      <th className="border-b border-slate-200 px-3 py-2">Gini</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...interactionFeatures]
                      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
                      .map((f, idx) => (
                        <tr key={f.name ?? idx} className={idx % 2 === 0 ? "bg-white" : "bg-slate-50/50"}>
                          <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-500">{idx + 1}</td>
                          <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-800">{f.feature_a}</td>
                          <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-800">{f.feature_b}</td>
                          <td className="border-b border-slate-200 px-3 py-2 text-xs text-slate-700">{f.interaction_type ?? f.type ?? "—"}</td>
                          <td className="border-b border-slate-200 px-3 py-2 text-xs text-slate-700">{f.score !== undefined ? f.score.toFixed(4) : "—"}</td>
                          <td className="border-b border-slate-200 px-3 py-2 text-xs text-slate-700">{f.gini !== undefined && f.gini !== null ? f.gini.toFixed(4) : "—"}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="mt-2 text-sm text-slate-500">No interaction terms passed evaluation for this dataset.</p>
          )}
        </section>
      </div>

      <FeatureOverviewTable rows={featureOverviewRows} />

      <FeatureImportanceBars items={importanceItems} metricLabel={importanceLabel} />

      {regulatoryAlerts.length > 0 && (
        <section className="rounded-2xl border border-amber-200 bg-amber-50 p-5 shadow-sm">
          <h2 className="text-base font-semibold text-amber-900">Regulatory insights</h2>
          <div className="mt-4 space-y-3">
            {regulatoryAlerts.map((alert: any, idx: number) => (
              <div key={idx} className="rounded-xl border border-amber-200 bg-white p-4">
                <div className="text-sm font-semibold text-amber-900">{alert.rule_id || alert.id || alert.code || alert.rule || `Alert ${idx + 1}`}</div>
                <div className="mt-1 text-sm text-slate-600">{alert.flag || alert.message || alert.detail || alert.description || JSON.stringify(alert)}</div>
                {alert.observed_value && (
                  <div className="mt-2 text-xs font-mono text-slate-500">Observed: {Array.isArray(alert.observed_value) ? alert.observed_value.join(", ") : String(alert.observed_value)}</div>
                )}
                {alert.suggestion && (
                  <div className="mt-2 text-[11px] text-slate-600">Recommendation: {alert.suggestion}</div>
                )}
                {alert.source && (
                  <div className="mt-1 text-[11px] text-slate-500">Reference: {alert.source} — {alert.principle || alert.section || ""}</div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {(engineeringResult.final_engineered_dataset_preview && Array.isArray(engineeringResult.final_engineered_dataset_preview) && engineeringResult.final_engineered_dataset_preview.length > 0) || (engineeringResult.x_engineered_preview && Array.isArray(engineeringResult.x_engineered_preview) && engineeringResult.x_engineered_preview.length > 0) ? (
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="text-base font-semibold text-slate-900">Engineered feature matrix preview</h2>
              <p className="text-xs text-slate-500">A sample of the transformed dataset after feature engineering.</p>
            </div>
          </div>
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full border-collapse text-sm">
              <thead>
                <tr>
                  <th className="border-b border-slate-200 px-3 py-2 text-left font-medium text-slate-500">#</th>
                  {Object.keys((engineeringResult.final_engineered_dataset_preview && Array.isArray(engineeringResult.final_engineered_dataset_preview) && engineeringResult.final_engineered_dataset_preview.length > 0 ? engineeringResult.final_engineered_dataset_preview : engineeringResult.x_engineered_preview ?? [])[0] ?? {}).map((key: string) => (
                    <th key={key} className="border-b border-slate-200 px-3 py-2 text-left font-medium text-slate-500">{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(engineeringResult.final_engineered_dataset_preview && Array.isArray(engineeringResult.final_engineered_dataset_preview) && engineeringResult.final_engineered_dataset_preview.length > 0 ? engineeringResult.final_engineered_dataset_preview : engineeringResult.x_engineered_preview ?? []).map((row: any, rowIndex: number) => (
                  <tr key={rowIndex} className={rowIndex % 2 === 0 ? "bg-white" : "bg-slate-50/50"}>
                    <td className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-500">{rowIndex + 1}</td>
                    {Object.values(row).map((cell: any, cellIndex: number) => (
                      <td key={cellIndex} className="border-b border-slate-200 px-3 py-2 font-mono text-xs text-slate-700">{String(cell)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <Button variant="outline" onClick={downloadDecisionLog} className="gap-2">
          <Download className="h-4 w-4" />
          Download feature decision log
        </Button>
        <Button
          disabled={!canProceed}
          className="gap-2"
          onClick={async () => {
            try {
              await navigate({ to: "/model-training-evaluation" });
            } catch (err) {
              console.error("Navigation failed:", err);
            }
          }}
        >
          Proceed to Model Training
          <ArrowRight className="h-4 w-4" />
        </Button>
      </section>
    </div>
  );
}
