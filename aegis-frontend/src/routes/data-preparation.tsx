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
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { computeFeatureRemovalProposal } from "@/lib/feature-removal";

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
  col_types?: Record<string, string>;
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

    const traces = [
      {
        type: "bar",
        name: "Class 0",
        x,
        y: classDistributionData.map((row: any) => Number(row["0"] ?? 0)),
        marker: { color: "#65A30D" },
        hovertemplate: "%{x}<br>%{y:.1%}<extra></extra>",
      },
      {
        type: "bar",
        name: "Class 1",
        x,
        y: classDistributionData.map((row: any) => Number(row["1"] ?? 0)),
        marker: { color: "#84CC16" },
        hovertemplate: "%{x}<br>%{y:.1%}<extra></extra>",
      },
    ];

    const layout: any = {
      barmode: "stack",
      margin: { t: 10, r: 20, l: 60, b: 60 },
      xaxis: { title: "", automargin: true },
      yaxis: { title: "", tickformat: ".0%", automargin: true, range: [0, 1] },
      legend: { orientation: "h", y: 1.12 },
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

  return (
    <div className="space-y-8">
      <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="text-sm font-semibold">Step 3 — Preprocessing Config &amp; Train/Val/Test Split</div>
        <p className="mt-2 text-sm text-muted-foreground">
          Finalize X/y, then split immediately so every learned statistic comes from training data only.
        </p>
      </div>

      <div className="rounded-xl border border-border bg-card p-6 shadow-elegant border-l-4 border-blue-500/80 bg-blue-500/10">
        <div className="text-sm font-semibold text-blue-900">Leakage control</div>
        <p className="mt-2 text-sm text-blue-900/90">
          The dataset is split before any feature engineering. Missing-value treatment, imputation strategy,
          skew/transform recommendations, IV/WOE, correlation/VIF, and feature-selection decisions are all
          learned on the training split only and applied unchanged to validation/test.
        </p>
      </div>

      <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="grid gap-6 xl:grid-cols-[1.2fr_1fr]">
          <div className="grid gap-4">
            <div>
              <div className="text-sm font-medium">Test Size (%)</div>
              <div className="mt-2 flex items-center gap-3">
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
                  className="flex-1"
                />
                <div className="w-16 text-right text-sm font-mono">{Math.round(testSize * 100)}%</div>
              </div>
              <div className="mt-2 text-xs text-muted-foreground">{splitStats.test_n ? `${splitStats.test_n.toLocaleString()} samples` : "Test split count"}</div>
            </div>

            <div>
              <div className="text-sm font-medium">Validation Size (%)</div>
              <div className="mt-2 flex items-center gap-3">
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
                  className="flex-1"
                />
                <div className="w-16 text-right text-sm font-mono">{Math.round(valSize * 100)}%</div>
              </div>
              <div className="mt-2 text-xs text-muted-foreground">{splitStats.val_n ? `${splitStats.val_n.toLocaleString()} samples` : "Validation split count"}</div>
            </div>
          </div>

          <div className="grid gap-4">
            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-xl border border-border bg-background p-4 text-center">
                <div className="text-xs uppercase tracking-wider text-muted-foreground">Train</div>
                <div className="mt-2 text-2xl font-semibold tabular-nums">{splitStats.train_n?.toLocaleString() ?? "—"}</div>
              </div>
              <div className="rounded-xl border border-border bg-background p-4 text-center">
                <div className="text-xs uppercase tracking-wider text-muted-foreground">Validation</div>
                <div className="mt-2 text-2xl font-semibold tabular-nums">{splitStats.val_n?.toLocaleString() ?? "—"}</div>
              </div>
              <div className="rounded-xl border border-border bg-background p-4 text-center">
                <div className="text-xs uppercase tracking-wider text-muted-foreground">Test</div>
                <div className="mt-2 text-2xl font-semibold tabular-nums">{splitStats.test_n?.toLocaleString() ?? "—"}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {loading && (
        <div className="rounded-xl border border-border bg-card p-6 text-center text-sm text-muted-foreground flex items-center justify-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          🔧 Building adaptive preprocessing pipeline...
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-border bg-card p-6 text-center text-sm text-destructive">
          {error}
        </div>
      )}

      {preprocess ? (
        <>
          {classDistributionData.length > 0 && (
            <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="text-sm font-semibold">Class Distribution per Split (stratified)</div>
                  <div className="mt-2 text-sm text-muted-foreground">Train, validation and test split proportions by class.</div>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {classKeys.map((label) => (
                    <div key={label} className="inline-flex items-center gap-2 rounded-full border border-border px-2 py-1 text-muted-foreground">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: label === "Y" ? "#65A30D" : label === "N" ? "#84CC16" : "#94a3b8" }} />
                      {label}
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

          {/* ── Missing Value Treatment ────────────────────────────────── */}
          <Card className="shadow-elegant">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Brain className="h-4 w-4" />
                Missing Value Treatment
              </CardTitle>
              <CardDescription>
                Each column is classified by its data shape alone — no column-name guessing. Review the
                proposal and override anything before it&apos;s applied.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {missingProposalEntries.length === 0 ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                  No missing values in the training features — imputation not required.
                </div>
              ) : (
                <>
                  <div className="rounded-lg border border-border bg-background p-3 text-xs text-muted-foreground leading-relaxed">
                    <span className="font-medium text-foreground">Unknown category</span> — categorical column, filled with an explicit &apos;Unknown&apos; value.{" "}
                    <span className="font-medium text-foreground">Zero-fill</span> — binary or structural-zero numeric column.{" "}
                    <span className="font-medium text-foreground">Statistical</span> — genuinely missing numeric values, filled jointly via MICE, KNN, or median.{" "}
                    <span className="font-medium text-foreground">Review</span> — over {Math.round(reviewMissingThreshold * 100)}% missing, too sparse to impute reliably.
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
                          className={`rounded-lg border p-3 ${isReviewFlag ? "border-amber-500/40 bg-amber-500/5" : "border-border bg-background"}`}
                        >
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="font-medium text-sm">{col}</span>
                            <span className="rounded-full border border-border px-2 py-0.5 text-xs text-muted-foreground">
                              {(missingPct * 100).toFixed(1)}% missing
                            </span>
                            {isReviewFlag && (
                              <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 text-xs text-amber-700">
                                <AlertTriangle className="h-3 w-3" />
                                Sparse
                              </span>
                            )}
                          </div>
                          <p className="mt-1.5 text-xs text-muted-foreground">{info.reason}</p>

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
                                <div className="mt-2 rounded-lg border border-border bg-background p-3">
                                  {dropImpactLoading[col] ? (
                                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                      Analyzing impact of dropping {col}...
                                    </div>
                                  ) : dropImpactError[col] ? (
                                    <div className="text-xs text-red-600">{dropImpactError[col]}</div>
                                  ) : dropImpact[col] ? (
                                    <>
                                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                        <div>
                                          <div className="text-xs text-muted-foreground">Predictive importance (IV)</div>
                                          <div className="text-sm font-semibold tabular-nums">
                                            {dropImpact[col].iv !== null && dropImpact[col].iv !== undefined
                                              ? dropImpact[col].iv.toFixed(3)
                                              : "n/a"}
                                          </div>
                                          {dropImpact[col].iv_label && (
                                            <div className="text-xs text-muted-foreground">{dropImpact[col].iv_label}</div>
                                          )}
                                        </div>
                                        <div>
                                          <div className="text-xs text-muted-foreground">Most correlated feature</div>
                                          {dropImpact[col].redundant_col ? (
                                            <>
                                              <div className="text-sm font-semibold">{dropImpact[col].redundant_col}</div>
                                              <div className="text-xs text-muted-foreground">
                                                {"|corr|="}{Math.abs(dropImpact[col].redundant_corr).toFixed(2)}
                                              </div>
                                            </>
                                          ) : (
                                            <>
                                              <div className="text-sm font-semibold">None found</div>
                                              <div className="text-xs text-muted-foreground">no redundancy ≥ 0.60</div>
                                            </>
                                          )}
                                        </div>
                                      </div>

                                      <div
                                        className={`mt-3 rounded-md border-l-4 p-2.5 text-xs ${
                                          dropImpact[col].verdict_tone === "safe"
                                            ? "border-emerald-500 bg-emerald-500/5 text-emerald-900"
                                            : dropImpact[col].verdict_tone === "caution"
                                            ? "border-amber-500 bg-amber-500/5 text-amber-900"
                                            : dropImpact[col].verdict_tone === "risk"
                                            ? "border-red-500 bg-red-500/5 text-red-900"
                                            : "border-border bg-background text-muted-foreground"
                                        }`}
                                      >
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
                              <span className="text-xs text-muted-foreground">Treatment</span>
                              <Select
                                value={currentTreatment}
                                disabled={isDropped}
                                onValueChange={(value) =>
                                  setTreatmentOverrides((prev) => ({ ...prev, [col]: value }))
                                }
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

                            <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer">
                              <Checkbox
                                checked={isDropped}
                                onCheckedChange={(checked) =>
                                  setDropCols((prev) => ({ ...prev, [col]: Boolean(checked) }))
                                }
                              />
                              Drop variable — removed entirely, not used in training or evaluation
                            </label>
                          </div>
                        </div>
                      );
                    })}

                  {recalibratedColumns.length > 0 && (
                    <div className="flex items-start gap-2 rounded-lg border border-sky-500/30 bg-sky-500/10 p-3 text-xs text-sky-900">
                      <Info className="h-4 w-4 mt-0.5 shrink-0" />
                      <div>
                        <span className="font-medium">Recalibrated</span> — kept despite being flagged for
                        review, so a real imputation method was found instead of leaving it untreated:{" "}
                        {recalibratedColumns.map((r) => `${r.column} → ${TREATMENT_LABELS[r.treatment] ?? r.treatment}`).join(", ")}
                      </div>
                    </div>
                  )}

                  {imputationStrategy && (
                    <div className="rounded-lg border border-border border-l-4 border-l-primary bg-background p-3">
                      <div className="text-sm font-medium">
                        Statistical imputation method: <span className="font-mono">{imputationStrategy.method?.toUpperCase()}</span>
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground">{imputationStrategy.reason}</p>
                      <div className="mt-2 flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">Override</span>
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

          {/* ── Skew-Driven Transforms ─────────────────────────────────── */}
          <Card className="shadow-elegant">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <BarChartIcon className="h-4 w-4" />
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
                    <div key={col} className="rounded-lg border border-border bg-background p-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-medium text-sm">{col}</span>
                        <span className={`rounded-full border px-2 py-0.5 text-xs ${badge.className}`}>
                          {badge.label} · {rec.skew.toFixed(2)}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          recommended: <span className="font-medium text-foreground">{TRANSFORM_LABELS[rec.transform]}</span>
                        </span>
                      </div>
                      <p className="mt-1.5 text-xs text-muted-foreground">{rec.reason}</p>
                      <div className="mt-3 flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">Apply</span>
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

          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="text-sm font-semibold flex items-center"><TableIcon className="h-4 w-4 mr-2" />Preprocessing Strategy Summary</div>
            <div className="mt-4 overflow-x-auto">
              {strategySummary.length > 0 ? (
                <table className="min-w-full border-collapse text-sm">
                  <thead>
                    <tr>
                      <th className="border-b border-border px-3 py-2 text-left font-medium text-muted-foreground">#</th>
                      <th className="border-b border-border px-3 py-2 text-left font-medium text-muted-foreground">Column</th>
                      <th className="border-b border-border px-3 py-2 text-left font-medium text-muted-foreground">Type</th>
                      <th className="border-b border-border px-3 py-2 text-left font-medium text-muted-foreground">Scaler</th>
                      <th className="border-b border-border px-3 py-2 text-left font-medium text-muted-foreground">Imputer</th>
                      <th className="border-b border-border px-3 py-2 text-left font-medium text-muted-foreground">Encoding</th>
                      <th className="border-b border-border px-3 py-2 text-left font-medium text-muted-foreground">Transform</th>
                    </tr>
                  </thead>
                  <tbody>
                    {strategySummary.map((row: any, index: number) => (
                      <tr key={index} className={index % 2 === 0 ? "bg-background" : "bg-background/50"}>
                        <td className="border-b border-border px-3 py-2 font-mono text-xs">{index + 1}</td>
                        <td className="border-b border-border px-3 py-2 font-mono text-xs">{row.feature}</td>
                        <td className="border-b border-border px-3 py-2 font-mono text-xs">{row.type}</td>
                        <td className="border-b border-border px-3 py-2 font-mono text-xs">{row.scaler}</td>
                        <td className="border-b border-border px-3 py-2 font-mono text-xs">{row.imputer}</td>
                        <td className="border-b border-border px-3 py-2 font-mono text-xs">{row.encoding}</td>
                        <td className="border-b border-border px-3 py-2 font-mono text-xs">{row.transform ?? "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="p-6 text-center text-sm text-muted-foreground">No preprocessing strategy summary available.</div>
              )}
            </div>
          </div>

          {/* ── Downloads ───────────────────────────────────────────────── */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-xl border border-border bg-card p-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="text-sm font-semibold">Original Dataset</div>
                <p className="text-xs text-muted-foreground">The dataset exactly as uploaded, before any processing.</p>
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
            <div className="rounded-xl border border-border bg-card p-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="text-sm font-semibold">Transformed Dataset</div>
                <p className="text-xs text-muted-foreground">Training split after imputation, scaling and encoding.</p>
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

          <Separator />

          <div className="flex gap-3 pt-4">
            <Button variant="outline" onClick={onBackToProfiling} className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to Profiling
            </Button>
          </div>
        </>
      ) : !loading && !error ? (
        <div className="rounded-xl border border-border bg-card p-6 text-center text-sm text-muted-foreground">
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

  return (
    <div className="space-y-8">
      <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="text-sm font-semibold">Step 4 — Feature Engineering</div>
        <p className="mt-2 text-sm text-muted-foreground">Engineered features, multicollinearity diagnostics, and importance preview.</p>
      </div>

      {preprocessingResult && (
        <div className={`grid grid-cols-1 gap-4 md:grid-cols-4${preprocessSummary.other_feature_count > 0 ? " xl:grid-cols-5" : ""}`}>
          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="flex items-center text-sm text-muted-foreground"><TableIcon className="h-4 w-4 mr-2" />Feature Count After Cleanup</div>
            <div className="mt-3 text-3xl font-semibold tabular-nums">{preprocessSummary.feature_count ?? "—"}</div>
            <div className="mt-1 text-xs text-muted-foreground">Columns remaining after removing sparse/ID columns</div>
          </div>
          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="flex items-center text-sm text-muted-foreground"><Trash2 className="h-4 w-4 mr-2" />Duplicate Rows Removed</div>
            <div className="mt-3 text-3xl font-semibold tabular-nums">{preprocessSummary.duplicates_removed ?? 0}</div>
            <div className="mt-1 text-xs text-muted-foreground">Exact-copy rows dropped before splitting</div>
          </div>
          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="flex items-center text-sm text-muted-foreground"><Hash className="h-4 w-4 mr-2" />Numeric Columns</div>
            <div className="mt-3 text-3xl font-semibold tabular-nums">{preprocessSummary.numeric_feature_count ?? "—"}</div>
            <div className="mt-1 text-xs text-muted-foreground">Continuous fields available for modeling</div>
          </div>
          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="flex items-center text-sm text-muted-foreground"><Tag className="h-4 w-4 mr-2" />Categorical Columns</div>
            <div className="mt-3 text-3xl font-semibold tabular-nums">{preprocessSummary.categorical_feature_count ?? "—"}</div>
            <div className="mt-1 text-xs text-muted-foreground">Non-numeric fields requiring encoding</div>
          </div>
          {preprocessSummary.other_feature_count > 0 && (
            <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <div className="flex items-center text-sm text-muted-foreground"><TableIcon className="h-4 w-4 mr-2" />Other Columns</div>
              <div className="mt-3 text-3xl font-semibold tabular-nums">{preprocessSummary.other_feature_count}</div>
              <div className="mt-1 text-xs text-muted-foreground">Boolean/datetime fields, engineered separately</div>
            </div>
          )}
        </div>
      )}

      <section className="grid grid-cols-1 gap-4 md:grid-cols-4">
        {originalFeatures !== null && (
          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="text-xs uppercase tracking-wider text-muted-foreground">Original features</div>
            <div className="mt-2 text-2xl font-semibold tabular-nums">{originalFeatures}</div>
            <div className="mt-1 text-xs text-muted-foreground">Columns before feature engineering</div>
          </div>
        )}
        {finalFeatures !== null && (
          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="text-xs uppercase tracking-wider text-muted-foreground">Final features</div>
            <div className="mt-2 text-2xl font-semibold tabular-nums">{finalFeatures}</div>
            <div className="mt-1 text-xs text-muted-foreground">Columns after feature engineering</div>
          </div>
        )}
        {addedFeatures.length > 0 && (
          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="text-xs uppercase tracking-wider text-muted-foreground">Features added</div>
            <div className="mt-2 text-2xl font-semibold tabular-nums">{addedFeatures.length}</div>
            <div className="mt-1 text-xs text-muted-foreground">New engineered columns created</div>
          </div>
        )}
        {removedFeatures.length > 0 && (
          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="text-xs uppercase tracking-wider text-muted-foreground">Features removed</div>
            <div className="mt-2 text-2xl font-semibold tabular-nums">{removedFeatures.length}</div>
            <div className="mt-1 text-xs text-muted-foreground">Columns dropped during feature engineering</div>
          </div>
        )}
      </section>

      <section className="flex flex-wrap gap-3">
        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-lg border border-border bg-background px-4 py-2 text-sm font-medium transition hover:border-primary hover:bg-primary-soft"
          onClick={downloadEngineeredDataset}
        >
          <Download className="h-4 w-4" />
          Download engineered dataset
        </button>
      </section>

      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <h2 className="text-base font-semibold">🗑️ Feature Removal Proposal</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Features proposed for removal by automated analysis. Untick any row to retain that feature. Click Apply
          to re-run feature engineering with your confirmed choices.
        </p>

        {removalProposal.rescueSet.size > 0 && (
          <div className="mt-4 flex items-start gap-2 rounded-xl border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900">
            <RefreshCw className="mt-0.5 h-4 w-4 flex-shrink-0" />
            <div>
              <strong>Cascade rescue</strong> — {Array.from(removalProposal.rescueSet).map((f) => `\`${f}\``).join(", ")}{" "}
              pre-retained: both members of a correlated pair were proposed for removal; the higher-IV member was
              kept so the information family doesn't vanish entirely.
            </div>
          </div>
        )}

        {removalProposal.rows.length > 0 ? (
          <>
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full border-collapse text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                    <th className="border-b border-border px-3 py-2">#</th>
                    <th className="border-b border-border px-3 py-2">Feature</th>
                    <th className="border-b border-border px-3 py-2">IV</th>
                    <th className="border-b border-border px-3 py-2">Reason</th>
                    <th className="border-b border-border px-3 py-2">Remove?</th>
                  </tr>
                </thead>
                <tbody>
                  {removalProposal.rows.map((row, rowIndex) => (
                    <tr key={row.feature} className="odd:bg-background">
                      <td className="border-b border-border px-3 py-2 font-mono text-xs text-muted-foreground">{rowIndex + 1}</td>
                      <td className="border-b border-border px-3 py-2 font-mono text-xs">{row.feature}</td>
                      <td className="border-b border-border px-3 py-2 text-xs">{row.iv !== null ? row.iv.toFixed(4) : "—"}</td>
                      <td className="border-b border-border px-3 py-2 text-xs text-muted-foreground">{row.reason}</td>
                      <td className="border-b border-border px-3 py-2 text-xs">
                        <input
                          type="checkbox"
                          checked={!!removeChecked[row.feature]}
                          onChange={(e) =>
                            setRemoveChecked((prev) => ({ ...prev, [row.feature]: e.target.checked }))
                          }
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
              className="mt-4 inline-flex items-center gap-2 rounded-lg border border-primary bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
              onClick={applyRemovalChoices}
            >
              {applyingRemoval ? <Loader className="h-4 w-4 animate-spin" /> : null}
              Apply removal choices
            </button>
          </>
        ) : (
          <p className="mt-4 text-sm text-muted-foreground">No features proposed for removal on this dataset.</p>
        )}
      </section>

      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <h2 className="text-base font-semibold">🔗 Interaction Terms Generated</h2>
        {interactionFeatures.length > 0 ? (
          <>
            <p className="mt-1 text-xs text-muted-foreground">
              IV and Gini are each interaction's own predictive power — the metrics that let it pass evaluation
              (min IV, redundancy filtering) — not a lift over the source features alone.
            </p>
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full border-collapse text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                    <th className="border-b border-border px-3 py-2">#</th>
                    <th className="border-b border-border px-3 py-2">Feature A</th>
                    <th className="border-b border-border px-3 py-2">Feature B</th>
                    <th className="border-b border-border px-3 py-2">Type</th>
                    <th className="border-b border-border px-3 py-2">IV</th>
                    <th className="border-b border-border px-3 py-2">Gini</th>
                  </tr>
                </thead>
                <tbody>
                  {[...interactionFeatures]
                    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
                    .map((f, idx) => (
                      <tr key={f.name ?? idx} className="odd:bg-background">
                        <td className="border-b border-border px-3 py-2 font-mono text-xs text-muted-foreground">{idx + 1}</td>
                        <td className="border-b border-border px-3 py-2 font-mono text-xs">{f.feature_a}</td>
                        <td className="border-b border-border px-3 py-2 font-mono text-xs">{f.feature_b}</td>
                        <td className="border-b border-border px-3 py-2 text-xs">{f.interaction_type ?? f.type ?? "—"}</td>
                        <td className="border-b border-border px-3 py-2 text-xs">{f.score !== undefined ? f.score.toFixed(4) : "—"}</td>
                        <td className="border-b border-border px-3 py-2 text-xs">{f.gini !== undefined && f.gini !== null ? f.gini.toFixed(4) : "—"}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p className="mt-2 text-sm text-muted-foreground">No interaction terms passed evaluation for this dataset.</p>
        )}
      </section>

      {regulatoryAlerts.length > 0 && (
        <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
          <h2 className="text-base font-semibold">Regulatory insights</h2>
          <div className="mt-4 space-y-3">
            {regulatoryAlerts.map((alert: any, idx: number) => (
              <div key={idx} className="rounded-xl border border-amber-200 bg-amber-50 p-4">
                <div className="text-sm font-semibold">{alert.rule_id || alert.id || alert.code || alert.rule || `Alert ${idx + 1}`}</div>
                <div className="mt-1 text-sm text-muted-foreground">{alert.flag || alert.message || alert.detail || alert.description || JSON.stringify(alert)}</div>
                {alert.observed_value && (
                  <div className="mt-2 text-xs font-mono text-muted-foreground">Observed: {Array.isArray(alert.observed_value) ? alert.observed_value.join(", ") : String(alert.observed_value)}</div>
                )}
                {alert.suggestion && (
                  <div className="mt-2 text-[11px] text-muted-foreground">Recommendation: {alert.suggestion}</div>
                )}
                {alert.source && (
                  <div className="mt-1 text-[11px] text-muted-foreground">Reference: {alert.source} — {alert.principle || alert.section || ''}</div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {(engineeringResult.final_engineered_dataset_preview && Array.isArray(engineeringResult.final_engineered_dataset_preview) && engineeringResult.final_engineered_dataset_preview.length > 0) || (engineeringResult.x_engineered_preview && Array.isArray(engineeringResult.x_engineered_preview) && engineeringResult.x_engineered_preview.length > 0) ? (
        <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="text-base font-semibold">Engineered feature matrix preview</h2>
              <p className="text-xs text-muted-foreground">A sample of the transformed dataset after feature engineering.</p>
            </div>
          </div>
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full border-collapse text-sm">
              <thead>
                <tr>
                  <th className="border-b border-border px-3 py-2 text-left font-medium text-muted-foreground">#</th>
                  {Object.keys((engineeringResult.final_engineered_dataset_preview && Array.isArray(engineeringResult.final_engineered_dataset_preview) && engineeringResult.final_engineered_dataset_preview.length > 0 ? engineeringResult.final_engineered_dataset_preview : engineeringResult.x_engineered_preview ?? [])[0] ?? {}).map((key: string) => (
                    <th key={key} className="border-b border-border px-3 py-2 text-left font-medium text-muted-foreground">{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(engineeringResult.final_engineered_dataset_preview && Array.isArray(engineeringResult.final_engineered_dataset_preview) && engineeringResult.final_engineered_dataset_preview.length > 0 ? engineeringResult.final_engineered_dataset_preview : engineeringResult.x_engineered_preview ?? []).map((row: any, rowIndex: number) => (
                  <tr key={rowIndex} className={rowIndex % 2 === 0 ? "bg-background" : ""}>
                    <td className="border-b border-border px-3 py-2 font-mono text-xs text-muted-foreground">{rowIndex + 1}</td>
                    {Object.values(row).map((cell: any, cellIndex: number) => (
                      <td key={cellIndex} className="border-b border-border px-3 py-2 font-mono text-xs">{String(cell)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <button
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-background px-4 py-2 text-sm font-medium transition hover:border-primary hover:bg-primary-soft"
          onClick={downloadDecisionLog}
        >
          <Download className="h-4 w-4" />
          Download feature decision log
        </button>
        <button
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-primary bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={!canProceed}
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
        </button>
      </section>
    </div>
  );
}
