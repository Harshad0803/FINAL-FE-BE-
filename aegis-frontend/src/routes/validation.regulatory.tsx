import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { PageHeader } from "@/components/app-shell";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ArrowRight, Loader2, Search } from "lucide-react";
import { ApiError, formUpload } from "@/lib/api";
import { useDataset } from "@/lib/app-context";
import PlotlyChart from "@/components/plotly-chart";
import { CheckSummaryTiles, deriveCheckTotal } from "@/components/check-summary";

export const Route = createFileRoute("/validation/regulatory")({
  head: () => ({ meta: [{ title: "Stage 6 — Explainability and Fairness — Aegis Credit" }] }),
  component: Regulatory,
});

type Status = "PASS" | "WARN" | "FAIL" | string;

type ThresholdCheck = {
  check_id: string;
  title: string;
  severity: string;
  status: Status;
  source: string;
  principle: string;
  observed: string;
  threshold: string;
  detail: string;
  check_type?: string;
};

type Stage7Response = {
  checks: ThresholdCheck[];
  summary: { total: number; pass: number; warn: number; fail: number; na?: number };
};

type BiasRow = { Group: string; Count: number; "Default Rate": number; "Avg Predicted PD": number; AUC: number | null };

type BiasCheckResult = {
  check_id: string;
  title: string;
  severity: string;
  status: Status;
  source: string;
  principle: string;
  observed: string;
  threshold: string;
  detail: string;
  check_type?: string;
};

// check_type "data"/"cross_reference" checks compute a number against an
// industry-standard statistical convention (e.g. the Fair Lending bias
// check's "AUC gap < 0.05") — the cited regulation requires that kind of
// check to exist, not that specific cutoff. The 7.1-7.10 regulatory checks
// are check_type "doc" (genuinely quoting/requiring regulatory text) and
// keep their existing combined citation.
function isQuantitativeConventionCheck(checkType: string | undefined): boolean {
  return checkType === "data" || checkType === "cross_reference";
}

type BiasResponse = {
  success: boolean;
  error: string | null;
  protected_columns: string[];
  bias_col: string | null;
  rows: BiasRow[];
  check: BiasCheckResult | null;
};

const STATUS_STYLES: Record<string, { border: string; bg: string; badge: string; icon: string }> = {
  PASS: { border: "border-emerald-500/40", bg: "bg-emerald-500/10", badge: "bg-emerald-500 text-emerald-950", icon: "✅" },
  WARN: { border: "border-amber-500/40", bg: "bg-amber-500/10", badge: "bg-amber-500 text-amber-950", icon: "🟡" },
  FAIL: { border: "border-red-500/40", bg: "bg-red-500/10", badge: "bg-red-500 text-red-950", icon: "🔴" },
  PENDING: { border: "border-border", bg: "bg-muted/30", badge: "bg-muted text-foreground", icon: "⏭️" },
};

const SEVERITY_STYLES: Record<string, string> = {
  HIGH: "bg-red-500 text-red-950",
  MEDIUM: "bg-amber-500 text-amber-950",
  LOW: "bg-emerald-500 text-emerald-950",
};

function statusStyle(status: string | undefined) {
  return STATUS_STYLES[status ?? ""] ?? { border: "border-border", bg: "bg-card", badge: "bg-muted text-foreground", icon: "⚪" };
}

function ThresholdCheckCard({ check }: { check: ThresholdCheck }) {
  const s = statusStyle(check.status);
  const sevClasses = SEVERITY_STYLES[check.severity?.toUpperCase()] ?? "bg-muted text-foreground";
  const isConvention = isQuantitativeConventionCheck(check.check_type);
  return (
    <div className={`min-w-0 rounded-r-lg border-l-4 ${s.border} ${s.bg} p-4`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1 break-words text-sm font-semibold text-foreground">
          {s.icon} <span className="text-muted-foreground">[{check.check_id}]</span> {check.title}{" "}
          <span className={`ml-1 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide ${sevClasses}`}>
            {check.severity}
          </span>
        </div>
        <span className={`shrink-0 rounded px-2 py-0.5 text-xs font-bold ${s.badge}`}>{check.status}</span>
      </div>
      {isConvention ? (
        <div className="mt-2 text-xs text-muted-foreground">
          📋 Regulatory basis: {check.source} {check.principle} — requires this to be assessed/documented
        </div>
      ) : (
        <div className="mt-2 text-xs text-muted-foreground">
          📋 {check.source} — {check.principle}
        </div>
      )}
      <div className="mt-2 text-sm text-foreground">
        📊 Observed: <code className="text-foreground/90">{check.observed}</code>
      </div>
      <div className="mt-1 text-xs text-muted-foreground">
        📐 Threshold: {check.threshold}
        {isConvention ? " — industry-standard convention" : ""}
      </div>
      {check.detail ? <div className="mt-2 text-sm text-muted-foreground">💡 {check.detail}</div> : null}
    </div>
  );
}

function Regulatory() {
  const ds = useDataset();
  const {
    validationIntakeData,
    validationMddText,
    validationStage4Result,
    validationStage7Result,
    setValidationStage7Result,
    validationStage7BiasResult,
    setValidationStage7BiasResult,
  } = ds;

  const [loading, setLoading] = useState(!validationStage7Result);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<Stage7Response | null>((validationStage7Result as Stage7Response | null) ?? null);
  // Tracks which sub-tab is active so the bottom button can tell whether
  // the reviewer is still on the first sub-tab (Explainability and
  // Fairness — button just advances to Regulatory Compliance) or the last
  // one (button navigates to Stage 7).
  const [activeSubTab, setActiveSubTab] = useState<string>("explainability");

  const skipInitialAutoRun = useRef(validationStage7Result !== null && validationStage7Result !== undefined);

  useEffect(() => {
    if (skipInitialAutoRun.current) {
      skipInitialAutoRun.current = false;
      setLoading(false);
      return;
    }

    let active = true;
    setLoading(true);
    setError(null);

    const form = new FormData();
    form.append("intake_json", JSON.stringify(validationIntakeData ?? {}));
    if (validationMddText) {
      const mddBlob = new Blob([validationMddText], { type: "text/plain" });
      form.append("mdd_file", new File([mddBlob], "mdd.txt", { type: "text/plain" }));
    }

    void formUpload<Stage7Response>("/validation/stage7/run", form)
      .then((resp) => {
        if (!active) return;
        setData(resp);
        setValidationStage7Result(resp as unknown as Record<string, any>);
      })
      .catch((err) => {
        console.error("Stage7 fetch error", err);
        if (!active) return;
        setError(err?.message ?? String(err));
      })
      .finally(() => {
        if (!active) return;
        setLoading(false);
      });

    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validationIntakeData, validationMddText]);

  const summary = data?.summary ?? { total: 0, pass: 0, warn: 0, fail: 0 };
  const totalChecks = deriveCheckTotal(summary);
  const progress = totalChecks > 0 ? Math.round((summary.pass / totalChecks) * 100) : 0;

  const featureImportance = useMemo(() => {
    const rows = (validationStage4Result as any)?.replication?.result?.feature_importance as
      | Array<{ Feature: string; Importance: number }>
      | undefined;
    if (!rows || !rows.length) return [];
    return [...rows]
      .sort((a, b) => (b.Importance ?? 0) - (a.Importance ?? 0))
      .slice(0, 15)
      .reverse();
  }, [validationStage4Result]);

  const stage4Available = featureImportance.length > 0;

  // Dataset/target/model resolution mirrors validation.performance.tsx
  // (Stage 5) — the bias check reruns the train/split server-side since the
  // backend is stateless, so it needs the same inputs Stage 3/4 use.
  const targetCol = ds.profile?.target_col || ds.trainingResult?.evaluation_data?.target_col || "default";
  const modelName = ds.selectedModel?.name || ds.trainingResult?.model_name || "Logistic Regression";
  const datasetFile = useMemo<File | null>(() => {
    if (ds.file) return ds.file;
    const csvText = typeof ds.profile?.csv_text === "string" ? ds.profile.csv_text : "";
    if (!csvText.trim()) return null;
    const resolvedName = ds.profile?.dataset_name ?? "validation_dataset.csv";
    const safeName = resolvedName.endsWith(".csv") || resolvedName.endsWith(".xlsx") ? resolvedName : `${resolvedName}.csv`;
    return new File([csvText], safeName, { type: "text/csv" });
  }, [ds.file, ds.profile?.csv_text, ds.profile?.dataset_name]);

  const [biasData, setBiasData] = useState<BiasResponse | null>((validationStage7BiasResult as BiasResponse | null) ?? null);
  const [biasCol, setBiasCol] = useState<string>(() => (validationStage7BiasResult as BiasResponse | null)?.bias_col ?? "");
  const [biasLoading, setBiasLoading] = useState(false);
  const [biasError, setBiasError] = useState<string | null>(null);
  const columnsFetched = useRef(false);

  // Fetch just the detected protected-characteristic columns on first load
  // (no protected_col posted yet) so the dropdown can populate itself.
  useEffect(() => {
    if (columnsFetched.current || biasData || !datasetFile) return;
    columnsFetched.current = true;
    const form = new FormData();
    form.append("file", datasetFile);
    form.append("target_col", targetCol);
    form.append("model_name", modelName);
    void formUpload<BiasResponse>("/validation/stage7/bias-check", form)
      .then((resp) => setBiasData(resp))
      .catch(() => {
        /* silent — the "Run Bias Check" button re-attempts with a protected_col */
      });
  }, [datasetFile, targetCol, modelName, biasData]);

  // Default the dropdown to the first detected column once columns arrive,
  // rather than leaving it on the disabled "Choose a column…" placeholder.
  // Only fires when nothing is selected yet, so it never overrides a
  // selection the user (or a restored cached result) already made.
  useEffect(() => {
    if (biasCol || !biasData?.protected_columns?.length) return;
    setBiasCol(biasData.protected_columns[0]);
  }, [biasCol, biasData]);

  const runBiasCheck = useCallback(async () => {
    if (!datasetFile || !biasCol) return;
    setBiasLoading(true);
    setBiasError(null);
    try {
      const form = new FormData();
      form.append("file", datasetFile);
      form.append("target_col", targetCol);
      form.append("model_name", modelName);
      form.append("protected_col", biasCol);
      const resp = await formUpload<BiasResponse>("/validation/stage7/bias-check", form);
      setBiasData(resp);
      setValidationStage7BiasResult(resp as unknown as Record<string, any>);
    } catch (err) {
      setBiasError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Bias check failed.");
    } finally {
      setBiasLoading(false);
    }
  }, [datasetFile, targetCol, modelName, biasCol, setValidationStage7BiasResult]);

  const biasAucRows = (biasData?.rows ?? []).filter((r) => r.AUC !== null) as Array<BiasRow & { AUC: number }>;
  const biasAucMean = biasAucRows.length ? biasAucRows.reduce((s, r) => s + r.AUC, 0) / biasAucRows.length : 0;

  const featureImportanceFigure = useMemo(() => {
    if (!featureImportance.length) return null;
    return {
      data: [
        {
          type: "bar",
          orientation: "h",
          x: featureImportance.map((row) => row.Importance),
          y: featureImportance.map((row) => row.Feature),
          marker: { color: "oklch(0.6 0.18 280)" },
          hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
          name: "Importance",
        },
      ],
      layout: {
        margin: { l: 140, r: 20, t: 20, b: 40 },
        xaxis: { title: { text: "Importance" }, tickfont: { size: 11 }, automargin: true },
        yaxis: { tickfont: { size: 11 }, automargin: true, autorange: "reversed" },
        height: 420,
      },
    };
  }, [featureImportance]);

  const biasAucFigure = useMemo(() => {
    if (!biasAucRows.length) return null;
    return {
      data: [
        {
          type: "bar",
          x: biasAucRows.map((row) => row.AUC),
          y: biasAucRows.map((row) => row.Group),
          orientation: "h",
          marker: {
            color: biasAucRows.map((row) =>
              Math.abs(row.AUC - biasAucMean) > 0.05 ? "oklch(0.6 0.22 27)" : "oklch(0.76 0.18 130)",
            ),
          },
          hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
          name: "AUC",
        },
        {
          type: "scatter",
          mode: "lines",
          x: Array(biasAucRows.length).fill(biasAucMean),
          y: biasAucRows.map((row) => row.Group),
          line: { color: "oklch(0.6 0.18 280)", dash: "dash" },
          hoverinfo: "skip",
          showlegend: false,
          name: "Mean AUC",
        },
      ],
      layout: {
        margin: { l: 140, r: 20, t: 20, b: 40 },
        xaxis: { title: { text: "AUC" }, tickfont: { size: 11 }, automargin: true, range: [0, 1] },
        yaxis: { tickfont: { size: 11 }, automargin: true, autorange: "reversed" },
        height: 320,
      },
    };
  }, [biasAucRows, biasAucMean]);

  return (
    <div className="space-y-8">
      <PageHeader
        title="Stage 6 — Explainability and Fairness"
        description="SS1/23 · SS11/13 · IFRS 9 · IFRS 7 — automated regulatory compliance checks and model explainability review."
      />

      {loading ? (
        <div className="rounded-xl border border-border bg-card p-6 text-center">Loading Stage 6 checks...</div>
      ) : error ? (
        <div className="rounded-xl border border-border bg-card p-6 text-destructive">Error loading Stage 6: {error}</div>
      ) : (
        <Tabs value={activeSubTab} onValueChange={setActiveSubTab} className="w-full">
          <TabsList>
            <TabsTrigger value="explainability">Explainability and Fairness</TabsTrigger>
            <TabsTrigger value="compliance">Regulatory Compliance</TabsTrigger>
          </TabsList>

          <TabsContent value="compliance" className="space-y-6 pt-4">
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h3 className="text-sm font-semibold">Regulatory Compliance Results (7.1–7.10)</h3>
              <CheckSummaryTiles summary={summary} checksLabel="Total Checks" className="mt-4" />
              <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-muted">
                <div className="h-full rounded-full bg-emerald-500 transition-all" style={{ width: `${progress}%` }} />
              </div>
            </section>

            <div className="mx-auto max-w-2xl space-y-3">
              {data?.checks && data.checks.length > 0 ? (
                data.checks.map((c) => <ThresholdCheckCard key={c.check_id} check={c} />)
              ) : (
                <div className="rounded-xl border border-border bg-card p-4 text-sm text-muted-foreground">
                  No regulatory compliance checks generated for this stage.
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="explainability" className="space-y-6 pt-4">
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h3 className="text-sm font-semibold">🔬 SHAP Feature Importance (from Stage 3 Replication)</h3>
              <p className="text-xs text-muted-foreground">
                Reuses the replicated model's feature importances computed in Stage 3 — no re-training here.
              </p>

              {featureImportanceFigure ? (
                <div className="mt-4 h-[420px]">
                  <PlotlyChart figure={featureImportanceFigure} style={{ height: "100%" }} />
                </div>
              ) : (
                <div className="mt-4 rounded-xl border border-dashed border-border bg-background p-8 text-center text-sm text-muted-foreground">
                  Feature importances not available. Run Stage 3 Model Replication first to populate this chart.
                </div>
              )}
              <p className="mt-2 text-xs text-muted-foreground">Top 15 Feature Importances (Replicated Model)</p>
            </section>

            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h3 className="text-sm font-semibold">⚖️ Fair Lending Bias Check</h3>
              <p className="text-xs text-muted-foreground">
                Check if model performance differs significantly across protected characteristics. Large gaps may
                indicate discriminatory bias.
              </p>

              {!datasetFile ? (
                <div className="mt-4 rounded-xl border border-dashed border-border bg-background p-8 text-center text-sm text-muted-foreground">
                  No dataset available. Complete Stage 1 Intake and Stage 2 Data Validation first.
                </div>
              ) : biasData && biasData.protected_columns.length === 0 ? (
                <div className="mt-4 rounded-xl border border-dashed border-border bg-background p-8 text-center text-sm text-muted-foreground">
                  No protected characteristic columns detected in the dataset. Common ones: age, gender, region,
                  employment, education.
                </div>
              ) : (
                <>
                  <div className="mt-4 flex flex-wrap items-end gap-3">
                    <label className="space-y-2 text-sm">
                      <span className="font-medium">Select characteristic to analyse for bias</span>
                      <select
                        value={biasCol}
                        onChange={(e) => setBiasCol(e.target.value)}
                        className="w-56 rounded-lg border border-border bg-background px-3 py-2"
                      >
                        <option value="" disabled>
                          Choose a column…
                        </option>
                        {(biasData?.protected_columns ?? []).map((col) => (
                          <option key={col} value={col}>
                            {col}
                          </option>
                        ))}
                      </select>
                    </label>
                    <button
                      type="button"
                      onClick={() => void runBiasCheck()}
                      disabled={!biasCol || biasLoading}
                      className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {biasLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                      Run Bias Check
                    </button>
                  </div>

                  {biasError ? <p className="mt-3 text-sm text-destructive">{biasError}</p> : null}

                  {biasData?.rows && biasData.rows.length > 0 ? (
                    <>
                      <div className="mt-4 overflow-x-auto rounded-lg border border-border">
                        <table className="w-full text-sm">
                          <thead className="bg-background text-[10px] uppercase tracking-wider text-muted-foreground">
                            <tr>
                              <th className="px-3 py-2 text-left">#</th>
                              <th className="px-3 py-2 text-left">Group</th>
                              <th className="px-3 py-2 text-right">Count</th>
                              <th className="px-3 py-2 text-right">Default Rate</th>
                              <th className="px-3 py-2 text-right">Avg Predicted PD</th>
                              <th className="px-3 py-2 text-right">AUC</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border">
                            {biasData.rows.map((r, rowIndex) => (
                              <tr key={r.Group}>
                                <td className="px-3 py-2 text-muted-foreground">{rowIndex + 1}</td>
                                <td className="px-3 py-2 font-medium">{r.Group}</td>
                                <td className="px-3 py-2 text-right tabular-nums">{r.Count.toLocaleString()}</td>
                                <td className="px-3 py-2 text-right tabular-nums">{(r["Default Rate"] * 100).toFixed(2)}%</td>
                                <td className="px-3 py-2 text-right tabular-nums">{(r["Avg Predicted PD"] * 100).toFixed(2)}%</td>
                                <td className="px-3 py-2 text-right tabular-nums">{r.AUC !== null ? r.AUC.toFixed(4) : "N/A"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      {biasAucFigure ? (
                        <div className="mt-4 h-64">
                          <PlotlyChart figure={biasAucFigure} style={{ height: "100%" }} />
                        </div>
                      ) : null}
                    </>
                  ) : null}

                  {biasData?.check ? (
                    <div className="mt-4">
                      <ThresholdCheckCard
                        check={{
                          check_id: biasData.check.check_id,
                          title: biasData.check.title,
                          severity: biasData.check.severity,
                          status: biasData.check.status,
                          source: biasData.check.source,
                          principle: biasData.check.principle,
                          observed: biasData.check.observed,
                          threshold: biasData.check.threshold,
                          detail: biasData.check.detail,
                          check_type: biasData.check.check_type,
                        }}
                      />
                    </div>
                  ) : null}
                </>
              )}
            </section>
          </TabsContent>
        </Tabs>
      )}

      <div className="text-right">
        {activeSubTab === "explainability" ? (
          <button
            type="button"
            onClick={() => setActiveSubTab("compliance")}
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90"
          >
            Continue
            <ArrowRight className="h-4 w-4" />
          </button>
        ) : (
          <Link
            to="/validation/findings"
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90"
          >
            Continue to Stage 7
            <ArrowRight className="h-4 w-4" />
          </Link>
        )}
      </div>
    </div>
  );
}
