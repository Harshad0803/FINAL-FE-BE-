import React from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { PageHeader } from "@/components/app-shell";
import { ArrowRight, AlertCircle, Loader2, PlayCircle, Upload } from "lucide-react";
import { useDataset } from "@/lib/app-context";
import PlotlyChart from "@/components/plotly-chart";
import { ApiError, formUpload } from "@/lib/api";

export const Route = createFileRoute("/validation/performance")({
  head: () => ({ meta: [{ title: "Benchmarking — Aegis Credit" }] }),
  component: Performance,
});

// Stage 4 — Benchmarking only. The full performance report (metrics, ROC/PR
// curves, confusion matrix, calibration, score distribution) that used to
// live on this page moved to Stage 3 (/validation/challenger, "Model
// Replication & Performance Testing"), since replication and performance
// testing fit the model the same way under the hood and showing them
// together avoids re-running that fit twice. This page keeps just the
// champion-vs-industry-benchmark comparison.
type PerformanceResponse = {
  stage: string;
  report: {
    metrics: Record<string, any>;
    // The re-fit replica's own computed metrics, kept for transparency —
    // NOT what's shown as the champion's reported performance (that's
    // `metrics`, sourced from the MDD). Useful if you need to sanity-check
    // how far a fresh replica diverges from the documented champion.
    replica_metrics?: Record<string, any>;
    roc_curve: { points: Array<Record<string, number>>; auc?: number | null };
    benchmark?: Record<string, any>;
  };
};

// Mirrors model_selector.py's CLASSIFICATION_MODELS registry keys (also
// available from the backend at GET /models/list) — the same candidate set
// offered during the model development pipeline's Model Selection step.
const CHALLENGER_CANDIDATES = [
  "Logistic Regression",
  "Random Forest",
  "XGBoost",
  "LightGBM",
  "Gradient Boosting",
];

type ChallengerComparisonRow = {
  model_name: string;
  roc_auc?: number;
  gini?: number;
  ks?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  training_time_s?: number;
  error?: string;
};

type CompareModelsResponse = {
  task_type: string;
  comparison: ChallengerComparisonRow[];
};

function formatValue(value: unknown, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return num.toFixed(digits);
}

function Performance() {
  const ds = useDataset();
  const [targetCol, setTargetCol] = React.useState(
    () => ds.profile?.target_col || ds.trainingResult?.evaluation_data?.target_col || "default",
  );
  const [modelName, setModelName] = React.useState(
    () => ds.selectedModel?.name || ds.trainingResult?.model_name || "Logistic Regression",
  );
  const [challengerModelName, setChallengerModelName] = React.useState("Logistic Regression");
  const [selectedCandidates, setSelectedCandidates] = React.useState<string[]>([...CHALLENGER_CANDIDATES]);
  const [comparisonRows, setComparisonRows] = React.useState<ChallengerComparisonRow[] | null>(null);
  const [comparing, setComparing] = React.useState(false);
  const [comparisonError, setComparisonError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  // Whether the reviewer has actually run "Compare Challengers" and clicked
  // "Select" on a row this session. The champion-vs-challenger benchmark
  // must not run against a guessed/default challenger — it can only run
  // once a challenger has been deliberately picked from real comparison
  // numbers.
  const [challengerPicked, setChallengerPicked] = React.useState(false);
  // Champion performance metrics come from the MDD that was uploaded and
  // parsed back in Stage 1 (Intake & Governance) — ds.validationMddMetrics
  // is populated there via POST /validation/parse-mdd and is exactly the
  // extract_metrics_from_mdd() output the benchmarking endpoint expects, so
  // it can be sent straight through as reported_json. No re-upload needed
  // here for the common case.
  const mddMetricsFromIntake = ds.validationMddMetrics;
  const hasMddMetrics = Boolean(mddMetricsFromIntake && Object.keys(mddMetricsFromIntake).length > 0);
  // Fallback only: lets a reviewer parse an MDD directly from this page if
  // Stage 1 was skipped or didn't have one yet. Writes back into the same
  // shared context (mirrors Stage 1's own upload handler) so it becomes the
  // one shared source of truth for every stage, not a page-local copy.
  const [mddParseError, setMddParseError] = React.useState<string | null>(null);
  const [mddParsing, setMddParsing] = React.useState(false);
  // Seed from shared context so returning to this page (e.g. via Back from
  // Stage 5) shows the already-computed result instead of resetting to the
  // bare input form — but ONLY if that cached result reflects a benchmark
  // that was actually run against a deliberately-picked challenger, not
  // just whatever happened to be sitting in shared state (e.g. left over
  // from an earlier dataset or an earlier visit). Previously this seeded
  // unconditionally, which is why the champion-vs-challenger comparison
  // could appear pre-loaded the moment the tab opened, before Compare
  // Challengers had ever been run.
  const cachedResult = ds.validationStage5Result as PerformanceResponse | null;
  const cachedIsUsable = Boolean(
    cachedResult?.report?.benchmark?.model_name && cachedResult?.report?.benchmark?.status === "OK",
  );
  const [payload, setPayload] = React.useState<PerformanceResponse | null>(
    cachedIsUsable ? cachedResult : null,
  );
  React.useEffect(() => {
    if (cachedIsUsable) setChallengerPicked(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const datasetName = ds.file?.name ?? ds.profile?.dataset_name ?? "uploaded dataset";
  const datasetReady = Boolean(ds.file || ds.profile?.csv_text || ds.profile?.dataset_name);

  // By the time a reviewer reaches Stage 4, the working dataset often only
  // exists as profile.csv_text (not a literal File object) — carried forward
  // through preprocessing/FE/macro-fetch as text, same as Stage 3's activeFile
  // pattern. Without this reconstruction, the auto-run below silently no-ops
  // whenever ds.file is null, leaving the page stuck with no data.
  const datasetFile = React.useMemo<File | null>(() => {
    if (ds.file) return ds.file;
    const csvText = typeof ds.profile?.csv_text === "string" ? ds.profile.csv_text : "";
    if (!csvText.trim()) return null;
    const resolvedName = ds.profile?.dataset_name ?? "validation_dataset.csv";
    const safeName = resolvedName.endsWith(".csv") || resolvedName.endsWith(".xlsx")
      ? resolvedName
      : `${resolvedName}.csv`;
    return new File([csvText], safeName, { type: "text/csv" });
  }, [ds.file, ds.profile?.csv_text, ds.profile?.dataset_name]);

  const handleRun = React.useCallback(async (fileOverride?: File | null) => {
    const fileToUse = fileOverride ?? datasetFile;
    if (!fileToUse) {
      setError("Upload a dataset or use the file from Intake before running Stage 4.");
      return;
    }
    if (!targetCol.trim()) {
      setError("Target column is required.");
      return;
    }
    if (!modelName.trim()) {
      setError("Model name is required.");
      return;
    }
    if (!hasMddMetrics) {
      setError("No MDD metrics found. Upload the MDD in Stage 1 (Intake & Governance), or use the fallback parser below — champion metrics are sourced from it, not from a re-fit replica.");
      return;
    }

    setLoading(true);
    setError(null);
    setPayload(null);
    try {
      const form = new FormData();
      form.append("file", fileToUse);
      form.append("model_name", modelName.trim());
      form.append("target_col", targetCol.trim());
      form.append("challenger_model_name", challengerModelName.trim());
      form.append("reported_json", JSON.stringify(mddMetricsFromIntake));
      const res = await formUpload<PerformanceResponse>("/validation/performance", form);
      setPayload(res);
      ds.setValidationStage5Result(res as unknown as Record<string, any>);
    } catch (err) {
      if (err instanceof ApiError) {
        const detail =
          err.body && typeof err.body === "object" && "detail" in (err.body as any)
            ? String((err.body as any).detail)
            : err.message;
        setError(detail);
      } else {
        setError(err instanceof Error ? err.message : "Benchmark analysis failed.");
      }
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetFile, targetCol, modelName, challengerModelName, hasMddMetrics, mddMetricsFromIntake]);

  // NOTE: Stage 4 deliberately does NOT auto-run on mount. Unlike Stage 2,
  // this page's main result is a champion-vs-challenger comparison, and a
  // challenger must be deliberately picked (via Compare Challengers ->
  // Select) before there's anything meaningful to benchmark against. An
  // earlier version auto-ran this against a hardcoded default challenger
  // ("Logistic Regression") the instant a dataset was ready, which is why
  // the comparison could appear already populated before the reviewer had
  // run Compare Challengers or picked a model — see handleRun's disabled
  // state below, which now requires challengerPicked.

  const toggleCandidate = React.useCallback((name: string) => {
    setSelectedCandidates((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name],
    );
  }, []);

  // Reuses /models/compare — the exact same lightweight, no-CV/no-hyperopt
  // comparison the model development pipeline's Model Selection step uses
  // — so picking a challenger here is grounded in the same fast read on
  // "which model is worth committing to" rather than a separate mechanism.
  const runComparison = React.useCallback(async () => {
    if (!datasetFile) {
      setComparisonError("Upload a dataset or use the file from Intake before comparing challengers.");
      return;
    }
    if (!targetCol.trim()) {
      setComparisonError("Target column is required.");
      return;
    }
    if (selectedCandidates.length === 0) {
      setComparisonError("Select at least one challenger candidate to compare.");
      return;
    }

    setComparing(true);
    setComparisonError(null);
    try {
      const form = new FormData();
      form.append("file", datasetFile);
      form.append("target_col", targetCol.trim());
      form.append("model_names", JSON.stringify(selectedCandidates));
      const res = await formUpload<CompareModelsResponse>("/models/compare", form);
      setComparisonRows(res.comparison ?? []);
    } catch (err) {
      if (err instanceof ApiError) {
        const detail =
          err.body && typeof err.body === "object" && "detail" in (err.body as any)
            ? String((err.body as any).detail)
            : err.message;
        setComparisonError(detail);
      } else {
        setComparisonError(err instanceof Error ? err.message : "Challenger comparison failed.");
      }
    } finally {
      setComparing(false);
    }
  }, [datasetFile, targetCol, selectedCandidates]);

  const rocPoints = payload?.report?.roc_curve?.points ?? [];
  const benchmarkResponse = payload?.report?.benchmark;
  const benchmarkComparison = benchmarkResponse?.comparison?.champion_vs_challenger;

  const benchmarkTableRows = React.useMemo(() => {
    const championMetrics = payload?.report?.metrics ?? {};
    const challengerMetrics = benchmarkResponse?.metrics ?? {};
    return [
      { model: "Champion", roc_auc: championMetrics.roc_auc, gini: championMetrics.gini, recall: championMetrics.recall },
      { model: benchmarkResponse?.model_name ?? "Benchmark", roc_auc: challengerMetrics.roc_auc, gini: challengerMetrics.gini, recall: challengerMetrics.recall },
    ];
  }, [payload, benchmarkResponse]);

  const benchmarkOverlayData = React.useMemo(() => {
    const championPoints = rocPoints ?? [];
    const benchmarkAuc = Number(benchmarkResponse?.metrics?.roc_auc ?? 0.5);
    const points = championPoints.length
      ? championPoints.map((point) => ({ fpr: point.fpr, champion: point.tpr, benchmark: Math.min(1, Math.max(0, point.fpr * benchmarkAuc)) }))
      : [{ fpr: 0, champion: 0, benchmark: 0 }, { fpr: 1, champion: 1, benchmark: benchmarkAuc }];
    return points;
  }, [rocPoints, benchmarkResponse]);

  const comparisonChartData = React.useMemo(() => {
    if (!benchmarkComparison) return [];
    return [
      { metric: "ROC-AUC", champion: benchmarkComparison.roc_auc?.champion, challenger: benchmarkComparison.roc_auc?.challenger },
      { metric: "Gini", champion: benchmarkComparison.gini?.champion, challenger: benchmarkComparison.gini?.challenger },
      { metric: "Recall", champion: benchmarkComparison.recall?.champion, challenger: benchmarkComparison.recall?.challenger },
    ];
  }, [benchmarkComparison]);

  const benchmarkComparisonFigure = React.useMemo(() => {
    const metrics = comparisonChartData.map((row) => row.metric);
    const champion = comparisonChartData.map((row) => row.champion);
    const challenger = comparisonChartData.map((row) => row.challenger);
    return {
      data: [
        {
          type: "bar",
          x: metrics,
          y: champion,
          name: "Champion",
          marker: { color: "oklch(0.76 0.18 130)" },
        },
        {
          type: "bar",
          x: metrics,
          y: challenger,
          name: "Challenger",
          marker: { color: "oklch(0.55 0.02 240)" },
        },
      ],
      layout: {
        barmode: "group",
        margin: { l: 40, r: 20, t: 25, b: 40 },
        xaxis: { title: "Metric", tickfont: { size: 11 }, showline: false },
        yaxis: { title: "Value", tickfont: { size: 11 }, showline: false, range: [0, 1] },
        height: 320,
      },
    };
  }, [comparisonChartData]);

  const benchmarkOverlayFigure = React.useMemo(() => {
    const fpr = benchmarkOverlayData.map((point) => point.fpr);
    const champion = benchmarkOverlayData.map((point) => point.champion);
    const benchmark = benchmarkOverlayData.map((point) => point.benchmark);
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: fpr,
          y: champion,
          line: { color: "oklch(0.6 0.18 135)", width: 2.5 },
          hovertemplate: "Champion TPR %{y:.3f}<br>FPR %{x:.3f}<extra></extra>",
          name: "Champion",
        },
        {
          type: "scatter",
          mode: "lines",
          x: fpr,
          y: benchmark,
          line: { color: "oklch(0.55 0.02 240)", width: 2.5 },
          hovertemplate: "Benchmark TPR %{y:.3f}<br>FPR %{x:.3f}<extra></extra>",
          name: "Benchmark",
        },
      ],
      layout: {
        margin: { l: 40, r: 20, t: 25, b: 40 },
        xaxis: { title: "FPR", tickfont: { size: 11 }, showline: false },
        yaxis: { title: "TPR", tickfont: { size: 11 }, showline: false },
        height: 320,
      },
    };
  }, [benchmarkOverlayData]);

  return (
    <div className="space-y-8">
      <PageHeader
        title="Stage 4 — Benchmarking"
        description="Compare the champion model against an industry-standard challenger before stress testing and regulatory review."
      />

      {loading ? (
        <section className="flex items-center gap-2 rounded-xl border border-border bg-card p-4 text-sm text-muted-foreground shadow-elegant">
          <Loader2 className="h-4 w-4 animate-spin" />
          Running Stage 4 benchmarking analysis on the shared dataset…
        </section>
      ) : null}

      {error ? (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      ) : null}

      <section className="rounded-xl border border-border bg-background p-4 text-sm text-muted-foreground">
        {datasetReady ? (
          <>Using the shared dataset from Stage 1 / Stage 2: <span className="font-semibold text-foreground">{datasetName}</span>.</>
        ) : (
          <>No active dataset is available in shared state yet. Complete Stage 1 Intake and Stage 2 Data Validation first.</>
        )}
      </section>

      {!payload && !loading ? (
        <div className="rounded-xl border border-dashed border-border bg-card p-8 text-center text-sm text-muted-foreground shadow-elegant">
          {datasetReady
            ? "Run Compare Challengers below, select a challenger model, then Run Benchmark to see the champion-vs-challenger comparison."
            : "Waiting on a dataset from Stage 1/Stage 2 to run the Stage 4 benchmark report."}
        </div>
      ) : null}

      {datasetReady ? (
        <div className="space-y-6">
          <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <h3 className="text-sm font-semibold">Compare challenger models</h3>
            <p className="mt-1 text-xs text-muted-foreground">
              Reuses the same lightweight, no-CV comparison from the model development pipeline's
              Model Selection step — fit → predict → summary metrics per model, so you can see which
              challenger is worth benchmarking against before committing to one.
            </p>

            <div className="mt-4 flex flex-wrap gap-2">
              {CHALLENGER_CANDIDATES.map((name) => (
                <label
                  key={name}
                  className={
                    "inline-flex cursor-pointer items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium " +
                    (selectedCandidates.includes(name)
                      ? "border-primary/40 bg-primary-soft text-foreground"
                      : "border-border bg-background text-muted-foreground")
                  }
                >
                  <input
                    type="checkbox"
                    className="hidden"
                    checked={selectedCandidates.includes(name)}
                    onChange={() => toggleCandidate(name)}
                  />
                  {name}
                </label>
              ))}
            </div>

            <div className="mt-4 flex items-center gap-3">
              <button
                type="button"
                onClick={() => void runComparison()}
                disabled={comparing || !datasetFile || selectedCandidates.length === 0}
                className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {comparing ? <Loader2 className="h-4 w-4 animate-spin" /> : <PlayCircle className="h-4 w-4" />}
                Compare Challengers
              </button>
              <span className="text-xs text-muted-foreground">
                Current challenger: <span className="font-semibold text-foreground">{challengerModelName}</span>
              </span>
            </div>

            {comparisonError ? (
              <div className="mt-3 flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{comparisonError}</span>
              </div>
            ) : null}

            {comparisonRows && comparisonRows.length > 0 ? (
              <div className="mt-4 overflow-x-auto rounded-lg border border-border">
                <table className="w-full text-sm">
                  <thead className="bg-background text-[10px] uppercase tracking-wider text-muted-foreground">
                    <tr>
                      <th className="px-3 py-2 text-left">Model</th>
                      <th className="px-3 py-2 text-right">ROC-AUC</th>
                      <th className="px-3 py-2 text-right">Gini</th>
                      <th className="px-3 py-2 text-right">Recall</th>
                      <th className="px-3 py-2 text-right">KS</th>
                      <th className="px-3 py-2 text-right">Fit time (s)</th>
                      <th className="px-3 py-2 text-left">Select challenger model</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {comparisonRows.map((row) => {
                      const isSelected = row.model_name === challengerModelName;
                      const gini = typeof row.gini === "number" ? row.gini : (typeof row.roc_auc === "number" ? 2 * row.roc_auc - 1 : undefined);
                      return (
                        <tr key={row.model_name} className={isSelected ? "bg-primary-soft" : undefined}>
                          <td className="px-3 py-2 font-medium">{row.model_name}</td>
                          {row.error ? (
                            <td colSpan={5} className="px-3 py-2 text-xs text-destructive">{row.error}</td>
                          ) : (
                            <>
                              <td className="px-3 py-2 text-right tabular-nums">{formatValue(row.roc_auc, 3)}</td>
                              <td className="px-3 py-2 text-right tabular-nums">{formatValue(gini, 3)}</td>
                              <td className="px-3 py-2 text-right tabular-nums">{formatValue(row.recall, 3)}</td>
                              <td className="px-3 py-2 text-right tabular-nums">{formatValue(row.ks, 3)}</td>
                              <td className="px-3 py-2 text-right tabular-nums">{formatValue(row.training_time_s, 2)}</td>
                            </>
                          )}
                          <td className="px-3 py-2">
                            <button
                              type="button"
                              onClick={() => {
                                setChallengerModelName(row.model_name);
                                setChallengerPicked(true);
                                // A newly-picked challenger invalidates any
                                // previously-run (or cached) benchmark result
                                // — don't let a stale comparison linger under
                                // the newly-selected challenger's name.
                                setPayload(null);
                              }}
                              disabled={Boolean(row.error)}
                              className={
                                "rounded-lg border px-3 py-1 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-50 " +
                                (isSelected
                                  ? "border-primary bg-primary text-primary-foreground"
                                  : "border-border bg-background hover:border-primary/40")
                              }
                            >
                              {isSelected ? "Selected" : "Select"}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : null}
          </section>

          <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="text-sm">
                <span className="font-medium">Champion vs Challenger</span>
                <p className="mt-1 text-xs text-muted-foreground">
                  {challengerPicked ? (
                    <>
                      Benchmarks the champion model against{" "}
                      <span className="font-semibold text-foreground">{challengerModelName}</span> — pick a
                      different row above and re-run to compare against another challenger.
                    </>
                  ) : (
                    "Run Compare Challengers above and select a row before running the benchmark."
                  )}
                </p>

                {hasMddMetrics ? (
                  <div className="mt-2 rounded-lg border border-border bg-background px-3 py-2 text-xs text-muted-foreground">
                    <span className="font-semibold text-foreground">Champion metrics sourced from MDD</span>{" "}
                    (Stage 1 — Intake &amp; Governance):{" "}
                    {Object.entries(mddMetricsFromIntake as Record<string, any>)
                      .map(([k, v]) => `${k.replace(/_/g, " ")}=${v}`)
                      .join(", ")}
                  </div>
                ) : (
                  <div className="mt-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    No MDD metrics found from Stage 1 (Intake &amp; Governance). Go back and upload the MDD
                    there, or parse one here as a fallback:
                    <label className="mt-2 flex cursor-pointer items-center gap-2 text-foreground hover:underline">
                      <Upload className="h-3.5 w-3.5" />
                      <span>{mddParsing ? "Parsing…" : "Upload & Parse MDD"}</span>
                      <input
                        type="file"
                        accept=".pdf,.docx,.txt"
                        className="hidden"
                        disabled={mddParsing}
                        onChange={async (e) => {
                          const f = e.target.files?.[0] ?? null;
                          if (!f) return;
                          setMddParseError(null);
                          setMddParsing(true);
                          try {
                            const form = new FormData();
                            form.append("mdd_file", f);
                            const resp = await formUpload<Record<string, any>>("/validation/parse-mdd", form);
                            // Write back to the same shared context Stage 1
                            // uses, so this becomes the one source of truth
                            // for every stage rather than a page-local copy.
                            ds.setValidationMddText(resp?.mdd_text ?? null);
                            ds.setValidationMddMetrics(resp?.metrics ?? null);
                          } catch (err) {
                            setMddParseError(err instanceof Error ? err.message : "Failed to parse MDD file.");
                          } finally {
                            setMddParsing(false);
                          }
                        }}
                      />
                    </label>
                    {mddParseError ? <div className="mt-1 text-destructive">{mddParseError}</div> : null}
                  </div>
                )}
              </div>
              <button
                type="button"
                onClick={() => handleRun()}
                disabled={loading || !datasetFile || !challengerPicked || !hasMddMetrics}
                className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <PlayCircle className="h-4 w-4" />}
                Run Benchmark
              </button>
            </div>
          </section>

          {payload ? (
            <>
              <Card title="Industry benchmark table" sub="Selected challenger benchmark versus champion model">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-background text-[10px] uppercase tracking-wider text-muted-foreground">
                      <tr>
                        <th className="px-3 py-2 text-left">#</th>
                        <th className="px-3 py-2 text-left">Model</th>
                        <th className="px-3 py-2 text-right">ROC-AUC</th>
                        <th className="px-3 py-2 text-right">Gini</th>
                        <th className="px-3 py-2 text-right">Recall</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {benchmarkTableRows.map((row, rowIndex) => (
                        <tr key={row.model}>
                          <td className="px-3 py-2 text-muted-foreground">{rowIndex + 1}</td>
                          <td className="px-3 py-2 font-medium">{row.model}</td>
                          <td className="px-3 py-2 text-right tabular-nums">{formatValue(row.roc_auc, 3)}</td>
                          <td className="px-3 py-2 text-right tabular-nums">{formatValue(row.gini, 3)}</td>
                          <td className="px-3 py-2 text-right tabular-nums">{formatValue(row.recall, 3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* Both cards render a 320px-tall Plotly figure, so pairing them
                  in their own row (rather than sharing a row with the much
                  shorter table above) keeps the two charts aligned instead of
                  one trailing below on a half-empty row. */}
              <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card title="Champion vs Challenger comparison" sub="Metric deltas from the benchmark response">
                  <PlotlyChart figure={benchmarkComparisonFigure} style={{ height: "100%" }} />
                </Card>

                <Card title="ROC overlay chart" sub="Champion vs selected benchmark model">
                  <PlotlyChart figure={benchmarkOverlayFigure} style={{ height: "100%" }} />
                </Card>
              </section>
            </>
          ) : null}
        </div>
      ) : null}

      <div className="text-right">
        <Link
          to="/validation/stress"
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90"
        >
          Continue to Stage 5
          <ArrowRight className="h-4 w-4" />
        </Link>
      </div>
    </div>
  );
}

function Card({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
      <h3 className="text-sm font-semibold">{title}</h3>
      {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
      <div className="mt-4 h-56">{children}</div>
    </div>
  );
}
