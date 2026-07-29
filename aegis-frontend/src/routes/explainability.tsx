import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { PageHeader } from "@/components/app-shell";
import { useDataset } from "@/lib/app-context";
import { formUpload } from "@/lib/api";
import { useEffect, useMemo, useState } from "react";
import PlotlyChart from "@/components/plotly-chart";
import { AlertCircle, ArrowLeft, ArrowRight, Loader, Download, Printer, BarChart3, Microscope, Search, ClipboardList } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import {
  missingTreatmentCounts,
  preprocessingColumnRows,
  computeGini,
  hyperparameterSummary,
  featureRemovalSummary,
  topInteractionTerms,
  woeFeatureSummary,
  rowsToCsv,
} from "@/lib/full-report";

type FeatureImportanceRow = {
  Feature: string;
  Importance: number;
};

type SampleShapRow = {
  Feature: string;
  SHAP: number | null;
  Value: unknown;
};

type ShapInfo = {
  shap_available: boolean;
  shap_mean_abs?: Array<{ Feature: string; MeanAbsSHAP: number }>;
  sample_idx?: number;
  sample_reasoning?: string;
  sample_shap?: SampleShapRow[];
  sample_features?: Record<string, unknown>;
};

type ExplainabilityResponse = {
  feature_importance: FeatureImportanceRow[];
  shap: ShapInfo;
  summary?: string | null;
};

export const Route = createFileRoute("/explainability")({
  head: () => ({ meta: [{ title: "Explainability — Aegis Credit" }] }),
  component: Explainability,
});

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function base64ToBlob(base64: string, mime = "application/octet-stream"): Blob {
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  return new Blob([bytes], { type: mime });
}

function formatValue(value: unknown) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (Math.abs(value) >= 1000) return Math.round(value).toString();
    return Number.isFinite(value) ? value.toFixed(4) : String(value);
  }
  return String(value);
}

// Remove markdown tokens for plain-text rendering but preserve line order —
// do not rewrite or summarise the backend-generated content.
function plainMarkdown(text?: string | null) {
  if (!text) return "";
  let t = text.replace(/\*\*/g, "").replace(/`/g, "").replace(/(^|\n)#+\s*/g, "$1");
  t = t.replace(/(^|\n)\s*-\s*/g, "$1");
  return t;
}

function Explainability() {
  const navigate = useNavigate();
  const { profile, file, trainingResult, preprocessingResult, featureEngineeringResult } = useDataset();

  const modelArtifact = trainingResult?.model_artifact ?? null;
  const trainingConfig = trainingResult?.training_config ?? {};
  const hasEngine = Boolean(modelArtifact && file);

  // Declared early (used by useMemo hooks below, which must not reference
  // consts declared later in the same function — that's a temporal-dead-zone
  // ReferenceError, not just a lint nit).
  const metrics = trainingResult?.evaluation_metrics ?? {};
  const metricsChartData = [
    { metric: "Accuracy", value: metrics.accuracy },
    { metric: "Precision", value: metrics.precision },
    { metric: "Recall", value: metrics.recall },
    { metric: "F1", value: metrics.f1 },
    { metric: "ROC-AUC", value: metrics.roc_auc },
    { metric: "PR-AUC", value: metrics.pr_auc },
  ].filter((row) => typeof row.value === "number");
  const classDistribution: Record<string, number> = profile?.class_distribution ?? {};
  const classDistributionChartData = Object.entries(classDistribution).map(([cls, count]) => ({ cls, count }));

  const targetColumn = profile?.target_col ??
    (Array.isArray(profile?.target_candidates) && profile.target_candidates.length > 0
      ? profile.target_candidates[0]
      : "loan_status");

  const [data, setData] = useState<ExplainabilityResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [topN, setTopN] = useState(15);
  const [shapSamples, setShapSamples] = useState(150);
  const [shapComputed, setShapComputed] = useState(false);
  const [shapLoading, setShapLoading] = useState(false);
  const [shapError, setShapError] = useState<string | null>(null);
  const [sampleIdx, setSampleIdx] = useState(0);

  const explainParams = () => {
    const form = new FormData();
    // Sent as a file part (not a plain text field): the base64-encoded
    // pipeline can be several MB, and multipart parsers cap plain form
    // fields much lower than file parts, which was causing 400s here.
    const modelArtifactBlob = new Blob([modelArtifact!], { type: "text/plain" });
    form.append("model_artifact", modelArtifactBlob, "model_artifact.b64");
    form.append("file", file!);
    form.append("target_col", targetColumn);
    form.append("use_feature_engineering", String(Boolean(trainingConfig.use_feature_engineering)));
    form.append("test_size", String(trainingConfig.test_size ?? 0.15));
    form.append("val_size", String(trainingConfig.val_size ?? 0.15));
    form.append("random_seed", String(trainingConfig.random_seed ?? 42));
    form.append("task_type", trainingResult?.task_type ?? "binary");
    if (trainingResult?.evaluation_metrics) {
      form.append("metrics", JSON.stringify(trainingResult.evaluation_metrics));
    }
    return form;
  };

  // Initial load: Feature Importance only (fast, no SHAP) — matches the old
  // app's tab behaviour where Feature Importance renders immediately and
  // SHAP is only computed on an explicit button click.
  const fetchFeatureImportance = async () => {
    if (!hasEngine) return;
    try {
      setLoading(true);
      setError(null);
      const form = explainParams();
      form.append("compute_shap", "false");
      const response = await formUpload<ExplainabilityResponse>("/models/explain", form);
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load explainability data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (hasEngine) void fetchFeatureImportance();
    setShapComputed(false);
    setSampleIdx(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasEngine, modelArtifact, file]);

  const computeShap = async () => {
    if (!hasEngine) return;
    try {
      setShapLoading(true);
      setShapError(null);
      const form = explainParams();
      form.append("compute_shap", "true");
      form.append("max_shap_samples", String(shapSamples));
      form.append("sample_idx", String(sampleIdx));
      const response = await formUpload<ExplainabilityResponse>("/models/explain", form);
      setData(response);
      if (response.shap.shap_available) {
        setShapComputed(true);
      } else {
        setShapError("SHAP computation failed or not supported for this model. Try the Feature Importance tab.");
      }
    } catch (err) {
      setShapError(err instanceof Error ? err.message : "Failed to compute SHAP values");
    } finally {
      setShapLoading(false);
    }
  };

  // Re-fetch reasoning when the reviewer picks a different sample, once SHAP
  // is already computed — cheap relative to a full SHAP recompute.
  const changeSample = async (idx: number) => {
    setSampleIdx(idx);
    if (!shapComputed || !hasEngine) return;
    try {
      const form = explainParams();
      form.append("compute_shap", "true");
      form.append("max_shap_samples", String(shapSamples));
      form.append("sample_idx", String(idx));
      const response = await formUpload<ExplainabilityResponse>("/models/explain", form);
      setData(response);
    } catch (err) {
      setShapError(err instanceof Error ? err.message : "Failed to load prediction reasoning");
    }
  };

  const importanceRows = data?.feature_importance ?? [];
  const maxTopN = Math.max(5, importanceRows.length);
  const chartRows = useMemo(
    () => [...importanceRows].slice(0, topN).sort((a, b) => a.Importance - b.Importance),
    [importanceRows, topN],
  );

  const shapSummaryRows = useMemo(
    () =>
      (data?.shap.shap_mean_abs ?? [])
        .slice()
        .sort((a, b) => (b.MeanAbsSHAP ?? 0) - (a.MeanAbsSHAP ?? 0))
        .slice(0, 15)
        .sort((a, b) => (a.MeanAbsSHAP ?? 0) - (b.MeanAbsSHAP ?? 0)),
    [data],
  );

  const sampleShapRows = data?.shap.sample_shap ?? [];
  const sampleCount = sampleShapRows.length > 0
    ? Math.max(sampleShapRows.length, sampleIdx + 1)
    : 0;
  const importanceFigure = useMemo(() => {
    const labels = chartRows.map((r) => r.Feature);
    const values = chartRows.map((r) => r.Importance);
    return {
      data: [
        {
          type: "bar",
          x: values,
          y: labels,
          orientation: "h",
          marker: { color: "oklch(0.6 0.18 275)" },
          hovertemplate: "%{x:.4f}<extra></extra>",
        },
      ],
      layout: {
        margin: { l: 160, r: 30, t: 20, b: 40 },
        xaxis: { tickfont: { size: 10 }, showline: false },
        yaxis: { tickfont: { size: 9 }, automargin: true, type: "category", autorange: "reversed" },
        height: 384,
      },
    };
  }, [chartRows]);

  const shapSummaryFigure = useMemo(() => {
    const labels = shapSummaryRows.map((r) => r.Feature);
    const values = shapSummaryRows.map((r) => r.MeanAbsSHAP ?? 0);
    return {
      data: [
        {
          type: "bar",
          x: values,
          y: labels,
          orientation: "h",
          marker: { color: "oklch(0.6 0.2 25)" },
          hovertemplate: "%{x:.5f}<extra></extra>",
        },
      ],
      layout: {
        margin: { l: 160, r: 30, t: 20, b: 40 },
        xaxis: { tickfont: { size: 10 }, showline: false },
        yaxis: { tickfont: { size: 9 }, automargin: true, type: "category", autorange: "reversed" },
        height: 384,
      },
    };
  }, [shapSummaryRows]);

  const sampleShapFigure = useMemo(() => {
    const rows = [...sampleShapRows].slice(0, 12).reverse();
    const labels = rows.map((r) => `${r.Feature} = ${formatValue(r.Value)}`);
    const values = rows.map((r) => (r.SHAP ?? 0));
    const colors = rows.map((r) => ((r.SHAP ?? 0) < 0 ? "oklch(0.7 0.15 150)" : "oklch(0.6 0.2 25)"));
    return {
      data: [
        {
          type: "bar",
          x: values,
          y: labels,
          orientation: "h",
          marker: { color: colors },
          hovertemplate: "%{x:.5f}<extra></extra>",
        },
      ],
      layout: {
        margin: { l: 220, r: 30, t: 20, b: 40 },
        xaxis: { tickfont: { size: 10 }, showline: false },
        yaxis: { tickfont: { size: 9 }, automargin: true, type: "category", autorange: "reversed" },
        height: 320,
      },
    };
  }, [sampleShapRows]);

  const classDistributionFigure = useMemo(() => {
    const labels = classDistributionChartData.map((r: any) => r.cls);
    const values = classDistributionChartData.map((r: any) => r.count);
    return {
      data: [
        {
          type: "bar",
          x: labels,
          y: values,
          marker: { color: "oklch(0.6 0.18 275)" },
          hovertemplate: "%{y}<extra></extra>",
        },
      ],
      layout: { margin: { l: 10, r: 20, t: 10, b: 40 }, height: 256 },
    };
  }, [classDistributionChartData]);

  const metricsFigure = useMemo(() => {
    const labels = metricsChartData.map((r: any) => r.metric);
    const values = metricsChartData.map((r: any) => r.value);
    return {
      data: [
        {
          type: "bar",
          x: labels,
          y: values,
          marker: { color: "oklch(0.6 0.2 25)" },
          hovertemplate: "%{y:.4f}<extra></extra>",
        },
      ],
      layout: { margin: { l: 10, r: 20, t: 10, b: 40 }, yaxis: { range: [0, 1] }, height: 256 },
    };
  }, [metricsChartData]);

  const importanceTop10Figure = useMemo(() => {
    const rows = [...importanceRows].slice(0, 10).sort((a, b) => a.Importance - b.Importance);
    const labels = rows.map((r) => r.Feature);
    const values = rows.map((r) => r.Importance);
    return {
      data: [
        {
          type: "bar",
          x: values,
          y: labels,
          orientation: "h",
          marker: { color: "oklch(0.6 0.18 275)" },
          hovertemplate: "%{x:.4f}<extra></extra>",
        },
      ],
      layout: {
        margin: { l: 130, r: 30, t: 20, b: 40 },
        xaxis: { tickfont: { size: 10 }, showline: false },
        yaxis: { tickfont: { size: 9 }, automargin: true, type: "category", autorange: "reversed" },
        height: 280,
      },
    };
  }, [importanceRows]);
  const downloadModel = () => {
    if (!modelArtifact) return;
    downloadBlob(base64ToBlob(modelArtifact), "final_credit_risk_model.pkl");
  };

  const downloadProcessedDataset = () => {
    const csv = preprocessingResult?.processed_dataset_csv ?? featureEngineeringResult?.x_engineered_csv;
    if (!csv) return;
    downloadBlob(new Blob([csv], { type: "text/csv;charset=utf-8" }), "processed_dataset.csv");
  };

  const downloadFeatureImportanceCsv = () => {
    if (importanceRows.length === 0) return;
    const header = "Feature,Importance";
    const rows = importanceRows.map((r) => `${r.Feature},${r.Importance}`);
    downloadBlob(new Blob([[header, ...rows].join("\n")], { type: "text/csv;charset=utf-8" }), "feature_importance.csv");
  };

  if (!hasEngine) {
    return (
      <div className="space-y-8">
        <PageHeader title="Model Explainability" description="Feature importance, SHAP values, and individual prediction reasoning." />
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-6">
          <div className="flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-amber-600" />
            <div>
              <div className="font-semibold text-amber-900">No trained model</div>
              <div className="text-sm text-amber-800">Complete model training first, then return here to explain the model.</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Full report data — every field below is read from state already
  // computed elsewhere in the pipeline; nothing here re-derives analysis. ──
  const trainingConfig2 = trainingResult?.training_config ?? {};
  const trainingInfo = trainingResult?.training_info ?? {};
  const classificationReport = metrics.classification_report ?? {};
  const feSummary = featureEngineeringResult?.feature_engineering_summary ?? {};
  const feAdded = Array.isArray(feSummary.added) ? feSummary.added : [];
  const feRemoved = Array.isArray(feSummary.removed) ? feSummary.removed : [];
  const feOriginalFeatures = Array.isArray(feSummary.original_shape) ? feSummary.original_shape[1] ?? null : null;
  const feFinalFeatures = Array.isArray(feSummary.final_shape) ? feSummary.final_shape[1] ?? null : null;
  const removalProposal = featureRemovalSummary(featureEngineeringResult);
  const rescuedFeatures = removalProposal.rows.filter((r) => r.rescued);
  const interactionRows = topInteractionTerms(featureEngineeringResult);
  const woeRows = woeFeatureSummary(featureEngineeringResult);
  const treatmentCounts = missingTreatmentCounts(preprocessingResult?.applied_treatment_map);
  const strategyRows = preprocessingColumnRows(preprocessingResult);
  const splitStats = preprocessingResult?.split_stats ?? {};
  const hyperparams = hyperparameterSummary(trainingResult);
  const gini = computeGini(metrics.roc_auc);
  const reportGeneratedAt = new Date().toLocaleString();

  const downloadReport = () => window.print();

  const downloadDataSummaryCsv = () => {
    const dict = profile?.data_dictionary ?? [];
    if (dict.length === 0) return;
    const headers = Object.keys(dict[0]);
    downloadBlob(
      new Blob([rowsToCsv(headers, dict.map((row: Record<string, any>) => headers.map((h) => row[h])))], { type: "text/csv;charset=utf-8" }),
      "data_summary.csv",
    );
  };

  const downloadOriginalDatasetCsv = () => {
    if (!preprocessingResult?.original_dataset_csv) return;
    downloadBlob(new Blob([preprocessingResult.original_dataset_csv], { type: "text/csv;charset=utf-8" }), "original_dataset.csv");
  };

  const downloadTransformedDatasetCsv = () => {
    if (!preprocessingResult?.processed_dataset_csv) return;
    downloadBlob(new Blob([preprocessingResult.processed_dataset_csv], { type: "text/csv;charset=utf-8" }), "transformed_dataset.csv");
  };

  const downloadEngineeredDatasetCsv = () => {
    if (!featureEngineeringResult?.x_engineered_csv) return;
    downloadBlob(new Blob([featureEngineeringResult.x_engineered_csv], { type: "text/csv;charset=utf-8" }), "engineered_dataset.csv");
  };

  const downloadFeatureDecisionLog = async () => {
    if (!file) return;
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("target_col", targetColumn);
      const res = await formUpload<any>("/data/feature-decision-log", form);
      if (res?.content_base64) {
        downloadBlob(base64ToBlob(res.content_base64, "text/csv"), res.file_name || "feature_decision_log.csv");
      }
    } catch (err) {
      console.error("Failed to download feature decision log:", err);
    }
  };

  const downloadMetricsCsv = () => {
    const rows = Object.entries(metrics).filter(([, v]) => typeof v === "number" && Number.isFinite(v));
    if (rows.length === 0) return;
    downloadBlob(
      new Blob([rowsToCsv(["Metric", "Value"], rows.map(([k, v]) => [k, v]))], { type: "text/csv;charset=utf-8" }),
      "metrics.csv",
    );
  };

  const downloadArtifacts: Array<{ label: string; filename: string; available: boolean; onClick: () => void }> = [
    { label: "Data summary (Data Profiling)", filename: "data_summary.csv", available: (profile?.data_dictionary ?? []).length > 0, onClick: downloadDataSummaryCsv },
    { label: "Original dataset (Preprocessing)", filename: "original_dataset.csv", available: Boolean(preprocessingResult?.original_dataset_csv), onClick: downloadOriginalDatasetCsv },
    { label: "Transformed dataset (Preprocessing)", filename: "transformed_dataset.csv", available: Boolean(preprocessingResult?.processed_dataset_csv), onClick: downloadTransformedDatasetCsv },
    { label: "Engineered dataset (Feature Engineering)", filename: "engineered_dataset.csv", available: Boolean(featureEngineeringResult?.x_engineered_csv), onClick: downloadEngineeredDatasetCsv },
    { label: "Feature decision log (Feature Engineering)", filename: "feature_decision_log.csv", available: Boolean(file), onClick: downloadFeatureDecisionLog },
    { label: "Evaluation metrics (Model Evaluation)", filename: "metrics.csv", available: Object.keys(metrics).length > 0, onClick: downloadMetricsCsv },
    { label: "Feature importance (Explainability)", filename: "feature_importance.csv", available: importanceRows.length > 0, onClick: downloadFeatureImportanceCsv },
    { label: "Trained model artifact", filename: "final_credit_risk_model.pkl", available: Boolean(modelArtifact), onClick: downloadModel },
    { label: "Full report (PDF via print)", filename: "full_report.pdf", available: true, onClick: downloadReport },
  ];

  return (
    <div className="space-y-8">
      <PageHeader title="💡 Model Explainability" description="Feature importance, SHAP values, and individual prediction reasoning." />

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">{error}</div>
      )}

      <Tabs defaultValue="importance" className="w-full">
        <TabsList>
          <TabsTrigger value="importance" className="gap-2"><BarChart3 className="h-4 w-4" />Feature Importance</TabsTrigger>
          <TabsTrigger value="shap" className="gap-2"><Microscope className="h-4 w-4" />SHAP Analysis</TabsTrigger>
          <TabsTrigger value="reasoning" className="gap-2"><Search className="h-4 w-4" />Prediction Reasoning</TabsTrigger>
          <TabsTrigger value="summary" className="gap-2"><ClipboardList className="h-4 w-4" />Summary</TabsTrigger>
        </TabsList>

        {/* ── Feature Importance ── */}
        <TabsContent value="importance" className="space-y-4 pt-4">
          <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <h2 className="text-base font-semibold">📊 Feature Importance</h2>
            {loading ? (
              <div className="mt-6 flex items-center gap-3 text-sm text-muted-foreground">
                <Loader className="h-4 w-4 animate-spin" />
                Extracting feature importance with real column names...
              </div>
            ) : importanceRows.length > 0 ? (
              <>
                <div className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900">
                  ✅ Extracted importance for {importanceRows.length} features with real column names.
                </div>
                <div className="mt-4 max-w-md">
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>Show top N features</span>
                    <span className="font-semibold text-foreground">{topN}</span>
                  </div>
                  <Slider
                    min={5}
                    max={maxTopN}
                    step={1}
                    value={[topN]}
                    onValueChange={(v) => setTopN(v[0])}
                    className="mt-2"
                  />
                </div>
                <div className="mt-6 h-96">
                  <PlotlyChart figure={importanceFigure} />
                </div>
                <div className="mt-6 overflow-x-auto">
                  <table className="min-w-full border-collapse text-sm">
                    <thead>
                      <tr className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                        <th className="border-b border-border px-3 py-2">#</th>
                        <th className="border-b border-border px-3 py-2">Feature</th>
                        <th className="border-b border-border px-3 py-2">Importance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {importanceRows.slice(0, topN + 5).map((row, rowIndex) => (
                        <tr key={row.Feature} className="odd:bg-background">
                          <td className="border-b border-border px-3 py-2 font-mono text-xs text-muted-foreground">{rowIndex + 1}</td>
                          <td className="border-b border-border px-3 py-2 font-mono text-xs">{row.Feature}</td>
                          <td className="border-b border-border px-3 py-2 text-xs">{row.Importance.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <p className="mt-4 text-sm text-muted-foreground">Feature importance not available for this model type.</p>
            )}
          </section>
        </TabsContent>

        {/* ── SHAP Analysis ── */}
        <TabsContent value="shap" className="space-y-4 pt-4">
          <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <h2 className="text-base font-semibold">🔬 SHAP Values</h2>
            <div className="mt-3 rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900">
              This shows how much each factor pushed an individual prediction up or down, using your actual column names.
            </div>

            <div className="mt-4 max-w-md">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Samples for SHAP analysis</span>
                <span className="font-semibold text-foreground">{shapSamples}</span>
              </div>
              <Slider
                min={50}
                max={300}
                step={50}
                value={[shapSamples]}
                onValueChange={(v) => setShapSamples(v[0])}
                className="mt-2"
              />
            </div>

            <Button onClick={computeShap} disabled={shapLoading} className="mt-4 w-full gap-2">
              {shapLoading ? <Loader className="h-4 w-4 animate-spin" /> : null}
              ▶ Compute SHAP Values
            </Button>

            {shapError && (
              <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">{shapError}</div>
            )}

            {shapComputed && shapSummaryRows.length > 0 && (
              <div className="mt-6">
                <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900">
                  ✅ SHAP values computed for {shapSummaryRows.length >= 15 ? "15+" : shapSummaryRows.length} features!
                </div>
                <div className="mt-4 h-96">
                  <PlotlyChart figure={shapSummaryFigure} />
                </div>
              </div>
            )}
          </section>
        </TabsContent>

        {/* ── Prediction Reasoning ── */}
        <TabsContent value="reasoning" className="space-y-4 pt-4">
          <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <h2 className="text-base font-semibold">🔍 Individual Prediction Reasoning</h2>
            <div className="mt-3 rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900">
              Inspect exactly why a specific customer was classified as risky or safe — with real feature names.
            </div>

            {!shapComputed ? (
              <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                Please compute SHAP values first (in the SHAP Analysis tab).
              </div>
            ) : (
              <>
                <div className="mt-4 max-w-xs">
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Customer index (row number)
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={Math.max(0, sampleCount - 1)}
                    value={sampleIdx}
                    onChange={(e) => void changeSample(Math.max(0, Number(e.target.value)))}
                    className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
                  />
                </div>

                {data?.shap.sample_reasoning && (
                  <div className="mt-4 whitespace-pre-line rounded-lg border border-border bg-background p-4 text-sm">
                    {plainMarkdown(data.shap.sample_reasoning)}
                  </div>
                )}

                {sampleShapRows.length > 0 && (
                  <div className="mt-6 h-80">
                    <PlotlyChart figure={sampleShapFigure} />
                  </div>
                )}

                {Object.keys(data?.shap.sample_features ?? {}).length > 0 && (
                  <details className="mt-6">
                    <summary className="cursor-pointer text-sm font-semibold text-foreground">
                      🔎 Raw Feature Values for This Customer
                    </summary>
                    <div className="mt-3 overflow-x-auto">
                      <table className="min-w-full border-collapse text-sm">
                        <thead>
                          <tr className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                            <th className="border-b border-border px-3 py-2">#</th>
                            <th className="border-b border-border px-3 py-2">Feature</th>
                            <th className="border-b border-border px-3 py-2">Value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(data?.shap.sample_features ?? {}).map(([feat, val], rowIndex) => (
                            <tr key={feat} className="odd:bg-background">
                              <td className="border-b border-border px-3 py-2 font-mono text-xs text-muted-foreground">{rowIndex + 1}</td>
                              <td className="border-b border-border px-3 py-2 font-mono text-xs">{feat}</td>
                              <td className="border-b border-border px-3 py-2 text-xs">{formatValue(val)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                )}
              </>
            )}
          </section>
        </TabsContent>

        {/* ── Summary ── */}
        <TabsContent value="summary" className="space-y-4 pt-4">
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground">Report generated {reportGeneratedAt}</p>
            <Button onClick={downloadReport} className="no-print gap-2">
              <Printer className="h-4 w-4" />
              Download Full Report
            </Button>
          </div>

          <div id="full-report-content" className="space-y-6">
            {/* 1. Model identity */}
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold">1. Model Identity</h2>
              <div className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
                <div><strong>Model:</strong> {trainingResult?.model_name ?? "—"}</div>
                <div><strong>Task type:</strong> {trainingResult?.task_type ?? "—"}</div>
                <div><strong>Target column:</strong> {targetColumn}</div>
                <div><strong>Report generated:</strong> {reportGeneratedAt}</div>
              </div>
            </section>

            {/* 2. Dataset summary */}
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold">2. Dataset Summary</h2>
              <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                <div><strong>Rows:</strong> {profile?.shape?.[0]?.toLocaleString() ?? "—"}</div>
                <div><strong>Columns:</strong> {profile?.shape?.[1]?.toLocaleString() ?? "—"}</div>
                <div><strong>Missing:</strong> {profile?.missing_percentage !== undefined ? `${profile.missing_percentage}%` : "—"}</div>
                <div><strong>Duplicates:</strong> {profile?.duplicate_rows ?? "—"}</div>
              </div>
              {Object.keys(classDistribution).length > 0 && (
                <div className="mt-4">
                  <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Class distribution</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {Object.entries(classDistribution).map(([cls, count]) => (
                      <span key={cls} className="rounded-full border border-border bg-background px-3 py-1 text-xs">
                        {cls}: {count.toLocaleString()}
                      </span>
                    ))}
                  </div>
                  {profile?.target_summary?.imbalance_ratio !== undefined && profile.target_summary.imbalance_ratio !== null && (
                    <p className="mt-2 text-xs text-muted-foreground">Imbalance ratio: {profile.target_summary.imbalance_ratio}:1</p>
                  )}
                  <div className="mt-4 h-64 w-full max-w-xl">
                    <PlotlyChart figure={classDistributionFigure} style={{ minHeight: 256 }} />
                  </div>
                </div>
              )}
            </section>

            {/* 3. Preprocessing decisions */}
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold">3. Preprocessing Decisions</h2>
              <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
                <div><strong>Train:</strong> {splitStats.train_n?.toLocaleString() ?? "—"}</div>
                <div><strong>Validation:</strong> {splitStats.val_n?.toLocaleString() ?? "—"}</div>
                <div><strong>Test:</strong> {splitStats.test_n?.toLocaleString() ?? "—"}</div>
              </div>
              {treatmentCounts.length > 0 && (
                <div className="mt-4">
                  <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Missing-value treatment</div>
                  <ul className="mt-2 space-y-1 text-sm">
                    {treatmentCounts.map((t) => (
                      <li key={t.treatment}>{t.label}: <strong>{t.count}</strong> column{t.count === 1 ? "" : "s"}</li>
                    ))}
                  </ul>
                </div>
              )}
              {strategyRows.length > 0 && (
                <div className="mt-4 overflow-x-auto">
                  <table className="min-w-full border-collapse text-sm">
                    <thead>
                      <tr className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                        <th className="border-b border-border px-3 py-2">Column</th>
                        <th className="border-b border-border px-3 py-2">Type</th>
                        <th className="border-b border-border px-3 py-2">Scaler</th>
                        <th className="border-b border-border px-3 py-2">Imputer</th>
                        <th className="border-b border-border px-3 py-2">Encoding</th>
                        <th className="border-b border-border px-3 py-2">Transform</th>
                      </tr>
                    </thead>
                    <tbody>
                      {strategyRows.map((row: any) => (
                        <tr key={row.feature} className="odd:bg-background">
                          <td className="border-b border-border px-3 py-2 font-mono text-xs">{row.feature}</td>
                          <td className="border-b border-border px-3 py-2 text-xs">{row.type}</td>
                          <td className="border-b border-border px-3 py-2 text-xs">{row.scaler}</td>
                          <td className="border-b border-border px-3 py-2 text-xs">{row.imputer}</td>
                          <td className="border-b border-border px-3 py-2 text-xs">{row.encoding}</td>
                          <td className="border-b border-border px-3 py-2 text-xs">{row.transform}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            {/* 4. Feature engineering summary */}
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold">4. Feature Engineering Summary</h2>
              <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                <div><strong>Before:</strong> {feOriginalFeatures ?? "—"}</div>
                <div><strong>After:</strong> {feFinalFeatures ?? "—"}</div>
                <div><strong>Added:</strong> {feAdded.length}</div>
                <div><strong>Removed:</strong> {feRemoved.length}</div>
              </div>
              {interactionRows.length > 0 && (
                <div className="mt-4">
                  <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Top interaction terms</div>
                  <ul className="mt-2 space-y-1 text-sm">
                    {interactionRows.slice(0, 5).map((f: any) => (
                      <li key={f.name}><code className="font-mono text-xs">{f.name}</code> ({f.feature_a} × {f.feature_b})</li>
                    ))}
                  </ul>
                </div>
              )}
              {woeRows.length > 0 && (
                <div className="mt-4">
                  <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">WOE-encoded features</div>
                  <ul className="mt-2 space-y-1 text-sm">
                    {woeRows.map((w) => (
                      <li key={w.feature}><code className="font-mono text-xs">{w.feature}</code> — {w.buckets} buckets{w.iv !== null ? `, IV ${w.iv.toFixed(4)}` : ""}</li>
                    ))}
                  </ul>
                </div>
              )}
              {rescuedFeatures.length > 0 && (
                <div className="mt-4">
                  <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Cascade-rescue actions</div>
                  <ul className="mt-2 space-y-1 text-sm">
                    {rescuedFeatures.map((r) => (
                      <li key={r.feature}><code className="font-mono text-xs">{r.feature}</code> — {r.reason}</li>
                    ))}
                  </ul>
                </div>
              )}
            </section>

            {/* 5. Model training configuration */}
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold">5. Model Training Configuration</h2>
              <div className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
                <div><strong>Model type:</strong> {trainingResult?.model_name ?? "—"}</div>
                <div><strong>Class balancing:</strong> {trainingInfo.class_weighting?.method ?? (trainingInfo.class_weighting?.applied ? "class_weight='balanced'" : "None")}</div>
                <div><strong>Cross-validation:</strong> {trainingConfig2.use_cv ? `${trainingConfig2.cv_folds}-fold (mean ${trainingInfo.cv_mean ?? "—"}, std ${trainingInfo.cv_std ?? "—"})` : "Not used"}</div>
                <div><strong>OOT holdout:</strong> {trainingConfig2.use_oot ? `Cutoff ${trainingInfo.oot?.cutoff_date ?? "—"}, ${trainingInfo.oot?.oot_n ?? 0} rows` : "Not used"}</div>
              </div>
              <div className="mt-4">
                <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Hyperparameters — {hyperparams.source}</div>
                {Object.keys(hyperparams.params).length > 0 ? (
                  <ul className="mt-2 grid grid-cols-2 gap-1 text-sm sm:grid-cols-3">
                    {Object.entries(hyperparams.params).map(([k, v]) => (
                      <li key={k}><code className="font-mono text-xs">{k}</code>: {String(v)}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-2 text-sm text-muted-foreground">No overrides recorded — the model used its library defaults.</p>
                )}
              </div>
            </section>

            {/* 6. Evaluation metrics */}
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold">6. Evaluation Metrics</h2>
              <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                <div><strong>Accuracy:</strong> {metrics.accuracy ?? "—"}</div>
                <div><strong>Precision:</strong> {metrics.precision ?? "—"}</div>
                <div><strong>Recall:</strong> {metrics.recall ?? "—"}</div>
                <div><strong>F1:</strong> {metrics.f1 ?? "—"}</div>
                <div><strong>ROC-AUC:</strong> {metrics.roc_auc ?? "—"}</div>
                <div><strong>PR-AUC:</strong> {metrics.pr_auc ?? "—"}</div>
                <div><strong>KS:</strong> {metrics.ks_statistic ?? "—"}</div>
                <div><strong>Brier:</strong> {metrics.brier_score ?? "—"}</div>
                <div><strong>Gini:</strong> {gini ?? "—"}</div>
              </div>
              {metricsChartData.length > 0 && (
                <div className="mt-4 h-64 w-full max-w-xl">
                  <PlotlyChart figure={metricsFigure} style={{ minHeight: 256 }} />
                </div>
              )}
              {metrics.threshold_used !== undefined && (
                <p className="mt-4 text-sm">
                  <strong>Decision threshold:</strong> {metrics.threshold_used}
                  {metrics.threshold_selection && (
                    <> — auto-selected to maximize {metrics.threshold_selection.metric} (F1 {metrics.threshold_selection.f1}, precision {metrics.threshold_selection.precision}, recall {metrics.threshold_selection.recall})</>
                  )}
                </p>
              )}
              {Object.keys(classificationReport).length > 0 && (
                <div className="mt-4 overflow-x-auto">
                  <table className="min-w-full border-collapse text-sm">
                    <thead>
                      <tr className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                        <th className="border-b border-border px-3 py-2">Class</th>
                        <th className="border-b border-border px-3 py-2">Precision</th>
                        <th className="border-b border-border px-3 py-2">Recall</th>
                        <th className="border-b border-border px-3 py-2">F1</th>
                        <th className="border-b border-border px-3 py-2">Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(classificationReport)
                        .filter(([key]) => key !== "accuracy")
                        .map(([key, row]: [string, any]) => (
                          <tr key={key} className="odd:bg-background">
                            <td className="border-b border-border px-3 py-2 font-mono text-xs">{key}</td>
                            <td className="border-b border-border px-3 py-2 text-xs">{row.precision?.toFixed?.(4) ?? "—"}</td>
                            <td className="border-b border-border px-3 py-2 text-xs">{row.recall?.toFixed?.(4) ?? "—"}</td>
                            <td className="border-b border-border px-3 py-2 text-xs">{row["f1-score"]?.toFixed?.(4) ?? "—"}</td>
                            <td className="border-b border-border px-3 py-2 text-xs">{row.support ?? "—"}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            {/* 7. Explainability findings */}
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold">7. Explainability Findings</h2>
              {importanceRows.length > 0 && (
                <div className="mt-4">
                  <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Top features by importance</div>
                  <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm">
                    {importanceRows.slice(0, 10).map((r) => (
                      <li key={r.Feature}><code className="font-mono text-xs">{r.Feature}</code> — {r.Importance.toFixed(4)}</li>
                    ))}
                  </ol>
                  <div className="mt-4 h-72 w-full max-w-xl">
                    <PlotlyChart figure={importanceTop10Figure} style={{ minHeight: 280 }} />
                  </div>
                </div>
              )}
              {data?.summary ? (
                <div className="mt-4 whitespace-pre-line text-sm text-foreground">{plainMarkdown(data.summary)}</div>
              ) : (
                <p className="mt-4 text-sm text-muted-foreground">Model performance summary is not available yet.</p>
              )}
            </section>
          </div>

          {/* Document Download Hub */}
          <section className="no-print rounded-xl border border-border bg-card p-6 shadow-elegant">
            <h3 className="text-sm font-semibold">📥 Document Download Hub</h3>
            <p className="mt-1 text-xs text-muted-foreground">Every downloadable artifact generated across the Model Development pipeline, in one place.</p>
            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {downloadArtifacts.map((artifact) => (
                <Button
                  key={artifact.filename}
                  variant="outline"
                  onClick={artifact.onClick}
                  disabled={!artifact.available}
                  className="gap-2 justify-start"
                >
                  <Download className="h-4 w-4 shrink-0" />
                  <span className="truncate">{artifact.label}</span>
                </Button>
              ))}
            </div>
          </section>
        </TabsContent>
      </Tabs>

      <div className="flex flex-wrap gap-3 pt-2">
        <Button variant="outline" onClick={() => navigate({ to: "/evaluation" })} className="gap-2">
          <ArrowLeft className="h-4 w-4" />
          Back to Evaluation
        </Button>
        <div className="ml-auto" />
        <Button onClick={() => navigate({ to: "/development" })} className="gap-2">
          Exit to Workspace
          <ArrowRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
