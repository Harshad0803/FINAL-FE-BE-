import React from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { PageHeader } from "@/components/app-shell";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  ArrowRight,
  PlayCircle,
  Loader2,
  UploadCloud,
  FileUp,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  MinusCircle,
  Database,
} from "lucide-react";
import PlotlyChart from "@/components/plotly-chart";
import { formUpload, ApiError } from "@/lib/api";
import { useDataset } from "@/lib/app-context";

export const Route = createFileRoute("/validation/challenger")({
  head: () => ({ meta: [{ title: "Model Replication & Performance — Aegis Credit" }] }),
  component: Challenger,
});

// --- Model Replication + Performance Testing (Stage 3) — real backend-connected panel ---
// Combines what used to be two separate pages: Stage 3 (Model Replication —
// R4.1-R4.8 checks, seed stability, feature ablation) and Stage 4's
// "Performance" tab (metrics, ROC/PR curves, confusion matrix, calibration,
// score distribution). Stage 4 (/validation/performance) is now
// benchmarking-only, since both pages already fit the model the same way
// under the hood (run_replication) — this just stops making the reviewer
// do it twice.

type CheckStatus = "PASS" | "WARN" | "FAIL" | "SKIP";

type ReplicationCheck = {
  id: string;
  title: string;
  severity: string;
  status: CheckStatus;
  observed?: string;
  threshold?: string;
  detail?: string;
  _table?: Array<Record<string, any>>;
  _ablation?: Record<string, number>;
  _seed_aucs?: number[];
  _seeds?: number[];
};

type ReplicationResult = {
  success: boolean;
  error?: string | null;
  metrics: Record<string, number>;
  seed_aucs: number[];
  cv_mean_auc?: number | null;
  cv_std_auc?: number | null;
  split_stats: Record<string, number>;
  ablation: Record<string, number>;
  timing_s: number;
};

type ReplicationResponse = {
  stage: string;
  flags: string[];
  report: {
    replication: { result: ReplicationResult; checks: ReplicationCheck[] };
    metrics: Record<string, any>;
    roc_curve: { points: Array<Record<string, number>>; auc?: number | null };
    pr_curve: { points: Array<Record<string, number>>; average_precision?: number | null };
    confusion_matrix: { labels: Array<number | string>; matrix: number[][] };
    score_distribution: { bins: Array<Record<string, any>> };
    calibration_chart: { points: Array<Record<string, any>> };
    train_test_auc_gap: { gap?: number | null; status?: string | null; cv_mean_auc?: number | null; test_auc?: number | null };
    threshold_selection?: { threshold: number; metric: string; f1?: number; precision?: number; recall?: number } | null;
    metric_checks?: Array<Record<string, any>>;
    compliance_findings?: Array<Record<string, any>>;
    threshold_analysis?: Array<Record<string, any>>;
  };
};

const metricDefinitions = [
  { key: "roc_auc", label: "ROC-AUC", digits: 3 },
  { key: "gini", label: "Gini", digits: 3 },
  { key: "ks", label: "KS", digits: 3 },
  { key: "accuracy", label: "Accuracy", digits: 3 },
  { key: "precision", label: "Precision", digits: 3 },
  { key: "recall", label: "Recall", digits: 3 },
  { key: "f1", label: "F1 Score", digits: 3 },
  { key: "brier_score", label: "Brier", digits: 3 },
  { key: "pr_auc", label: "PR-AUC", digits: 3 },
];

function formatValue(value: unknown, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return num.toFixed(digits);
}

const statusStyles: Record<CheckStatus, string> = {
  PASS: "bg-primary-soft text-foreground border-primary/30",
  WARN: "bg-warning/20 text-warning-foreground border-warning/40",
  FAIL: "bg-destructive/10 text-destructive border-destructive/30",
  SKIP: "bg-background text-muted-foreground border-border",
};

function StatusIcon({ s }: { s: CheckStatus }) {
  if (s === "PASS") return <CheckCircle2 className="h-3.5 w-3.5" />;
  if (s === "WARN") return <AlertTriangle className="h-3.5 w-3.5" />;
  if (s === "FAIL") return <XCircle className="h-3.5 w-3.5" />;
  return <MinusCircle className="h-3.5 w-3.5" />;
}

const REPORTED_METRIC_FIELDS: Array<{ key: string; label: string }> = [
  { key: "roc_auc", label: "ROC-AUC" },
  { key: "gini", label: "Gini" },
  { key: "ks", label: "KS" },
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
  { key: "cv_mean_auc", label: "CV Mean AUC" },
];

function ModelReplicationPanel({
  activeSubTab,
  setActiveSubTab,
}: {
  activeSubTab: string;
  setActiveSubTab: (tab: string) => void;
}) {
  const ds = useDataset();

  const [localFile, setLocalFile] = React.useState<File | null>(null);
  const [mddFile, setMddFile] = React.useState<File | null>(null);
  const [targetCol, setTargetCol] = React.useState("");
  const [modelName, setModelName] = React.useState("");
  const [testSize, setTestSize] = React.useState(0.15);
  const [valSize, setValSize] = React.useState(0.15);
  const [reported, setReported] = React.useState<Record<string, string>>({});
  const [availableModels, setAvailableModels] = React.useState<string[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  // Seed from shared context so returning to this page (e.g. via Back from
  // Stage 4 — Benchmarking) shows the already-computed R4.1-R4.8 checks,
  // model ranking, and performance report instead of forcing a full rerun —
  // this previously lived only in local state and was lost on every remount.
  const [replication, setReplication] = React.useState<{ result: ReplicationResult; checks: ReplicationCheck[] } | null>(
    (ds.validationStage4Result?.replication as { result: ReplicationResult; checks: ReplicationCheck[] } | null) ?? null,
  );
  const [flags, setFlags] = React.useState<string[]>((ds.validationStage4Result?.flags as string[] | null) ?? []);
  const [performanceReport, setPerformanceReport] = React.useState<ReplicationResponse["report"] | null>(
    (ds.validationStage4Result?.performanceReport as ReplicationResponse["report"] | null) ?? null,
  );

  // profile / trainingResult shapes aren't strictly typed on the context
  // (Record<string, any>), so field access below is defensive with fallbacks.
  const profile = ds.profile as Record<string, any> | null | undefined;
  const trainingResult = ds.trainingResult as Record<string, any> | null | undefined;
  const trainingConfig = ds.trainingConfig as Record<string, any> | null | undefined;
  const validationMddMetrics = ds.validationMddMetrics as Record<string, any> | null | undefined;
  const validationIntakeData = ds.validationIntakeData as Record<string, any> | null | undefined;
  const selectedModelName = ds.selectedModel?.name as string | undefined;

  const targetCandidates: string[] = React.useMemo(() => {
    const c = profile?.target_candidates ?? profile?.targetCandidates ?? profile?.candidate_targets ?? [];
    return Array.isArray(c) ? c.filter((x) => typeof x === "string") : [];
  }, [profile]);

  const allColumns: string[] = React.useMemo(() => {
    const c = profile?.columns ?? profile?.column_names ?? profile?.all_columns ?? [];
    return Array.isArray(c) ? c.filter((x) => typeof x === "string") : [];
  }, [profile]);

  const datasetName: string | null =
    profile?.dataset_name ?? profile?.name ?? ds.file?.name ?? null;

  const contextFileAvailable = Boolean(ds.file || profile?.csv_text || profile?.dataset_name || profile?.name);
  const resolvedAlgorithmName = React.useMemo(() => {
    const fromIntake = typeof validationIntakeData?.algorithm === "string" ? validationIntakeData.algorithm.trim() : "";
    if (fromIntake) return fromIntake;

    const fromSelected = typeof selectedModelName === "string" ? selectedModelName.trim() : "";
    if (fromSelected) return fromSelected;

    const fromTraining = typeof trainingResult?.model_name === "string" ? trainingResult.model_name.trim() : "";
    if (fromTraining) return fromTraining;

    return modelName.trim();
  }, [modelName, selectedModelName, trainingResult?.model_name, validationIntakeData?.algorithm]);
  const activeFile = React.useMemo<File | null>(() => {
    if (localFile) return localFile;
    if (ds.file) return ds.file;

    const csvText = typeof profile?.csv_text === "string" ? profile.csv_text : "";
    if (!csvText.trim()) return null;

    const resolvedName = datasetName ?? "validation_dataset.csv";
    const safeName = resolvedName.endsWith(".csv") || resolvedName.endsWith(".xlsx")
      ? resolvedName
      : `${resolvedName}.csv`;
    return new File([csvText], safeName, { type: "text/csv" });
  }, [datasetName, ds.file, localFile, profile?.csv_text]);

  // Fetch the classification model registry once for the datalist.
  React.useEffect(() => {
    const contextModels = [selectedModelName, trainingResult?.model_name, validationIntakeData?.model_name]
      .filter((value): value is string => Boolean(value))
      .map((value) => String(value));

    if (contextModels.length > 0) {
      setAvailableModels((prev) => Array.from(new Set([...prev, ...contextModels])));
    }
  }, [selectedModelName, trainingResult?.model_name, validationIntakeData?.model_name]);

  // Prefill from whatever the earlier stages already put in context, once.
  const prefilledRef = React.useRef(false);
  React.useEffect(() => {
    if (prefilledRef.current) return;
    const hasContextModel = Boolean(selectedModelName || trainingResult?.model_name || validationIntakeData?.model_name || profile);
    if (!hasContextModel) return;
    prefilledRef.current = true;

    if (targetCandidates[0]) setTargetCol(targetCandidates[0]);

    const contextModelName = selectedModelName ?? trainingResult?.model_name ?? validationIntakeData?.model_name;
    if (contextModelName) {
      setModelName(String(contextModelName));
    }

    if (trainingConfig) {
      if (typeof trainingConfig.test_size === "number") setTestSize(trainingConfig.test_size);
      if (typeof trainingConfig.val_size === "number") setValSize(trainingConfig.val_size);
    }

    const sourceMetrics = validationMddMetrics ?? (trainingResult?.evaluation_metrics as Record<string, any> | undefined);
    if (sourceMetrics) {
      setReported((prev) => {
        const next = { ...prev };
        for (const { key } of REPORTED_METRIC_FIELDS) {
          const v = sourceMetrics[key] ?? (key === "cv_mean_auc" ? sourceMetrics.cv_mean : undefined);
          if (v !== undefined && v !== null) next[key] = String(v);
        }
        return next;
      });
    }
  }, [profile, trainingResult, trainingConfig, validationMddMetrics, validationIntakeData?.model_name, selectedModelName, targetCandidates]);

  const runReplication = async () => {
    setError(null);

    if (!activeFile) {
      setError("No active dataset is available in shared state. Complete Stage 1 Intake and Stage 2 Data Validation first, or upload a file below.");
      return;
    }
    if (!targetCol.trim()) {
      setError("Target column is required.");
      return;
    }
    if (!modelName.trim()) {
      setError("Model is required.");
      return;
    }

    const algorithmName = resolvedAlgorithmName.trim();
    if (!algorithmName) {
      setError("Algorithm is required.");
      return;
    }

    setLoading(true);
    setReplication(null);
    setFlags([]);
    setPerformanceReport(null);
    try {
      const form = new FormData();
      if (activeFile) {
        form.append("file", activeFile);
      } else if (typeof profile?.csv_text === "string") {
        form.append("csv_text", profile.csv_text);
      }
      form.append("model_name", modelName.trim());
      form.append("algorithm", algorithmName);
      form.append("target_col", targetCol.trim());
      form.append("test_size", String(testSize));
      form.append("val_size", String(valSize));
      if (mddFile) form.append("mdd_file", mddFile);

      const reportedPayload = Object.fromEntries(
        Object.entries(reported)
          .filter(([, v]) => v !== "" && v !== undefined && v !== null)
          .map(([k, v]) => [k, Number(v)])
          .filter(([, v]) => !Number.isNaN(v as number)),
      );
      if (Object.keys(reportedPayload).length > 0) {
        form.append("reported_json", JSON.stringify(reportedPayload));
      }

      const res = await formUpload<ReplicationResponse>("/validation/replication", form);
      setReplication(res.report.replication);
      setFlags(res.flags ?? []);
      setPerformanceReport(res.report);
      ds.setValidationStage4Result({
        replication: res.report.replication,
        flags: res.flags ?? [],
        performanceReport: res.report,
      });
    } catch (err) {
      if (err instanceof ApiError) {
        const detail =
          err.body && typeof err.body === "object" && "detail" in (err.body as any)
            ? String((err.body as any).detail)
            : err.message;
        setError(detail);
      } else {
        setError(err instanceof Error ? err.message : "Replication run failed.");
      }
    } finally {
      setLoading(false);
    }
  };

  const seedChartData = React.useMemo(() => {
    if (!replication?.result?.seed_aucs?.length) return [];
    return replication.result.seed_aucs.map((auc, i) => ({
      seed: `#${i + 1}`,
      auc,
    }));
  }, [replication]);

  const ablationChartData = React.useMemo(() => {
    const abl = replication?.result?.ablation;
    if (!abl) return [];
    return Object.entries(abl)
      .filter(([, v]) => typeof v === "number" && !Number.isNaN(v))
      .sort((a, b) => (b[1] as number) - (a[1] as number))
      .slice(0, 10)
      .map(([feature, drop]) => ({ feature, drop: Number((drop as number).toFixed(4)) }));
  }, [replication]);

  const seedFigure = React.useMemo(() => {
    if (!seedChartData.length) return null;
    return {
      data: [
        {
          type: "bar",
          x: seedChartData.map((d) => d.seed),
          y: seedChartData.map((d) => d.auc),
          marker: { color: "oklch(0.55 0.02 240)" },
          hovertemplate: "%{y:.4f}<extra></extra>",
          name: "AUC",
        },
      ],
      layout: {
        margin: { l: 40, r: 20, t: 20, b: 40 },
        xaxis: { tickfont: { size: 11 }, automargin: true },
        yaxis: { title: { text: "AUC" }, tickfont: { size: 11 }, range: [0, 1] },
        height: 224,
      },
    };
  }, [seedChartData]);

  const ablationFigure = React.useMemo(() => {
    if (!ablationChartData.length) return null;
    return {
      data: [
        {
          type: "bar",
          orientation: "h",
          x: ablationChartData.map((d) => d.drop),
          y: ablationChartData.map((d) => d.feature),
          marker: { color: "oklch(0.76 0.18 130)" },
          hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
          name: "AUC drop",
        },
      ],
      layout: {
        margin: { l: 140, r: 20, t: 20, b: 40 },
        xaxis: { tickfont: { size: 11 }, automargin: true },
        yaxis: { tickfont: { size: 11 }, automargin: true, autorange: "reversed" },
        height: 224,
      },
    };
  }, [ablationChartData]);

  const metrics = replication?.result?.metrics ?? {};

  // --- Performance tab data (ported from the old Stage 4 "Performance" tab) ---
  const metricCards = React.useMemo(() => {
    const m = performanceReport?.metrics ?? {};
    return metricDefinitions
      .map((item) => ({ label: item.label, value: formatValue(m[item.key], item.digits) }))
      .filter((item) => item.value !== "—");
  }, [performanceReport]);

  const gap = performanceReport?.train_test_auc_gap;
  const rocPoints = performanceReport?.roc_curve?.points ?? [];
  const prPoints = performanceReport?.pr_curve?.points ?? [];
  const scoreBins = performanceReport?.score_distribution?.bins ?? [];
  const calibrationPoints = performanceReport?.calibration_chart?.points ?? [];
  const confusionMatrix = performanceReport?.confusion_matrix;
  const thresholdSelection = performanceReport?.threshold_selection;

  const rocFigure = React.useMemo(() => {
    const fpr = rocPoints.map((point) => point.fpr);
    const tpr = rocPoints.map((point) => point.tpr);
    const diagonal = rocPoints.map((point) => point.fpr);
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: fpr,
          y: tpr,
          line: { color: "oklch(0.6 0.18 135)", width: 2.5 },
          hovertemplate: "TPR %{y:.3f}<br>FPR %{x:.3f}<extra></extra>",
          name: "ROC",
        },
        {
          type: "scatter",
          mode: "lines",
          x: diagonal,
          y: diagonal,
          line: { color: "oklch(0.6 0.01 240)", dash: "dash" },
          hoverinfo: "skip",
          showlegend: false,
          name: "Diagonal",
        },
      ],
      layout: {
        margin: { l: 40, r: 20, t: 25, b: 40 },
        xaxis: { title: "FPR", tickfont: { size: 11 }, showline: false },
        yaxis: { title: "TPR", tickfont: { size: 11 }, showline: false },
        height: 320,
      },
    };
  }, [rocPoints]);

  const prFigure = React.useMemo(() => {
    const recall = prPoints.map((point) => point.recall);
    const precision = prPoints.map((point) => point.precision);
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: recall,
          y: precision,
          line: { color: "oklch(0.6 0.18 135)", width: 2.5 },
          hovertemplate: "Precision %{y:.3f}<br>Recall %{x:.3f}<extra></extra>",
          name: "PR",
        },
      ],
      layout: {
        margin: { l: 40, r: 20, t: 25, b: 40 },
        xaxis: { title: "Recall", tickfont: { size: 11 }, showline: false },
        yaxis: { title: "Precision", tickfont: { size: 11 }, showline: false },
        height: 320,
      },
    };
  }, [prPoints]);

  const calibrationFigure = React.useMemo(() => {
    const pred = calibrationPoints.map((point) => point.predicted_rate);
    const actual = calibrationPoints.map((point) => point.actual_rate);
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: pred,
          y: actual,
          line: { color: "oklch(0.6 0.18 135)", width: 2.5 },
          hovertemplate: "Actual %{y:.3f}<br>Pred %{x:.3f}<extra></extra>",
          name: "Actual",
        },
        {
          type: "scatter",
          mode: "lines",
          x: pred,
          y: pred,
          line: { color: "oklch(0.6 0.01 240)", dash: "dash" },
          hoverinfo: "skip",
          showlegend: false,
          name: "Perfect",
        },
      ],
      layout: {
        margin: { l: 40, r: 20, t: 25, b: 40 },
        xaxis: { title: "Predicted rate", tickfont: { size: 11 }, showline: false },
        yaxis: { title: "Observed rate", tickfont: { size: 11 }, showline: false },
        height: 320,
      },
    };
  }, [calibrationPoints]);

  const scoreDistributionFigure = React.useMemo(() => {
    const bins = scoreBins.map((bin) => bin.bin);
    const good = scoreBins.map((bin) => bin.good ?? 0);
    const bad = scoreBins.map((bin) => bin.bad ?? 0);
    return {
      data: [
        {
          type: "bar",
          x: bins,
          y: good,
          name: "Good",
          marker: { color: "oklch(0.76 0.18 130)" },
        },
        {
          type: "bar",
          x: bins,
          y: bad,
          name: "Bad",
          marker: { color: "oklch(0.6 0.22 27)" },
        },
      ],
      layout: {
        barmode: "stack",
        margin: { l: 40, r: 20, t: 25, b: 40 },
        xaxis: { title: "Score bin", tickfont: { size: 11 }, showline: false },
        yaxis: { title: "Count", tickfont: { size: 11 }, showline: false },
        height: 320,
      },
    };
  }, [scoreBins]);

  return (
    <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold">Model Replication</h3>
          <p className="mt-1 text-sm text-foreground/80">
            Independently re-train the submitted model on the validation dataset and run checks R4.1–R4.8
            against developer-reported metrics.
          </p>
        </div>
      </div>

      {/* Dataset source */}
      <div className="mt-5 rounded-lg border border-border bg-background p-4">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          <Database className="h-3.5 w-3.5" /> Dataset
        </div>
        {contextFileAvailable && !localFile && (
          <p className="mt-2 text-sm">
            Using <span className="font-medium">{datasetName ?? ds.file?.name}</span> from Intake.{" "}
            <label className="cursor-pointer text-primary underline underline-offset-2">
              Use a different file
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                className="hidden"
                onChange={(e) => setLocalFile(e.target.files?.[0] ?? null)}
              />
            </label>
          </p>
        )}
        {(!contextFileAvailable || localFile) && (
          <div className="mt-2">
            <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-border px-3 py-2 text-sm text-muted-foreground hover:border-primary/40">
              <UploadCloud className="h-4 w-4" />
              {localFile ? localFile.name : "Upload dataset (CSV or XLSX)"}
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                className="hidden"
                onChange={(e) => setLocalFile(e.target.files?.[0] ?? null)}
              />
            </label>
            {contextFileAvailable && localFile && (
              <button
                className="ml-3 text-xs text-primary underline underline-offset-2"
                onClick={() => setLocalFile(null)}
              >
                Revert to Intake file ({ds.file?.name})
              </button>
            )}
          </div>
        )}
      </div>

      {/* Config form */}
      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <label className="text-xs font-medium text-muted-foreground">Target column</label>
          <input
            list="replication-target-candidates"
            value={targetCol}
            onChange={(e) => setTargetCol(e.target.value)}
            placeholder="e.g. default_flag"
            className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
          />
          <datalist id="replication-target-candidates">
            {[...targetCandidates, ...allColumns].map((c) => (
              <option key={c} value={c} />
            ))}
          </datalist>
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">Model</label>
          <input
            list="replication-model-options"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g. xgboost"
            className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
          />
          <datalist id="replication-model-options">
            {availableModels.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">Test size</label>
          <input
            type="number"
            step={0.01}
            min={0.05}
            max={0.4}
            value={testSize}
            onChange={(e) => setTestSize(Number(e.target.value))}
            className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
          />
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">Validation size</label>
          <input
            type="number"
            step={0.01}
            min={0.05}
            max={0.4}
            value={valSize}
            onChange={(e) => setValSize(Number(e.target.value))}
            className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
          />
        </div>
      </div>

      {/* Reported metrics + MDD */}
      <div className="mt-4 rounded-lg border border-border bg-background p-4">
        <div className="flex items-center justify-between gap-3">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Developer-reported metrics (for R4.2 / R4.3 / R4.4 / R4.8)
          </p>
          <label className="inline-flex cursor-pointer items-center gap-2 text-xs text-primary underline underline-offset-2">
            <FileUp className="h-3.5 w-3.5" />
            {mddFile ? mddFile.name : "Upload MDD (PDF/DOCX/TXT) to auto-extract"}
            <input
              type="file"
              accept=".pdf,.docx,.txt"
              className="hidden"
              onChange={(e) => setMddFile(e.target.files?.[0] ?? null)}
            />
          </label>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          Auto-filled from this dataset's training run where available. Edit any value, or leave blank to skip that check.
          Values extracted from an uploaded MDD override these on run.
        </p>
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {REPORTED_METRIC_FIELDS.map(({ key, label }) => (
            <div key={key}>
              <label className="text-[11px] text-muted-foreground">{label}</label>
              <input
                value={reported[key] ?? ""}
                onChange={(e) => setReported((prev) => ({ ...prev, [key]: e.target.value }))}
                placeholder="—"
                className="mt-1 w-full rounded-md border border-border bg-card px-2 py-1.5 text-sm"
              />
            </div>
          ))}
        </div>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <XCircle className="h-4 w-4 shrink-0" /> {error}
        </div>
      )}

      <div className="mt-4">
        <button
          onClick={runReplication}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-lg gradient-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant disabled:opacity-60"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <PlayCircle className="h-4 w-4" />}
          {loading ? "Running replication…" : "Run replication"}
        </button>
      </div>

      {/* Results */}
      {replication && (
        <div className="mt-6 space-y-6">
          {replication.result.success ? (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  {flags.length === 0 ? "No failing checks" : `${flags.length} failing check${flags.length === 1 ? "" : "s"}`}
                </span>
                {flags.map((f) => (
                  <span key={f} className="rounded-full border border-destructive/30 bg-destructive/10 px-2 py-0.5 text-[11px] font-semibold text-destructive">
                    {f}
                  </span>
                ))}
                <span className="ml-auto text-xs text-muted-foreground">
                  Completed in {replication.result.timing_s}s
                </span>
              </div>

              <Tabs value={activeSubTab} onValueChange={setActiveSubTab} className="w-full">
                <TabsList>
                  <TabsTrigger value="replication">Replication Checks</TabsTrigger>
                  <TabsTrigger value="performance">Performance</TabsTrigger>
                </TabsList>

                <TabsContent value="replication" className="space-y-6 pt-4">
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
                    {["roc_auc", "gini", "ks", "accuracy", "precision", "recall", "f1"].map((k) => (
                      <div key={k} className="rounded-lg border border-border bg-background p-3">
                        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{k.replace("_", " ")}</div>
                        <div className="mt-1 text-lg font-semibold tabular-nums">
                          {typeof metrics[k] === "number" ? metrics[k].toFixed(3) : "—"}
                        </div>
                      </div>
                    ))}
                    <div className="rounded-lg border border-border bg-background p-3">
                      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">cv mean auc</div>
                      <div className="mt-1 text-lg font-semibold tabular-nums">
                        {typeof replication.result.cv_mean_auc === "number" ? replication.result.cv_mean_auc.toFixed(3) : "—"}
                      </div>
                    </div>
                  </div>

                  {/* Checks table */}
                  <div className="overflow-x-auto rounded-lg border border-border">
                    <table className="w-full text-sm">
                      <thead className="bg-background text-[10px] uppercase tracking-wider text-muted-foreground">
                        <tr>
                          <th className="px-4 py-2 text-left">#</th>
                          <th className="px-4 py-2 text-left">ID</th>
                          <th className="px-4 py-2 text-left">Check</th>
                          <th className="px-4 py-2 text-left">Observed</th>
                          <th className="px-4 py-2 text-left">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {replication.checks.map((c, rowIndex) => (
                          <tr key={c.id}>
                            <td className="px-4 py-2 font-mono text-xs text-muted-foreground">{rowIndex + 1}</td>
                            <td className="px-4 py-2 font-mono text-xs text-muted-foreground">{c.id}</td>
                            <td className="px-4 py-2 font-medium">{c.title}</td>
                            <td className="px-4 py-2 text-xs text-foreground/80">{c.observed ?? "—"}</td>
                            <td className="px-4 py-2">
                              <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold ${statusStyles[c.status]}`}>
                                <StatusIcon s={c.status} />
                                {c.status}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Seed stability + ablation charts */}
                  <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                    {seedFigure && (
                      <div>
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Seed stability (R4.6)</h4>
                        <div className="mt-2 h-56">
                          <PlotlyChart figure={seedFigure} style={{ height: "100%" }} />
                        </div>
                      </div>
                    )}

                    {ablationFigure && (
                      <div>
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                          Top feature ablation — AUC drop (R4.5)
                        </h4>
                        <div className="mt-2 h-56">
                          <PlotlyChart figure={ablationFigure} style={{ height: "100%" }} />
                        </div>
                      </div>
                    )}
                  </div>
                </TabsContent>

                <TabsContent value="performance" className="space-y-8 pt-4">
                  <section className="grid grid-cols-2 gap-3 md:grid-cols-5">
                    {metricCards.map((m) => (
                      <div key={m.label} className="rounded-xl border border-border bg-background p-4">
                        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{m.label}</div>
                        <div className="mt-2 text-xl font-semibold tracking-tight tabular-nums">{m.value}</div>
                      </div>
                    ))}
                    <div className="rounded-xl border border-border bg-background p-4">
                      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Train/Test AUC Gap</div>
                      <div className="mt-2 text-xl font-semibold tracking-tight tabular-nums">
                        {gap ? formatValue(gap.gap, 3) : "—"}
                      </div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        Status: {gap?.status ?? "—"}
                      </div>
                    </div>
                  </section>

                  <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                    <div className="rounded-xl border border-border bg-background p-6">
                      <h3 className="text-sm font-semibold">ROC curve</h3>
                      <p className="text-xs text-muted-foreground">AUC {formatValue(performanceReport?.roc_curve?.auc, 3)}</p>
                      <div className="mt-4 h-56">
                        <PlotlyChart figure={rocFigure} style={{ height: "100%" }} />
                      </div>
                    </div>

                    <div className="rounded-xl border border-border bg-background p-6">
                      <h3 className="text-sm font-semibold">Precision–Recall</h3>
                      <p className="text-xs text-muted-foreground">
                        Average precision {formatValue(performanceReport?.pr_curve?.average_precision, 3)}
                      </p>
                      <div className="mt-4 h-56">
                        <PlotlyChart figure={prFigure} style={{ height: "100%" }} />
                      </div>
                    </div>

                    <div className="rounded-xl border border-border bg-background p-6">
                      <h3 className="text-sm font-semibold">Confusion matrix</h3>
                      <p className="text-xs text-muted-foreground">
                        {thresholdSelection
                          ? `Threshold ${formatValue(thresholdSelection.threshold, 2)} (auto-calibrated for max F1)`
                          : "Threshold —"}
                      </p>
                      <div className="mt-4 flex h-56 flex-col">
                        <div className="mb-1 text-center text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                          Predicted
                        </div>
                        <div className="flex flex-1 items-stretch gap-2">
                          <div className="flex items-center">
                            <span className="w-4 origin-center -rotate-90 whitespace-nowrap text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                              Actual
                            </span>
                          </div>
                          <div className="grid flex-1 grid-cols-2 gap-3">
                            {confusionMatrix?.matrix?.length
                              ? confusionMatrix.matrix.flatMap((row, rowIndex) =>
                                  row.map((value, colIndex) => {
                                    const label = confusionMatrix.labels?.[colIndex] ?? colIndex;
                                    const tone = rowIndex === 1 && colIndex === 1 ? "primary" : rowIndex === 0 && colIndex === 1 ? "warning" : "destructive";
                                    return (
                                      <div
                                        key={`${rowIndex}-${colIndex}`}
                                        className={
                                          "flex flex-col justify-between rounded-xl border p-4 " +
                                          (tone === "primary"
                                            ? "border-primary/30 bg-primary-soft"
                                            : tone === "warning"
                                              ? "border-warning/40 bg-warning/15"
                                              : "border-destructive/30 bg-destructive/10")
                                        }
                                      >
                                        <span className="text-[11px] uppercase tracking-wider text-muted-foreground">
                                          Predicted {label} · Actual {confusionMatrix.labels?.[rowIndex] ?? rowIndex}
                                        </span>
                                        <span className="text-2xl font-semibold tabular-nums">{value.toLocaleString()}</span>
                                      </div>
                                    );
                                  }),
                                )
                              : null}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-xl border border-border bg-background p-6">
                      <h3 className="text-sm font-semibold">Calibration</h3>
                      <p className="text-xs text-muted-foreground">Predicted vs observed default rate</p>
                      <div className="mt-4 h-56">
                        <PlotlyChart figure={calibrationFigure} style={{ height: "100%" }} />
                      </div>
                    </div>

                    <div className="rounded-xl border border-border bg-background p-6">
                      <h3 className="text-sm font-semibold">Score distribution</h3>
                      <p className="text-xs text-muted-foreground">
                        Hold-out set · KS {formatValue(performanceReport?.metrics?.ks, 3)}
                      </p>
                      <div className="mt-4 h-56">
                        <PlotlyChart figure={scoreDistributionFigure} style={{ height: "100%" }} />
                      </div>
                    </div>
                  </section>
                </TabsContent>
              </Tabs>
            </>
          ) : (
            <div className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              <XCircle className="h-4 w-4 shrink-0" /> Training failed: {replication.result.error ?? "Unknown error"}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function Challenger() {
  // Tracks which sub-tab is active so the bottom button can tell whether
  // the reviewer is still on the first sub-tab (Replication Checks —
  // button just advances to Performance) or the last one (button
  // navigates to Stage 4).
  const [activeSubTab, setActiveSubTab] = React.useState<string>("replication");

  return (
    <div className="space-y-8">
      <PageHeader
        title="Stage 3 — Model Replication & Performance Testing"
        description="Independently reproduce the developer's submitted model, verify results against the R4.1-R4.8 replication checks, and review its full performance profile on data the model has never seen."
      />

      <ModelReplicationPanel activeSubTab={activeSubTab} setActiveSubTab={setActiveSubTab} />

      <div className="text-right">
        {activeSubTab === "replication" ? (
          <button
            type="button"
            onClick={() => setActiveSubTab("performance")}
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90"
          >
            Continue
            <ArrowRight className="h-4 w-4" />
          </button>
        ) : (
        <Link
          to="/validation/performance"
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90"
        >
          Continue to Stage 4 — Benchmarking
          <ArrowRight className="h-4 w-4" />
        </Link>
        )}
      </div>
    </div>
  );
}
