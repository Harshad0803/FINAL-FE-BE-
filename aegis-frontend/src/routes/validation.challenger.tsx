import React from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
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
  GitCompareArrows,
  ListChecks,
  Clock,
} from "lucide-react";
import PlotlyChart from "@/components/plotly-chart";
import { formUpload, ApiError } from "@/lib/api";
import { useDataset } from "@/lib/app-context";
import { useResumeState } from "@/hooks/use-resume-state";
import { StageHero, HeroChip, VCard, VEmptyState, KpiStrip } from "@/components/validation-ui";

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

const MODEL_OPTIONS = [
  "Logistic Regression",
  "Random Forest",
  "XGBoost",
  "LightGBM",
  "Gradient Boosting",
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

// Reference (developer-reported) vs Replicated (independently re-trained)
// comparison tile — the actual pass/fail verdict for this metric still comes
// only from the real R4.x check below; this tile shows the real delta
// between the two real numbers, nothing more.
function MetricComparisonTile({ label, replicated, reported }: { label: string; replicated: number | null; reported: number | null }) {
  const delta = replicated !== null && reported !== null ? Math.abs(replicated - reported) : null;
  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
      <div className="h-[3px] bg-gradient-to-r from-blue-500 to-indigo-500" />
      <div className="px-4 py-3.5">
        <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">{label}</div>
        <div className="mt-2 font-mono text-2xl font-bold leading-none text-slate-900">
          {replicated !== null ? replicated.toFixed(3) : "—"}
        </div>
        {reported !== null ? (
          <div className="mt-2 text-[11px] font-semibold text-slate-500">
            Δ {delta !== null ? delta.toFixed(3) : "—"} <span className="font-normal text-slate-400">vs reported {reported.toFixed(3)}</span>
          </div>
        ) : (
          <div className="mt-2 text-[11px] text-slate-400">No reported value</div>
        )}
      </div>
    </div>
  );
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
  // `modelIdentity` is the business/model name; `algorithm` is the technical framework.
  const [modelIdentity, setModelIdentity] = React.useState("");
  const [algorithm, setAlgorithm] = React.useState("XGBoost");
  const [testSize, setTestSize] = React.useState(0.15);
  const [valSize, setValSize] = React.useState(0.15);
  const [reported, setReported] = React.useState<Record<string, string>>({});
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

  // Resume where the reviewer left off: if this session has no replication
  // result yet, pull the last saved /validation/replication run from the
  // backend (this page maps to the "replication" stage since it's the one
  // that actually calls POST /validation/replication).
  const { data: resumedReplication } = useResumeState<ReplicationResponse>(
    "validation_pipeline_log.csv",
    "replication",
  );
  React.useEffect(() => {
    if (!replication && resumedReplication?.report?.replication) {
      setReplication(resumedReplication.report.replication);
      setFlags(resumedReplication.flags ?? []);
      setPerformanceReport(resumedReplication.report);
      ds.setValidationStage4Result({
        replication: resumedReplication.report.replication,
        flags: resumedReplication.flags ?? [],
        performanceReport: resumedReplication.report,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumedReplication]);

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
    return algorithm.trim();
  }, [algorithm]);
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

  // Prefill from whatever the earlier stages already put in context, once.
  const prefilledRef = React.useRef(false);
  React.useEffect(() => {
    if (prefilledRef.current) return;
    const hasContextModel = Boolean(selectedModelName || trainingResult?.model_name || validationIntakeData?.model_name || profile);
    if (!hasContextModel) return;
    prefilledRef.current = true;

    if (targetCandidates[0]) setTargetCol(targetCandidates[0]);

    // Prefill the model identity (business name) if present in prior stages.
    const contextIdentity = [selectedModelName, trainingResult?.model_name, validationIntakeData?.model_name]
      .filter((value): value is string => Boolean(value))
      .map((value) => value.trim())
      .find(Boolean);
    if (contextIdentity) setModelIdentity(contextIdentity);

    // Prefill algorithm if any context value matches a known algorithm option.
    const contextAlg = [selectedModelName, trainingResult?.model_name, validationIntakeData?.model_name]
      .filter((value): value is string => Boolean(value))
      .map((value) => value.trim())
      .find((value) => MODEL_OPTIONS.includes(value));
    if (contextAlg) setAlgorithm(contextAlg);

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
    if (!modelIdentity.trim()) {
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
      form.append("model_name", modelIdentity.trim());
      form.append("algorithm", algorithmName);
      if (validationIntakeData) {
        form.append("intake_json", JSON.stringify(validationIntakeData));
      }
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

  // Real reported values, parsed to numbers where the reviewer actually
  // entered/extracted one — used only for the informational Reference vs
  // Replicated delta tiles, never for a derived pass/fail verdict (that
  // comes solely from the real R4.x checks below).
  const reportedNumeric = React.useMemo(() => {
    const out: Record<string, number | null> = {};
    for (const { key } of REPORTED_METRIC_FIELDS) {
      const raw = reported[key];
      const num = raw !== undefined && raw !== "" ? Number(raw) : NaN;
      out[key] = Number.isFinite(num) ? num : null;
    }
    return out;
  }, [reported]);

  const checkCounts = React.useMemo(() => {
    const checks = replication?.checks ?? [];
    return {
      total: checks.length,
      pass: checks.filter((c) => c.status === "PASS").length,
      fail: checks.filter((c) => c.status === "FAIL").length,
      warn: checks.filter((c) => c.status === "WARN").length,
    };
  }, [replication]);

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
    <VCard
      icon={GitCompareArrows}
      title="Model Replication"
      sub="Independently re-train the submitted model on the validation dataset and run checks R4.1–R4.8 against developer-reported metrics."
    >
      {/* Dataset source */}
      <div className="rounded-lg border border-slate-200 bg-slate-50/60 p-4">
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
            <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-slate-300 px-3 py-2 text-sm text-slate-500 hover:border-blue-400">
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
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
          />
          <datalist id="replication-target-candidates">
            {[...targetCandidates, ...allColumns].map((c) => (
              <option key={c} value={c} />
            ))}
          </datalist>
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">Model name (business identity)</label>
          <input
            value={modelIdentity}
            onChange={(e) => setModelIdentity(e.target.value)}
            placeholder="e.g. Credit Risk PD Model"
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
          />
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">Algorithm / Framework</label>
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
          >
            {MODEL_OPTIONS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
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
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
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
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
          />
        </div>
      </div>

      {/* Reported metrics + MDD */}
      <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50/60 p-4">
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
                className="mt-1 w-full rounded-md border border-slate-200 bg-white px-2 py-1.5 text-sm"
              />
            </div>
          ))}
        </div>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          <XCircle className="h-4 w-4 shrink-0" /> {error}
        </div>
      )}

      <div className="mt-4">
        <button
          onClick={runReplication}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-lg bg-gradient-to-br from-blue-700 via-blue-600 to-indigo-500 px-4 py-2 text-sm font-semibold text-white shadow-[0_6px_20px_rgba(37,99,235,0.35)] transition-opacity hover:opacity-90 disabled:opacity-60"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <PlayCircle className="h-4 w-4" />}
          {loading ? "Running replication…" : "Run replication"}
        </button>
      </div>

      {/* Purposeful empty state before the first run, instead of a blank gap */}
      {!replication && !loading && (
        <div className="mt-6">
          <VEmptyState
            icon={GitCompareArrows}
            title="Replication has not been run yet"
            description="Configure the dataset, target column, and model above, then run replication to see the R4.1–R4.8 checks and performance profile."
          />
        </div>
      )}

      {/* Results */}
      {replication && (
        <div className="mt-6 space-y-6">
          {replication.result.success ? (
            <>
              <KpiStrip
                tiles={[
                  {
                    icon: checkCounts.fail === 0 ? CheckCircle2 : XCircle,
                    label: "Replication Status",
                    value: checkCounts.fail === 0 ? "Passed" : "Failed",
                    sub: checkCounts.fail === 0 ? "All checks satisfied" : `${checkCounts.fail} check${checkCounts.fail === 1 ? "" : "s"} failing`,
                    tone: checkCounts.fail === 0 ? "emerald" : "rose",
                  },
                  {
                    icon: ListChecks, label: "Checks Passed", value: `${checkCounts.pass}/${checkCounts.total}`,
                    sub: "R4.1–R4.8 automated checks", tone: checkCounts.pass === checkCounts.total ? "emerald" : "amber",
                  },
                  {
                    icon: AlertTriangle, label: "Failing Checks", value: flags.length,
                    sub: flags.length ? flags.join(", ") : "None", tone: flags.length ? "rose" : "emerald",
                  },
                  {
                    icon: Clock, label: "Elapsed Time", value: `${replication.result.timing_s}s`,
                    sub: "Independent re-training run", tone: "slate",
                  },
                ]}
              />

              <Tabs value={activeSubTab} onValueChange={setActiveSubTab} className="w-full">
                <TabsList>
                  <TabsTrigger value="replication">Replication Checks</TabsTrigger>
                  <TabsTrigger value="performance">Performance</TabsTrigger>
                </TabsList>

                <TabsContent value="replication" className="space-y-6 pt-4">
                  {/* Reference (developer-reported) vs Replicated — the central
                      question this stage answers, made explicit rather than a
                      flat list of only the replicated numbers. */}
                  <div>
                    <div className="mb-3 flex items-center justify-between">
                      <div>
                        <h4 className="text-sm font-bold text-slate-900">Reference vs Replicated</h4>
                        <p className="text-xs text-slate-400">Developer-reported metrics vs this independent re-training run</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      {["roc_auc", "gini", "ks", "accuracy", "precision", "recall", "f1", "cv_mean_auc"].map((k) => {
                        const rawValue = k === "cv_mean_auc" ? replication.result.cv_mean_auc : metrics[k];
                        const replicatedValue = typeof rawValue === "number" ? rawValue : null;
                        return (
                          <MetricComparisonTile
                            key={k}
                            label={k.replace(/_/g, " ")}
                            replicated={replicatedValue}
                            reported={reportedNumeric[k] ?? null}
                          />
                        );
                      })}
                    </div>
                  </div>

                  {/* Checks table */}
                  <div className="overflow-hidden rounded-xl border border-slate-200">
                    <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50 px-4 py-3">
                      <div>
                        <div className="text-sm font-bold text-slate-900">Replication Check Results</div>
                        <div className="text-xs text-slate-400">R4.1 – R4.8 automated validation checks</div>
                      </div>
                      <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-bold text-slate-600">
                        {checkCounts.pass}/{checkCounts.total} passed
                      </span>
                    </div>
                    <table className="w-full text-sm">
                      <thead className="bg-slate-50 text-[10.5px] font-bold uppercase tracking-wider text-slate-400">
                        <tr>
                          <th className="px-4 py-2 text-left">#</th>
                          <th className="px-4 py-2 text-left">ID</th>
                          <th className="px-4 py-2 text-left">Check</th>
                          <th className="px-4 py-2 text-left">Observed</th>
                          <th className="px-4 py-2 text-left">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {replication.checks.map((c, rowIndex) => (
                          <tr key={c.id} className={c.status === "FAIL" ? "bg-red-50/40" : undefined}>
                            <td className="px-4 py-2 font-mono text-xs text-slate-400">{rowIndex + 1}</td>
                            <td className="px-4 py-2 font-mono text-xs text-slate-400">{c.id}</td>
                            <td className="px-4 py-2 font-medium text-slate-900">{c.title}</td>
                            <td className="px-4 py-2 text-xs text-slate-600">{c.observed ?? "—"}</td>
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
                      <div key={m.label} className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
                        <div className="h-[3px] bg-gradient-to-r from-blue-500 to-indigo-500" />
                        <div className="px-4 py-3.5">
                          <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">{m.label}</div>
                          <div className="mt-2 font-mono text-2xl font-bold leading-none text-slate-900">{m.value}</div>
                        </div>
                      </div>
                    ))}
                    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
                      <div className={`h-[3px] ${gap?.status === "FAIL" ? "bg-gradient-to-r from-red-400 to-red-500" : "bg-gradient-to-r from-blue-500 to-indigo-500"}`} />
                      <div className="px-4 py-3.5">
                        <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">Train/Test AUC Gap</div>
                        <div className="mt-2 font-mono text-2xl font-bold leading-none text-slate-900">{gap ? formatValue(gap.gap, 3) : "—"}</div>
                        <div className="mt-2 text-[11px] font-semibold text-slate-500">Status: {gap?.status ?? "—"}</div>
                      </div>
                    </div>
                  </section>

                  <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                      <h3 className="text-sm font-semibold">ROC curve</h3>
                      <p className="text-xs text-muted-foreground">AUC {formatValue(performanceReport?.roc_curve?.auc, 3)}</p>
                      <div className="mt-4 h-56">
                        <PlotlyChart figure={rocFigure} style={{ height: "100%" }} />
                      </div>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                      <h3 className="text-sm font-semibold">Precision–Recall</h3>
                      <p className="text-xs text-muted-foreground">
                        Average precision {formatValue(performanceReport?.pr_curve?.average_precision, 3)}
                      </p>
                      <div className="mt-4 h-56">
                        <PlotlyChart figure={prFigure} style={{ height: "100%" }} />
                      </div>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
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
                            {confusionMatrix?.matrix?.length === 2
                              ? confusionMatrix.matrix.flatMap((row, rowIndex) =>
                                  row.map((value, colIndex) => {
                                    // Standard binary quadrant labels — row 0 = Actual
                                    // negative, col 0 = Predicted negative, matching
                                    // the real backend's labels/matrix ordering.
                                    const quadrant =
                                      rowIndex === 0 && colIndex === 0 ? { label: "TN", classes: "border-blue-200 bg-blue-50 text-blue-800" }
                                      : rowIndex === 0 && colIndex === 1 ? { label: "FP", classes: "border-red-200 bg-red-50 text-red-700" }
                                      : rowIndex === 1 && colIndex === 0 ? { label: "FN", classes: "border-amber-200 bg-amber-50 text-amber-700" }
                                      : { label: "TP", classes: "border-emerald-200 bg-emerald-50 text-emerald-800" };
                                    return (
                                      <div key={`${rowIndex}-${colIndex}`} className={`flex flex-col justify-between rounded-xl border p-4 ${quadrant.classes}`}>
                                        <span className="text-[10px] font-bold uppercase tracking-wider opacity-70">{quadrant.label}</span>
                                        <span className="font-mono text-2xl font-bold tabular-nums">{value.toLocaleString()}</span>
                                      </div>
                                    );
                                  }),
                                )
                              : confusionMatrix?.matrix?.length
                              ? confusionMatrix.matrix.flatMap((row, rowIndex) =>
                                  row.map((value, colIndex) => {
                                    const label = confusionMatrix.labels?.[colIndex] ?? colIndex;
                                    const tone = rowIndex === colIndex ? "primary" : "destructive";
                                    return (
                                      <div
                                        key={`${rowIndex}-${colIndex}`}
                                        className={
                                          "flex flex-col justify-between rounded-xl border p-4 " +
                                          (tone === "primary" ? "border-primary/30 bg-primary-soft" : "border-destructive/30 bg-destructive/10")
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

                    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                      <h3 className="text-sm font-semibold">Calibration</h3>
                      <p className="text-xs text-muted-foreground">Predicted vs observed default rate</p>
                      <div className="mt-4 h-56">
                        <PlotlyChart figure={calibrationFigure} style={{ height: "100%" }} />
                      </div>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
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
    </VCard>
  );
}

function Challenger() {
  // Tracks which sub-tab is active so the bottom button can tell whether
  // the reviewer is still on the first sub-tab (Replication Checks —
  // button just advances to Performance) or the last one (button
  // navigates to Stage 4).
  const [activeSubTab, setActiveSubTab] = React.useState<string>("replication");

  return (
    <div className="space-y-6">
      <StageHero
        eyebrow="STAGE 3 · MODEL VALIDATION"
        title="Model Replication & Performance Testing"
        description="Independently reproduce the developer's submitted model, verify results against the R4.1-R4.8 replication checks, and review its full performance profile on data the model has never seen."
        chips={
          <>
            <HeroChip tone={activeSubTab === "replication" ? "success" : "neutral"}>Replication</HeroChip>
            <HeroChip tone={activeSubTab === "performance" ? "success" : "neutral"}>Performance review</HeroChip>
          </>
        }
      />

      <ModelReplicationPanel activeSubTab={activeSubTab} setActiveSubTab={setActiveSubTab} />

      <div className="text-right">
        {activeSubTab === "replication" ? (
          <button
            type="button"
            onClick={() => setActiveSubTab("performance")}
            className="inline-flex items-center gap-2 rounded-lg bg-[#2f67ff] px-4 py-2 text-sm font-semibold text-white shadow-[0_4px_10px_rgba(47,103,255,0.18)] hover:bg-[#285ee6]"
          >
            Continue
            <ArrowRight className="h-4 w-4" />
          </button>
        ) : (
        <Link
          to="/validation/performance"
          className="inline-flex items-center gap-2 rounded-lg bg-[#2f67ff] px-4 py-2 text-sm font-semibold text-white shadow-[0_4px_10px_rgba(47,103,255,0.18)] hover:bg-[#285ee6]"
        >
          Continue to Stage 4 — Benchmarking
          <ArrowRight className="h-4 w-4" />
        </Link>
        )}
      </div>
    </div>
  );
}
