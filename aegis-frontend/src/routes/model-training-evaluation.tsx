import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { PageHeader } from "@/components/app-shell";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  CheckCircle2, Loader2, ArrowLeft, ArrowRight, Zap, BarChart3, AlertCircle, Info, Download,
  Database, Layers, AlertTriangle, Cpu, Activity, Target, SlidersHorizontal, Settings2,
  CalendarClock, ClipboardList, ShieldAlert, Workflow, Gauge,
  TrendingUp, TrendingDown, Percent, Crosshair, FileDown,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Switch } from "@/components/ui/switch";
import { useDataset } from "@/lib/app-context";
import { formUpload } from "@/lib/api";
import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { useAuth } from "@/lib/auth-context";
import PlotlyChart from "@/components/plotly-chart";
import AnimatedNumber from "@/components/animated-number";
import { useResumeState } from "@/hooks/use-resume-state";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/model-training-evaluation")({
  head: () => ({ meta: [{ title: "Model Training & Evaluation — Aegis Credit" }] }),
  component: ModelTrainingEvaluation,
  validateSearch: (search: Record<string, unknown>) => ({
    tab: search.tab === "evaluation" ? "evaluation" : "training",
  }),
});

function ModelTrainingEvaluation() {
  const { trainingResult } = useDataset();
  const search = Route.useSearch();
  const [tab, setTab] = useState<string>(search.tab);

  // Gate: a model must have actually been trained (trainingResult populated
  // in shared context, not just local component state) before Evaluation can
  // open — this persists across remounts, unlike the Training page's own
  // local trainingInfo state.
  const trainingComplete = Boolean(trainingResult);

  return (
    <div className="space-y-8">
      <PageHeader
        title="Model Training & Evaluation"
        description="Select and train a model, then evaluate its performance on held-out data."
      />

      <Tabs value={tab} onValueChange={setTab} className="w-full">
        <TabsList>
          <TabsTrigger value="training">Model Training</TabsTrigger>
          <TabsTrigger
            value="evaluation"
            disabled={!trainingComplete}
            className={!trainingComplete ? "cursor-not-allowed opacity-50" : ""}
          >
            Model Evaluation
          </TabsTrigger>
        </TabsList>

        <TabsContent value="training" className="space-y-8 pt-4">
          <TrainingTab onProceed={() => setTab("evaluation")} />
        </TabsContent>

        <TabsContent value="evaluation" className="space-y-8 pt-4">
          <EvaluationTab onBackToTraining={() => setTab("training")} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Sub-tab 1 — Model Training (moved from training.tsx, unchanged logic)
// ═══════════════════════════════════════════════════════════════════════

interface TrainingConfig {
  test_size: number;
  val_size: number;
  random_seed: number;
  use_cv: boolean;
  cv_folds: number;
  use_hyperopt: boolean;
  use_feature_engineering: boolean;
  manual_params: Record<string, any>;
  use_oot: boolean;
  date_col: string | null;
}

interface ComparisonResult {
  model_name: string;
  roc_auc?: number;
  recall?: number;
  precision?: number;
  f1?: number;
  pr_auc?: number;
  accuracy?: number;
  training_time_s?: number;
  error?: string;
}

// ── Merged in from models.tsx (Model Selection step, now folded into Training) ──
interface ModelRecommendation {
  name: string;
  description: string;
  icon?: string;
  reasons?: string[]; // only populated for the backend-recommended model
}

interface ModelCard extends ModelRecommendation {
  selected?: boolean;
}

// ═══════════════════════════════════════════════════════════════════════
// Visual design-system helpers (local to Model Training) — mirror the
// slate/blue "KPI strip / gradient hero / pipeline chip" language used in
// the redesigned Preprocessing & Feature Engineering tabs (data-preparation.tsx)
// so this stage reads as part of the same product, not a one-off.
// ═══════════════════════════════════════════════════════════════════════

// Same blue/emerald pair data-preparation.tsx uses for its own "class
// distribution per split" chart — ties this chart's class colors to the
// app's primary accent instead of a generic red/green traffic-light pair.
const SPLIT_CLASS_COLORS: Record<string, string> = { "0": "#059669", "1": "#2563EB" };
const SPLIT_CLASS_FALLBACK = "#94A3B8";

type KpiTone = "primary" | "amber" | "emerald" | "rose" | "violet" | "slate";

function KpiTile({
  icon: Icon,
  label,
  value,
  sub,
  tone = "primary",
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: React.ReactNode;
  sub?: string;
  tone?: KpiTone;
}) {
  const toneClasses: Record<KpiTone, string> = {
    primary: "bg-blue-500/10 text-blue-600",
    amber: "bg-amber-500/10 text-amber-600",
    emerald: "bg-emerald-500/10 text-emerald-600",
    rose: "bg-rose-500/10 text-rose-600",
    violet: "bg-violet-500/10 text-violet-600",
    slate: "bg-slate-500/10 text-slate-600",
  };
  return (
    <div className="bg-white p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-500">{label}</span>
        <span className={cn("flex h-6 w-6 shrink-0 items-center justify-center rounded-md transition-colors duration-300", toneClasses[tone])}>
          <Icon className="h-3.5 w-3.5" />
        </span>
      </div>
      <div className="mt-2 text-2xl font-bold tabular-nums text-slate-900">{value}</div>
      {sub && <div className="mt-0.5 truncate text-[11px] text-slate-500">{sub}</div>}
    </div>
  );
}

function KpiStrip({
  tiles,
}: {
  tiles: Array<{ icon: React.ComponentType<{ className?: string }>; label: string; value: React.ReactNode; sub?: string; tone?: KpiTone }>;
}) {
  return (
    <div className="grid grid-cols-2 divide-x divide-y divide-slate-100 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm sm:grid-cols-3 lg:grid-cols-6 lg:divide-y-0">
      {tiles.map((tile) => (
        <KpiTile key={tile.label} {...tile} />
      ))}
    </div>
  );
}

const TRAINING_SPLIT_COLORS = [
  { bg: "bg-blue-600", border: "border-blue-600", text: "text-blue-600" },
  { bg: "bg-violet-600", border: "border-violet-600", text: "text-violet-600" },
  { bg: "bg-emerald-600", border: "border-emerald-600", text: "text-emerald-600" },
] as const;

function TrainingSplitBar({ trainN, valN, testN }: { trainN: number; valN: number; testN: number }) {
  const total = trainN + valN + testN;
  const pct = (n: number) => (total ? (n / total) * 100 : 0);
  const segments = [
    { label: "Train", n: trainN, role: "Model fitting", c: TRAINING_SPLIT_COLORS[0] },
    { label: "Validation", n: valN, role: "Hyperparameter tuning", c: TRAINING_SPLIT_COLORS[1] },
    { label: "Test", n: testN, role: "Final evaluation", c: TRAINING_SPLIT_COLORS[2] },
  ];
  return (
    <div>
      <div className="flex h-8 gap-0.5 overflow-hidden rounded-lg">
        {segments.map((s) => (
          <div
            key={s.label}
            className={cn("flex items-center justify-center text-[10.5px] font-bold text-white transition-all duration-500 ease-out", s.c.bg)}
            style={{ width: `${pct(s.n)}%` }}
            title={`${s.label}: ${pct(s.n).toFixed(1)}%`}
          >
            {pct(s.n) > 12 ? s.label.slice(0, 5).toUpperCase() : ""}
          </div>
        ))}
      </div>
      <div className="mt-4 grid grid-cols-3 gap-3">
        {segments.map((s) => (
          <div key={s.label} className={cn("rounded-lg border-l-4 bg-slate-50 p-3", s.c.border)}>
            <div className="flex items-baseline gap-1.5">
              <span className="text-lg font-bold tabular-nums text-slate-900"><AnimatedNumber value={s.n} /></span>
              <span className={cn("text-xs font-bold", s.c.text)}><AnimatedNumber value={pct(s.n)} formatter={(n) => `${n.toFixed(1)}%`} /></span>
            </div>
            <div className="text-xs font-semibold text-slate-900">{s.label}</div>
            <div className="text-[11px] text-slate-500">{s.role}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

type PipelineNodeStatus = "complete" | "active" | "pending";

function TrainingPipeline({ nodes }: { nodes: Array<{ label: string; status: PipelineNodeStatus; sub?: string }> }) {
  const activeIndex = nodes.findIndex((n) => n.status === "active");
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-1.5 text-sm font-semibold text-slate-900">
          <Workflow className="h-4 w-4 text-slate-400" />
          Model Development Pipeline
        </div>
        <span className="rounded-full bg-blue-50 px-2.5 py-1 text-[10.5px] font-bold uppercase tracking-wide text-blue-700">
          Step {(activeIndex >= 0 ? activeIndex : nodes.length - 1) + 1} of {nodes.length}
        </span>
      </div>
      <div className="mt-5 flex items-center gap-1 overflow-x-auto pb-1">
        {nodes.map((n, i) => (
          <div key={n.label} className="flex items-center">
            <div className="flex flex-col items-center gap-1.5">
              <div
                className={cn(
                  "flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border text-sm font-bold transition-all duration-500",
                  n.status === "complete" && "border-emerald-600 bg-emerald-600 text-white",
                  n.status === "active" && "border-blue-600 bg-blue-600 text-white shadow-[0_0_0_4px_rgba(37,99,235,0.15)]",
                  n.status === "pending" && "border-slate-200 bg-slate-50 text-slate-400",
                )}
              >
                {n.status === "complete" ? <CheckCircle2 className="h-5 w-5" /> : n.status === "active" ? <Zap className="h-4 w-4" /> : i + 1}
              </div>
              <div className="text-center">
                <div className="whitespace-nowrap text-[10.5px] font-semibold text-slate-700">{n.label}</div>
                {n.sub && <div className="whitespace-nowrap text-[9.5px] text-slate-400">{n.sub}</div>}
              </div>
            </div>
            {i < nodes.length - 1 && <ArrowRight className="mx-2 mb-5 h-3.5 w-3.5 shrink-0 text-slate-300" />}
          </div>
        ))}
      </div>
    </div>
  );
}

function ConfigToggleCard({
  icon: Icon,
  title,
  sub,
  enabled,
  onToggle,
  badge,
  children,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  sub: string;
  enabled: boolean;
  onToggle: (value: boolean) => void;
  badge?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className={cn("rounded-xl border p-4 transition", enabled ? "border-blue-200 bg-blue-50/50" : "border-slate-200 bg-slate-50/60")}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className={cn("flex h-9 w-9 shrink-0 items-center justify-center rounded-lg", enabled ? "bg-blue-600 text-white" : "bg-slate-200 text-slate-500")}>
            <Icon className="h-4 w-4" />
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-900">{title}</div>
            <p className="mt-0.5 text-xs text-slate-500">{sub}</p>
          </div>
        </div>
        <Switch checked={enabled} onCheckedChange={onToggle} />
      </div>
      {badge && (
        <div className="mt-3 flex items-center gap-2">
          <span className={cn("inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide", enabled ? "bg-blue-100 text-blue-700" : "bg-slate-200 text-slate-500")}>
            {badge}
          </span>
          <span className={cn("text-[10px] font-bold", enabled ? "text-emerald-600" : "text-slate-400")}>
            {enabled ? "● Enabled" : "○ Disabled"}
          </span>
        </div>
      )}
      {children && <div className="mt-3">{children}</div>}
    </div>
  );
}

function HyperparamSlider({
  label,
  value,
  min,
  max,
  step,
  decimals = 0,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  decimals?: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50/60 p-4">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-slate-700">{label}</label>
        <span className="text-lg font-bold tabular-nums text-blue-600">{value.toFixed(decimals)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="mt-3 w-full accent-blue-600"
      />
      <div className="mt-1 flex justify-between text-[10.5px] text-slate-400">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function TrainingTab({ onProceed }: { onProceed: () => void }) {
  const navigate = useNavigate();
  const {
    profile,
    file,
    selectedModel,
    recommendations,
    setRecommendations,
    compareModels,
    setCompareModels,
    trainingConfig,
    trainingResult,
    comparisonResults,
    selectedComparisonModel,
    setSelectedModel,
    setTrainingConfig,
    setTrainingResult,
    setComparisonResults,
    setSelectedComparisonModel,
    preprocessingResult,
    validationIntakeData,
  } = useDataset();

  // ── Split config: owned by Preprocessing (Step 3). Training only reads it. ──
  const splitConfig = preprocessingResult?.split_config ?? { test_size: 0.15, val_size: 0.15, random_seed: 42 };

  // Training configuration state (split fields kept for API-call compatibility,
  // but sourced from preprocessing's locked-in split rather than user input here)
  const [config, setConfig] = useState<TrainingConfig>(trainingConfig ?? {
    test_size: splitConfig.test_size,
    val_size: splitConfig.val_size,
    random_seed: splitConfig.random_seed,
    use_cv: false,
    cv_folds: 5,
    use_hyperopt: false,
    use_feature_engineering: false,
    manual_params: {},
    use_oot: false,
    date_col: null,
  });

  // Keep config's split fields in sync with whatever Preprocessing locked in,
  // in case the reviewer changed it there after this page already mounted.
  useEffect(() => {
    setConfig((prev) => ({
      ...prev,
      test_size: splitConfig.test_size,
      val_size: splitConfig.val_size,
      random_seed: splitConfig.random_seed,
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [splitConfig.test_size, splitConfig.val_size, splitConfig.random_seed]);

  const [trainingInfo, setTrainingInfo] = useState<Record<string, any> | null>(trainingResult?.training_info ?? null);
  const [splitStats, setSplitStats] = useState<Record<string, any> | null>(trainingResult?.split_stats ?? preprocessingResult?.split_stats ?? null);
  const [evaluationMetrics, setEvaluationMetrics] = useState<Record<string, any> | null>(trainingResult?.evaluation_metrics ?? null);
  const [modelArtifact, setModelArtifact] = useState<string | null>(trainingResult?.model_artifact ?? null);
  const [taskType, setTaskType] = useState<string | null>(trainingResult?.task_type ?? null);
  const [trainingModelName, setTrainingModelName] = useState<string | null>(trainingResult?.model_name ?? "");
  const [trainingConfigResult, setTrainingConfigResult] = useState<Record<string, any> | null>(trainingResult?.training_config ?? null);
  // Which ESTIMATOR (LightGBM, XGBoost, ...) the current evaluationMetrics
  // actually belong to — decoupled from `selectedModel`, which the reviewer
  // can click to a different card after training without re-running it.
  // Without this, Model Selection could misattribute one model's real
  // metrics to a different model's card. Best-effort on resume/reload
  // (selectedModel is persisted alongside trainingResult and is normally
  // still the trained one at that point); set precisely on every fresh
  // train in handleTrain below.
  const [trainedEstimatorName, setTrainedEstimatorName] = useState<string | null>(
    trainingResult ? (selectedModel?.name ?? null) : null,
  );
  const { user } = useAuth();
  const [modelOwner, setModelOwner] = useState<string>(validationIntakeData?.model_owner ?? "");

  useEffect(() => {
    if ((!modelOwner || modelOwner.trim() === "") && user && typeof user.name === "string" && user.name.trim()) {
      setModelOwner(user.name);
    }
    // Only update when user or validationIntakeData changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);
  const [businessUnit, setBusinessUnit] = useState<string>(validationIntakeData?.owning_team ?? "");
  const [modelPurpose, setModelPurpose] = useState<string>(validationIntakeData?.model_purpose ?? "");
  const [modelVersion, setModelVersion] = useState<string>(validationIntakeData?.model_version ?? "1.0");
  const [modelType, setModelType] = useState<string>(validationIntakeData?.model_type ?? "Custom");
  const [developmentDate, setDevelopmentDate] = useState<string>("");
  const [status, setStatus] = useState<string>("In Development");
  const [documentationPath, setDocumentationPath] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelComparison, setModelComparison] = useState<boolean>(false);
  const [modelsToCompare, setModelsToCompare] = useState<string[]>(compareModels ?? []);

  // Resume where the reviewer left off: if this session has no training
  // result yet, pull the last saved /models/train run from the backend.
  const { data: resumedTraining } = useResumeState<Record<string, any>>("dev_pipeline_log.csv", "training");
  useEffect(() => {
    if (!trainingResult && resumedTraining) {
      setTrainingResult(resumedTraining as any);
      setTrainingInfo(resumedTraining?.training_info ?? null);
      setSplitStats(resumedTraining?.split_stats ?? null);
      setEvaluationMetrics(resumedTraining?.evaluation_metrics ?? null);
      setModelArtifact(resumedTraining?.model_artifact ?? null);
      setTaskType(resumedTraining?.task_type ?? null);
      setTrainingModelName(resumedTraining?.model_name ?? null);
      setTrainingConfigResult(resumedTraining?.training_config ?? null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumedTraining]);

  // ── Decision threshold: auto (F1-maximizing, computed by the backend) by
  // default, with an optional manual override via the slider below. When
  // `useAutoThreshold` is true, no `threshold` field is sent to /models/train
  // at all — main.py's _build_evaluation_data() treats a missing/None
  // threshold as "auto-select the F1-maximizing cut-off" and every chart
  // (confusion matrix, threshold curve, evaluation metrics) is built
  // against that resolved value. Flipping the toggle off pins the exact
  // value the slider shows instead. ──
  const [useAutoThreshold, setUseAutoThreshold] = useState(true);
  const [manualThreshold, setManualThreshold] = useState<number>(0.5);
  const [resolvedThreshold, setResolvedThreshold] = useState<number | null>(null);

  // ── Flow mode: user picks one of two paths before training ─────────────
  // 'choose'  → initial fork, nothing configured yet
  // 'compare' → lightweight side-by-side comparison across candidate models
  // 'direct'  → configure (class balancing, CV) and train a single model,
  //             either chosen directly or carried over from a comparison run
  const [flowMode, setFlowMode] = useState<"choose" | "compare" | "direct">(
    trainingResult ? "direct" : "choose",
  );
  const [comparisonLoading, setComparisonLoading] = useState(false);

  // Hyperparameter preset controls
  const [hyperparams, setHyperparams] = useState<Record<string, any>>({
    learning_rate: 0.05,
    max_depth: 6,
    n_estimators: 200,
    subsample: 0.8,
    colsample_bytree: 0.8,
    reg_lambda: 1.0,
    reg_alpha: 0.0,
  });

  // ── Model recommendations (merged in from models.tsx) ──────────────────
  const [trainingStats, setTrainingStats] = useState<{ train_n: number; train_features: number; imbalance_ratio: number } | null>(null);
  const [recommendationTaskType, setRecommendationTaskType] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [recommendedModel, setRecommendedModel] = useState<ModelRecommendation | null>(null);
  const fetchRef = useRef(false);

  const datasetSummary = useMemo(() => {
    // Prefer training stats returned by the backend (train split after FE)
    if (trainingStats) {
      return { sampleCount: trainingStats.train_n, featureCount: trainingStats.train_features, imbalanceRatio: trainingStats.imbalance_ratio };
    }
    if (!profile) return null;
    const shape = profile.shape ?? [0, 0];
    const sampleCount = shape[0] ?? 0;
    const featureCount = shape[1] ?? 0;
    let imbalanceRatio = 1.0;

    if (profile.class_distribution && typeof profile.class_distribution === "object") {
      const values = Object.values(profile.class_distribution) as number[];
      if (values.length >= 2) {
        const sorted = values.sort((a, b) => b - a);
        imbalanceRatio = sorted[0] / (sorted[1] || 1);
      }
    }

    return { sampleCount, featureCount, imbalanceRatio };
  }, [profile, trainingStats]);

  // Columns detected as datetime by profiling — used to pick an origination/
  // observation date for Out-of-Time (OOT) validation. Falls back to
  // backend auto-detection (first datetime column) if none is chosen here.
  const datetimeColumns: string[] = useMemo(
    () => (profile as any)?.col_types?.datetime ?? [],
    [profile]
  );

  const transformedModels: ModelCard[] = useMemo(() => {
    if (!recommendations || !Array.isArray(recommendations) || recommendations.length === 0) return [];
    return recommendations.map((rec, idx) => ({
      ...rec,
      selected: rec.name === selectedModel?.name || (!selectedModel && idx === 0),
    }));
  }, [recommendations, selectedModel]);

  useEffect(() => {
    if (!profile || !file) return;
    // Reset guard when dataset changes so we fetch for new dataset
    fetchRef.current = false;
    if (recommendations && recommendations.length > 0) return; // already loaded
    if (fetchRef.current) return; // already in-flight or fetched

    let isMounted = true;
    const loadRecommendations = async () => {
      setModelsLoading(true);
      setModelsError(null);

      try {
        const form = new FormData();
        form.append("file", file);
        form.append("target_col", profile.target_col || "loan_status");

        const response = await formUpload("/models/recommend", form);
        if (!isMounted) return;
        // Use backend-provided training stats when available
        if (response?.training) {
          setTrainingStats(response.training as any);
        }
        if (response?.task_type) {
          setRecommendationTaskType(response.task_type);
        }

        const recommended = response?.recommended_model ?? null;
        const allModels = response?.all_models ?? null;

        if (recommended && Array.isArray(allModels)) {
          const transformed: ModelCard[] = allModels.map((m: any) => ({
            name: m.name,
            description: m.description ?? "",
            icon: m.icon,
            reasons: m.name === recommended.name ? (recommended.reasons ?? []) : [],
          }));

          setRecommendations(transformed);
          setRecommendedModel({
            name: recommended.name,
            description: recommended.description ?? "",
            icon: recommended.icon,
            reasons: recommended.reasons ?? [],
          });

          const currentModelName = selectedModel?.name;
          const hasCurrentSelection = currentModelName
            ? transformed.some((m) => m.name === currentModelName)
            : false;

          if (!hasCurrentSelection) {
            const preselect = transformed.find((m) => m.name === recommended.name) ?? transformed[0];
            if (preselect) setSelectedModel(preselect);
          }

          const validCompareModels = (compareModels ?? []).filter((name) =>
            transformed.some((m) => m.name === name),
          );

          if (validCompareModels.length === 0 && transformed.length > 0) {
            setCompareModels(transformed.slice(0, Math.min(3, transformed.length)).map((m) => m.name));
          } else if (validCompareModels.length !== (compareModels ?? []).length) {
            setCompareModels(validCompareModels);
          }
        } else {
          setModelsError("No recommendation returned by backend.");
        }
      } catch (err: any) {
        console.error("Training: failed to load model recommendations", err);
        if (!isMounted) return;
        setModelsError(err?.body?.detail ?? err?.message ?? "Failed to load model recommendations.");
      } finally {
        if (isMounted) setModelsLoading(false);
      }
    };

    fetchRef.current = true;
    loadRecommendations();
    return () => {
      isMounted = false;
    };
  }, [file, profile, recommendations, selectedModel, compareModels, setRecommendations, setSelectedModel, setCompareModels]);

  const toggleModelToCompare = useCallback((modelName: string) => {
    const current = compareModels ?? [];
    const next = current.includes(modelName)
      ? current.filter((m) => m !== modelName)
      : [...current, modelName];
    setCompareModels(next);
  }, [compareModels, setCompareModels]);

  // Split stats to display: prefer the exact stats a training run returned,
  // otherwise fall back to the split already computed in Preprocessing.
  const displaySplitStats = splitStats ?? preprocessingResult?.split_stats ?? null;

  // Handle training execution
  const trainModel = async (modelName: string) => {
    if (!profile || !file) {
      throw new Error("Missing profile or file");
    }

    const trainForm = new FormData();
    trainForm.append("file", file);
    trainForm.append("target_col", profile.target_col || "loan_status");
    // If the user didn't provide an explicit business `model_name`, send
    // an empty string rather than defaulting to the estimator name. This
    // prevents accidental updates to existing inventory entries that use
    // the algorithm name as their `model_name`.
    const modelNameToSend = trainingModelName && trainingModelName.trim() ? trainingModelName.trim() : "";
    trainForm.append("model_name", modelNameToSend);
    trainForm.append("estimator_name", modelName);
    trainForm.append("test_size", String(config.test_size));
    trainForm.append("val_size", String(config.val_size));
    trainForm.append("random_seed", String(config.random_seed));
    trainForm.append("use_cv", String(config.use_cv));
    trainForm.append("cv_folds", String(config.cv_folds));
    trainForm.append("use_hyperopt", String(config.use_hyperopt));
    trainForm.append("use_oot", String(config.use_oot));
    if (config.date_col) {
      trainForm.append("date_col", config.date_col);
    }
    trainForm.append("use_feature_engineering", String(config.use_feature_engineering));
    if (Object.keys(config.manual_params).length > 0) {
      trainForm.append("manual_params", JSON.stringify(config.manual_params));
    }
    trainForm.append("model_owner", modelOwner);
    trainForm.append("business_unit", businessUnit);
    trainForm.append("model_purpose", modelPurpose);
    trainForm.append("model_version", modelVersion);
    trainForm.append("model_type", modelType);
    trainForm.append("development_date", developmentDate);
    trainForm.append("status", status);
    trainForm.append("documentation_path", documentationPath);
    // PD classification cut-off. Omitted entirely when "Auto" is on, so the
    // backend (main.py's _build_evaluation_data, threshold=None) auto-selects
    // the F1-maximizing threshold. Only send an explicit value when the
    // reviewer has overridden it via the slider below.
    if (!useAutoThreshold) {
      trainForm.append("threshold", String(manualThreshold));
    }

    const trainResponse = await formUpload("/models/train", trainForm);
    if (!trainResponse?.training_info || !trainResponse?.split_stats || !trainResponse?.model_artifact) {
      throw new Error("Training response missing required fields.");
    }

    return {
      model_name: trainResponse.model_name ?? (trainingModelName?.trim() || modelName),
      task_type: trainResponse.task_type ?? "binary",
      real_feature_names: trainResponse.real_feature_names ?? [],
      training_config: trainResponse.training_config ?? null,
      training_info: trainResponse.training_info,
      split_stats: trainResponse.split_stats,
      feature_engineering_summary: trainResponse.feature_engineering_summary ?? null,
      model_artifact: trainResponse.model_artifact,
      evaluation_metrics: trainResponse.evaluation_metrics ?? null,
      evaluation_data: trainResponse.evaluation_data ?? null,
    };
  };

  const handleTrain = async () => {
    if (!profile || !file || !selectedModel) {
      setError("Missing profile, file, or model selection");
      return;
    }

    // Require an explicit Business Model Name before training to avoid
    // accidental updates of existing inventory entries when the field is
    // left blank.
    if (!trainingModelName || (typeof trainingModelName === "string" && trainingModelName.trim() === "")) {
      setError("Please enter a Business Model Name before starting development.");
      return;
    }

    setLoading(true);
    setError(null);
    setModelComparison(false);

    try {
      const result = await trainModel(selectedModel.name);
      setTrainingInfo(result.training_info);
      setSplitStats(result.split_stats);
      setEvaluationMetrics(result.evaluation_metrics);
      setModelArtifact(result.model_artifact);
      setTaskType(result.task_type);
      setTrainingModelName(result.model_name);
      setTrainedEstimatorName(selectedModel.name);
      setTrainingConfigResult(result.training_config ?? null);
      setResolvedThreshold(
        typeof result.evaluation_data?.threshold === "number" ? result.evaluation_data.threshold : null,
      );
      setComparisonResults([{ model_name: result.model_name, ...result.evaluation_metrics, training_time_s: result.training_info.training_time_s }]);
      setTrainingResult({
        task_type: result.task_type,
        model_name: result.model_name,
        real_feature_names: result.real_feature_names ?? [],
        training_config: result.training_config ?? null,
        training_info: result.training_info,
        split_stats: result.split_stats,
        feature_engineering_summary: result.feature_engineering_summary,
        evaluation_metrics: result.evaluation_metrics,
        evaluation_data: result.evaluation_data,
        model_artifact: result.model_artifact,
      });
    } catch (err: any) {
      console.error("Training: failed", err);
      setError(err?.body?.detail ?? err?.message ?? "Failed to train model.");
    } finally {
      setLoading(false);
    }
  };

  // Lightweight comparison: hits /models/compare, which skips CV, hyperopt,
  // OOT, evaluation curves, and model-artifact serialization for every
  // candidate — just a quick fit + test-set metrics per model, so this stays
  // fast even with several candidates selected. Nothing here is treated as
  // a "trained" model — trainingInfo/trainingResult are untouched, so
  // "Proceed to Evaluation" stays locked until the user actually trains a
  // chosen model via the direct-training path.
  const handleRunComparison = async () => {
    if (!profile || !file) {
      setError("Missing profile or file");
      return;
    }

    if (modelsToCompare.length < 2) {
      setError("Select at least two models to compare");
      return;
    }

    setComparisonLoading(true);
    setError(null);
    setModelComparison(true);

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("target_col", profile.target_col || "loan_status");
      form.append("model_names", JSON.stringify(modelsToCompare));
      form.append("test_size", String(config.test_size));
      form.append("val_size", String(config.val_size));
      form.append("random_seed", String(config.random_seed));
      form.append("use_feature_engineering", String(config.use_feature_engineering));

      const response = await formUpload("/models/compare", form);
      const rows: ComparisonResult[] = (response?.comparison ?? []).map((row: any) => ({
        model_name: row.model_name,
        roc_auc: row.roc_auc,
        recall: row.recall,
        precision: row.precision,
        f1: row.f1,
        pr_auc: row.pr_auc,
        accuracy: row.accuracy,
        training_time_s: row.training_time_s,
        error: row.error,
      }));

      setComparisonResults(rows);
      if (rows.length > 0 && !rows[0].error) {
        setSelectedComparisonModel(rows[0].model_name);
      }
    } catch (err: any) {
      console.error("Comparison: failed", err);
      setError(err?.body?.detail ?? err?.message ?? "Failed to run comparison.");
    } finally {
      setComparisonLoading(false);
    }
  };

  // Carry a comparison winner into the direct-training path: select it as
  // the model to train, then flip to 'direct' so the user can set class
  // balancing / CV before running the real, full-fidelity training run.
  const handleUseComparisonModel = (modelName: string) => {
    setSelectedComparisonModel(modelName);
    const chosen = recommendations?.find((rec) => rec.name === modelName);
    if (chosen) setSelectedModel(chosen);
    setFlowMode("direct");
  };

  // Calculate class imbalance for recommendations
  const usedTrainingConfig = trainingConfigResult ?? trainingResult?.training_config ?? config;

  const classImbalance = useMemo(() => {
    const trainingDist = splitStats?.train_class_dist ?? profile?.class_distribution;
    if (!trainingDist) return 1.0;
    const values = Object.values(trainingDist) as number[];
    if (values.length < 2) return 1.0;
    const sorted = [...values].sort((a, b) => b - a);
    return sorted[0] / (sorted[1] || 1);
  }, [splitStats?.train_class_dist, profile?.class_distribution]);

  const classLabels = useMemo(() => {
    if (splitStats?.train_class_dist) {
      return Object.keys(splitStats.train_class_dist);
    }
    return [];
  }, [splitStats]);

  const splitClassData = useMemo(() => {
    if (!splitStats?.train_class_dist || !splitStats?.val_class_dist || !splitStats?.test_class_dist) {
      return [];
    }

    return [
      { split: "Train", dist: splitStats.train_class_dist },
      { split: "Val", dist: splitStats.val_class_dist },
      { split: "Test", dist: splitStats.test_class_dist },
    ].map((item) => {
      const row: Record<string, number | string> = { split: item.split };
      classLabels.forEach((label) => {
        row[label] = Number(item.dist?.[label] ?? 0);
      });
      return row;
    });
  }, [splitStats, classLabels]);

  const splitClassFigure = useMemo(() => {
    if (!splitClassData || splitClassData.length === 0) return null;
    const x = splitClassData.map((row) => row.split as string);

    const traces = classLabels.map((label) => ({
      type: "bar",
      name: label === "0" ? "Class 0" : label === "1" ? "Class 1" : label,
      x,
      y: splitClassData.map((row) => Number(row[label] ?? 0)),
      marker: {
        color: SPLIT_CLASS_COLORS[label] ?? SPLIT_CLASS_FALLBACK,
      },
      hovertemplate: "%{x}<br>%{y:.0f}<extra></extra>",
    }));

    return {
      data: traces,
      layout: {
        barmode: "stack",
        margin: { t: 8, r: 12, l: 36, b: 28 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "inherit", size: 11, color: "#64748b" },
        bargap: 0.35,
        xaxis: { title: "", automargin: true, tickfont: { color: "#64748b", size: 11.5 }, linecolor: "#e2e8f0", showgrid: false, zeroline: false },
        yaxis: { title: "", automargin: true, tickfont: { color: "#64748b", size: 11.5 }, gridcolor: "#f1f5f9", zeroline: false },
        legend: { orientation: "h", y: 1.16, font: { size: 11.5, color: "#475569" } },
      },
    };
  }, [splitClassData, classLabels]);

  // Per-model comparison metrics — populated either by a "Compare models
  // first" run, or (for whichever model was actually trained) by
  // handleTrain's setComparisonResults call. Used to enrich the Model
  // Selection list with real, model-specific AUC/F1/Recall figures; a
  // model with no entry here genuinely hasn't been evaluated yet, so the
  // UI says so instead of rendering empty dashes.
  const comparisonByModel = useMemo(() => {
    const map = new Map<string, ComparisonResult>();
    (comparisonResults ?? []).forEach((row) => map.set(row.model_name, row));
    return map;
  }, [comparisonResults]);

  useEffect(() => {
    setTrainingConfig(config);
  }, [config, setTrainingConfig]);

  useEffect(() => {
    if (compareModels && compareModels.length > 0) {
      setModelsToCompare(compareModels);
    }
  }, [compareModels]);

  useEffect(() => {
    setCompareModels(modelsToCompare);
  }, [modelsToCompare, setCompareModels]);

  if (!profile) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 text-center">
        <h3 className="text-lg font-semibold">No dataset available</h3>
        <p className="mt-2 text-sm text-muted-foreground">Upload and preprocess a dataset before training.</p>
      </div>
    );
  }

  // ── Presentation-only derivations. Every input here already exists in
  // state above; nothing new is fetched or computed business-side. ──────
  const statusLabel = loading ? "Training" : trainingInfo ? "Trained" : selectedModel ? "Ready" : "Pending";
  const statusTone: KpiTone = loading ? "amber" : trainingInfo ? "emerald" : selectedModel ? "primary" : "slate";
  const thresholdDisplay = resolvedThreshold != null
    ? resolvedThreshold.toFixed(3)
    : useAutoThreshold ? "Auto" : manualThreshold.toFixed(2);
  const thresholdSub = resolvedThreshold != null
    ? (useAutoThreshold ? "Last run · max F1" : "Last run · manual")
    : (useAutoThreshold ? "Auto max-F1" : "Manual override");

  const pipelineNodes: Array<{ label: string; status: PipelineNodeStatus; sub?: string }> = [
    { label: "Dataset", status: "complete", sub: `${datasetSummary?.sampleCount.toLocaleString() ?? "—"} rows` },
    { label: "Model Selection", status: selectedModel ? "complete" : "active", sub: selectedModel?.name },
    { label: "Configuration", status: trainingInfo ? "complete" : selectedModel && flowMode === "direct" ? "active" : "pending" },
    { label: "Training", status: trainingInfo ? "complete" : loading ? "active" : "pending" },
    { label: "Model Ready", status: trainingInfo ? "complete" : "pending" },
  ];

  return (
    <div className="space-y-6">
      {/* ── Executive header ─────────────────────────────────────────── */}
      <div className="rounded-2xl border border-slate-200 bg-gradient-to-r from-slate-900 via-sky-900 to-blue-800 p-6 text-white shadow-[0_16px_36px_rgba(15,23,42,0.16)]">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="text-[11px] uppercase tracking-[0.2em] text-sky-200">Step 5 · Model Development</div>
            <h3 className="mt-2 text-2xl font-semibold">Model Training</h3>
            <p className="mt-2 max-w-2xl text-sm text-slate-200">
              {selectedModel ? `Configure and train the ${selectedModel.name} model with optimized parameters.` : "Models ranked by suitability for your dataset — pick one to train."}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {datasetSummary && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-300/40 bg-white/90 px-3 py-1 text-xs font-semibold text-slate-700">
                {datasetSummary.sampleCount.toLocaleString()} samples
              </span>
            )}
            <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-300/50 bg-emerald-400/15 px-3 py-1 text-xs font-semibold text-emerald-200">
              <CheckCircle2 className="h-3 w-3" /> Dataset Ready
            </span>
            {selectedModel && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-sky-300/50 bg-sky-400/15 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-sky-100">
                {selectedModel.name}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* ── KPI strip ─────────────────────────────────────────────────── */}
      {datasetSummary && (
        <KpiStrip
          tiles={[
            { icon: Database, label: "Total Samples", value: <AnimatedNumber value={datasetSummary.sampleCount} />, sub: "Rows available for training", tone: "primary" },
            { icon: Layers, label: "Features", value: <AnimatedNumber value={datasetSummary.featureCount} />, sub: "Model input columns", tone: "violet" },
            { icon: AlertTriangle, label: "Imbalance", value: <AnimatedNumber value={datasetSummary.imbalanceRatio} formatter={(n) => `${n.toFixed(1)}:1`} />, sub: "Class 0 : Class 1", tone: datasetSummary.imbalanceRatio > 1.5 ? "amber" : "emerald" },
            { icon: Cpu, label: "Model", value: selectedModel?.name ?? "—", sub: recommendedModel?.name === selectedModel?.name ? "Auto-recommended" : selectedModel ? "Manually selected" : "Not selected", tone: "primary" },
            { icon: Activity, label: "Status", value: statusLabel, sub: trainingInfo ? "Training run complete" : "Config complete", tone: statusTone },
            { icon: Target, label: "Threshold", value: thresholdDisplay, sub: thresholdSub, tone: "violet" },
          ]}
        />
      )}

      {/* ── Training workflow pipeline (visual only) ─────────────────── */}
      <TrainingPipeline nodes={pipelineNodes} />

      {error && (
        <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 flex gap-3 shadow-sm">
          <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
          <div>{error}</div>
        </div>
      )}

      {modelsError && (
        <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 flex gap-3 shadow-sm">
          <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
          <div>{modelsError}</div>
        </div>
      )}

      {modelsLoading && (
        <div className="flex items-center justify-center gap-2 rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-600 shadow-sm">
          <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
          Loading model recommendations...
        </div>
      )}

      {transformedModels.length === 0 && !modelsLoading && !modelsError && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 text-center text-sm text-slate-500 shadow-sm">
          No model recommendations available.
        </div>
      )}

      {/* ── Model Selection (merged in from models.tsx) ──────────────────── */}
      {transformedModels.length > 0 && (
        <>
          <section className={cn("grid gap-5", recommendedModel ? "lg:grid-cols-5" : "lg:grid-cols-1")}>
            {recommendedModel && (
              <div className="lg:col-span-2 rounded-2xl border border-blue-700 bg-gradient-to-br from-blue-600 to-blue-800 p-6 text-white shadow-[0_16px_36px_rgba(29,78,216,0.25)]">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.2em] text-blue-100">Recommended Model</div>
                    <h3 className="mt-2 text-2xl font-bold">{recommendedModel.name}</h3>
                  </div>
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-white/15 text-lg font-bold">
                    {recommendedModel.icon ?? recommendedModel.name.charAt(0)}
                  </div>
                </div>
                <div className="mt-2 inline-flex w-fit items-center gap-1 rounded-full bg-white/15 px-2 py-1 text-[11px] font-medium text-white">
                  <CheckCircle2 className="h-3 w-3" /> Recommended
                </div>
                <p className="mt-3 text-sm text-blue-50">{recommendedModel.description}</p>
                <p className="mt-3 text-[11px] uppercase tracking-wider text-blue-200">
                  Chosen from dataset size, missingness, class imbalance, feature mix, correlation, and non-linearity.
                </p>

                {recommendedModel.reasons?.length ? (
                  <div className="mt-4 space-y-2 border-t border-white/15 pt-4">
                    {recommendedModel.reasons.map((reason, i) => (
                      <div key={i} className="flex items-start gap-2 text-sm text-blue-50">
                        <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-300" />
                        <span>{reason}</span>
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>
            )}

            <div className="lg:col-span-3 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <div className="text-base font-semibold text-slate-900">Model Selection</div>
                  <div className="text-xs text-slate-500">Click to select — compare performance at a glance</div>
                </div>
                {selectedModel && (
                  <span className="whitespace-nowrap rounded-full bg-blue-50 px-2.5 py-1 text-[10.5px] font-bold uppercase tracking-wide text-blue-700">
                    {selectedModel.name} active
                  </span>
                )}
              </div>
              <div className="mt-4 space-y-2">
                {transformedModels.map((model) => {
                  const isSelected = model.name === selectedModel?.name;
                  const isRecommended = recommendedModel?.name === model.name;
                  // Prefer this model's own /models/compare row when one
                  // exists; otherwise, if THIS is the exact estimator that
                  // produced the current evaluationMetrics (not just
                  // whatever's selected right now), use those — the same
                  // real numbers already shown in Evaluation Metrics below.
                  const isJustTrained = Boolean(trainingInfo) && Boolean(evaluationMetrics) && trainedEstimatorName === model.name;
                  const cmp: ComparisonResult | undefined = comparisonByModel.get(model.name)
                    ?? (isJustTrained
                      ? { model_name: model.name, roc_auc: evaluationMetrics?.roc_auc, f1: evaluationMetrics?.f1, recall: evaluationMetrics?.recall }
                      : undefined);
                  return (
                    <button
                      key={model.name}
                      type="button"
                      onClick={() => setSelectedModel(model)}
                      className={cn(
                        "w-full rounded-xl border p-3.5 text-left transition",
                        isSelected ? "border-blue-600 bg-blue-50/60 ring-1 ring-blue-600/20" : "border-slate-200 hover:border-blue-300",
                      )}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex min-w-0 items-center gap-2.5">
                          <div className={cn("flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-xs font-bold", isSelected ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-600")}>
                            {model.icon ?? model.name.charAt(0)}
                          </div>
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-1.5 text-sm font-semibold text-slate-900">
                              {model.name}
                              {isRecommended && <span className="rounded-full bg-emerald-100 px-1.5 py-0.5 text-[9.5px] font-bold uppercase tracking-wide text-emerald-700">Recommended</span>}
                              {isSelected && <span className="rounded-full bg-blue-100 px-1.5 py-0.5 text-[9.5px] font-bold uppercase tracking-wide text-blue-700">Selected</span>}
                            </div>
                            {model.description && <div className="truncate text-xs text-slate-500">{model.description}</div>}
                          </div>
                        </div>
                        <div className="hidden shrink-0 items-center sm:flex">
                          {(() => {
                            const hasMetrics = Boolean(cmp) && !cmp?.error && [cmp?.roc_auc, cmp?.f1, cmp?.recall].some((v) => typeof v === "number");
                            if (!hasMetrics) {
                              return (
                                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10.5px] font-medium text-slate-500">
                                  {cmp?.error ? "Evaluation failed" : "Not evaluated"}
                                </span>
                              );
                            }
                            return (
                              <div className="flex items-center gap-3">
                                {[["AUC", cmp?.roc_auc], ["F1", cmp?.f1], ["Recall", cmp?.recall]].map(([label, value]) => (
                                  <div key={label as string} className="w-12 text-right">
                                    <div className="text-[9.5px] font-semibold uppercase text-slate-400">{label}</div>
                                    <div className="text-xs font-bold tabular-nums text-slate-800">
                                      {typeof value === "number" ? <AnimatedNumber value={value} formatter={(n) => n.toFixed(3)} /> : "—"}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            );
                          })()}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
              <p className="mt-4 text-[11px] text-slate-500">
                The recommended model is pre-selected. Override it here, or run a side-by-side comparison below.
              </p>
            </div>
          </section>

          {recommendationTaskType === "binary" && (
            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center gap-2">
                <ShieldAlert className="h-4 w-4 text-blue-600" />
                <h2 className="text-base font-semibold text-slate-900">Credit Risk Evaluation Strategy</h2>
              </div>
              <p className="mt-1 text-xs text-slate-500">Metric priority for credit PD model — minimize missed defaults</p>

              <div className="mt-5 grid gap-5 lg:grid-cols-[0.9fr_1.1fr_0.9fr]">
                <div className="rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-700 p-5 text-white">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-blue-100">Primary Target</div>
                  <div className="mt-1 text-3xl font-bold">Recall</div>
                  <p className="mt-3 text-xs text-blue-100">
                    Minimize missed defaults (FN). A missed defaulter costs far more than a false alarm in credit risk.
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-500">Optimization Targets</div>
                  {[
                    { label: "ROC-AUC", value: evaluationMetrics?.roc_auc, sub: "Overall discrimination" },
                    { label: "PR-AUC", value: evaluationMetrics?.pr_auc, sub: "Imbalance-robust" },
                    { label: "F1", value: evaluationMetrics?.f1, sub: "Recall–precision balance" },
                  ].map((m) => (
                    <div key={m.label} className="rounded-lg border border-slate-200 p-3">
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium text-slate-700">{m.label}</span>
                        <span className="font-mono font-semibold text-slate-900">
                          {typeof m.value === "number" ? <AnimatedNumber value={m.value} formatter={(n) => n.toFixed(3)} /> : "—"}
                        </span>
                      </div>
                      <div className="mt-2 h-1.5 rounded-full bg-slate-100">
                        <div className="h-full rounded-full bg-blue-600 transition-all duration-500 ease-out" style={{ width: typeof m.value === "number" ? `${Math.min(m.value * 100, 100)}%` : "0%" }} />
                      </div>
                      <div className="mt-1 text-[10.5px] text-slate-400">{m.sub}</div>
                    </div>
                  ))}
                </div>

                <div className="space-y-2">
                  <div className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-500">Risk Signals</div>
                  <div className={cn("rounded-lg border px-3 py-2 text-xs font-medium", classImbalance > 1.5 ? "border-amber-200 bg-amber-50 text-amber-800" : "border-slate-200 bg-slate-50 text-slate-600")}>
                    {classImbalance > 1.5 ? `⚠ Class imbalance ${classImbalance.toFixed(1)}:1` : "Class balance within range"}
                  </div>
                  <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-medium text-emerald-800">✓ Recall-first threshold</div>
                  <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-medium text-emerald-800">✓ Class weights applied</div>
                  <div className={cn("rounded-lg border px-3 py-2 text-xs font-medium", useAutoThreshold ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-slate-200 bg-slate-50 text-slate-500")}>
                    {useAutoThreshold ? "✓ F1 threshold optimized" : "Manual threshold override"}
                  </div>
                </div>
              </div>

              <div className="mt-5 rounded-xl bg-slate-900 p-4">
                <div className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-400">Model Cost Tradeoff</div>
                <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="flex items-center justify-between rounded-lg bg-white/5 px-3 py-2.5">
                    <div>
                      <div className="text-xs font-semibold text-white">False Negative</div>
                      <div className="text-[10.5px] text-slate-400">Missed default</div>
                    </div>
                    <span className="rounded-full bg-red-500/15 px-2 py-0.5 text-[10.5px] font-bold text-red-400">Very High</span>
                  </div>
                  <div className="flex items-center justify-between rounded-lg bg-white/5 px-3 py-2.5">
                    <div>
                      <div className="text-xs font-semibold text-white">False Positive</div>
                      <div className="text-[10.5px] text-slate-400">False alarm</div>
                    </div>
                    <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10.5px] font-bold text-emerald-400">Low</span>
                  </div>
                </div>
              </div>

              <p className="mt-4 text-xs text-slate-500">
                We optimize for: <strong className="text-slate-700">ROC-AUC → Recall → PR-AUC → F1</strong>
              </p>
            </section>
          )}
        </>
      )}

      {!selectedModel && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 text-center text-sm text-slate-500 shadow-sm">
          Select a model above to configure and run training.
        </div>
      )}

      {selectedModel && (
      <>

      {/* ── Choose a path: compare candidates first, or go straight to configuring/training the selected model ── */}
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-base font-semibold text-slate-900 mb-1">How do you want to proceed?</h2>
        <p className="text-sm text-slate-500 mb-4">
          Run a quick, lightweight comparison across a few candidates first, or go straight to configuring and training the selected model.
        </p>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <button
            type="button"
            onClick={() => setFlowMode("compare")}
            className={cn(
              "flex flex-col items-start gap-1 rounded-xl border p-4 text-left transition",
              flowMode === "compare" ? "border-blue-600 bg-blue-50/60 ring-1 ring-blue-600/20" : "border-slate-200 hover:border-blue-300",
            )}
          >
            <div className="flex items-center gap-2 font-semibold text-slate-900">
              <Zap className="h-4 w-4 text-blue-600" /> Compare models first
            </div>
            <p className="text-xs text-slate-500">
              Quick fit on a few candidates (no CV, no tuning) so you can pick a winner before committing to a full training run.
            </p>
          </button>
          <button
            type="button"
            onClick={() => setFlowMode("direct")}
            className={cn(
              "flex flex-col items-start gap-1 rounded-xl border p-4 text-left transition",
              flowMode === "direct" ? "border-blue-600 bg-blue-50/60 ring-1 ring-blue-600/20" : "border-slate-200 hover:border-blue-300",
            )}
          >
            <div className="flex items-center gap-2 font-semibold text-slate-900">
              <CheckCircle2 className="h-4 w-4 text-blue-600" /> Configure &amp; train "{selectedModel.name}"
            </div>
            <p className="text-xs text-slate-500">
              Go straight to class balancing, cross-validation, and a full training run on the selected model.
            </p>
          </button>
        </div>
      </section>

      {/* ── Compare path ── */}
      {flowMode === "compare" && (
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="h-5 w-5 text-blue-600" />
            <h2 className="text-base font-semibold text-slate-900">Model Comparison</h2>
          </div>
          <p className="text-sm text-slate-500 mb-4">
            Pick at least two candidates to compare on the same split. This runs a quick fit — no cross-validation or tuning — so it stays fast.
          </p>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {(recommendations ?? []).map((rec) => (
              <label
                key={rec.name}
                className={cn(
                  "p-3 border rounded-lg cursor-pointer transition",
                  modelsToCompare.includes(rec.name) ? "border-blue-600 bg-blue-50/60" : "border-slate-200 hover:border-blue-300",
                )}
              >
                <input
                  type="checkbox"
                  checked={modelsToCompare.includes(rec.name)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setModelsToCompare((prev) => [...prev, rec.name]);
                    } else {
                      setModelsToCompare((prev) => prev.filter((m) => m !== rec.name));
                    }
                  }}
                  className="w-4 h-4 accent-blue-600"
                />
                <div className="text-sm font-medium mt-2 text-slate-900">{rec.name}</div>
              </label>
            ))}
          </div>

          <Button
            onClick={handleRunComparison}
            disabled={comparisonLoading || modelsToCompare.length < 2}
            className="mt-4 gap-2"
          >
            {comparisonLoading && <Loader2 className="h-4 w-4 animate-spin" />}
            <Zap className="h-4 w-4" />
            {comparisonLoading ? "Comparing..." : "Run Comparison"}
          </Button>

          {comparisonResults && comparisonResults.length > 0 && (
            <div className="mt-6 overflow-x-auto rounded-xl border border-slate-200">
              <table className="min-w-full divide-y divide-slate-200 text-sm">
                <thead>
                  <tr className="bg-slate-50 text-left text-[11px] uppercase tracking-wider text-slate-500">
                    <th className="px-3 py-2">#</th>
                    <th className="px-3 py-2">Model</th>
                    <th className="px-3 py-2">ROC-AUC</th>
                    <th className="px-3 py-2">Recall</th>
                    <th className="px-3 py-2">Precision</th>
                    <th className="px-3 py-2">F1</th>
                    <th className="px-3 py-2">PR-AUC</th>
                    <th className="px-3 py-2">Accuracy</th>
                    <th className="px-3 py-2">Fit Time</th>
                    <th className="px-3 py-2">Use</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {comparisonResults.map((row, rowIndex) => (
                    <tr key={row.model_name} className={row.model_name === selectedComparisonModel ? "bg-blue-50/60" : undefined}>
                      <td className="px-3 py-3 text-slate-400">{rowIndex + 1}</td>
                      <td className="px-3 py-3 font-medium text-slate-900">{row.model_name}</td>
                      {row.error ? (
                        <td className="px-3 py-3 text-red-600 text-xs" colSpan={6}>{row.error}</td>
                      ) : (
                        <>
                          <td className="px-3 py-3 text-slate-700">{row.roc_auc?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3 text-slate-700">{row.recall?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3 text-slate-700">{row.precision?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3 text-slate-700">{row.f1?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3 text-slate-700">{row.pr_auc?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3 text-slate-700">{row.accuracy?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3 text-slate-700">{row.training_time_s ? `${row.training_time_s.toFixed(2)}s` : "—"}</td>
                        </>
                      )}
                      <td className="px-3 py-3">
                        {!row.error && (
                          <Button
                            variant={row.model_name === selectedModel?.name ? "secondary" : "outline"}
                            size="sm"
                            onClick={() => handleUseComparisonModel(row.model_name)}
                          >
                            {row.model_name === selectedModel?.name ? "Selected" : "Use this model"}
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="p-3 text-xs text-slate-500">
                Pick a model above to move to configuration and run the full training pass (with class balancing / CV) on that model.
              </p>
            </div>
          )}
        </section>
      )}

      {/* ── Direct path: configure + train a single model ── */}
      {flowMode === "direct" && (
      <>

      {/* Data Split — read-only here. The split itself happens in Preprocessing (Step 3); Training just reuses it. */}
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between gap-2 mb-4">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-blue-600" />
            <h2 className="text-base font-semibold text-slate-900">Data Split</h2>
          </div>
          <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10.5px] font-bold uppercase tracking-wide text-slate-500">
            Locked in Preprocessing
          </span>
        </div>

        {displaySplitStats ? (
          <>
            <TrainingSplitBar
              trainN={Number(displaySplitStats.train_n ?? 0)}
              valN={Number(displaySplitStats.val_n ?? 0)}
              testN={Number(displaySplitStats.test_n ?? 0)}
            />
            <p className="mt-4 text-xs text-slate-500">
              Random seed: <span className="font-mono text-slate-700">{config.random_seed}</span>. Feature engineering is re-learned on the training split only, then applied unchanged to validation/test.
              To change the split ratio or seed, go back to Preprocessing.
            </p>
          </>
        ) : (
          <p className="text-sm text-slate-500">
            No split found yet. <Button variant="link" className="px-0 h-auto" onClick={() => navigate({ to: "/data-preparation", search: { tab: "preprocessing" } })}>Run Preprocessing</Button> first — the split happens there and Training reuses it.
          </p>
        )}
      </section>

      {/* Class Distribution Visualization */}
      {splitStats && classLabels.length > 0 && (
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-base font-semibold text-slate-900 mb-4">Class Distribution per Split</h2>
          <div className="h-64 rounded-xl border border-slate-100 bg-slate-50/40 p-2">
            <PlotlyChart
              figure={splitClassFigure}
              style={{ height: "100%", minHeight: "100%" }}
              config={{ displayModeBar: false }}
            />
          </div>
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            {(() => {
              const distributionSource = splitStats?.train_class_counts
                ? Object.entries(splitStats.train_class_counts).map(([label, count]) => ({ label, count: Number(count) }))
                : [];
              const maxCount = Math.max(...distributionSource.map((item) => item.count), 1);
              return distributionSource.map(({ label, count }) => (
                <div key={label} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <div className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-500">Class {label}</div>
                  <div className="mt-2 text-2xl font-bold tabular-nums text-slate-900"><AnimatedNumber value={count} /></div>
                  <div className="mt-2 h-2 rounded-full bg-blue-100 overflow-hidden">
                    <div className="h-full bg-blue-600 transition-all duration-500 ease-out" style={{ width: `${Math.min((count / maxCount) * 100, 100)}%` }} />
                  </div>
                </div>
              ));
            })()}
          </div>
          {classImbalance > 1.5 && (
            <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg flex gap-2">
              <Info className="h-4 w-4 text-amber-600 flex-shrink-0 mt-0.5" />
              <div className="text-xs text-amber-900">
                <strong>Class Imbalance Detected:</strong> {classImbalance.toFixed(2)}x ratio. Balanced class weights are applied automatically during training to compensate.
              </div>
            </div>
          )}
        </section>
      )}

      {/* Training Configuration */}
      <div className="grid gap-5 xl:grid-cols-[1.3fr_0.7fr]">
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-1">
            <Settings2 className="h-4 w-4 text-blue-600" />
            <h2 className="text-base font-semibold text-slate-900">Validation & Tuning Strategy</h2>
          </div>
          <p className="text-xs text-slate-500 mb-4">Configure model robustness and search strategy</p>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <ConfigToggleCard
              icon={BarChart3}
              title="Cross Validation"
              sub="Assess model stability across data splits"
              enabled={config.use_cv}
              onToggle={(value) => setConfig((prev) => ({ ...prev, use_cv: value }))}
              badge={config.use_cv ? `${config.cv_folds}-FOLD CV` : "Disabled"}
            >
              {config.use_cv && (
                <div>
                  <label className="text-xs font-medium text-slate-600 block mb-1">CV Folds</label>
                  <input
                    type="number"
                    min="2"
                    max="10"
                    value={config.cv_folds}
                    onChange={(e) => setConfig((prev) => ({ ...prev, cv_folds: parseInt(e.target.value) || 5 }))}
                    className="w-full px-3 py-1.5 border border-slate-200 rounded-lg text-sm bg-white"
                  />
                </div>
              )}
            </ConfigToggleCard>

            <ConfigToggleCard
              icon={SlidersHorizontal}
              title="Hyperparameter Tuning"
              sub="Randomized search for optimal parameters"
              enabled={config.use_hyperopt}
              onToggle={(value) => setConfig((prev) => ({ ...prev, use_hyperopt: value }))}
              badge={config.use_hyperopt ? "Enabled" : "Disabled"}
            />

            <ConfigToggleCard
              icon={CalendarClock}
              title="Out-of-Time (OOT) Validation"
              sub="Holds out the most recent slice of data by date as an untouched final check"
              enabled={config.use_oot}
              onToggle={(value) => setConfig((prev) => ({ ...prev, use_oot: value }))}
              badge={config.use_oot ? "Enabled" : "Disabled"}
            >
              {config.use_oot && (
                <div>
                  <label className="text-xs font-medium text-slate-600 block mb-1">Origination / Observation Date Column</label>
                  {datetimeColumns.length > 0 ? (
                    <select
                      value={config.date_col ?? ""}
                      onChange={(e) => setConfig((prev) => ({ ...prev, date_col: e.target.value || null }))}
                      className="w-full px-3 py-1.5 border border-slate-200 rounded-lg text-sm bg-white"
                    >
                      <option value="">Auto-detect ({datetimeColumns[0]})</option>
                      {datetimeColumns.map((col) => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  ) : (
                    <p className="text-[11px] text-slate-500">
                      No datetime column was detected in profiling. OOT will be skipped unless one is available at training time.
                    </p>
                  )}
                  <p className="text-[11px] text-slate-500 mt-1">
                    CV (if enabled) and the final fit only ever see development data — the OOT holdout is scored once, after training.
                  </p>
                </div>
              )}
            </ConfigToggleCard>
          </div>
        </section>

        {/* Decision Threshold */}
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-1">
            <Gauge className="h-4 w-4 text-blue-600" />
            <h2 className="text-base font-semibold text-slate-900">Decision Threshold</h2>
          </div>
          <p className="text-xs text-slate-500 mb-4">Prediction cutoff for class assignment</p>

          <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50/60 p-3.5">
            <div>
              <label className="text-sm font-medium text-slate-800">Auto-select (max F1)</label>
              <p className="text-[11px] text-slate-500 mt-0.5">Optimize threshold automatically</p>
            </div>
            <Switch checked={useAutoThreshold} onCheckedChange={setUseAutoThreshold} />
          </div>

          {useAutoThreshold ? (
            <div className="mt-5 text-center">
              <div className="text-4xl font-bold tabular-nums text-blue-600">
                {resolvedThreshold !== null ? <AnimatedNumber value={resolvedThreshold} formatter={(n) => n.toFixed(2)} /> : "—"}
              </div>
              <div className="mt-1 text-[10.5px] font-semibold uppercase tracking-wider text-slate-400">Current Threshold</div>
            </div>
          ) : (
            <div className="mt-5">
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-slate-700">Manual threshold</label>
                <span className="text-lg font-bold tabular-nums text-blue-600">{manualThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={manualThreshold}
                onChange={(e) => setManualThreshold(parseFloat(e.target.value))}
                className="w-full accent-blue-600"
              />
              <div className="flex justify-between text-[10.5px] text-slate-400 mt-1">
                <span>Permissive 0.0</span>
                <span>1.0 Strict</span>
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Predictions with PD ≥ {manualThreshold.toFixed(2)} are classified as default. Lower values
                catch more defaults (higher recall, more false positives); higher values do the opposite.
              </p>
            </div>
          )}

          {resolvedThreshold !== null && (
            <div className="mt-4 rounded-lg bg-slate-900 px-3 py-2.5 text-[11px] text-slate-200">
              Last run: threshold = <strong className="text-white"><AnimatedNumber value={resolvedThreshold} formatter={(n) => n.toFixed(3)} /></strong>
              {useAutoThreshold ? " → maximized F1" : " → manual override"}
            </div>
          )}
        </section>
      </div>

      <div className="grid gap-5 lg:grid-cols-5">
      <section className="lg:col-span-3 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between gap-2 mb-4">
          <div className="flex items-center gap-2">
            <ClipboardList className="h-4 w-4 text-blue-600" />
            <div>
              <h2 className="text-base font-semibold text-slate-900">Development Model Metadata</h2>
              <p className="text-xs text-slate-500">Governance metadata for model documentation</p>
            </div>
          </div>
          <span className="whitespace-nowrap rounded-full bg-amber-100 px-2.5 py-1 text-[10.5px] font-bold uppercase tracking-wide text-amber-700">
            {status || "In Development"}
          </span>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="md:col-span-2">
            <label className="text-xs font-medium text-slate-600 block mb-1.5">Model Name</label>
            <input
              value={trainingModelName ?? selectedModel?.name ?? ""}
              onChange={(e) => setTrainingModelName(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
              placeholder="Business model name"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-600 block mb-1.5">Model Owner</label>
            <input
              value={modelOwner}
              onChange={(e) => setModelOwner(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
              placeholder="Name of the development owner"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-600 block mb-1.5">Business Unit</label>
            <input
              value={businessUnit}
              onChange={(e) => setBusinessUnit(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
              placeholder="Business unit or team"
            />
          </div>
          <div className="md:col-span-2">
            <label className="text-xs font-medium text-slate-600 block mb-1.5">Model Purpose</label>
            <input
              value={modelPurpose}
              onChange={(e) => setModelPurpose(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
              placeholder="Purpose of the development model"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-600 block mb-1.5">Model Version</label>
            <input
              value={modelVersion}
              onChange={(e) => setModelVersion(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
              placeholder="1.0"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-600 block mb-1.5">Model Type</label>
            <input
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
              placeholder="PD (Probability of Default)"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-600 block mb-1.5">Documentation Path</label>
            <input
              value={documentationPath}
              onChange={(e) => setDocumentationPath(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
              placeholder="docs/pd_model_internal_v1_mdd.pdf"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-600 block mb-1.5">Development Date</label>
            <input
              type="date"
              value={developmentDate}
              onChange={(e) => setDevelopmentDate(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-600 block mb-1.5">Status</label>
            <input
              value={status}
              onChange={(e) => setStatus(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
              placeholder="In Development"
            />
          </div>
        </div>
      </section>

      {/* Manual Hyperparameter Controls */}
      <section className="lg:col-span-2 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center gap-2 mb-1">
          <SlidersHorizontal className="h-4 w-4 text-blue-600" />
          <h2 className="text-base font-semibold text-slate-900">Hyperparameter Controls</h2>
        </div>
        <p className="text-xs text-slate-500 mb-4">Manual configuration for {selectedModel.name}</p>
        <div className="grid grid-cols-1 gap-4">
          {selectedModel.name !== "Logistic Regression" && (
            <>
              <HyperparamSlider
                label="Learning Rate"
                value={hyperparams.learning_rate}
                min={0.001}
                max={1}
                step={0.01}
                decimals={3}
                onChange={(v) => setHyperparams((prev) => ({ ...prev, learning_rate: v }))}
              />

              <HyperparamSlider
                label="Max Depth"
                value={hyperparams.max_depth}
                min={1}
                max={30}
                step={1}
                decimals={0}
                onChange={(v) => setHyperparams((prev) => ({ ...prev, max_depth: Math.round(v) || 6 }))}
              />

              <HyperparamSlider
                label="N Estimators"
                value={hyperparams.n_estimators}
                min={10}
                max={1000}
                step={10}
                decimals={0}
                onChange={(v) => setHyperparams((prev) => ({ ...prev, n_estimators: Math.round(v) || 200 }))}
              />

              {selectedModel.name === "XGBoost" && (
                <>
                  <HyperparamSlider
                    label="Subsample"
                    value={hyperparams.subsample}
                    min={0.1}
                    max={1}
                    step={0.1}
                    decimals={1}
                    onChange={(v) => setHyperparams((prev) => ({ ...prev, subsample: v }))}
                  />

                  <HyperparamSlider
                    label="Colsample Bytree"
                    value={hyperparams.colsample_bytree}
                    min={0.1}
                    max={1}
                    step={0.1}
                    decimals={1}
                    onChange={(v) => setHyperparams((prev) => ({ ...prev, colsample_bytree: v }))}
                  />

                  <HyperparamSlider
                    label="Reg Lambda"
                    value={hyperparams.reg_lambda}
                    min={0}
                    max={10}
                    step={0.1}
                    decimals={1}
                    onChange={(v) => setHyperparams((prev) => ({ ...prev, reg_lambda: v }))}
                  />

                  <HyperparamSlider
                    label="Reg Alpha"
                    value={hyperparams.reg_alpha}
                    min={0}
                    max={10}
                    step={0.1}
                    decimals={1}
                    onChange={(v) => setHyperparams((prev) => ({ ...prev, reg_alpha: v }))}
                  />
                </>
              )}
            </>
          )}
        </div>
        <div className="mt-4">
          <button
            type="button"
            onClick={() => {
              setConfig(prev => ({
                ...prev,
                manual_params: hyperparams,
              }));
            }}
            className="w-full rounded-xl border-2 border-dashed border-blue-200 py-2.5 text-xs font-bold text-blue-600 transition hover:bg-blue-50"
          >
            Apply Manual Parameters
          </button>
          <p className="mt-2 text-center text-[11px] text-slate-500">
            {Object.keys(config.manual_params || {}).length > 0 ? "Custom parameters applied to the next run" : "Using model defaults until applied"}
          </p>
        </div>
      </section>
      </div>

      {/* Current Parameters Summary */}
      <Accordion type="single" collapsible className="rounded-2xl border border-slate-200 bg-white shadow-sm">
        <AccordionItem value="current-params" className="border-b-0">
          <AccordionTrigger className="px-6 py-4 text-sm font-semibold text-slate-900 hover:no-underline">
            Current Parameters Summary
          </AccordionTrigger>
          <AccordionContent className="px-6 pt-0 pb-6">
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 text-sm text-slate-700">
              <div><strong className="text-slate-900">Model:</strong> {selectedModel.name}</div>
              <div><strong className="text-slate-900">Random Seed:</strong> {usedTrainingConfig.random_seed}</div>
              <div><strong className="text-slate-900">Train / Val / Test:</strong> {((1 - usedTrainingConfig.test_size - usedTrainingConfig.val_size) * 100).toFixed(0)}% / {(usedTrainingConfig.val_size * 100).toFixed(0)}% / {(usedTrainingConfig.test_size * 100).toFixed(0)}%</div>
              <div><strong className="text-slate-900">CV:</strong> {usedTrainingConfig.use_cv ? `Yes (${usedTrainingConfig.cv_folds} folds)` : "No"}</div>
              <div><strong className="text-slate-900">Hyperopt:</strong> {usedTrainingConfig.use_hyperopt ? "Yes" : "No"}</div>
              <div><strong className="text-slate-900">OOT Validation:</strong> {usedTrainingConfig.use_oot ? `Yes${usedTrainingConfig.date_col ? ` (${usedTrainingConfig.date_col})` : " (auto-detected date column)"}` : "No"}</div>
              <div><strong className="text-slate-900">Feature engineering:</strong> {usedTrainingConfig.use_feature_engineering ? "Enabled" : "Disabled"}</div>
              <div><strong className="text-slate-900">Class Weight:</strong> Automatic (balanced)</div>
              {Object.keys(usedTrainingConfig.manual_params || {}).length > 0 && (
                <div className="md:col-span-2"><strong className="text-slate-900">Manual Params:</strong> <code className="rounded bg-slate-100 px-2 py-1 text-xs">{JSON.stringify(usedTrainingConfig.manual_params)}</code></div>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      </>
      )}

      {/* Training Results */}
      {trainingInfo && (
        <>
          {evaluationMetrics && (
            <section className="rounded-2xl border border-amber-200 bg-amber-50 p-6 shadow-sm">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-amber-900">Regulatory risk warning</p>
                  <p className="text-sm text-amber-900/90">
                    {evaluationMetrics.roc_auc !== undefined && evaluationMetrics.roc_auc < 0.70 && (
                      <>ROC-AUC is below 0.70, which may signal weak discrimination for credit risk. </>
                    )}
                    {evaluationMetrics.recall !== undefined && evaluationMetrics.recall < 0.60 && (
                      <>Recall is below 0.60, which may indicate an elevated missed-default risk. </>
                    )}
                    Review model performance before promotion to production.
                  </p>
                </div>
              </div>
            </section>
          )}

          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-600" />
              <h2 className="text-base font-semibold text-slate-900">Training Summary</h2>
            </div>
            <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-3">
              <div>
                <div className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-500">Hyperparameters</div>
                <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                  {Object.entries(trainingInfo.best_params || {}).map(([k, v]) => (
                    <div key={k} className="contents">
                      <dt className="text-slate-500">{k}</dt>
                      <dd className="text-right font-mono text-xs text-slate-800">{String(v)}</dd>
                    </div>
                  ))}
                  {trainingInfo.training_time_s && (
                    <>
                      <dt className="text-slate-500">training_time_s</dt>
                      <dd className="text-right font-mono text-xs text-slate-800">{trainingInfo.training_time_s.toFixed(2)}</dd>
                    </>
                  )}
                </dl>
              </div>

              <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-gradient-to-br from-slate-900 to-slate-800 p-5 font-mono text-xs text-slate-200">
                <div className="mb-3 font-sans text-sm font-semibold text-white">Model Summary</div>
                <div className="space-y-1.5">
                  <div><strong className="text-white">Model:</strong> {trainingModelName ?? selectedModel.name}</div>
                  <div><strong className="text-white">Training Time:</strong> {trainingInfo.training_time_s?.toFixed(2)}s</div>
                  {trainingInfo.cv_mean && <div><strong className="text-white">CV Mean Score:</strong> {trainingInfo.cv_mean.toFixed(4)}</div>}
                  {trainingInfo.cv_std && <div><strong className="text-white">CV Std Dev:</strong> {trainingInfo.cv_std.toFixed(4)}</div>}
                  {trainingInfo.oot?.oot_available && trainingInfo.oot?.oot_roc_auc !== undefined && (
                    <div><strong className="text-white">OOT ROC-AUC:</strong> {trainingInfo.oot.oot_roc_auc.toFixed(4)}</div>
                  )}
                  {trainingInfo.oot?.oot_available && trainingInfo.oot?.oot_gini !== undefined && (
                    <div><strong className="text-white">OOT Gini:</strong> {trainingInfo.oot.oot_gini.toFixed(4)}</div>
                  )}
                </div>
              </div>
            </div>
          </section>

          {trainingInfo.oot && (
            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-base font-semibold text-slate-900">Out-of-Time (OOT) Validation</h2>
              {trainingInfo.oot.oot_available ? (
                <>
                  <p className="mt-1 text-xs text-slate-500">
                    The most recent {trainingInfo.oot.oot_n?.toLocaleString()} dated row(s) were held out
                    (cutoff: {trainingInfo.oot.cutoff_date}) and scored once against the final model,
                    fit on the remaining {trainingInfo.oot.dev_n?.toLocaleString()} development row(s).
                  </p>
                  <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <div className="text-xs text-slate-500">OOT ROC-AUC</div>
                      <div className="mt-1 text-lg font-semibold text-slate-900">
                        {trainingInfo.oot.oot_roc_auc !== undefined ? trainingInfo.oot.oot_roc_auc.toFixed(4) : "—"}
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <div className="text-xs text-slate-500">OOT Gini</div>
                      <div className="mt-1 text-lg font-semibold text-slate-900">
                        {trainingInfo.oot.oot_gini !== undefined ? trainingInfo.oot.oot_gini.toFixed(4) : "—"}
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <div className="text-xs text-slate-500">OOT Rows Scored</div>
                      <div className="mt-1 text-lg font-semibold text-slate-900">{trainingInfo.oot.oot_n_eval?.toLocaleString() ?? "—"}</div>
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <div className="text-xs text-slate-500">Development Rows</div>
                      <div className="mt-1 text-lg font-semibold text-slate-900">{trainingInfo.oot.dev_n?.toLocaleString() ?? "—"}</div>
                    </div>
                  </div>
                  {trainingInfo.oot.oot_eval_note && (
                    <p className="mt-3 text-xs text-slate-500">{trainingInfo.oot.oot_eval_note}</p>
                  )}
                  {trainingInfo.oot.oot_eval_error && (
                    <p className="mt-3 text-xs text-red-600">Evaluation error: {trainingInfo.oot.oot_eval_error}</p>
                  )}
                </>
              ) : (
                <p className="mt-1 text-xs text-slate-500">
                  {trainingInfo.oot.oot_reason ?? "OOT validation was not run for this training config."}
                </p>
              )}
            </section>
          )}

          {splitStats && (
            <KpiStrip
              tiles={[
                { icon: Database, label: "Total Samples", value: splitStats.total?.toLocaleString() ?? "—", tone: "primary" },
                { icon: Layers, label: "Train", value: String(splitStats.train_n ?? "—"), tone: "primary" },
                { icon: Layers, label: "Validation", value: String(splitStats.val_n ?? "—"), tone: "violet" },
                { icon: Layers, label: "Test", value: String(splitStats.test_n ?? "—"), tone: "emerald" },
              ]}
            />
          )}

          {evaluationMetrics && (
            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-base font-semibold text-slate-900">Evaluation Metrics</h2>
              <div className="mt-4 grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
                {[
                  ["ROC-AUC", evaluationMetrics.roc_auc],
                  ["Recall", evaluationMetrics.recall],
                  ["Precision", evaluationMetrics.precision],
                  ["F1", evaluationMetrics.f1],
                  ["PR-AUC", evaluationMetrics.pr_auc],
                  ["Accuracy", evaluationMetrics.accuracy],
                ].map(([label, value]) => (
                  value !== undefined && (
                    <div key={label} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                      <div className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-500">{label}</div>
                      <div className="mt-2 text-2xl font-bold tabular-nums text-slate-900">{(typeof value === "number" ? value.toFixed(3) : String(value))}</div>
                    </div>
                  )
                ))}
              </div>
            </section>
          )}

        </>
      )}

      </>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3 pt-4">
        <Button variant="outline" onClick={() => navigate({ to: "/data-preparation", search: { tab: "preprocessing" } })} className="gap-2">
          <ArrowLeft className="h-4 w-4" />
          Back to Feature Engineering
        </Button>
        {flowMode === "direct" && (
          <Button
            onClick={handleTrain}
            disabled={loading || !selectedModel}
            className="gap-2"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            {loading ? "Training..." : "Train Model Now"}
          </Button>
        )}
        <Button
          onClick={onProceed}
          disabled={loading || !trainingInfo}
          className="gap-2 ml-auto"
        >
          Proceed to Evaluation
          <ArrowRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Sub-tab 2 — Model Evaluation (moved from evaluation.tsx, unchanged logic)
// ═══════════════════════════════════════════════════════════════════════

function makeCsvRows(metrics: Record<string, any>) {
  return Object.entries(metrics)
    .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
    .map(([key, value]) => [key, String(value)]);
}

function downloadCsv(metrics: Record<string, any>, fileName: string) {
  const rows = makeCsvRows(metrics);
  const csv = ["Metric,Value", ...rows.map(([key, value]) => `${JSON.stringify(key)},${JSON.stringify(value)}`)].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function downloadBase64File(base64: string, fileName: string) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function formatMetricValue(value: unknown) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  if (typeof value === "number") {
    return value.toFixed(3);
  }
  return String(value);
}

const AXIS_LINE_COLOR = "#e2e8f0"; // slate-200 — thin, open axis line (not a boxed border)
const AXIS_TICK_COLOR = "#64748b"; // slate-500
const AXIS_TITLE_COLOR = "#475569"; // slate-600
const AXIS_GRID_COLOR = "#f1f5f9"; // slate-100 — hairline grid, not a heavy ruled grid
const CHART_FONT_FAMILY = "'Inter', system-ui, -apple-system, sans-serif";

function withTitleFont(title: unknown) {
  const text = typeof title === "string" ? title : (title as any)?.text;
  if (!text) return title;
  return {
    text,
    standoff: 16,
    font: { size: 12.5, color: AXIS_TITLE_COLOR, family: CHART_FONT_FAMILY, ...((title as any)?.font ?? {}) },
  };
}

// Re-skins a backend-generated Plotly figure to the redesigned Model
// Evaluation dashboard's visual language. This ONLY touches presentation
// (fonts, grid weight, axis chrome, legend placement, margins) — every
// trace's real x/y data, line/fill colors, dashes, shapes and annotations
// (e.g. the "Best F1 @ τ" marker) come straight from the backend untouched,
// so the chart stays fully dynamic. The backend renders these figures for a
// dark panel (light text); this UI shows them on a white card, so the light
// axis chrome below intentionally wins over whatever the backend sent.
// `plotly.js-basic-dist` (the trimmed bundle this app ships) does not
// register the "histogram" trace module at all — it silently falls back to
// plotting each raw value as its own scatter point, which is what produced
// the jagged/streaky Score Distribution chart. Rather than pull in a larger
// Plotly bundle (a shared-infrastructure change), bin the trace's own real
// x values client-side — same bin count and normalization the backend
// already requested — and hand Plotly a "bar" trace instead, which the
// basic bundle renders correctly. Every bar height is computed straight
// from the real predicted-probability values; nothing is invented.
function histogramTraceToBar(trace: any) {
  if (trace?.type !== "histogram" || !Array.isArray(trace.x) || trace.x.length === 0) return trace;

  const values: number[] = trace.x.filter((v: any) => typeof v === "number" && Number.isFinite(v));
  if (values.length === 0) return trace;

  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return trace;

  const bins = typeof trace.nbinsx === "number" && trace.nbinsx > 0 ? trace.nbinsx : 30;
  const width = (max - min) / bins || 1;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    let idx = Math.floor((v - min) / width);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx] += 1;
  }

  const isDensity = trace.histnorm === "probability density";
  const isProbability = trace.histnorm === "probability";
  const y = counts.map((c) => (isDensity ? c / (values.length * width) : isProbability ? c / values.length : c));
  const x = counts.map((_, i) => min + width * (i + 0.5));
  // Real bin edges + raw sample count per bin, for a tooltip that shows the
  // actual probability range and how many real predictions fall in it —
  // not just the bin's center point.
  const customdata = counts.map((c, i) => [min + width * i, min + width * (i + 1), c]);
  const yLabel = isDensity ? "Density" : isProbability ? "Share" : "Count";

  return {
    type: "bar",
    name: trace.name,
    x,
    y,
    width: width * 0.92,
    marker: { color: trace.marker?.color, line: { width: 0 } },
    opacity: trace.opacity,
    customdata,
    hovertemplate:
      `<b>${trace.name}</b><br>` +
      "Probability: %{customdata[0]:.3f}–%{customdata[1]:.3f}<br>" +
      "Samples: %{customdata[2]}<br>" +
      `${yLabel}: %{y:.3f}<extra></extra>`,
  };
}

// Gives every real (x, y) trace a rich, labeled tooltip built from the
// figure's OWN real axis titles ("False Positive Rate", "Recall", "Decision
// Threshold", ...) instead of Plotly's generic "x: 0.12, y: 0.85" default.
// Traces that already carry a purpose-built hovertemplate (e.g. the binned
// histogram-to-bar conversion below) are left alone. No values are touched.
function withDynamicHover(trace: any, xTitle?: string, yTitle?: string) {
  if (trace?.hovertemplate) return trace;
  if (trace?.type !== "scatter" && trace?.type !== "bar") return trace;
  const xLabel = xTitle || "X";
  const yLabel = yTitle || "Y";
  const nameLine = trace.name ? `<b>${trace.name}</b><br>` : "";
  return {
    ...trace,
    hovertemplate: `${nameLine}${xLabel}: %{x:.3f}<br>${yLabel}: %{y:.3f}<extra></extra>`,
  };
}

function axisTitleText(axis: any): string | undefined {
  const t = axis?.title;
  return typeof t === "string" ? t : t?.text;
}

function enhanceFigureAxes(figure: any) {
  if (!figure) return figure;

  let data = Array.isArray(figure.data) ? figure.data.map(histogramTraceToBar) : figure.data;

  const baseAxis = {
    showline: true,
    linecolor: AXIS_LINE_COLOR,
    linewidth: 1,
    mirror: false,
    ticks: "outside",
    tickcolor: AXIS_LINE_COLOR,
    ticklen: 5,
    tickfont: { size: 12, color: AXIS_TICK_COLOR, family: CHART_FONT_FAMILY },
    showgrid: true,
    gridcolor: AXIS_GRID_COLOR,
    gridwidth: 1,
    zeroline: false,
    // Hover crosshair — a thin dashed guide that tracks the cursor across
    // the plot, so a specific real data point is easy to line up with its
    // axis values. Pure interaction chrome; doesn't touch any trace data.
    showspikes: true,
    spikemode: "across",
    spikethickness: 1,
    spikedash: "dot",
    spikecolor: "#94a3b8",
    spikesnap: "cursor",
    // Lets Plotly grow the margin as needed so the axis title never
    // overlaps long tick labels (paired with `standoff` on the title).
    automargin: true,
  };

  const layout = { ...(figure.layout ?? {}) };

  layout.xaxis = { ...baseAxis, ...(layout.xaxis ?? {}), title: withTitleFont(layout.xaxis?.title) };
  layout.yaxis = { ...baseAxis, ...(layout.yaxis ?? {}), title: withTitleFont(layout.yaxis?.title) };
  if (layout.xaxis2) {
    layout.xaxis2 = { ...baseAxis, ...layout.xaxis2, title: withTitleFont(layout.xaxis2?.title) };
  }
  if (layout.yaxis2) {
    layout.yaxis2 = { ...baseAxis, ...layout.yaxis2, title: withTitleFont(layout.yaxis2?.title) };
  }

  // Real, per-trace tooltip labels — read the figure's own axis titles
  // rather than guessing per chart, so this generalizes to every figure
  // (ROC, PR, Threshold, Score Distribution, Gain/Lift) without hardcoding
  // any chart's identity.
  if (Array.isArray(data)) {
    const primaryX = axisTitleText(layout.xaxis);
    const primaryY = axisTitleText(layout.yaxis);
    const secondaryX = axisTitleText(layout.xaxis2) ?? primaryX;
    const secondaryY = axisTitleText(layout.yaxis2) ?? primaryY;
    data = data.map((trace: any) => {
      const onSecondAxis = trace?.xaxis === "x2" || trace?.yaxis === "y2";
      return withDynamicHover(trace, onSecondAxis ? secondaryX : primaryX, onSecondAxis ? secondaryY : primaryY);
    });

    // A single-subplot figure with more than one real series (e.g. ROC
    // curve + random baseline, or Precision/Recall/F1 vs threshold) gets a
    // unified hover: moving the cursor shows every real series' value at
    // that x position in one tooltip, so trade-offs are readable at a
    // glance. Skipped for split-subplot figures like Gain/Lift, where a
    // shared x position spans two unrelated axes.
    if (!layout.xaxis2 && data.length >= 2 && !layout.hovermode) {
      layout.hovermode = "x unified";
    }
  }

  layout.font = { family: CHART_FONT_FAMILY, size: 12.5, color: AXIS_TICK_COLOR };
  // Generous, consistent margins (deliberately winning over the backend's
  // tighter defaults) — the chart no longer needs top margin for an
  // in-canvas title since the surrounding EvalCard already shows one.
  layout.margin = { t: 16, r: 24, b: 56, l: 60 };
  layout.paper_bgcolor = "rgba(0,0,0,0)";
  layout.plot_bgcolor = "rgba(0,0,0,0)";
  layout.hoverlabel = {
    bgcolor: "#0f172a",
    bordercolor: "#0f172a",
    font: { color: "#f1f5f9", size: 12, family: CHART_FONT_FAMILY },
  };
  layout.transition = { duration: 300, easing: "cubic-in-out" };

  // Redundant with the EvalCard's own header above the chart — drop it so
  // the plot area gets the full card instead of a second title row.
  delete layout.title;

  layout.legend = {
    ...(layout.legend ?? {}),
    orientation: "h",
    x: 1,
    xanchor: "right",
    y: 1.14,
    yanchor: "bottom",
    font: { size: 12, color: AXIS_TITLE_COLOR, family: CHART_FONT_FAMILY },
    bgcolor: "rgba(0,0,0,0)",
    borderwidth: 0,
  };

  // Threshold/best-F1 style reference annotations already carry real,
  // backend-computed values (e.g. "Best F1 @ 0.66") — restyle to match the
  // light card without altering their position or text.
  if (Array.isArray(layout.annotations)) {
    layout.annotations = layout.annotations.map((a: any) => ({
      ...a,
      font: { size: 11, color: a.font?.color ?? "#b91c1c", family: CHART_FONT_FAMILY, ...(a.font ?? {}) },
    }));
  }

  return { ...figure, data, layout };
}

const EVAL_BADGE_TONE: Record<string, string> = {
  primary: "bg-blue-50 text-blue-700",
  amber: "bg-amber-50 text-amber-700",
  emerald: "bg-emerald-50 text-emerald-700",
  rose: "bg-rose-50 text-rose-700",
  violet: "bg-violet-50 text-violet-700",
  slate: "bg-slate-100 text-slate-600",
};

function EvalCard({
  title,
  sub,
  children,
  className,
  badge,
  right,
}: {
  title: string;
  sub?: string;
  children: React.ReactNode;
  className?: string;
  badge?: { text: string; tone?: "primary" | "amber" | "emerald" | "rose" | "violet" | "slate" };
  right?: React.ReactNode;
}) {
  return (
    <div className={cn("rounded-2xl border border-slate-200 bg-white p-6 shadow-sm", className)}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
          {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
        </div>
        {(badge || right) && (
          <div className="flex shrink-0 flex-col items-end gap-1.5">
            {right}
            {badge && (
              <span className={cn("whitespace-nowrap rounded-full px-2.5 py-1 text-[10.5px] font-bold uppercase tracking-wide", EVAL_BADGE_TONE[badge.tone ?? "primary"])}>
                {badge.text}
              </span>
            )}
          </div>
        )}
      </div>
      <div className="mt-4">{children}</div>
    </div>
  );
}

function EvaluationTab({ onBackToTraining }: { onBackToTraining: () => void }) {
  const navigate = useNavigate();
  const { trainingResult, setTrainingResult } = useDataset();

  // Resume where the reviewer left off: if this session has no training/
  // evaluation result yet, pull the last saved /models/evaluate run from the
  // backend and merge its evaluation_metrics/evaluation_data in.
  const { data: resumedEvaluation } = useResumeState<Record<string, any>>("dev_pipeline_log.csv", "evaluation");
  useEffect(() => {
    if (!trainingResult && resumedEvaluation) {
      setTrainingResult(resumedEvaluation as any);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumedEvaluation]);

  const [activeTab, setActiveTab] = useState<"summary" | "roc" | "pr" | "confusion" | "score" | "threshold" | "lift" | "residual" | "temporal">("summary");
  const [temporalDateColumn, setTemporalDateColumn] = useState<string | null>(null);
  const [temporalFrequency, setTemporalFrequency] = useState<string>("Quarterly");

  const evaluationMetrics = trainingResult?.evaluation_metrics && typeof trainingResult.evaluation_metrics === "object"
    ? trainingResult.evaluation_metrics
    : null;
  const evaluationData = trainingResult?.evaluation_data && typeof trainingResult.evaluation_data === "object"
    ? trainingResult.evaluation_data
    : null;
  const modelArtifact = typeof trainingResult?.model_artifact === "string" ? trainingResult.model_artifact : null;
  const taskType = typeof trainingResult?.task_type === "string" ? trainingResult.task_type : "binary";
  const threshold = typeof evaluationMetrics?.threshold_used === "number"
    ? evaluationMetrics.threshold_used
    : typeof evaluationData?.threshold === "number"
      ? evaluationData.threshold
      : 0.5;
  // Present only when the backend auto-picked the threshold (i.e. no explicit
  // override was passed to /models/train) — see evaluate_new.select_best_threshold.
  const thresholdSelection = evaluationMetrics?.threshold_selection ?? evaluationData?.threshold_selection ?? null;
  const isAutoThreshold = thresholdSelection != null;

  const confusion = useMemo(() => {
    const matrix = evaluationMetrics?.confusion_matrix;
    if (Array.isArray(matrix) && matrix.length === 2 && Array.isArray(matrix[0]) && Array.isArray(matrix[1])) {
      return [
        ["True Negative", matrix[0][0], "primary"],
        ["False Positive", matrix[0][1], "warning"],
        ["False Negative", matrix[1][0], "destructive"],
        ["True Positive", matrix[1][1], "primary"],
      ] as const;
    }
    return [
      ["True Negative", 0, "primary"],
      ["False Positive", 0, "warning"],
      ["False Negative", 0, "destructive"],
      ["True Positive", 0, "primary"],
    ] as const;
  }, [evaluationMetrics?.confusion_matrix]);

  const rocFigure = useMemo(() => enhanceFigureAxes(evaluationData?.roc_curve_figure ?? null), [evaluationData?.roc_curve_figure]);
  const prFigure = useMemo(() => enhanceFigureAxes(evaluationData?.pr_curve_figure ?? null), [evaluationData?.pr_curve_figure]);
  const thresholdFigure = useMemo(() => enhanceFigureAxes(evaluationData?.threshold_analysis_figure ?? null), [evaluationData?.threshold_analysis_figure]);
  const scoreDistributionFigure = useMemo(() => enhanceFigureAxes(evaluationData?.score_distribution_figure ?? null), [evaluationData?.score_distribution_figure]);
  const liftChartFigure = useMemo(() => enhanceFigureAxes(evaluationData?.lift_chart_figure ?? null), [evaluationData?.lift_chart_figure]);
  const heteroscedasticityCheck = useMemo(() => evaluationData?.heteroscedasticity_check ?? null, [evaluationData?.heteroscedasticity_check]);
  const temporalAnalysis = useMemo(() => evaluationData?.temporal_analysis ?? null, [evaluationData?.temporal_analysis]);
  const temporalRows = useMemo(() => {
    if (!temporalAnalysis) {
      return [] as Array<{ period: string; actual_rate: number; predicted_rate: number; gap: number; flagged: boolean }>;
    }

    if (temporalFrequency === "Quarterly") {
      return temporalAnalysis.plot_data ?? [];
    }

    return temporalAnalysis.plot_data_by_freq?.[temporalFrequency] ?? [];
  }, [temporalAnalysis, temporalFrequency]);

  const temporalFigure = useMemo(() => {
    if (!temporalRows || temporalRows.length === 0) return null;

    const periods = temporalRows.map((r: any) => r.period);
    const actual = temporalRows.map((r: any) => r.actual_rate);
    const predicted = temporalRows.map((r: any) => r.predicted_rate);
    const gap = temporalRows.map((r: any) => r.gap);
    // Flagged periods (significant actual-vs-predicted drift) get a red
    // marker so they stand out against the rest of the series at a glance.
    const markerColors = temporalRows.map((r: any) => (r.flagged ? "#ef4444" : "#0ea5e9"));

    return enhanceFigureAxes({
      data: [
        {
          type: "bar",
          name: "Gap",
          x: periods,
          y: gap,
          yaxis: "y2",
          marker: { color: "rgba(100,116,139,0.25)" },
          hovertemplate: "%{x}<br>Gap: %{y:.3f}<extra></extra>",
        },
        {
          type: "scatter",
          mode: "lines+markers",
          name: "Actual Rate",
          x: periods,
          y: actual,
          line: { color: "#0ea5e9", width: 3 },
          marker: { color: markerColors, size: 8 },
          hovertemplate: "%{x}<br>Actual: %{y:.3f}<extra></extra>",
        },
        {
          type: "scatter",
          mode: "lines+markers",
          name: "Predicted Rate",
          x: periods,
          y: predicted,
          line: { color: "#6366f1", width: 3, dash: "dot" },
          marker: { size: 6 },
          hovertemplate: "%{x}<br>Predicted: %{y:.3f}<extra></extra>",
        },
      ],
      layout: {
        title: "Actual vs Predicted Default Rate",
        xaxis: { title: "Period" },
        yaxis: { title: "Rate" },
        yaxis2: { title: "Gap", overlaying: "y", side: "right", showgrid: false },
        legend: { orientation: "h", y: 1.18 },
        barmode: "overlay",
      },
    });
  }, [temporalRows]);

  const temporalSummary = useMemo(() => {
    if (!temporalAnalysis) {
      return null;
    }

    if (temporalFrequency === "Quarterly") {
      return temporalAnalysis.summary ?? null;
    }

    return temporalAnalysis.summaries_by_freq?.[temporalFrequency] ?? null;
  }, [temporalAnalysis, temporalFrequency]);

  const summaryMetricRows = useMemo(() => {
    if (taskType !== "binary") {
      return [
        { label: "R²", value: evaluationMetrics?.r2 },
        { label: "MAE", value: evaluationMetrics?.mae },
        { label: "MSE", value: evaluationMetrics?.mse },
        { label: "RMSE", value: evaluationMetrics?.rmse },
      ];
    }

    return [
      { label: "Accuracy", value: evaluationMetrics?.accuracy },
      { label: "Precision", value: evaluationMetrics?.precision },
      { label: "Recall", value: evaluationMetrics?.recall },
      { label: "F1 score", value: evaluationMetrics?.f1 },
      { label: "ROC AUC", value: evaluationMetrics?.roc_auc },
      { label: "PR AUC", value: evaluationMetrics?.pr_auc },
      { label: "KS statistic", value: evaluationMetrics?.ks_statistic },
      { label: "Brier score", value: evaluationMetrics?.brier_score },
    ];
  }, [evaluationMetrics, taskType]);

  const classificationReportRows = useMemo(() => {
    const report = evaluationMetrics?.classification_report;
    if (!report || typeof report !== "object") {
      return [] as Array<{ label: string; precision?: number; recall?: number; f1?: number; support?: number }>;
    }

    return Object.entries(report)
      .filter(([, value]) => value && typeof value === "object" && "precision" in value)
      .map(([label, value]) => {
        const row = value as Record<string, unknown>;
        return {
          label,
          precision: typeof row.precision === "number" ? row.precision : undefined,
          recall: typeof row.recall === "number" ? row.recall : undefined,
          f1: typeof row["f1-score"] === "number" ? row["f1-score"] : undefined,
          support: typeof row.support === "number" ? row.support : undefined,
        };
      });
  }, [evaluationMetrics?.classification_report]);

  useMemo(() => {
    if (!temporalAnalysis || temporalDateColumn) {
      return;
    }
    const available = temporalAnalysis.date_columns ?? [];
    if (available.length > 0) {
      setTemporalDateColumn(available[0]);
    }
  }, [temporalAnalysis, temporalDateColumn]);

  if (!trainingResult) {
    return (
      <div className="rounded-xl border border-border bg-card p-6 text-center">
        <h3 className="text-lg font-semibold">No trained model available</h3>
        <p className="mt-2 text-sm text-muted-foreground">Run training first to populate evaluation metrics and compliance checks.</p>
        <Button onClick={onBackToTraining} className="mt-4">
          Go to Training
        </Button>
      </div>
    );
  }

  // ── Presentation-only derivations. Everything here is arithmetic on data
  // already computed above (confusion, classificationReportRows, temporalSummary,
  // etc.) — no new API calls, no invented numbers. ──────────────────────────
  const [[, tnRaw], [, fpRaw], [, fnRaw], [, tpRaw]] = confusion;
  const tn = Number(tnRaw) || 0;
  const fp = Number(fpRaw) || 0;
  const fn = Number(fnRaw) || 0;
  const tp = Number(tpRaw) || 0;
  const specificity = tn + fp > 0 ? tn / (tn + fp) : null;
  const sensitivity = tp + fn > 0 ? tp / (tp + fn) : null;
  const falseAlarmRate = fp + tn > 0 ? fp / (fp + tn) : null;
  const missRate = fn + tp > 0 ? fn / (fn + tp) : null;
  const confusionTotal = tn + fp + fn + tp;

  const class0Row = classificationReportRows.find((r) => r.label === "0");
  const class1Row = classificationReportRows.find((r) => r.label === "1");
  const classImbalanceRatio = class0Row?.support && class1Row?.support
    ? Math.max(class0Row.support, class1Row.support) / Math.max(1, Math.min(class0Row.support, class1Row.support))
    : null;

  const holdoutSamples = trainingResult.split_stats?.test_n
    ?? classificationReportRows.find((r) => r.label === "weighted avg")?.support
    ?? null;

  const METRIC_VISUALS: Record<string, { icon: React.ComponentType<{ className?: string }>; tone: KpiTone }> = {
    "Accuracy": { icon: Target, tone: "primary" },
    "Precision": { icon: Crosshair, tone: "violet" },
    "Recall": { icon: Activity, tone: "amber" },
    "F1 score": { icon: Gauge, tone: "violet" },
    "ROC AUC": { icon: TrendingUp, tone: "primary" },
    "PR AUC": { icon: BarChart3, tone: "emerald" },
    "KS statistic": { icon: Percent, tone: "emerald" },
    "Brier score": { icon: Target, tone: "rose" },
    "R²": { icon: TrendingUp, tone: "primary" },
    "MAE": { icon: Target, tone: "amber" },
    "MSE": { icon: Target, tone: "rose" },
    "RMSE": { icon: Target, tone: "violet" },
  };
  const heroKpiTiles = summaryMetricRows
    .filter((m): m is { label: string; value: number } => typeof m.value === "number")
    .map((m) => ({
      icon: METRIC_VISUALS[m.label]?.icon ?? Activity,
      label: m.label,
      value: <AnimatedNumber value={m.value} formatter={(n) => n.toFixed(3)} />,
      tone: (METRIC_VISUALS[m.label]?.tone ?? "primary") as KpiTone,
    }));

  return (
    <div className="space-y-6">
      {/* ── Executive header ─────────────────────────────────────────── */}
      <div className="rounded-2xl border border-slate-200 bg-gradient-to-r from-slate-900 via-indigo-900 to-blue-800 p-6 text-white shadow-[0_16px_36px_rgba(15,23,42,0.16)]">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="text-[11px] uppercase tracking-[0.2em] text-indigo-200">Step 5 · Model Development</div>
            <h3 className="mt-2 text-2xl font-semibold">Model Evaluation</h3>
            <p className="mt-2 max-w-2xl text-sm text-slate-200">
              Evaluate how {trainingResult.model_name || "the trained model"} performs on held-out data it never saw during training.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {holdoutSamples != null && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-300/40 bg-white/90 px-3 py-1 text-xs font-semibold text-slate-700">
                {Number(holdoutSamples).toLocaleString()} samples
              </span>
            )}
            <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-300/50 bg-emerald-400/15 px-3 py-1 text-xs font-semibold text-emerald-200">
              <CheckCircle2 className="h-3 w-3" /> Evaluation Ready
            </span>
            {trainingResult.model_name && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-sky-300/50 bg-sky-400/15 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-sky-100">
                {trainingResult.model_name}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* ── KPI strip — every real evaluation metric Aegis returned ──── */}
      {heroKpiTiles.length > 0 && <KpiStrip tiles={heroKpiTiles} />}

      {/* ── Action row ───────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div>
          <div className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-400">Model Evaluation</div>
          <div className="text-base font-semibold text-slate-900">Performance Report</div>
          <p className="mt-0.5 text-xs text-slate-500">Explore how the model performs on data it never saw during training.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {evaluationMetrics && (
            <Button variant="outline" onClick={() => downloadCsv(evaluationMetrics, "evaluation_metrics.csv")} className="gap-2">
              <FileDown className="h-4 w-4" />
              Download metrics CSV
            </Button>
          )}
          {modelArtifact && (
            <Button variant="outline" onClick={() => downloadBase64File(modelArtifact, "trained_model.pkl")} className="gap-2">
              <FileDown className="h-4 w-4" />
              Download model artifact
            </Button>
          )}
        </div>
      </div>

      {/* ── Sub-nav ──────────────────────────────────────────────────── */}
      <div className="flex flex-wrap gap-1 overflow-x-auto rounded-2xl border border-slate-200 bg-white p-1.5 shadow-sm">
        {(["summary","roc","pr","confusion","score","threshold","lift","residual","temporal"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            className={cn(
              "whitespace-nowrap rounded-xl px-3.5 py-2 text-[12.5px] font-semibold transition",
              activeTab === t ? "bg-blue-600 text-white shadow-sm" : "text-slate-500 hover:bg-slate-50 hover:text-slate-700",
            )}
          >
            {t === "summary" ? "Summary" : t === "roc" ? "ROC Curve" : t === "pr" ? "PR Curve" : t === "confusion" ? "Confusion" : t === "score" ? "Score Dist" : t === "threshold" ? "Thresholds" : t === "lift" ? "Lift" : t === "residual" ? "Residuals" : "Temporal"}
          </button>
        ))}
      </div>

      {/* ── Summary ──────────────────────────────────────────────────── */}
      {activeTab === "summary" && (
        <div className="space-y-5">
          <div className="grid gap-5 xl:grid-cols-[1.6fr_1fr]">
            <EvalCard
              title="Threshold Insight"
              sub={isAutoThreshold ? "Auto-selected to maximise F1 across 99 candidates" : "Manually specified decision threshold"}
            >
              {isAutoThreshold ? (
                <>
                  <div className="mb-4 flex items-start gap-2.5 rounded-xl border border-blue-200 bg-blue-50 p-3.5">
                    <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-600 text-[10px] font-bold text-white">i</div>
                    <p className="text-xs leading-relaxed text-blue-900">
                      Threshold <strong className="font-mono">{threshold.toFixed(2)}</strong> was chosen automatically — it's the cut-off
                      that maximizes F1 ({(thresholdSelection.f1 * 100).toFixed(1)}%) on this test data, out of
                      99 candidate thresholds swept from 0.01 to 0.99. At this cut-off: <strong>precision {(thresholdSelection.precision * 100).toFixed(1)}%</strong>,{" "}
                      <strong>recall {(thresholdSelection.recall * 100).toFixed(1)}%</strong>.
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    {[
                      { label: "Threshold", value: threshold, formatter: (n: number) => n.toFixed(2), sub: "selected", tone: "text-blue-600" },
                      { label: "Max F1", value: thresholdSelection.f1 * 100, formatter: (n: number) => `${n.toFixed(1)}%`, sub: "at threshold", tone: "text-emerald-600" },
                      { label: "Precision", value: thresholdSelection.precision * 100, formatter: (n: number) => `${n.toFixed(1)}%`, sub: `at τ=${threshold.toFixed(2)}`, tone: "text-cyan-600" },
                      { label: "Recall", value: thresholdSelection.recall * 100, formatter: (n: number) => `${n.toFixed(1)}%`, sub: `at τ=${threshold.toFixed(2)}`, tone: "text-amber-600" },
                    ].map((t) => (
                      <div key={t.label} className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-center">
                        <div className={cn("font-mono text-xl font-bold", t.tone)}><AnimatedNumber value={t.value} formatter={t.formatter} /></div>
                        <div className="mt-1 text-[10px] font-bold uppercase tracking-wider text-slate-500">{t.label}</div>
                        <div className="text-[9.5px] text-slate-400">{t.sub}</div>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <p className="text-sm text-slate-500">
                  Threshold <strong className="font-mono text-slate-900">{threshold.toFixed(2)}</strong> was manually specified for this training run rather than auto-selected.
                </p>
              )}
            </EvalCard>

            <EvalCard title="Model Status">
              <div className="space-y-2.5">
                {[
                  { k: "Model", v: trainingResult.model_name || "—" },
                  { k: "Dataset", v: "Hold-out Test" },
                  { k: "Samples", v: holdoutSamples != null ? Number(holdoutSamples).toLocaleString() : "—", mono: true },
                  { k: "Threshold", v: `${threshold.toFixed(2)}${isAutoThreshold ? " (auto)" : ""}`, mono: true },
                  ...(isAutoThreshold ? [{ k: "Candidates", v: "99 swept", mono: false }] : []),
                ].map((row) => (
                  <div key={row.k} className="flex items-center justify-between border-b border-slate-100 pb-2.5 text-sm last:border-0 last:pb-0">
                    <span className="text-slate-500">{row.k}</span>
                    <span className={cn("font-semibold text-slate-900", row.mono && "font-mono")}>{row.v}</span>
                  </div>
                ))}
              </div>
              {classImbalanceRatio !== null && (
                <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-3">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-amber-700">⚠ Risk Signal</div>
                  <p className="mt-1 text-xs leading-relaxed text-amber-900">
                    Class imbalance {classImbalanceRatio.toFixed(1)}:1 in the hold-out set.
                    {typeof evaluationMetrics?.recall === "number" && ` Recall ${(evaluationMetrics.recall * 100).toFixed(1)}% on the minority class.`}
                  </p>
                </div>
              )}
            </EvalCard>
          </div>

          <EvalCard title="Classification Report" sub="Per-class precision, recall, F1 and support — hold-out set">
            {classificationReportRows.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">
                      <th className="py-2 pr-3">#</th>
                      <th className="py-2 pr-3">Class</th>
                      <th className="py-2 pr-3 text-right">Precision</th>
                      <th className="py-2 pr-3 text-right">Recall</th>
                      <th className="py-2 pr-3 text-right">F1</th>
                      <th className="py-2 pr-3 text-right">Support</th>
                      <th className="py-2">Recall bar</th>
                    </tr>
                  </thead>
                  <tbody>
                    {classificationReportRows.map((row, rowIndex) => {
                      const isAvg = row.label.includes("avg");
                      return (
                        <tr key={row.label} className={cn("border-t border-slate-100", isAvg && "bg-slate-50")}>
                          <td className="py-2.5 pr-3 text-slate-400">{isAvg ? "" : rowIndex + 1}</td>
                          <td className="py-2.5 pr-3">
                            <div className="flex items-center gap-2">
                              {!isAvg && <span className={cn("h-2 w-2 rounded-full", row.label === "0" ? "bg-emerald-500" : "bg-blue-500")} />}
                              <span className={cn("font-medium", isAvg ? "text-slate-700" : "text-slate-900")}>{row.label}</span>
                            </div>
                          </td>
                          <td className="py-2.5 pr-3 text-right font-mono font-semibold text-slate-800">{formatMetricValue(row.precision)}</td>
                          <td className="py-2.5 pr-3 text-right font-mono font-semibold text-slate-800">{formatMetricValue(row.recall)}</td>
                          <td className="py-2.5 pr-3 text-right font-mono font-semibold text-slate-800">{formatMetricValue(row.f1)}</td>
                          <td className="py-2.5 pr-3 text-right font-mono text-slate-500">{row.support?.toLocaleString() ?? "—"}</td>
                          <td className="py-2.5 pr-0" style={{ minWidth: 120 }}>
                            {!isAvg && typeof row.recall === "number" && (
                              <div className="h-1.5 rounded-full bg-slate-100">
                                <div className={cn("h-full rounded-full transition-all duration-500 ease-out", row.label === "0" ? "bg-emerald-500" : "bg-blue-500")} style={{ width: `${Math.min(row.recall * 100, 100)}%` }} />
                              </div>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-slate-500">Classification report will appear after a completed binary classification run.</p>
            )}
          </EvalCard>
        </div>
      )}

      {/* ── ROC Curve ────────────────────────────────────────────────── */}
      {activeTab === "roc" && (
        <EvalCard
          title="ROC Curve"
          sub={rocFigure ? "Receiver Operating Characteristic — discrimination ability across all thresholds" : "Probability output unavailable"}
          right={typeof evaluationMetrics?.roc_auc === "number" ? (
            <div className="text-right">
              <div className="font-mono text-2xl font-bold text-blue-600">{evaluationMetrics.roc_auc.toFixed(3)}</div>
              <div className="text-[10px] uppercase tracking-wider text-slate-400">ROC-AUC</div>
            </div>
          ) : undefined}
        >
          {rocFigure ? (
            <PlotlyChart figure={rocFigure} style={{ minHeight: 380 }} config={{ displayModeBar: false }} />
          ) : (
            <p className="text-sm text-slate-500">ROC curve requires probability predictions from a binary classification model.</p>
          )}
        </EvalCard>
      )}

      {/* ── PR Curve ─────────────────────────────────────────────────── */}
      {activeTab === "pr" && (
        <EvalCard
          title="Precision–Recall Curve"
          sub={prFigure ? "Trade-off between precision and recall across thresholds" : "Probability output unavailable"}
          right={typeof evaluationMetrics?.pr_auc === "number" ? (
            <div className="text-right">
              <div className="font-mono text-2xl font-bold text-amber-600">{evaluationMetrics.pr_auc.toFixed(3)}</div>
              <div className="text-[10px] uppercase tracking-wider text-slate-400">PR-AUC</div>
            </div>
          ) : undefined}
        >
          {prFigure ? (
            <PlotlyChart figure={prFigure} style={{ minHeight: 380 }} config={{ displayModeBar: false }} />
          ) : (
            <p className="text-sm text-slate-500">Precision–Recall curve requires probability predictions from a binary classification model.</p>
          )}
        </EvalCard>
      )}

      {/* ── Confusion Matrix ─────────────────────────────────────────── */}
      {activeTab === "confusion" && (
        <EvalCard
          title="Confusion Matrix"
          sub={`Threshold ${threshold.toFixed(2)}${isAutoThreshold ? " (auto-selected, max F1)" : ""}${confusionTotal ? ` — ${confusionTotal.toLocaleString()} hold-out samples` : ""}`}
          badge={{ text: `τ = ${threshold.toFixed(2)}`, tone: "primary" }}
        >
          <div className="flex flex-wrap items-center justify-center gap-8">
            <div>
              <div className="ml-16 mb-1.5 flex gap-2 sm:ml-20">
                <div className="w-28 text-center text-[10.5px] font-bold uppercase tracking-wider text-slate-400 sm:w-32">Predicted 0</div>
                <div className="w-28 text-center text-[10.5px] font-bold uppercase tracking-wider text-slate-400 sm:w-32">Predicted 1</div>
              </div>
              <div className="space-y-2">
                {[
                  { rowLabel: "Actual 0", cells: [
                    { label: "True Negative", value: tn, sub: "Correctly rejected", border: "border-indigo-200", bg: "bg-indigo-50", hoverBg: "hover:bg-indigo-100", text: "text-indigo-700" },
                    { label: "False Positive", value: fp, sub: "Type I error", border: "border-amber-200", bg: "bg-amber-50", hoverBg: "hover:bg-amber-100", text: "text-amber-700" },
                  ] },
                  { rowLabel: "Actual 1", cells: [
                    { label: "False Negative", value: fn, sub: "Missed defaults", border: "border-red-200", bg: "bg-red-50", hoverBg: "hover:bg-red-100", text: "text-red-700" },
                    { label: "True Positive", value: tp, sub: "Correctly flagged", border: "border-emerald-200", bg: "bg-emerald-50", hoverBg: "hover:bg-emerald-100", text: "text-emerald-700" },
                  ] },
                ].map((row) => (
                  <div key={row.rowLabel} className="flex items-center gap-2">
                    <div className="w-16 shrink-0 text-right text-[10.5px] font-bold uppercase tracking-wider text-slate-400">{row.rowLabel}</div>
                    {row.cells.map((cell) => {
                      const pct = confusionTotal ? (cell.value / confusionTotal) * 100 : null;
                      return (
                        <div
                          key={cell.label}
                          title={`${cell.label}: ${cell.value.toLocaleString()} sample(s)${pct !== null ? ` (${pct.toFixed(1)}% of hold-out set)` : ""}`}
                          className={cn(
                            "flex h-24 w-28 cursor-default flex-col items-center justify-center gap-1 rounded-xl border text-center transition-all duration-150 sm:h-28 sm:w-32",
                            cell.border, cell.bg, cell.hoverBg, "hover:-translate-y-0.5 hover:shadow-md",
                          )}
                        >
                          <div className={cn("text-[8.5px] font-bold uppercase tracking-wider opacity-70", cell.text)}>{cell.label}</div>
                          <div className={cn("font-mono text-2xl font-extrabold sm:text-3xl", cell.text)}><AnimatedNumber value={cell.value} /></div>
                          <div className={cn("text-[9px] opacity-60", cell.text)}>
                            {cell.sub}{pct !== null && <> · <AnimatedNumber value={pct} formatter={(n) => `${n.toFixed(1)}%`} /></>}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>

            <div className="flex min-w-[180px] flex-col gap-2.5">
              {[
                { label: "Specificity", note: "TN / (TN+FP)", value: specificity, tone: "text-indigo-700", bg: "bg-indigo-50" },
                { label: "Sensitivity", note: "TP / (TP+FN)", value: sensitivity, tone: "text-emerald-700", bg: "bg-emerald-50" },
                { label: "False Alarm", note: `${fp.toLocaleString()} false positives`, value: falseAlarmRate, tone: "text-amber-700", bg: "bg-amber-50" },
                { label: "Miss Rate", note: `${fn.toLocaleString()} missed defaults`, value: missRate, tone: "text-red-700", bg: "bg-red-50" },
              ].map((s) => (
                <div key={s.label} className={cn("flex items-center justify-between rounded-xl p-3", s.bg)}>
                  <div>
                    <div className={cn("text-[10px] font-bold uppercase tracking-wider", s.tone)}>{s.label}</div>
                    <div className={cn("mt-0.5 text-[10px] opacity-70", s.tone)}>{s.note}</div>
                  </div>
                  <div className={cn("font-mono text-xl font-extrabold", s.tone)}>
                    {typeof s.value === "number" ? <AnimatedNumber value={s.value * 100} formatter={(n) => `${n.toFixed(1)}%`} /> : "—"}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </EvalCard>
      )}

      {/* ── Score Distribution ───────────────────────────────────────── */}
      {activeTab === "score" && (
        <EvalCard title="Score Distribution" sub="Predicted probability distribution by class — hold-out set" badge={{ text: `τ = ${threshold.toFixed(2)}`, tone: "primary" }}>
          {scoreDistributionFigure ? (
            <PlotlyChart figure={scoreDistributionFigure} style={{ minHeight: 380 }} config={{ displayModeBar: false }} />
          ) : (
            <p className="text-sm text-slate-500">Score distribution requires probability predictions from a binary classification model.</p>
          )}
        </EvalCard>
      )}

      {/* ── Threshold Analysis ───────────────────────────────────────── */}
      {activeTab === "threshold" && (
        <EvalCard
          title="Threshold Analysis"
          sub="Precision · Recall · F1 across all decision cut-offs"
          badge={isAutoThreshold ? { text: `Best F1 @ ${threshold.toFixed(2)}`, tone: "rose" } : undefined}
        >
          {thresholdFigure ? (
            <PlotlyChart figure={thresholdFigure} style={{ minHeight: 380 }} config={{ displayModeBar: false }} />
          ) : (
            <p className="text-sm text-slate-500">Threshold analysis requires probability predictions from a binary classification model.</p>
          )}
        </EvalCard>
      )}

      {/* ── Lift / Gain ──────────────────────────────────────────────── */}
      {activeTab === "lift" && (
        <EvalCard title="Gain &amp; Lift" sub="Cumulative gain and lift by decile — model vs. random selection">
          {liftChartFigure ? (
            <PlotlyChart figure={liftChartFigure} style={{ minHeight: 380 }} config={{ displayModeBar: false }} />
          ) : (
            <p className="text-sm text-slate-500">Gain and lift charts require probability predictions from a binary classification model.</p>
          )}
        </EvalCard>
      )}

      {/* ── Residuals ────────────────────────────────────────────────── */}
      {activeTab === "residual" && (
        <div className="space-y-5">
          <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr_1fr]">
            <div className={cn("rounded-2xl border p-5", heteroscedasticityCheck?.risk_flag ? "border-amber-200 bg-amber-50" : "border-slate-200 bg-white shadow-sm")}>
              <div className="flex items-start gap-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-amber-100 text-amber-700">
                  <AlertTriangle className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-wider text-amber-700">Risk Signal</div>
                  <div className="mt-1 text-base font-semibold text-slate-900">{heteroscedasticityCheck?.risk_flag ?? "Not available"}</div>
                  <p className="mt-1 text-xs leading-relaxed text-slate-600">Overall heteroscedasticity risk flag for this model, derived from residual variance across score bins.</p>
                </div>
              </div>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-white p-5 text-center shadow-sm">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Residual Correlation</div>
              <div className="mt-2 font-mono text-3xl font-extrabold text-blue-600">
                {typeof heteroscedasticityCheck?.spearman_abs_resid_vs_score === "number"
                  ? <AnimatedNumber value={heteroscedasticityCheck.spearman_abs_resid_vs_score} formatter={(n) => n.toFixed(3)} />
                  : formatMetricValue(heteroscedasticityCheck?.spearman_abs_resid_vs_score)}
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-slate-500">Correlation between prediction error size and predicted score</p>
              {typeof heteroscedasticityCheck?.spearman_abs_resid_vs_score === "number" && (
                <div className="mt-3 h-1.5 rounded-full bg-slate-100">
                  <div className="h-full rounded-full bg-blue-600 transition-all duration-500 ease-out" style={{ width: `${Math.min(Math.abs(heteroscedasticityCheck.spearman_abs_resid_vs_score) * 100, 100)}%` }} />
                </div>
              )}
            </div>
            <div className="rounded-2xl border border-slate-200 bg-white p-5 text-center shadow-sm">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Variance Ratio</div>
              <div className="mt-2 font-mono text-3xl font-extrabold text-amber-600">
                {typeof heteroscedasticityCheck?.variance_ratio === "number" ? <AnimatedNumber value={heteroscedasticityCheck.variance_ratio} formatter={(n) => `${n.toFixed(1)}×`} /> : "—"}
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-slate-500">Spread of residual variance across score bins</p>
            </div>
          </div>

          <EvalCard title="Residual Variance by Score Bin" sub="Heteroscedasticity-style residual checks across equal-count bins">
            {Array.isArray(heteroscedasticityCheck?.bin_variance) && heteroscedasticityCheck.bin_variance.length > 0 ? (
              (() => {
                const bins = heteroscedasticityCheck.bin_variance;
                const maxVar = Math.max(...bins.map((r: any) => Number(r.residual_variance) || 0), 0.0001);
                const relColor = (v: number) => (v / maxVar > 0.85 ? "#f59e0b" : v / maxVar > 0.4 ? "#2563eb" : "#10b981");
                const residualBinFigure = enhanceFigureAxes({
                  data: [{
                    type: "bar",
                    x: bins.map((r: any, i: number) => r.score_bin ?? `Bin ${i + 1}`),
                    y: bins.map((r: any) => Number(r.residual_variance) || 0),
                    marker: { color: bins.map((r: any) => relColor(Number(r.residual_variance) || 0)), line: { width: 0 } },
                    customdata: bins.map((r: any) => [r.n != null ? Number(r.n) : null]),
                    hovertemplate: "<b>Bin %{x}</b><br>Residual variance: %{y:.4f}<br>Samples: %{customdata[0]}<extra></extra>",
                  }],
                  layout: {
                    xaxis: { title: "Predicted-score bin" },
                    yaxis: { title: "Residual variance" },
                    showlegend: false,
                  },
                });
                return (
                  <>
                    <div className="mb-5 h-56">
                      <PlotlyChart figure={residualBinFigure} style={{ height: "100%" }} config={{ displayModeBar: false }} />
                    </div>
                    <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">
                          <th className="py-2 pr-3">#</th>
                          <th className="py-2 pr-3">Bin</th>
                          <th className="py-2 pr-3 text-right">Count</th>
                          <th className="py-2 pr-3 text-right">Variance</th>
                          <th className="py-2">Relative</th>
                        </tr>
                      </thead>
                      <tbody>
                        {heteroscedasticityCheck.bin_variance.map((row: Record<string, any>, index: number) => {
                          const variance = Number(row.residual_variance) || 0;
                          const pct = variance / maxVar;
                          const barColor = pct > 0.85 ? "bg-amber-500" : pct > 0.4 ? "bg-blue-500" : "bg-emerald-500";
                          return (
                            <tr key={row.score_bin ?? index} className="border-t border-slate-100">
                              <td className="py-2.5 pr-3 text-slate-400">{index + 1}</td>
                              <td className="py-2.5 pr-3 font-mono text-slate-700">{row.score_bin ?? `Bin ${index + 1}`}</td>
                              <td className="py-2.5 pr-3 text-right font-mono text-slate-700">{row.n != null ? Number(row.n).toLocaleString() : "—"}</td>
                              <td className="py-2.5 pr-3 text-right font-mono font-semibold text-slate-800">{formatMetricValue(row.residual_variance)}</td>
                              <td className="py-2.5" style={{ minWidth: 140 }}>
                                <div className="flex items-center gap-2">
                                  <div className="h-1.5 flex-1 rounded-full bg-slate-100">
                                    <div className={cn("h-full rounded-full transition-all duration-500 ease-out", barColor)} style={{ width: `${Math.min(pct * 100, 100)}%` }} />
                                  </div>
                                  <span className="w-9 text-right text-[10.5px] font-bold text-slate-500">{(pct * 100).toFixed(0)}%</span>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                    </div>
                  </>
                );
              })()
            ) : (
              <p className="text-sm text-slate-500">Residual variance breakdown is not available for this model.</p>
            )}
          </EvalCard>
        </div>
      )}

      {/* ── Temporal ─────────────────────────────────────────────────── */}
      {activeTab === "temporal" && (
        <EvalCard title="Actual vs. Predicted Default Rate" sub="Temporal stability by period">
          <div className="space-y-4">
            {temporalAnalysis?.frequency_options?.length ? (
              <div className="flex justify-end">
                <div className="inline-flex gap-1 rounded-full bg-slate-100 p-1">
                  {temporalAnalysis.frequency_options.map((f: string) => (
                    <button
                      key={f}
                      onClick={() => setTemporalFrequency(f)}
                      className={cn(
                        "rounded-full px-3.5 py-1.5 text-xs font-semibold transition",
                        temporalFrequency === f ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700",
                      )}
                    >
                      {f}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            {temporalFigure ? (
              <PlotlyChart figure={temporalFigure} style={{ minHeight: 380 }} config={{ displayModeBar: false }} />
            ) : (
              <p className="text-sm text-slate-500">No temporal data available for the selected period.</p>
            )}

            {temporalSummary && (temporalSummary.max_overestimation_period || typeof temporalSummary.n_periods_flagged === "number") && (
              <div className="grid gap-3 sm:grid-cols-2">
                {temporalSummary.max_overestimation_period && (
                  <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-3.5">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-indigo-700">Largest Overestimation</div>
                    <p className="mt-1 text-xs leading-relaxed text-indigo-900">
                      {temporalSummary.max_overestimation_period} — predicted exceeded actual by <AnimatedNumber value={temporalSummary.max_overestimation_gap * 100} formatter={(n) => `${n.toFixed(1)}pp`} />.
                    </p>
                  </div>
                )}
                {typeof temporalSummary.n_periods_flagged === "number" && typeof temporalSummary.n_periods_total === "number" && (
                  <div className="rounded-xl border border-amber-200 bg-amber-50 p-3.5">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-amber-700">Stability Signal</div>
                    <p className="mt-1 text-xs leading-relaxed text-amber-900">
                      <AnimatedNumber value={temporalSummary.n_periods_flagged} /> of <AnimatedNumber value={temporalSummary.n_periods_total} /> periods flagged (&gt;5pp gap)
                      {typeof temporalSummary.mean_absolute_gap === "number" && (
                        <> — mean absolute gap <AnimatedNumber value={temporalSummary.mean_absolute_gap * 100} formatter={(n) => `${n.toFixed(1)}pp`} /></>
                      )}.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </EvalCard>
      )}

      {/* ── Action bar ───────────────────────────────────────────────── */}
      <div className="flex gap-3 pt-2">
        <Button variant="outline" onClick={onBackToTraining} className="gap-2">
          <ArrowLeft className="h-4 w-4" />
          Back to Training
        </Button>
        <Button onClick={() => navigate({ to: "/explainability" })} className="gap-2 ml-auto">
          Proceed to Explainability
          <ArrowRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
