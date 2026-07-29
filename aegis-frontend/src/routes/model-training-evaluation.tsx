import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { PageHeader } from "@/components/app-shell";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CheckCircle2, Loader2, ArrowLeft, ArrowRight, Zap, BarChart3, AlertCircle, Info, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { useDataset } from "@/lib/app-context";
import { formUpload } from "@/lib/api";
import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import PlotlyChart from "@/components/plotly-chart";

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
  const [trainingModelName, setTrainingModelName] = useState<string | null>(trainingResult?.model_name ?? selectedModel?.name ?? null);
  const [trainingConfigResult, setTrainingConfigResult] = useState<Record<string, any> | null>(trainingResult?.training_config ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelComparison, setModelComparison] = useState<boolean>(false);
  const [modelsToCompare, setModelsToCompare] = useState<string[]>(compareModels ?? []);

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
    trainForm.append("model_name", modelName);
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
      model_name: modelName,
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

    const traces = classLabels.map((label, index) => ({
      type: "bar",
      name: label === "0" ? "Class 0" : label === "1" ? "Class 1" : label,
      x,
      y: splitClassData.map((row) => Number(row[label] ?? 0)),
      marker: {
        color: index === 0 ? "#22c55e" : index === 1 ? "#ef4444" : index === 2 ? "#3b82f6" : "#f59e0b",
      },
      hovertemplate: "%{x}<br>%{y:.0f}<extra></extra>",
    }));

    return {
      data: traces,
      layout: {
        barmode: "stack",
        margin: { t: 10, r: 12, l: 0, b: 0 },
        xaxis: { title: "", automargin: true, tickfont: { color: "#9ca3af" }, linecolor: "#9ca3af" },
        yaxis: { title: "", automargin: true, tickfont: { color: "#9ca3af" }, linecolor: "#9ca3af" },
        legend: { orientation: "h", y: 1.2 },
      },
    };
  }, [splitClassData, classLabels]);

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

  return (
    <div className="space-y-8">
      <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="text-sm font-semibold">Step 5 — Model Training</div>
        <p className="mt-2 text-sm text-muted-foreground">
          {selectedModel ? `Configure and train the ${selectedModel.name} model with optimized parameters.` : "Models ranked by suitability for your dataset — pick one to train."}
        </p>
      </div>

      {error && (
        <div className="rounded-xl border border-destructive bg-destructive/5 p-4 text-sm text-destructive flex gap-3">
          <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
          <div>{error}</div>
        </div>
      )}

      {modelsError && (
        <div className="rounded-xl border border-destructive bg-destructive/5 p-4 text-sm text-destructive flex gap-3">
          <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
          <div>{modelsError}</div>
        </div>
      )}

      {/* ── Model Selection (merged in from models.tsx) ──────────────────── */}
      {datasetSummary && (
        <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
          <div className="text-sm text-muted-foreground">Dataset summary</div>
          <div className="mt-2 text-lg font-semibold">
            Dataset: {datasetSummary.sampleCount.toLocaleString()} samples × {datasetSummary.featureCount} features | Imbalance ratio: {datasetSummary.imbalanceRatio.toFixed(1)}:1
          </div>
        </section>
      )}

      {modelsLoading && (
        <div className="rounded-xl border border-border bg-card p-6 text-center text-sm text-muted-foreground">
          Loading model recommendations...
        </div>
      )}

      {transformedModels.length === 0 && !modelsLoading && !modelsError && (
        <div className="rounded-xl border border-border bg-card p-6 text-center text-sm text-muted-foreground">
          No model recommendations available.
        </div>
      )}

      {transformedModels.length > 0 && (
        <>
          <section>
            <h2 className="mb-4 text-base font-semibold">Recommended Model</h2>
            <p className="mb-4 text-sm text-muted-foreground">
              Chosen automatically from your dataset's characteristics (size, missingness, class imbalance,
              feature mix, correlation, and non-linearity). You can still compare or switch to any other model below.
            </p>
            {recommendedModel && (
              <div className="rounded-2xl border border-primary bg-primary/5 p-6 shadow-elegant">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-2">
                    {recommendedModel.icon ? <span className="text-xl">{recommendedModel.icon}</span> : null}
                    <h3 className="text-lg font-semibold">{recommendedModel.name}</h3>
                  </div>
                  <div className="inline-flex w-fit items-center gap-1 rounded-full bg-primary/10 px-2 py-1 text-[11px] font-medium text-primary">
                    <CheckCircle2 className="h-3 w-3" /> Recommended
                  </div>
                </div>

                <p className="mt-3 text-sm text-muted-foreground">{recommendedModel.description}</p>

                {recommendedModel.reasons?.length ? (
                  <div className="mt-4">
                    <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Why this model, for this dataset</div>
                    <ul className="mt-2 space-y-1.5 text-sm text-foreground/90">
                      {recommendedModel.reasons.map((reason, i) => (
                        <li key={i} className="flex gap-2">
                          <span className="text-primary">•</span>
                          <span>{reason}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            )}
          </section>

          <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <h2 className="mb-4 text-base font-semibold">Select model to train</h2>
            <select
              value={selectedModel?.name ?? transformedModels[0]?.name}
              onChange={(e) => {
                const next = transformedModels.find((m) => m.name === e.target.value);
                if (next) setSelectedModel(next);
              }}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
            >
              {transformedModels.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name}{recommendedModel?.name === model.name ? " (recommended)" : ""}
                </option>
              ))}
            </select>
            <p className="mt-2 text-sm text-muted-foreground">
              The recommended model is pre-selected. You can override it and train any other model directly,
              or compare a few side by side below.
            </p>
          </section>

          {recommendationTaskType === "binary" && (
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant border-l-4 border-blue-500/80 bg-blue-500/10">
              <h2 className="text-base font-semibold text-blue-900">Credit Risk Evaluation Strategy</h2>
              <p className="mt-3 text-sm text-blue-900/90">
                In credit risk, <strong>Recall</strong> is the most critical metric because failing to identify a truly risky customer (false negative) is far more costly than incorrectly flagging a safe one.
              </p>
              <p className="mt-2 text-sm text-blue-900/90">
                We optimize for: <strong>ROC-AUC → Recall → PR-AUC → F1</strong>
              </p>
            </section>
          )}
        </>
      )}

      {!selectedModel && (
        <div className="rounded-xl border border-border bg-card p-6 text-center text-sm text-muted-foreground">
          Select a model above to configure and run training.
        </div>
      )}

      {selectedModel && (
      <>

      {/* ── Choose a path: compare candidates first, or go straight to configuring/training the selected model ── */}
      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <h2 className="text-base font-semibold mb-1">How do you want to proceed?</h2>
        <p className="text-sm text-muted-foreground mb-4">
          Run a quick, lightweight comparison across a few candidates first, or go straight to configuring and training the selected model.
        </p>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <button
            type="button"
            onClick={() => setFlowMode("compare")}
            className={`flex flex-col items-start gap-1 rounded-xl border p-4 text-left transition ${
              flowMode === "compare" ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
            }`}
          >
            <div className="flex items-center gap-2 font-semibold">
              <Zap className="h-4 w-4 text-primary" /> Compare models first
            </div>
            <p className="text-xs text-muted-foreground">
              Quick fit on a few candidates (no CV, no tuning) so you can pick a winner before committing to a full training run.
            </p>
          </button>
          <button
            type="button"
            onClick={() => setFlowMode("direct")}
            className={`flex flex-col items-start gap-1 rounded-xl border p-4 text-left transition ${
              flowMode === "direct" ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
            }`}
          >
            <div className="flex items-center gap-2 font-semibold">
              <CheckCircle2 className="h-4 w-4 text-primary" /> Configure &amp; train "{selectedModel.name}"
            </div>
            <p className="text-xs text-muted-foreground">
              Go straight to class balancing, cross-validation, and a full training run on the selected model.
            </p>
          </button>
        </div>
      </section>

      {/* ── Compare path ── */}
      {flowMode === "compare" && (
        <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="h-5 w-5 text-primary" />
            <h2 className="text-base font-semibold">Model Comparison</h2>
          </div>
          <p className="text-sm text-muted-foreground mb-4">
            Pick at least two candidates to compare on the same split. This runs a quick fit — no cross-validation or tuning — so it stays fast.
          </p>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {(recommendations ?? []).map((rec) => (
              <label
                key={rec.name}
                className={`p-3 border rounded-lg cursor-pointer transition ${
                  modelsToCompare.includes(rec.name)
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50"
                }`}
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
                  className="w-4 h-4"
                />
                <div className="text-sm font-medium mt-2">{rec.name}</div>
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
            <div className="mt-6 overflow-x-auto">
              <table className="min-w-full divide-y divide-border text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-muted-foreground">
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
                <tbody className="divide-y divide-border">
                  {comparisonResults.map((row, rowIndex) => (
                    <tr key={row.model_name} className={row.model_name === selectedComparisonModel ? "bg-primary/5" : undefined}>
                      <td className="px-3 py-3 text-muted-foreground">{rowIndex + 1}</td>
                      <td className="px-3 py-3 font-medium">{row.model_name}</td>
                      {row.error ? (
                        <td className="px-3 py-3 text-destructive text-xs" colSpan={6}>{row.error}</td>
                      ) : (
                        <>
                          <td className="px-3 py-3">{row.roc_auc?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3">{row.recall?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3">{row.precision?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3">{row.f1?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3">{row.pr_auc?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3">{row.accuracy?.toFixed(3) ?? "—"}</td>
                          <td className="px-3 py-3">{row.training_time_s ? `${row.training_time_s.toFixed(2)}s` : "—"}</td>
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
              <p className="mt-3 text-xs text-muted-foreground">
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
      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="h-5 w-5 text-primary" />
          <h2 className="text-base font-semibold">Data Split (locked in during Preprocessing)</h2>
        </div>

        {displaySplitStats ? (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {(() => {
                const totalN = displaySplitStats.total
                  ?? (Number(displaySplitStats.train_n ?? 0) + Number(displaySplitStats.val_n ?? 0) + Number(displaySplitStats.test_n ?? 0));
                const pct = (n: number) => (totalN ? (n / totalN) * 100 : 0);
                return (
                  <>
                    <div>
                      <div className="text-xs uppercase tracking-wider text-muted-foreground">Train</div>
                      <div className="mt-1 text-2xl font-semibold tabular-nums">{pct(displaySplitStats.train_n).toFixed(0)}%</div>
                      <p className="text-xs text-muted-foreground mt-1">{Number(displaySplitStats.train_n ?? 0).toLocaleString()} samples</p>
                    </div>
                    <div>
                      <div className="text-xs uppercase tracking-wider text-muted-foreground">Validation</div>
                      <div className="mt-1 text-2xl font-semibold tabular-nums">{pct(displaySplitStats.val_n).toFixed(0)}%</div>
                      <p className="text-xs text-muted-foreground mt-1">{Number(displaySplitStats.val_n ?? 0).toLocaleString()} samples</p>
                    </div>
                    <div>
                      <div className="text-xs uppercase tracking-wider text-muted-foreground">Test</div>
                      <div className="mt-1 text-2xl font-semibold tabular-nums">{pct(displaySplitStats.test_n).toFixed(0)}%</div>
                      <p className="text-xs text-muted-foreground mt-1">{Number(displaySplitStats.test_n ?? 0).toLocaleString()} samples</p>
                    </div>
                  </>
                );
              })()}
            </div>
            <p className="mt-4 text-sm text-muted-foreground">
              Random seed: <span className="font-mono">{config.random_seed}</span>. Feature engineering is re-learned on the training split only, then applied unchanged to validation/test.
              To change the split ratio or seed, go back to Preprocessing.
            </p>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            No split found yet. <Button variant="link" className="px-0 h-auto" onClick={() => navigate({ to: "/data-preparation", search: { tab: "preprocessing" } })}>Run Preprocessing</Button> first — the split happens there and Training reuses it.
          </p>
        )}
      </section>

      {/* Class Distribution Visualization */}
{splitStats && classLabels.length > 0 && (
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold mb-4">Class Distribution per Split</h2>
              <div className="h-72">
                <div className="h-72">
                  <PlotlyChart figure={splitClassFigure} style={{ height: "100%", minHeight: "100%" }} />
                </div>
              </div>
              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                {(() => {
                  const distributionSource = splitStats?.train_class_counts
                    ? Object.entries(splitStats.train_class_counts).map(([label, count]) => ({ label, count: Number(count) }))
                    : [];
              const maxCount = Math.max(...distributionSource.map((item) => item.count), 1);
              return distributionSource.map(({ label, count }) => (
                <div key={label} className="bg-muted rounded-lg p-4">
                  <div className="text-xs uppercase tracking-wider text-muted-foreground">Class {label}</div>
                  <div className="mt-2 text-2xl font-semibold">{count.toLocaleString()}</div>
                  <div className="mt-2 h-2 bg-primary/20 rounded-full overflow-hidden">
                    <div className="h-full bg-primary" style={{ width: `${Math.min((count / maxCount) * 100, 100)}%` }} />
                  </div>
                </div>
              ));
            })()}
          </div>
          {classImbalance > 1.5 && (
            <div className="mt-4 p-3 bg-orange-500/10 border border-orange-200 rounded-lg flex gap-2">
              <Info className="h-4 w-4 text-orange-600 flex-shrink-0 mt-0.5" />
              <div className="text-xs text-orange-900">
                <strong>Class Imbalance Detected:</strong> {classImbalance.toFixed(2)}x ratio. Balanced class weights are applied automatically during training to compensate.
              </div>
            </div>
          )}
        </section>
      )}

      {/* Cross Validation & Hyperparameter Tuning */}
      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <h2 className="text-base font-semibold mb-4">Cross Validation & Hyperparameter Tuning</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium">Enable Cross Validation</label>
              <p className="text-xs text-muted-foreground mt-1">Assess model stability across data splits</p>
            </div>
            <input
              type="checkbox"
              checked={config.use_cv}
              onChange={(e) => setConfig(prev => ({ ...prev, use_cv: e.target.checked }))}
              className="w-5 h-5"
            />
          </div>

          {config.use_cv && (
            <div>
              <label className="text-sm font-medium block mb-2">CV Folds</label>
              <input
                type="number"
                min="2"
                max="10"
                value={config.cv_folds}
                onChange={(e) => setConfig(prev => ({ ...prev, cv_folds: parseInt(e.target.value) || 5 }))}
                className="w-full px-3 py-2 border border-input rounded-lg text-sm bg-background"
              />
            </div>
          )}

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium">Enable Hyperparameter Tuning</label>
              <p className="text-xs text-muted-foreground mt-1">Randomized search for optimal parameters</p>
            </div>
            <input
              type="checkbox"
              checked={config.use_hyperopt}
              onChange={(e) => setConfig(prev => ({ ...prev, use_hyperopt: e.target.checked }))}
              className="w-5 h-5"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium">Enable Out-of-Time (OOT) Validation</label>
              <p className="text-xs text-muted-foreground mt-1">
                Holds out the most recent slice of data by date as an untouched final check
              </p>
            </div>
            <input
              type="checkbox"
              checked={config.use_oot}
              onChange={(e) => setConfig(prev => ({ ...prev, use_oot: e.target.checked }))}
              className="w-5 h-5"
            />
          </div>

          {config.use_oot && (
            <div>
              <label className="text-sm font-medium block mb-2">Origination / Observation Date Column</label>
              {datetimeColumns.length > 0 ? (
                <select
                  value={config.date_col ?? ""}
                  onChange={(e) => setConfig(prev => ({ ...prev, date_col: e.target.value || null }))}
                  className="w-full px-3 py-2 border border-input rounded-lg text-sm bg-background"
                >
                  <option value="">Auto-detect ({datetimeColumns[0]})</option>
                  {datetimeColumns.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              ) : (
                <p className="text-xs text-muted-foreground">
                  No datetime column was detected in profiling. OOT will be skipped unless one is available at training time.
                </p>
              )}
              <p className="text-xs text-muted-foreground mt-1">
                CV (if enabled) and the final fit only ever see development data — the OOT holdout is scored once, after training.
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Decision Threshold */}
      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <h2 className="text-base font-semibold mb-4">Decision Threshold</h2>
        <div className="flex items-center justify-between">
          <div>
            <label className="text-sm font-medium">Auto-select (maximize F1)</label>
            <p className="text-xs text-muted-foreground mt-1">
              Lets the backend pick the PD cut-off that maximizes F1 on the hold-out set.
              Turn off to pin a specific value instead.
            </p>
          </div>
          <input
            type="checkbox"
            checked={useAutoThreshold}
            onChange={(e) => setUseAutoThreshold(e.target.checked)}
            className="w-5 h-5"
          />
        </div>

        {!useAutoThreshold && (
          <div className="mt-5">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium">Manual threshold</label>
              <span className="text-sm font-mono tabular-nums">{manualThreshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={manualThreshold}
              onChange={(e) => setManualThreshold(parseFloat(e.target.value))}
              className="w-full accent-primary"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>0.00</span>
              <span>0.50</span>
              <span>1.00</span>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Predictions with PD ≥ {manualThreshold.toFixed(2)} are classified as default. Lower values
              catch more defaults (higher recall, more false positives); higher values do the opposite.
            </p>
          </div>
        )}

        {resolvedThreshold !== null && (
          <p className="text-xs text-muted-foreground mt-4 pt-4 border-t border-border">
            Last training run used a threshold of <strong>{resolvedThreshold.toFixed(4)}</strong>
            {useAutoThreshold ? " (auto-selected to maximize F1)" : " (manual override)"}.
          </p>
        )}
      </section>

      {/* Manual Hyperparameter Controls */}
      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <h2 className="text-base font-semibold mb-4">Manual Hyperparameter Controls</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {selectedModel.name !== "Logistic Regression" && (
            <>
              <div>
                <label className="text-sm font-medium block mb-2">Learning Rate</label>
                <input
                  type="number"
                  min="0.001"
                  max="1"
                  step="0.01"
                  value={hyperparams.learning_rate}
                  onChange={(e) => setHyperparams(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
                  className="w-full px-3 py-2 border border-input rounded-lg text-sm bg-background"
                />
              </div>

              <div>
                <label className="text-sm font-medium block mb-2">Max Depth</label>
                <input
                  type="number"
                  min="1"
                  max="30"
                  value={hyperparams.max_depth}
                  onChange={(e) => setHyperparams(prev => ({ ...prev, max_depth: parseInt(e.target.value) || 6 }))}
                  className="w-full px-3 py-2 border border-input rounded-lg text-sm bg-background"
                />
              </div>

              <div>
                <label className="text-sm font-medium block mb-2">N Estimators</label>
                <input
                  type="number"
                  min="10"
                  max="1000"
                  step="10"
                  value={hyperparams.n_estimators}
                  onChange={(e) => setHyperparams(prev => ({ ...prev, n_estimators: parseInt(e.target.value) || 200 }))}
                  className="w-full px-3 py-2 border border-input rounded-lg text-sm bg-background"
                />
              </div>

              {selectedModel.name === "XGBoost" && (
                <>
                  <div>
                    <label className="text-sm font-medium block mb-2">Subsample</label>
                    <input
                      type="number"
                      min="0.1"
                      max="1"
                      step="0.1"
                      value={hyperparams.subsample}
                      onChange={(e) => setHyperparams(prev => ({ ...prev, subsample: parseFloat(e.target.value) }))}
                      className="w-full px-3 py-2 border border-input rounded-lg text-sm bg-background"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium block mb-2">Colsample Bytree</label>
                    <input
                      type="number"
                      min="0.1"
                      max="1"
                      step="0.1"
                      value={hyperparams.colsample_bytree}
                      onChange={(e) => setHyperparams(prev => ({ ...prev, colsample_bytree: parseFloat(e.target.value) }))}
                      className="w-full px-3 py-2 border border-input rounded-lg text-sm bg-background"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium block mb-2">Reg Lambda</label>
                    <input
                      type="number"
                      min="0"
                      max="10"
                      step="0.1"
                      value={hyperparams.reg_lambda}
                      onChange={(e) => setHyperparams(prev => ({ ...prev, reg_lambda: parseFloat(e.target.value) }))}
                      className="w-full px-3 py-2 border border-input rounded-lg text-sm bg-background"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium block mb-2">Reg Alpha</label>
                    <input
                      type="number"
                      min="0"
                      max="10"
                      step="0.1"
                      value={hyperparams.reg_alpha}
                      onChange={(e) => setHyperparams(prev => ({ ...prev, reg_alpha: parseFloat(e.target.value) }))}
                      className="w-full px-3 py-2 border border-input rounded-lg text-sm bg-background"
                    />
                  </div>
                </>
              )}
            </>
          )}
        </div>
        <Button
          variant="outline"
          size="sm"
          className="mt-4"
          onClick={() => {
            setConfig(prev => ({
              ...prev,
              manual_params: hyperparams,
            }));
          }}
        >
          Apply Manual Parameters
        </Button>
      </section>

      {/* Current Parameters Summary */}
      <Accordion type="single" collapsible className="rounded-xl border border-border bg-card shadow-elegant">
        <AccordionItem value="current-params">
          <AccordionTrigger className="px-6 py-4">
            Current Parameters Summary
          </AccordionTrigger>
          <AccordionContent className="px-6 pt-0 pb-6">
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 text-sm">
              <div><strong>Model:</strong> {selectedModel.name}</div>
              <div><strong>Random Seed:</strong> {usedTrainingConfig.random_seed}</div>
              <div><strong>Train / Val / Test:</strong> {((1 - usedTrainingConfig.test_size - usedTrainingConfig.val_size) * 100).toFixed(0)}% / {(usedTrainingConfig.val_size * 100).toFixed(0)}% / {(usedTrainingConfig.test_size * 100).toFixed(0)}%</div>
              <div><strong>CV:</strong> {usedTrainingConfig.use_cv ? `Yes (${usedTrainingConfig.cv_folds} folds)` : "No"}</div>
              <div><strong>Hyperopt:</strong> {usedTrainingConfig.use_hyperopt ? "Yes" : "No"}</div>
              <div><strong>OOT Validation:</strong> {usedTrainingConfig.use_oot ? `Yes${usedTrainingConfig.date_col ? ` (${usedTrainingConfig.date_col})` : " (auto-detected date column)"}` : "No"}</div>
              <div><strong>Feature engineering:</strong> {usedTrainingConfig.use_feature_engineering ? "Enabled" : "Disabled"}</div>
              <div><strong>Class Weight:</strong> Automatic (balanced)</div>
              {Object.keys(usedTrainingConfig.manual_params || {}).length > 0 && (
                <div className="md:col-span-2"><strong>Manual Params:</strong> <code className="rounded bg-background px-2 py-1 text-xs">{JSON.stringify(usedTrainingConfig.manual_params)}</code></div>
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
            <section className="rounded-xl border border-warning/40 bg-warning/10 p-6 shadow-elegant">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-warning-foreground flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-warning-foreground">Regulatory risk warning</p>
                  <p className="text-sm text-warning-foreground">
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

          <section className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold">Hyperparameters</h2>
              <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                {Object.entries(trainingInfo.best_params || {}).map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-muted-foreground">{k}</dt>
                    <dd className="text-right font-mono text-xs">{String(v)}</dd>
                  </div>
                ))}
                {trainingInfo.training_time_s && (
                  <>
                    <dt className="text-muted-foreground">training_time_s</dt>
                    <dd className="text-right font-mono text-xs">{trainingInfo.training_time_s.toFixed(2)}</dd>
                  </>
                )}
              </dl>
            </div>

            <div className="lg:col-span-2 rounded-xl border border-border bg-sidebar p-6 font-mono text-xs text-sidebar-foreground shadow-elegant">
              <div className="mb-3 flex items-center justify-between">
                <span className="font-sans text-sm font-semibold text-sidebar-foreground">Model Summary</span>
              </div>
              <div className="space-y-1.5">
                <div><strong>Model:</strong> {trainingModelName ?? selectedModel.name}</div>
                <div><strong>Training Time:</strong> {trainingInfo.training_time_s?.toFixed(2)}s</div>
                {trainingInfo.cv_mean && <div><strong>CV Mean Score:</strong> {trainingInfo.cv_mean.toFixed(4)}</div>}
                {trainingInfo.cv_std && <div><strong>CV Std Dev:</strong> {trainingInfo.cv_std.toFixed(4)}</div>}
                {trainingInfo.oot?.oot_available && trainingInfo.oot?.oot_roc_auc !== undefined && (
                  <div><strong>OOT ROC-AUC:</strong> {trainingInfo.oot.oot_roc_auc.toFixed(4)}</div>
                )}
                {trainingInfo.oot?.oot_available && trainingInfo.oot?.oot_gini !== undefined && (
                  <div><strong>OOT Gini:</strong> {trainingInfo.oot.oot_gini.toFixed(4)}</div>
                )}
              </div>
            </div>
          </section>

          {trainingInfo.oot && (
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold">Out-of-Time (OOT) Validation</h2>
              {trainingInfo.oot.oot_available ? (
                <>
                  <p className="mt-1 text-xs text-muted-foreground">
                    The most recent {trainingInfo.oot.oot_n?.toLocaleString()} dated row(s) were held out
                    (cutoff: {trainingInfo.oot.cutoff_date}) and scored once against the final model,
                    fit on the remaining {trainingInfo.oot.dev_n?.toLocaleString()} development row(s).
                  </p>
                  <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                    <div className="rounded-lg border border-border bg-background p-3">
                      <div className="text-xs text-muted-foreground">OOT ROC-AUC</div>
                      <div className="mt-1 text-lg font-semibold">
                        {trainingInfo.oot.oot_roc_auc !== undefined ? trainingInfo.oot.oot_roc_auc.toFixed(4) : "—"}
                      </div>
                    </div>
                    <div className="rounded-lg border border-border bg-background p-3">
                      <div className="text-xs text-muted-foreground">OOT Gini</div>
                      <div className="mt-1 text-lg font-semibold">
                        {trainingInfo.oot.oot_gini !== undefined ? trainingInfo.oot.oot_gini.toFixed(4) : "—"}
                      </div>
                    </div>
                    <div className="rounded-lg border border-border bg-background p-3">
                      <div className="text-xs text-muted-foreground">OOT Rows Scored</div>
                      <div className="mt-1 text-lg font-semibold">{trainingInfo.oot.oot_n_eval?.toLocaleString() ?? "—"}</div>
                    </div>
                    <div className="rounded-lg border border-border bg-background p-3">
                      <div className="text-xs text-muted-foreground">Development Rows</div>
                      <div className="mt-1 text-lg font-semibold">{trainingInfo.oot.dev_n?.toLocaleString() ?? "—"}</div>
                    </div>
                  </div>
                  {trainingInfo.oot.oot_eval_note && (
                    <p className="mt-3 text-xs text-muted-foreground">{trainingInfo.oot.oot_eval_note}</p>
                  )}
                  {trainingInfo.oot.oot_eval_error && (
                    <p className="mt-3 text-xs text-destructive">Evaluation error: {trainingInfo.oot.oot_eval_error}</p>
                  )}
                </>
              ) : (
                <p className="mt-1 text-xs text-muted-foreground">
                  {trainingInfo.oot.oot_reason ?? "OOT validation was not run for this training config."}
                </p>
              )}
            </section>
          )}

          {splitStats && (
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold">Data Split Statistics</h2>
              <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                <div>
                  <dt className="text-[11px] uppercase tracking-wider text-muted-foreground">Total Samples</dt>
                  <dd className="mt-1 text-lg font-semibold">{splitStats.total?.toLocaleString()}</dd>
                </div>
                <div>
                  <dt className="text-[11px] uppercase tracking-wider text-muted-foreground">Train / Val / Test</dt>
                  <dd className="mt-1 text-sm">{splitStats.train_n} / {splitStats.val_n} / {splitStats.test_n}</dd>
                </div>
              </div>
            </section>
          )}

          {evaluationMetrics && (
            <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <h2 className="text-base font-semibold">Evaluation Metrics</h2>
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
                    <div key={label} className="rounded-lg border border-border p-4">
                      <div className="text-xs uppercase tracking-wider text-muted-foreground">{label}</div>
                      <div className="mt-2 text-2xl font-semibold">{(typeof value === "number" ? value.toFixed(3) : String(value))}</div>
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

const AXIS_LINE_COLOR = "#334155"; // slate-700
const AXIS_TICK_COLOR = "#1e293b"; // slate-800
const AXIS_GRID_COLOR = "#cbd5e1"; // slate-300

function withTitleFont(title: unknown) {
  const text = typeof title === "string" ? title : (title as any)?.text;
  if (!text) return title;
  return {
    text,
    standoff: 14,
    font: { size: 15, color: AXIS_TICK_COLOR, ...((title as any)?.font ?? {}) },
  };
}

// Makes axis lines, ticks, and titles bolder and higher-contrast so charts are easier to read.
function enhanceFigureAxes(figure: any) {
  if (!figure) return figure;

  const baseAxis = {
    showline: true,
    linecolor: AXIS_LINE_COLOR,
    linewidth: 2,
    mirror: true,
    ticks: "outside",
    tickcolor: AXIS_LINE_COLOR,
    ticklen: 6,
    tickfont: { size: 13, color: AXIS_TICK_COLOR },
    showgrid: true,
    gridcolor: AXIS_GRID_COLOR,
    gridwidth: 1,
    zerolinecolor: AXIS_LINE_COLOR,
    zerolinewidth: 2,
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
  // Backend figures are styled for a dark card (light-gray text) — this UI
  // renders charts on a light card, so our high-contrast color must win
  // over whatever the backend sent, not the other way around.
  layout.font = { ...(layout.font ?? {}), size: 13, color: AXIS_TICK_COLOR };
  layout.margin = { t: 30, r: 20, b: 60, l: 65, ...(layout.margin ?? {}) };

  // Same light-on-dark mismatch applies to the chart title and legend,
  // neither of which inherit cleanly from layout.font since the backend
  // sets its own explicit colors on them.
  if (layout.title) {
    const titleObj = typeof layout.title === "string" ? { text: layout.title } : { ...layout.title };
    layout.title = { ...titleObj, font: { size: 16, ...(titleObj.font ?? {}), color: AXIS_TICK_COLOR } };
  }
  layout.legend = {
    ...(layout.legend ?? {}),
    font: { size: 12, color: AXIS_TICK_COLOR },
    bgcolor: "rgba(255,255,255,0.85)",
    bordercolor: AXIS_GRID_COLOR,
    borderwidth: 1,
  };

  return { ...figure, layout };
}

function toneClasses(tone: string) {
  if (tone === "warning") return "border-amber-500/40 bg-amber-500/5";
  if (tone === "destructive") return "border-destructive/40 bg-destructive/5";
  return "border-primary/40 bg-primary/5";
}

function EvalCard({ title, sub, children, className }: { title: string; sub?: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-border bg-card p-6 shadow-elegant ${className ?? ""}`.trim()}>
      <h3 className="text-sm font-semibold">{title}</h3>
      {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
      <div className="mt-4">{children}</div>
    </div>
  );
}

function EvaluationTab({ onBackToTraining }: { onBackToTraining: () => void }) {
  const navigate = useNavigate();
  const { trainingResult } = useDataset();
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

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold">Model Evaluation</div>
          <p className="mt-1 text-sm text-muted-foreground">Explore how the model performs on data it never saw during training.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {evaluationMetrics && (
            <Button variant="outline" onClick={() => downloadCsv(evaluationMetrics, "evaluation_metrics.csv")} className="gap-2">
              <Download className="h-4 w-4" />
              Download metrics CSV
            </Button>
          )}
          {modelArtifact && (
            <Button variant="outline" onClick={() => downloadBase64File(modelArtifact, "trained_model.pkl")} className="gap-2">
              <Download className="h-4 w-4" />
              Download model artifact
            </Button>
          )}
        </div>
      </div>

      <section className="grid grid-cols-1 gap-6">
          <div className="space-y-6">
          <div className="flex items-center gap-2">
            <div className="inline-flex rounded-lg bg-background/60 p-1">
              {(["summary","roc","pr","confusion","score","threshold","lift","residual","temporal"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setActiveTab(t)}
                  className={`px-3 py-1 text-sm rounded ${activeTab===t?"bg-primary text-white":"text-muted-foreground"}`}>
                  {t === "summary" ? "Summary" : t === "roc" ? "ROC Curve" : t === "pr" ? "PR Curve" : t === "confusion" ? "Confusion" : t === "score" ? "Score Dist" : t === "threshold" ? "Thresholds" : t === "lift" ? "Lift" : t === "residual" ? "Residuals" : "Temporal"}
                </button>
              ))}
            </div>
          </div>

          {activeTab === "summary" ? (
            <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
              <EvalCard
                title="Hold-out performance"
                sub={`Key evaluation metrics from the test split · decision threshold ${threshold.toFixed(2)}${isAutoThreshold ? " (auto-selected, max F1)" : ""}`}
              >
                {isAutoThreshold && (
                  <div className="mb-4 flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-500/10 p-3">
                    <Info className="h-4 w-4 text-blue-600 flex-shrink-0 mt-0.5" />
                    <p className="text-xs text-blue-900">
                      Threshold <strong>{threshold.toFixed(2)}</strong> was chosen automatically — it's the cut-off
                      that maximizes F1 ({(thresholdSelection.f1 * 100).toFixed(1)}%) on this test data, out of
                      99 candidate thresholds swept from 0.01 to 0.99. At this cut-off: precision{" "}
                      {(thresholdSelection.precision * 100).toFixed(1)}%, recall {(thresholdSelection.recall * 100).toFixed(1)}%.
                    </p>
                  </div>
                )}
                <div className="grid gap-3 sm:grid-cols-2">
                  {summaryMetricRows.map((metric) => (
                    <div key={metric.label} className="rounded-lg border border-border bg-background/70 p-3">
                      <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{metric.label}</div>
                      <div className="mt-1 text-lg font-semibold tabular-nums">{formatMetricValue(metric.value)}</div>
                    </div>
                  ))}
                </div>
              </EvalCard>

              <EvalCard title="Classification report" sub="Per-class precision, recall, F1 and support">
                <div className="overflow-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted-foreground">
                        <th className="py-2 pr-3">#</th>
                        <th className="py-2 pr-3">Class</th>
                        <th className="py-2 pr-3">Precision</th>
                        <th className="py-2 pr-3">Recall</th>
                        <th className="py-2 pr-3">F1</th>
                        <th className="py-2">Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {classificationReportRows.length > 0 ? (
                        classificationReportRows.map((row, rowIndex) => (
                          <tr key={row.label} className="border-b border-border/60 text-sm">
                            <td className="py-2 pr-3 text-muted-foreground">{rowIndex + 1}</td>
                            <td className="py-2 pr-3 font-medium">{row.label}</td>
                            <td className="py-2 pr-3">{formatMetricValue(row.precision)}</td>
                            <td className="py-2 pr-3">{formatMetricValue(row.recall)}</td>
                            <td className="py-2 pr-3">{formatMetricValue(row.f1)}</td>
                            <td className="py-2">{formatMetricValue(row.support)}</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={6} className="py-3 text-sm text-muted-foreground">Classification report will appear after a completed binary classification run.</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </EvalCard>
            </div>
          ) : (
          <div className="grid gap-6">
            {activeTab === "roc" && (
              <EvalCard title="ROC curve" sub={evaluationMetrics?.roc_auc ? `AUC ${evaluationMetrics.roc_auc}` : "Probability output unavailable"}>
                {rocFigure ? (
                  <PlotlyChart figure={rocFigure} style={{ minHeight: 360 }} />
                ) : (
                  <p className="text-sm text-muted-foreground">ROC curve requires probability predictions from a binary classification model.</p>
                )}
              </EvalCard>
            )}

            {activeTab === "pr" && (
              <EvalCard title="Precision–Recall" sub={evaluationMetrics?.pr_auc ? `Average precision ${evaluationMetrics.pr_auc}` : "Probability output unavailable"}>
                {prFigure ? (
                  <PlotlyChart figure={prFigure} style={{ minHeight: 360 }} />
                ) : (
                  <p className="text-sm text-muted-foreground">Precision–Recall curve requires probability predictions from a binary classification model.</p>
                )}
              </EvalCard>
            )}

            {activeTab === "confusion" && (
              <EvalCard title="Confusion matrix" sub={`Threshold ${threshold.toFixed(2)}${isAutoThreshold ? " (auto-selected, max F1)" : ""}`}>
                <div className="flex justify-center overflow-auto py-2">
                  <table className="border-separate" style={{ borderSpacing: 8 }}>
                    <thead>
                      <tr>
                        <th />
                        <th />
                        <th colSpan={2} className="pb-1 text-center text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                          Predicted
                        </th>
                      </tr>
                      <tr>
                        <th />
                        <th />
                        <th className="px-2 pb-2 text-center text-xs font-medium text-muted-foreground">0</th>
                        <th className="px-2 pb-2 text-center text-xs font-medium text-muted-foreground">1</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th
                          rowSpan={2}
                          className="pr-1 text-center align-middle text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
                        >
                          Actual
                        </th>
                        <th className="pr-2 text-center text-xs font-medium text-muted-foreground">0</th>
                        {[confusion[0], confusion[1]].map(([label, count, tone]) => (
                          <td key={label} className={`rounded-lg border p-4 text-center ${toneClasses(tone)}`}>
                            <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</div>
                            <div className="mt-1 text-lg font-semibold tabular-nums">{Number(count).toLocaleString()}</div>
                          </td>
                        ))}
                      </tr>
                      <tr>
                        <th className="pr-2 text-center text-xs font-medium text-muted-foreground">1</th>
                        {[confusion[2], confusion[3]].map(([label, count, tone]) => (
                          <td key={label} className={`rounded-lg border p-4 text-center ${toneClasses(tone)}`}>
                            <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</div>
                            <div className="mt-1 text-lg font-semibold tabular-nums">{Number(count).toLocaleString()}</div>
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </EvalCard>
            )}

            {activeTab === "lift" && (
              <EvalCard title="Gain & lift" sub="Cumulative gain and lift by decile">
                {liftChartFigure ? (
                  <PlotlyChart figure={liftChartFigure} style={{ minHeight: 360 }} />
                ) : (
                  <p className="text-sm text-muted-foreground">Gain and lift charts require probability predictions from a binary classification model.</p>
                )}
              </EvalCard>
            )}

            {activeTab === "threshold" && (
              <EvalCard
                title="Threshold analysis"
                sub={isAutoThreshold ? `Precision · Recall · F1 across cut-offs · best F1 at ${threshold.toFixed(2)}` : "Precision · Recall · F1 across cut-offs"}
              >
                {thresholdFigure ? (
                  <PlotlyChart figure={thresholdFigure} style={{ minHeight: 360 }} />
                ) : (
                  <p className="text-sm text-muted-foreground">Threshold analysis requires probability predictions from a binary classification model.</p>
                )}
              </EvalCard>
            )}

            {activeTab === "score" && (
              <EvalCard title="Score distribution" sub="Hold-out set">
                {scoreDistributionFigure ? (
                  <PlotlyChart figure={scoreDistributionFigure} style={{ minHeight: 360 }} />
                ) : (
                  <p className="text-sm text-muted-foreground">Score distribution requires probability predictions from a binary classification model.</p>
                )}
              </EvalCard>
            )}

            {activeTab === "residual" && (
              <EvalCard title="Residual diagnostics" sub="Heteroscedasticity-style residual checks">
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="rounded-lg border border-border bg-background/70 p-3">
                    <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Risk Signal</div>
                    <div className="mt-1 text-base font-semibold">{heteroscedasticityCheck?.risk_flag ?? "N/A"}</div>
                    <div className="mt-1 text-xs text-muted-foreground">Overall heteroscedasticity risk flag for this model</div>
                  </div>
                  <div className="rounded-lg border border-border bg-background/70 p-3">
                    <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Residual Correlation</div>
                    <div className="mt-1 text-base font-semibold">{formatMetricValue(heteroscedasticityCheck?.spearman_abs_resid_vs_score)}</div>
                    <div className="mt-1 text-xs text-muted-foreground">Correlation between prediction error size and predicted score</div>
                  </div>
                  <div className="rounded-lg border border-border bg-background/70 p-3 sm:col-span-2">
                    <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Variance Ratio</div>
                    <div className="mt-1 text-base font-semibold">{formatMetricValue(heteroscedasticityCheck?.variance_ratio)}</div>
                    <div className="mt-1 text-xs text-muted-foreground">Spread of residual variance across score bins</div>
                  </div>
                </div>
                {Array.isArray(heteroscedasticityCheck?.bin_variance) && heteroscedasticityCheck.bin_variance.length > 0 && (
                  <div className="mt-4 overflow-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted-foreground">
                          <th className="py-2 pr-3">#</th>
                          <th className="py-2 pr-3">Bin</th>
                          <th className="py-2 pr-3">Count</th>
                          <th className="py-2">Variance</th>
                        </tr>
                      </thead>
                      <tbody>
                        {heteroscedasticityCheck.bin_variance.map((row: Record<string, any>, index: number) => (
                          <tr key={`${row.score_bin ?? index}`} className="border-b border-border/60 text-sm">
                            <td className="py-2 pr-3 text-muted-foreground">{index + 1}</td>
                            <td className="py-2 pr-3">{row.score_bin ?? `Bin ${index + 1}`}</td>
                            <td className="py-2 pr-3">{row.n ?? "—"}</td>
                            <td className="py-2">{formatMetricValue(row.residual_variance)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </EvalCard>
            )}

            {activeTab === "temporal" && (
              <EvalCard title="Actual vs Predicted" sub="Temporal stability by selected period">
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <select value={temporalFrequency} onChange={(e) => setTemporalFrequency(e.target.value)} className="text-sm p-1 rounded bg-background">
                      {temporalAnalysis?.frequency_options?.map((f: string) => <option key={f} value={f}>{f}</option>)}
                    </select>
                  </div>
                  {temporalFigure ? (
                    <PlotlyChart figure={temporalFigure} style={{ minHeight: 380 }} />
                  ) : (
                    <p className="text-sm text-muted-foreground">No temporal data available for the selected period.</p>
                  )}
                </div>
              </EvalCard>
            )}
          </div>
          )}

        </div>
      </section>

      <div className="flex gap-3 pt-4">
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
