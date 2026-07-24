import React from "react";

type DatasetProfile = Record<string, any>;
type ModelRecommendation = Record<string, any>;

type TrainingConfig = {
  test_size: number;
  val_size: number;
  random_seed: number;
  use_cv: boolean;
  cv_folds: number;
  use_hyperopt: boolean;
  scale_pos_weight: number;
  use_feature_engineering: boolean;
  manual_params: Record<string, any>;
};

type ComparisonResult = {
  model_name: string;
  roc_auc?: number;
  recall?: number;
  precision?: number;
  f1?: number;
  pr_auc?: number;
  accuracy?: number;
  training_time_s?: number;
};

type TrainingResult = {
  task_type: string;
  model_name: string;
  real_feature_names: string[];
  training_config?: Record<string, any>;
  training_info: Record<string, any>;
  split_stats: Record<string, any>;
  feature_engineering_summary?: Record<string, any> | null;
  evaluation_metrics?: Record<string, any> | null;
  evaluation_data?: Record<string, any> | null;
  model_artifact?: string;
};

type DatasetState = {
  file?: File | null;
  profile?: DatasetProfile | null;
  recommendations?: ModelRecommendation[] | null;
  selectedModel?: ModelRecommendation | null;
  compareModels?: string[] | null;
  preprocessingResult?: Record<string, any> | null;
  featureEngineeringResult?: Record<string, any> | null;
  trainingConfig?: TrainingConfig | null;
  trainingResult?: TrainingResult | null;
  comparisonResults?: ComparisonResult[] | null;
  selectedComparisonModel?: string | null;
  validationIntakeData?: Record<string, any> | null;
  validationMddText?: string | null;
  validationMddMetrics?: Record<string, any> | null;
  validationProfile?: Record<string, any> | null;
  validationResults?: Record<string, any> | null;
  validationStage3Result?: Record<string, any> | null;
  validationStage4Result?: Record<string, any> | null;
  validationStage5Result?: Record<string, any> | null;
  validationStage7Result?: Record<string, any> | null;
  validationStage7BiasResult?: Record<string, any> | null;
  validationStage8Result?: Record<string, any> | null;
  setUploadResult: (file: File | null, profile: DatasetProfile | null) => void;
  setProfile: (profile: DatasetProfile | null) => void;
  setRecommendations: (recommendations: ModelRecommendation[] | null) => void;
  setSelectedModel: (model: ModelRecommendation | null) => void;
  setPreprocessingResult: (result: Record<string, any> | null) => void;
  setFeatureEngineeringResult: (result: Record<string, any> | null) => void;
  setTrainingConfig: (config: TrainingConfig | null) => void;
  setTrainingResult: (result: TrainingResult | null) => void;
  setComparisonResults: (results: ComparisonResult[] | null) => void;
  setSelectedComparisonModel: (modelName: string | null) => void;
  setCompareModels: (models: string[] | null) => void;
  setValidationIntakeData: (data: Record<string, any> | null) => void;
  setValidationMddText: (text: string | null) => void;
  setValidationMddMetrics: (metrics: Record<string, any> | null) => void;
  setValidationProfile: (profile: Record<string, any> | null) => void;
  setValidationResults: (results: Record<string, any> | null) => void;
  setValidationStage3Result: (result: Record<string, any> | null) => void;
  setValidationStage4Result: (result: Record<string, any> | null) => void;
  setValidationStage5Result: (result: Record<string, any> | null) => void;
  setValidationStage7Result: (result: Record<string, any> | null) => void;
  setValidationStage7BiasResult: (result: Record<string, any> | null) => void;
  setValidationStage8Result: (result: Record<string, any> | null) => void;
  // Clears all uploaded/derived dataset + validation state (and the
  // localStorage blob it's persisted to) without touching app-level
  // preferences like theme.
  resetSession: () => void;
};

const DatasetContext = React.createContext<DatasetState | null>(null);

export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const [file, setFile] = React.useState<File | null>(null);
  const [profile, setProfile] = React.useState<DatasetProfile | null>(null);
  const [recommendations, setRecommendations] = React.useState<ModelRecommendation[] | null>(null);
  const [selectedModel, setSelectedModelState] = React.useState<ModelRecommendation | null>(null);
  const [compareModels, setCompareModelsState] = React.useState<string[] | null>(null);
  const [preprocessingResult, setPreprocessingResultState] = React.useState<Record<string, any> | null>(null);
  const [featureEngineeringResult, setFeatureEngineeringResultState] = React.useState<Record<string, any> | null>(null);
  const [trainingConfig, setTrainingConfigState] = React.useState<TrainingConfig | null>(null);
  const [trainingResult, setTrainingResultState] = React.useState<TrainingResult | null>(null);
  const [comparisonResults, setComparisonResultsState] = React.useState<ComparisonResult[] | null>(null);
  const [selectedComparisonModel, setSelectedComparisonModelState] = React.useState<string | null>(null);
  const [validationIntakeData, setValidationIntakeDataState] = React.useState<Record<string, any> | null>(null);
  const [validationMddText, setValidationMddTextState] = React.useState<string | null>(null);
  const [validationMddMetrics, setValidationMddMetricsState] = React.useState<Record<string, any> | null>(null);
  const [validationProfile, setValidationProfileState] = React.useState<Record<string, any> | null>(null);
  const [validationResults, setValidationResultsState] = React.useState<Record<string, any> | null>(null);
  const [validationStage3Result, setValidationStage3ResultState] = React.useState<Record<string, any> | null>(null);
  const [validationStage4Result, setValidationStage4ResultState] = React.useState<Record<string, any> | null>(null);
  const [validationStage5Result, setValidationStage5ResultState] = React.useState<Record<string, any> | null>(null);
  const [validationStage7Result, setValidationStage7ResultState] = React.useState<Record<string, any> | null>(null);
  const [validationStage7BiasResult, setValidationStage7BiasResultState] = React.useState<Record<string, any> | null>(null);
  const [validationStage8Result, setValidationStage8ResultState] = React.useState<Record<string, any> | null>(null);
  const [isHydrated, setIsHydrated] = React.useState(false);

  React.useEffect(() => {
    if (typeof window === "undefined") {
      setIsHydrated(true);
      return;
    }

    try {
      const stored = window.localStorage.getItem("aegis_dataset_state");
      if (!stored) {
        setIsHydrated(true);
        return;
      }

      const parsed = JSON.parse(stored) as {
        profile?: DatasetProfile | null;
        recommendations?: ModelRecommendation[] | null;
        preprocessingResult?: Record<string, any> | null;
        featureEngineeringResult?: Record<string, any> | null;
        trainingConfig?: TrainingConfig | null;
        trainingResult?: TrainingResult | null;
        comparisonResults?: ComparisonResult[] | null;
        compareModels?: string[] | null;
        selectedModel?: ModelRecommendation | null;
        selectedComparisonModel?: string | null;
        validationIntakeData?: Record<string, any> | null;
        validationMddText?: string | null;
        validationMddMetrics?: Record<string, any> | null;
        validationProfile?: Record<string, any> | null;
        validationResults?: Record<string, any> | null;
        validationStage3Result?: Record<string, any> | null;
        validationStage4Result?: Record<string, any> | null;
        validationStage5Result?: Record<string, any> | null;
        validationStage7Result?: Record<string, any> | null;
        validationStage7BiasResult?: Record<string, any> | null;
        validationStage8Result?: Record<string, any> | null;
      };

      // `file` (a raw File object) is deliberately not persisted/restored —
      // it can't survive JSON serialization anyway. Every page that needs
      // the dataset already falls back to `profile.csv_text` (which /data/
      // upload always echoes back) whenever `file` is null, so restoring
      // just `profile` here is enough to carry the dataset itself forward.
      if (parsed.profile) {
        setProfile(parsed.profile);
      }
      if (parsed.recommendations) {
        setRecommendations(parsed.recommendations);
      }
      if (parsed.preprocessingResult) {
        setPreprocessingResultState(parsed.preprocessingResult);
      }
      if (parsed.featureEngineeringResult) {
        setFeatureEngineeringResultState(parsed.featureEngineeringResult);
      }
      if (parsed.trainingConfig) {
        setTrainingConfigState(parsed.trainingConfig);
      }
      if (parsed.trainingResult) {
        setTrainingResultState(parsed.trainingResult);
      }
      if (parsed.comparisonResults) {
        setComparisonResultsState(parsed.comparisonResults);
      }
      if (parsed.compareModels) {
        setCompareModelsState(parsed.compareModels);
      }
      if (parsed.selectedModel) {
        setSelectedModelState(parsed.selectedModel);
      }
      if (parsed.selectedComparisonModel) {
        setSelectedComparisonModelState(parsed.selectedComparisonModel);
      }
      if (parsed.validationIntakeData) {
        setValidationIntakeDataState(parsed.validationIntakeData);
      }
      if (parsed.validationMddText) {
        setValidationMddTextState(parsed.validationMddText);
      }
      if (parsed.validationMddMetrics) {
        setValidationMddMetricsState(parsed.validationMddMetrics);
      }
      if (parsed.validationProfile) {
        setValidationProfileState(parsed.validationProfile);
      }
      if (parsed.validationResults) {
        setValidationResultsState(parsed.validationResults);
      }
      if (parsed.validationStage3Result) {
        setValidationStage3ResultState(parsed.validationStage3Result);
      }
      if (parsed.validationStage4Result) {
        setValidationStage4ResultState(parsed.validationStage4Result);
      }
      if (parsed.validationStage5Result) {
        setValidationStage5ResultState(parsed.validationStage5Result);
      }
      if (parsed.validationStage7Result) {
        setValidationStage7ResultState(parsed.validationStage7Result);
      }
      if (parsed.validationStage7BiasResult) {
        setValidationStage7BiasResultState(parsed.validationStage7BiasResult);
      }
      if (parsed.validationStage8Result) {
        setValidationStage8ResultState(parsed.validationStage8Result);
      }
    } catch {
      // Ignore invalid stored state
    } finally {
      setIsHydrated(true);
    }
  }, []);

  React.useEffect(() => {
    if (typeof window === "undefined" || !isHydrated) return;

    // model_artifact is a base64-encoded, pickled sklearn pipeline — it can
    // easily be several MB and blows past localStorage's ~5-10MB per-origin
    // quota. It only needs to live in memory for this session (it's used
    // to call /models/evaluate), so it's deliberately excluded here rather
    // than persisted. A page reload will require re-training to get a
    // fresh artifact, but that's a much better trade-off than crashing the
    // whole app on every training run.
    const { model_artifact, ...trainingResultToPersist } = trainingResult ?? {};
    const persisted = {
      profile,
      recommendations,
      preprocessingResult,
      featureEngineeringResult,
      trainingConfig,
      trainingResult: trainingResult ? trainingResultToPersist : null,
      comparisonResults,
      compareModels,
      selectedModel,
      selectedComparisonModel,
      validationIntakeData,
      validationMddText,
      validationMddMetrics,
      validationProfile,
      validationResults,
      validationStage3Result,
      validationStage4Result,
      validationStage5Result,
      validationStage7Result,
      validationStage7BiasResult,
      validationStage8Result,
    };

    try {
      window.localStorage.setItem("aegis_dataset_state", JSON.stringify(persisted));
    } catch (err) {
      // Quota exceeded (or any other storage failure) — state still works
      // fine in memory for this session, it just won't survive a refresh.
      console.warn("Failed to persist dataset state to localStorage:", err);
    }
  }, [
    profile,
    recommendations,
    preprocessingResult,
    featureEngineeringResult,
    trainingConfig,
    trainingResult,
    comparisonResults,
    compareModels,
    selectedModel,
    selectedComparisonModel,
    validationIntakeData,
    validationMddText,
    validationMddMetrics,
    validationProfile,
    validationResults,
    validationStage3Result,
    validationStage4Result,
    validationStage5Result,
    validationStage7Result,
    validationStage7BiasResult,
    validationStage8Result,
    isHydrated,
  ]);

  const setUploadResult = React.useCallback((f: File | null, p: DatasetProfile | null) => {
    setFile(f);
    setProfile(p);
    setRecommendations(null);
    setSelectedModelState(null);
    setCompareModelsState(null);
    setPreprocessingResultState(null);
    setFeatureEngineeringResultState(null);
    setTrainingConfigState(null);
    setTrainingResultState(null);
    setComparisonResultsState(null);
    setSelectedComparisonModelState(null);
  }, []);

  const setProfileState = React.useCallback((p: DatasetProfile | null) => {
    setProfile(p);
  }, []);

  const setSelectedModel = React.useCallback((model: ModelRecommendation | null) => {
    setSelectedModelState(model);
  }, []);

  const setCompareModels = React.useCallback((models: string[] | null) => {
    setCompareModelsState(models);
  }, []);

  const setPreprocessingResult = React.useCallback((result: Record<string, any> | null) => {
    setPreprocessingResultState(result);
  }, []);

  const setFeatureEngineeringResult = React.useCallback((result: Record<string, any> | null) => {
    setFeatureEngineeringResultState(result);
  }, []);

  const setTrainingConfig = React.useCallback((config: TrainingConfig | null) => {
    setTrainingConfigState(config);
  }, []);

  const setTrainingResult = React.useCallback((result: TrainingResult | null) => {
    setTrainingResultState(result);
  }, []);

  const setComparisonResults = React.useCallback((results: ComparisonResult[] | null) => {
    setComparisonResultsState(results);
  }, []);

  const setSelectedComparisonModel = React.useCallback((modelName: string | null) => {
    setSelectedComparisonModelState(modelName);
  }, []);

  const setValidationIntakeData = React.useCallback((data: Record<string, any> | null) => {
    setValidationIntakeDataState(data);
  }, []);

  const setValidationMddText = React.useCallback((text: string | null) => {
    setValidationMddTextState(text);
  }, []);

  const setValidationMddMetrics = React.useCallback((metrics: Record<string, any> | null) => {
    setValidationMddMetricsState(metrics);
  }, []);

  const setValidationProfile = React.useCallback((profileState: Record<string, any> | null) => {
    setValidationProfileState(profileState);
  }, []);

  const setValidationResults = React.useCallback((results: Record<string, any> | null) => {
    setValidationResultsState(results);
  }, []);

  const setValidationStage3Result = React.useCallback((result: Record<string, any> | null) => {
    setValidationStage3ResultState(result);
  }, []);

  const setValidationStage4Result = React.useCallback((result: Record<string, any> | null) => {
    setValidationStage4ResultState(result);
  }, []);

  const setValidationStage5Result = React.useCallback((result: Record<string, any> | null) => {
    setValidationStage5ResultState(result);
  }, []);

  const setValidationStage7Result = React.useCallback((result: Record<string, any> | null) => {
    setValidationStage7ResultState(result);
  }, []);

  const setValidationStage7BiasResult = React.useCallback((result: Record<string, any> | null) => {
    setValidationStage7BiasResultState(result);
  }, []);

  const setValidationStage8Result = React.useCallback((result: Record<string, any> | null) => {
    setValidationStage8ResultState(result);
  }, []);

  const resetSession = React.useCallback(() => {
    setFile(null);
    setProfile(null);
    setRecommendations(null);
    setSelectedModelState(null);
    setCompareModelsState(null);
    setPreprocessingResultState(null);
    setFeatureEngineeringResultState(null);
    setTrainingConfigState(null);
    setTrainingResultState(null);
    setComparisonResultsState(null);
    setSelectedComparisonModelState(null);
    setValidationIntakeDataState(null);
    setValidationMddTextState(null);
    setValidationMddMetricsState(null);
    setValidationProfileState(null);
    setValidationResultsState(null);
    setValidationStage3ResultState(null);
    setValidationStage4ResultState(null);
    setValidationStage5ResultState(null);
    setValidationStage7ResultState(null);
    setValidationStage7BiasResultState(null);
    setValidationStage8ResultState(null);
    if (typeof window !== "undefined") {
      try {
        window.localStorage.removeItem("aegis_dataset_state");
        window.localStorage.removeItem("aegis_intake_form_state");
      } catch {
        // Ignore storage failures — in-memory state is already cleared.
      }
    }
  }, []);

  const value = React.useMemo(
    () => ({
      file,
      profile,
      recommendations,
      selectedModel,
      compareModels,
      preprocessingResult,
      featureEngineeringResult,
      trainingConfig,
      trainingResult,
      comparisonResults,
      selectedComparisonModel,
      validationIntakeData,
      validationMddText,
      validationMddMetrics,
      validationProfile,
      validationResults,
      validationStage3Result,
      validationStage4Result,
      validationStage5Result,
      validationStage7Result,
      validationStage7BiasResult,
      validationStage8Result,
      resetSession,
      setUploadResult,
      setProfile: setProfileState,
      setRecommendations,
      setSelectedModel,
      setCompareModels,
      setPreprocessingResult,
      setFeatureEngineeringResult,
      setTrainingConfig,
      setTrainingResult,
      setComparisonResults,
      setSelectedComparisonModel,
      setValidationIntakeData,
      setValidationMddText,
      setValidationMddMetrics,
      setValidationProfile,
      setValidationResults,
      setValidationStage3Result,
      setValidationStage4Result,
      setValidationStage5Result,
      setValidationStage7Result,
      setValidationStage7BiasResult,
      setValidationStage8Result,
    }),
    [
      file,
      profile,
      recommendations,
      selectedModel,
      compareModels,
      preprocessingResult,
      featureEngineeringResult,
      trainingConfig,
      trainingResult,
      comparisonResults,
      selectedComparisonModel,
      validationIntakeData,
      validationMddText,
      validationMddMetrics,
      validationProfile,
      validationResults,
      validationStage3Result,
      validationStage4Result,
      validationStage5Result,
      validationStage7Result,
      validationStage7BiasResult,
      validationStage8Result,
      resetSession,
      setUploadResult,
      setRecommendations,
      setSelectedModel,
      setCompareModels,
      setPreprocessingResult,
      setFeatureEngineeringResult,
      setTrainingConfig,
      setTrainingResult,
      setComparisonResults,
      setSelectedComparisonModel,
      setValidationIntakeData,
      setValidationMddText,
      setValidationMddMetrics,
      setValidationProfile,
      setValidationResults,
      setValidationStage3Result,
      setValidationStage4Result,
      setValidationStage5Result,
      setValidationStage7Result,
      setValidationStage7BiasResult,
      setValidationStage8Result,
    ],
  );

  return <DatasetContext.Provider value={value}>{children}</DatasetContext.Provider>;
}

export function useDataset() {
  const ctx = React.useContext(DatasetContext);
  if (!ctx) throw new Error("useDataset must be used within DatasetProvider");
  return ctx;
}

export default DatasetContext;
