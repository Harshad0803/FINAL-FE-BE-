import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { PageHeader } from "@/components/app-shell";
import { useDataset } from "@/lib/app-context";
import { formUpload } from "@/lib/api";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import PlotlyChart from "@/components/plotly-chart";
import AnimatedNumber from "@/components/animated-number";
import {
  AlertCircle, ArrowLeft, ArrowRight, Loader, Download, Printer, Microscope, Search,
  AlertTriangle, TrendingUp, TrendingDown, ChevronRight, Lock,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
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
import { useResumeState } from "@/hooks/use-resume-state";
import { cn } from "@/lib/utils";

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

// ─── Presentational primitives — shared visual language with the Model
// Training & Evaluation page (rounded-2xl white cards, icon-badge headers,
// slate/blue tables, native range sliders, dense analytics grids). Purely
// presentational: none of these touch data, calculations, or API calls. ────

function SectionCard({
  icon,
  title,
  sub,
  eyebrow,
  actions,
  children,
  className,
  contentClassName,
}: {
  // Accepts either a lucide component (rare, kept for future use) or an
  // emoji glyph — the Figma reference uses colorful emoji inside a soft
  // gradient badge for every card header, so that's the default going forward.
  icon?: React.ComponentType<{ className?: string }> | string;
  title: ReactNode;
  sub?: ReactNode;
  eyebrow?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const Icon = typeof icon === "function" ? icon : null;
  return (
    <section className={cn("rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6", className)}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          {icon && (
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-blue-100/60 bg-gradient-to-br from-blue-50 to-indigo-100">
              {Icon ? <Icon className="h-4 w-4 text-blue-600" /> : <span className="text-base leading-none">{icon as string}</span>}
            </div>
          )}
          <div>
            {eyebrow && <div className="text-[10px] font-bold uppercase tracking-wider text-blue-600">{eyebrow}</div>}
            <h2 className="text-[15px] font-bold tracking-tight text-slate-900">{title}</h2>
            {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
          </div>
        </div>
        {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
      </div>
      <div className={cn("mt-4", contentClassName)}>{children}</div>
    </section>
  );
}

const STATUS_BANNER_TONES = {
  success: { wrap: "border-emerald-300/70 bg-gradient-to-r from-emerald-50 to-green-50 text-emerald-900", icon: "text-emerald-600", dotCore: "bg-emerald-500", dotPing: "bg-emerald-400" },
  info: { wrap: "border-blue-200 bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-900", icon: "text-blue-600", dotCore: "bg-blue-500", dotPing: "bg-blue-400" },
  warning: { wrap: "border-amber-300/70 bg-amber-50 text-amber-900", icon: "text-amber-600", dotCore: "bg-amber-500", dotPing: "bg-amber-400" },
  error: { wrap: "border-red-200 bg-red-50 text-red-800", icon: "text-red-600", dotCore: "bg-red-500", dotPing: "bg-red-400" },
} as const;

function StatusBanner({
  tone,
  icon: Icon,
  dot,
  children,
  className,
}: {
  tone: keyof typeof STATUS_BANNER_TONES;
  icon?: React.ComponentType<{ className?: string }>;
  // Pulsing dot indicator (matches the header's "live" model chip) instead
  // of a static icon — this is what the Figma reference uses for its
  // success banners ("36 features extracted…").
  dot?: boolean;
  children: ReactNode;
  className?: string;
}) {
  const t = STATUS_BANNER_TONES[tone];
  return (
    <div className={cn("flex items-start gap-2.5 rounded-xl border px-3.5 py-2.5 text-sm", t.wrap, className)}>
      {dot ? (
        <span className="relative mt-1 flex h-2.5 w-2.5 shrink-0">
          <span className={cn("absolute inline-flex h-full w-full animate-ping rounded-full opacity-75", t.dotPing)} />
          <span className={cn("relative inline-flex h-2.5 w-2.5 rounded-full", t.dotCore)} />
        </span>
      ) : (
        Icon && <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", t.icon)} />
      )}
      <div className="font-medium leading-relaxed">{children}</div>
    </div>
  );
}

function RangeSliderRow({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">{label}</span>
        <span className="min-w-[2.75rem] rounded-lg bg-gradient-to-br from-blue-700 to-blue-500 px-2.5 py-1 text-center text-sm font-extrabold tabular-nums text-white shadow-sm">
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-3 w-full accent-blue-600"
      />
      <div className="mt-1 flex justify-between text-[10.5px] text-slate-400">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function InsightPanel({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-2.5 rounded-r-xl border border-blue-200 border-l-4 border-l-blue-600 bg-gradient-to-r from-blue-50 to-indigo-50 p-3.5">
      <span className="mt-0.5 shrink-0 text-base leading-none">💡</span>
      <div>
        <div className="text-[10px] font-bold uppercase tracking-wider text-blue-700">Key Insight</div>
        <p className="mt-1 text-xs leading-relaxed text-indigo-950">{children}</p>
      </div>
    </div>
  );
}

function StatTile({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-center">
      <div className="font-mono text-lg font-bold tabular-nums text-slate-900">{value}</div>
      <div className="mt-1 text-[10px] font-bold uppercase tracking-wider text-slate-500">{label}</div>
    </div>
  );
}

function DefList({ items }: { items: Array<{ label: string; value: ReactNode; mono?: boolean; highlight?: boolean }> }) {
  return (
    <div className="space-y-2">
      {items.map((item) => (
        <div key={item.label} className="flex items-center justify-between gap-4 border-b border-slate-100 pb-2 text-sm last:border-0 last:pb-0">
          <span className="text-slate-500">{item.label}</span>
          <span
            className={cn(
              "rounded-md border px-2 py-0.5 text-right font-semibold",
              item.mono && "font-mono",
              item.highlight ? "border-blue-200 bg-blue-50 text-blue-700" : "border-slate-100 bg-slate-50 text-slate-900",
            )}
          >
            {item.value}
          </span>
        </div>
      ))}
    </div>
  );
}

// Semicircle probability gauge for the Prediction Reasoning hero tiles —
// purely presentational SVG driven by the real parsed probability value.
function Gauge({ pct, color, size = 92 }: { pct: number; color: string; size?: number }) {
  const r = size * 0.37;
  const circ = 2 * Math.PI * r;
  const halfCirc = circ / 2;
  const cx = size / 2;
  const cy = size / 2;
  const sw = 9;
  const safePct = Number.isFinite(pct) ? Math.min(100, Math.max(0, pct)) : 0;
  const fillLen = (safePct / 100) * halfCirc;
  const svgH = cy + r + sw / 2 + 4;
  return (
    <svg width={size} height={svgH} viewBox={`0 0 ${size} ${svgH}`}>
      <circle
        cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth={sw}
        strokeDasharray={`${halfCirc} ${halfCirc}`} strokeLinecap="round" transform={`rotate(180 ${cx} ${cy})`}
      />
      <circle
        cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth={sw}
        strokeDasharray={`${halfCirc} ${halfCirc}`} strokeDashoffset={halfCirc - fillLen}
        strokeLinecap="round" transform={`rotate(180 ${cx} ${cy})`}
        style={{ transition: "stroke-dashoffset 0.9s cubic-bezier(0.22,1,0.36,1)" }}
      />
    </svg>
  );
}

// Interpolated blue marker-color array for ranked horizontal bar charts —
// purely a presentation detail for Plotly's marker.color; never touches the
// underlying values/order of the bars themselves.
function blueBarColors(n: number): string[] {
  if (n <= 0) return [];
  if (n === 1) return ["rgb(29,78,216)"];
  return Array.from({ length: n }, (_, i) => {
    const t = i / (n - 1);
    const r = Math.round(29 + t * (96 - 29));
    const g = Math.round(78 + t * (165 - 78));
    const b = Math.round(216 + t * (250 - 216));
    return `rgb(${r},${g},${b})`;
  });
}

const CHART_AXIS_COLOR = "#94a3b8";
const CHART_GRID_COLOR = "#eef1f7";

// Purposeful empty state: for when data hasn't been computed yet (an action
// unlocks it) rather than a blank card. Never used to paper over genuinely
// missing/unavailable data — those metrics are simply not rendered.
function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-300 bg-slate-50/70 px-6 py-10 text-center">
      <div className="flex h-11 w-11 items-center justify-center rounded-full bg-blue-50 text-blue-600">
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <div className="text-sm font-semibold text-slate-900">{title}</div>
        <p className="mx-auto mt-1 max-w-sm text-xs leading-relaxed text-slate-500">{description}</p>
      </div>
      {action}
    </div>
  );
}

function TablePagination({
  page,
  totalPages,
  onChange,
  totalCount,
  label,
}: {
  page: number;
  totalPages: number;
  onChange: (page: number) => void;
  totalCount: number;
  label: string;
}) {
  if (totalPages <= 1) return null;
  return (
    <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4 text-xs text-slate-500">
      <span>{totalCount} {label} · page {page} of {totalPages}</span>
      <div className="flex items-center gap-1">
        <button
          onClick={() => onChange(Math.max(1, page - 1))}
          disabled={page === 1}
          className="rounded-lg border border-slate-200 px-2.5 py-1.5 font-semibold text-slate-600 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Prev
        </button>
        {Array.from({ length: totalPages }).map((_, i) => (
          <button
            key={i}
            onClick={() => onChange(i + 1)}
            className={cn(
              "h-7 w-7 rounded-lg text-[12px] font-semibold transition",
              page === i + 1 ? "bg-blue-600 text-white shadow-sm" : "border border-slate-200 text-slate-600 hover:bg-slate-50",
            )}
          >
            {i + 1}
          </button>
        ))}
        <button
          onClick={() => onChange(Math.min(totalPages, page + 1))}
          disabled={page === totalPages}
          className="rounded-lg border border-slate-200 px-2.5 py-1.5 font-semibold text-slate-600 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </div>
  );
}

// Fades/slides a tab's content in on activation — Radix unmounts inactive
// TabsContent, so this replays on every switch. Ties the "dynamic" tab
// navigation to a visible transition instead of an instant hard cut.
const TAB_TRANSITION = "animate-in fade-in-0 slide-in-from-bottom-2 duration-300";

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

  // Resume where the reviewer left off: if this session has no explainability
  // result yet, pull the last saved /models/explain run from the backend.
  const { data: resumedExplain } = useResumeState<ExplainabilityResponse>("dev_pipeline_log.csv", "explainability");
  useEffect(() => {
    if (!data && resumedExplain) {
      setData(resumedExplain);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumedExplain]);

  const [topN, setTopN] = useState(15);
  const [shapSamples, setShapSamples] = useState(150);
  const [shapComputed, setShapComputed] = useState(false);
  const [shapLoading, setShapLoading] = useState(false);
  const [shapError, setShapError] = useState<string | null>(null);
  const [sampleIdx, setSampleIdx] = useState(0);
  // UI-only state for the redesigned Feature Importance table (search +
  // pagination) — does not affect the underlying feature_importance data.
  const [featureSearch, setFeatureSearch] = useState("");
  const [featurePage, setFeaturePage] = useState(1);
  // Controlled tab state — lets Prediction Reasoning's empty state deep-link
  // the reviewer straight into SHAP Analysis instead of leaving them stuck.
  const [activeTab, setActiveTab] = useState("importance");

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
          marker: { color: blueBarColors(values.length) },
          hovertemplate: "%{x:.4f}<extra></extra>",
        },
      ],
      layout: {
        margin: { l: 160, r: 30, t: 20, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#475569" },
        xaxis: { tickfont: { size: 10, color: CHART_AXIS_COLOR }, showline: false, gridcolor: CHART_GRID_COLOR, zerolinecolor: CHART_GRID_COLOR },
        yaxis: { tickfont: { size: 9, color: "#334155" }, automargin: true, type: "category", autorange: "reversed" },
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
          marker: { color: blueBarColors(values.length) },
          hovertemplate: "%{x:.5f}<extra></extra>",
        },
      ],
      layout: {
        margin: { l: 160, r: 30, t: 20, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#475569" },
        xaxis: { tickfont: { size: 10, color: CHART_AXIS_COLOR }, showline: false, gridcolor: CHART_GRID_COLOR, zerolinecolor: CHART_GRID_COLOR },
        yaxis: { tickfont: { size: 9, color: "#334155" }, automargin: true, type: "category", autorange: "reversed" },
        height: 384,
      },
    };
  }, [shapSummaryRows]);

  const sampleShapFigure = useMemo(() => {
    const rows = [...sampleShapRows].slice(0, 12).reverse();
    const labels = rows.map((r) => `${r.Feature} = ${formatValue(r.Value)}`);
    const values = rows.map((r) => (r.SHAP ?? 0));
    const colors = rows.map((r) => ((r.SHAP ?? 0) < 0 ? "#10b981" : "#f43f5e"));
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
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#475569" },
        xaxis: { tickfont: { size: 10, color: CHART_AXIS_COLOR }, showline: false, gridcolor: CHART_GRID_COLOR, zerolinecolor: "#cbd5e1" },
        yaxis: { tickfont: { size: 9, color: "#334155" }, automargin: true, type: "category", autorange: "reversed" },
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
          marker: { color: blueBarColors(values.length) },
          hovertemplate: "%{y}<extra></extra>",
        },
      ],
      layout: {
        margin: { l: 10, r: 20, t: 10, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#475569" },
        xaxis: { tickfont: { size: 10, color: CHART_AXIS_COLOR }, gridcolor: CHART_GRID_COLOR },
        yaxis: { tickfont: { size: 10, color: CHART_AXIS_COLOR }, gridcolor: CHART_GRID_COLOR },
        height: 256,
      },
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
          marker: { color: blueBarColors(values.length) },
          hovertemplate: "%{y:.4f}<extra></extra>",
        },
      ],
      layout: {
        margin: { l: 10, r: 20, t: 10, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#475569" },
        xaxis: { tickfont: { size: 10, color: CHART_AXIS_COLOR }, gridcolor: CHART_GRID_COLOR },
        yaxis: { range: [0, 1], tickfont: { size: 10, color: CHART_AXIS_COLOR }, gridcolor: CHART_GRID_COLOR },
        height: 256,
      },
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
          marker: { color: blueBarColors(values.length) },
          hovertemplate: "%{x:.4f}<extra></extra>",
        },
      ],
      layout: {
        margin: { l: 130, r: 30, t: 20, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#475569" },
        xaxis: { tickfont: { size: 10, color: CHART_AXIS_COLOR }, showline: false, gridcolor: CHART_GRID_COLOR, zerolinecolor: CHART_GRID_COLOR },
        yaxis: { tickfont: { size: 9, color: "#334155" }, automargin: true, type: "category", autorange: "reversed" },
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

  // ── Presentation-only derived values (Feature Importance table search /
  // pagination, SHAP "at a glance" stats, Prediction Reasoning structured
  // summary, Summary KPIs). None of these touch the underlying explainability
  // data or logic — every value here is read straight from real state. ─────
  const featureImportanceFiltered = importanceRows.filter((r) =>
    (r.Feature ?? "").toLowerCase().includes(featureSearch.trim().toLowerCase()),
  );
  const featureTablePageSize = 10;
  const featureTableTotalPages = Math.max(1, Math.ceil(featureImportanceFiltered.length / featureTablePageSize));
  const featureTablePageSafe = Math.min(featurePage, featureTableTotalPages);
  const featureImportancePaged = featureImportanceFiltered.slice(
    (featureTablePageSafe - 1) * featureTablePageSize,
    featureTablePageSafe * featureTablePageSize,
  );
  const maxImportanceValue = importanceRows.reduce((m, r) => Math.max(m, r.Importance), 0) || 1;

  const featureImportanceInsight = (() => {
    if (importanceRows.length === 0) return null;
    const top = importanceRows.slice(0, 3);
    const top3Sum = top.reduce((s, r) => s + r.Importance, 0) * 100;
    if (top.length === 1) {
      return (
        <>
          <strong>{top[0].Feature}</strong> is the dominant driver of model predictions at {(top[0].Importance * 100).toFixed(2)}%.
        </>
      );
    }
    if (top.length === 2) {
      return (
        <>
          <strong>{top[0].Feature}</strong> is the most influential factor at {(top[0].Importance * 100).toFixed(2)}%, followed by <strong>{top[1].Feature}</strong> ({(top[1].Importance * 100).toFixed(2)}%).
        </>
      );
    }
    return (
      <>
        <strong>{top[0].Feature}</strong> is the most influential factor at {(top[0].Importance * 100).toFixed(2)}%, followed by <strong>{top[1].Feature}</strong> ({(top[1].Importance * 100).toFixed(2)}%) and <strong>{top[2].Feature}</strong> ({(top[2].Importance * 100).toFixed(2)}%). The top 3 features collectively drive {top3Sum.toFixed(1)}% of total model influence.
      </>
    );
  })();

  // Real, model-context chip for the page header — only rendered when the
  // underlying values actually exist (never fabricated).
  const headerChipParts: string[] = [];
  if (trainingResult?.model_name) headerChipParts.push(trainingResult.model_name);
  if (typeof metrics.roc_auc === "number") headerChipParts.push(`ROC-AUC ${metrics.roc_auc.toFixed(4)}`);

  // The backend generates sample_reasoning as deterministic templated text
  // (see explainability.py::generate_prediction_reasoning) — parsing it for
  // the structured summary tiles below is safe because the format is
  // code-generated, not free text. If it ever doesn't match, we fall back to
  // rendering the raw text untouched, so nothing is ever silently dropped.
  const reasoningText = data?.shap.sample_reasoning ?? null;
  const reasoningMatch = reasoningText?.match(
    /\*\*Prediction:\*\*\s*(?:🔴|🟢)\s*(HIGH RISK|LOW RISK)\s*\(([^)]*)\)[\s\S]*?\*\*Default Probability:\*\*\s*([\d.]+)%\s*\(threshold = ([\d.]+)%\)/,
  );
  const reasoningProbability = reasoningMatch ? parseFloat(reasoningMatch[3]) : NaN;
  const reasoningThreshold = reasoningMatch ? parseFloat(reasoningMatch[4]) : NaN;
  const parsedReasoning = reasoningMatch && Number.isFinite(reasoningProbability) && Number.isFinite(reasoningThreshold)
    ? {
        isHighRisk: reasoningMatch[1] === "HIGH RISK",
        riskLabel: reasoningMatch[1],
        detail: reasoningMatch[2],
        probability: reasoningProbability,
        threshold: reasoningThreshold,
      }
    : null;

  const shapRowsWithNumber = sampleShapRows.filter(
    (r): r is SampleShapRow & { SHAP: number } => typeof r.SHAP === "number",
  );
  const topDriversReduces = shapRowsWithNumber.filter((r) => r.SHAP < 0).sort((a, b) => a.SHAP - b.SHAP).slice(0, 3);
  const topDriversIncreases = shapRowsWithNumber.filter((r) => r.SHAP >= 0).sort((a, b) => b.SHAP - a.SHAP).slice(0, 3);

  // Highest mean-|SHAP| feature — last element, since shapSummaryRows is
  // sorted ascending (matches the horizontal-bar chart's top row).
  const topShapFeature = shapSummaryRows.length > 0 ? shapSummaryRows[shapSummaryRows.length - 1] : null;

  const summaryTopFeature = importanceRows[0]?.Feature ?? "—";
  const summaryTopFeatureSub = importanceRows[0] ? `${(importanceRows[0].Importance * 100).toFixed(2)}% importance` : undefined;
  const summaryKeyRiskDriver = importanceRows[1]?.Feature ?? "—";
  const summaryFeaturesAnalyzed = importanceRows.length > 0 ? String(importanceRows.length) : "—";
  const summaryShapAvailable = Boolean(data?.shap.shap_available);

  return (
    <div className="space-y-6">
      <PageHeader
        title="💡 Model Explainability"
        description="Feature importance, SHAP values, and individual prediction reasoning."
        actions={
          headerChipParts.length > 0 ? (
            <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-600 shadow-sm">
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
              </span>
              Model: {headerChipParts.join(" · ")}
            </div>
          ) : undefined
        }
      />

      {error && (
        <StatusBanner tone="error" icon={AlertCircle}>{error}</StatusBanner>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList>
          {[
            { value: "importance", icon: "📊", label: "Feature Importance" },
            { value: "shap", icon: "🔬", label: "SHAP Analysis" },
            { value: "reasoning", icon: "🔍", label: "Prediction Reasoning" },
            { value: "summary", icon: "📋", label: "Summary" },
          ].map((t) => (
            <TabsTrigger
              key={t.value}
              value={t.value}
              className="gap-2 data-[state=active]:bg-gradient-to-br data-[state=active]:from-slate-900 data-[state=active]:via-blue-800 data-[state=active]:to-blue-600 data-[state=active]:shadow-[0_4px_16px_rgba(37,99,235,0.4)]"
            >
              <span className="text-base leading-none">{t.icon}</span>
              {t.label}
            </TabsTrigger>
          ))}
        </TabsList>

        {/* ── Feature Importance — one dense analytics workspace: controls,
             then a chart+insights row, then the table beneath a divider so
             chart and table read as one connected surface. ── */}
        <TabsContent value="importance" className={cn("pt-5", TAB_TRANSITION)}>
          <SectionCard
            icon="📊"
            title="Feature Importance Overview"
            sub="Ranked features by their contribution to model predictions"
            actions={importanceRows.length > 0 ? (
              <Button variant="outline" size="sm" onClick={downloadFeatureImportanceCsv} className="gap-1.5">
                <Download className="h-3.5 w-3.5" />
                Export CSV
              </Button>
            ) : undefined}
          >
            {loading ? (
              <div className="flex items-center gap-3 text-sm text-slate-500">
                <Loader className="h-4 w-4 animate-spin" />
                Extracting feature importance with real column names...
              </div>
            ) : importanceRows.length > 0 ? (
              <>
                <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
                  <StatusBanner tone="success" dot className="lg:flex-1">
                    Extracted importance for {importanceRows.length} features with real column names.
                  </StatusBanner>
                  <div className="lg:w-[340px] lg:shrink-0">
                    <RangeSliderRow label="Show top N features" value={topN} min={5} max={maxTopN} onChange={setTopN} />
                  </div>
                </div>

                {/* Main analytics row — chart left, insights right */}
                <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-[1fr_280px]">
                  <div className="overflow-hidden">
                    <PlotlyChart figure={importanceFigure} style={{ minHeight: 384 }} config={{ displayModeBar: false }} />
                  </div>
                  <div className="flex flex-col gap-3">
                    {featureImportanceInsight && <InsightPanel>{featureImportanceInsight}</InsightPanel>}
                    <div className="rounded-xl border border-slate-200 bg-slate-50/60 p-4">
                      <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">At a Glance</div>
                      <div className="mt-3">
                        <DefList
                          items={[
                            { label: "Top feature", value: importanceRows[0]?.Feature ?? "—" },
                            { label: "Top weight", value: `${(importanceRows[0].Importance * 100).toFixed(2)}%`, mono: true },
                            { label: "Features analyzed", value: String(importanceRows.length), mono: true },
                            { label: "Shown in chart", value: String(chartRows.length), mono: true },
                          ]}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Table — same card, divided by a hairline so it reads as
                    connected to the chart above rather than a separate block */}
                <div className="mt-5 border-t border-slate-100 pt-5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Full Ranking</h3>
                    <div className="relative">
                      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
                      <input
                        value={featureSearch}
                        onChange={(e) => { setFeatureSearch(e.target.value); setFeaturePage(1); }}
                        placeholder="Search feature…"
                        className="w-44 rounded-lg border border-slate-200 py-1.5 pl-8 pr-3 text-xs outline-none focus:border-blue-400"
                      />
                    </div>
                  </div>

                  <div className="mt-3 overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">
                          <th className="py-2 pr-3">#</th>
                          <th className="py-2 pr-3">Feature</th>
                          <th className="py-2 pr-3 text-right">Importance</th>
                          <th className="py-2 pl-3">Relative Weight</th>
                        </tr>
                      </thead>
                      <tbody>
                        {featureImportancePaged.map((row, i) => {
                          const rank = (featureTablePageSafe - 1) * featureTablePageSize + i + 1;
                          const barPct = Math.max(2, (row.Importance / maxImportanceValue) * 100);
                          return (
                            <tr key={row.Feature} className="border-t border-slate-100">
                              <td className="py-2.5 pr-3">
                                <span className={cn("inline-flex h-5 min-w-[20px] items-center justify-center rounded-md border px-1 text-[11px] font-bold", rank <= 3 ? "border-blue-200 bg-blue-50 text-blue-600" : "border-transparent text-slate-400")}>
                                  {rank}
                                </span>
                              </td>
                              <td className={cn("py-2.5 pr-3 font-mono text-xs", rank <= 3 ? "font-semibold text-slate-900" : "text-slate-700")}>{row.Feature}</td>
                              <td className="py-2.5 pr-3 text-right">
                                <span className={cn("font-mono font-semibold tabular-nums", rank === 1 ? "rounded-md border border-blue-100 bg-blue-50 px-2 py-0.5 text-blue-700" : "text-slate-800")}>
                                  {row.Importance.toFixed(4)}
                                </span>
                              </td>
                              <td className="py-2.5 pl-3" style={{ minWidth: 160 }}>
                                <div className="flex items-center gap-2.5">
                                  <div className="h-1.5 flex-1 rounded-full bg-slate-100">
                                    <div className="h-full rounded-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-500" style={{ width: `${barPct}%` }} />
                                  </div>
                                  <span className="w-12 shrink-0 text-right text-[11px] font-semibold tabular-nums text-slate-500">
                                    {(row.Importance * 100).toFixed(2)}%
                                  </span>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  <TablePagination
                    page={featureTablePageSafe}
                    totalPages={featureTableTotalPages}
                    onChange={setFeaturePage}
                    totalCount={featureImportanceFiltered.length}
                    label="features"
                  />
                </div>
              </>
            ) : (
              <p className="text-sm text-slate-500">Feature importance not available for this model type.</p>
            )}
          </SectionCard>
        </TabsContent>

        {/* ── SHAP Analysis — controls always visible; a purposeful empty
             state before compute, a real analytics grid after. ── */}
        <TabsContent value="shap" className={cn("pt-5", TAB_TRANSITION)}>
          <SectionCard
            icon="🔬"
            title="SHAP Values"
            sub="Understand how individual features push predictions higher or lower using SHAP analysis."
          >
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
              <div className="flex-1">
                <RangeSliderRow label="Samples for SHAP analysis" value={shapSamples} min={50} max={300} step={50} onChange={setShapSamples} />
              </div>
              <Button
                onClick={computeShap}
                disabled={shapLoading}
                className="gap-2 bg-gradient-to-br from-blue-700 via-blue-600 to-indigo-500 shadow-[0_6px_20px_rgba(37,99,235,0.35)] transition-opacity hover:opacity-90 sm:w-60"
              >
                {shapLoading ? <Loader className="h-4 w-4 animate-spin" /> : null}
                {shapLoading ? "Computing…" : "▶ Compute SHAP Values"}
              </Button>
            </div>

            {shapError && <StatusBanner tone="warning" icon={AlertTriangle} className="mt-4">{shapError}</StatusBanner>}

            <div className="mt-4">
              {shapLoading ? (
                <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-slate-200 bg-slate-50/70 px-6 py-12 text-center">
                  <Loader className="h-6 w-6 animate-spin text-blue-600" />
                  <div className="text-sm font-semibold text-slate-900">Computing SHAP values…</div>
                  <p className="max-w-sm text-xs text-slate-500">
                    Analyzing {shapSamples} samples against the trained model. This can take a moment for larger sample sizes.
                  </p>
                </div>
              ) : shapComputed && shapSummaryRows.length > 0 ? (
                <>
                  <StatusBanner tone="success" dot>
                    SHAP values computed for {shapSummaryRows.length >= 15 ? "15+" : shapSummaryRows.length} features using {shapSamples} samples.
                  </StatusBanner>
                  <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-[1fr_280px]">
                    <div className="overflow-hidden">
                      <PlotlyChart figure={shapSummaryFigure} style={{ minHeight: 384 }} config={{ displayModeBar: false }} />
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-slate-50/60 p-4">
                      <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">At a Glance</div>
                      <div className="mt-3">
                        <DefList
                          items={[
                            { label: "Highest impact", value: topShapFeature?.Feature ?? "—" },
                            { label: "Mean |SHAP|", value: topShapFeature ? (topShapFeature.MeanAbsSHAP ?? 0).toFixed(4) : "—", mono: true },
                            { label: "Features covered", value: String(shapSummaryRows.length), mono: true },
                            { label: "Samples used", value: String(shapSamples), mono: true },
                          ]}
                        />
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <EmptyState
                  icon={Microscope}
                  title="SHAP analysis is ready to run"
                  description="Select a sample size above, then compute SHAP values to see which features push predictions higher or lower."
                />
              )}
            </div>
          </SectionCard>
        </TabsContent>

        {/* ── Prediction Reasoning — a single numbered workflow: select →
             result → drivers → contribution chart → raw values. Gated
             behind SHAP being computed, with a deep-link CTA instead of a
             dead-end banner. ── */}
        <TabsContent value="reasoning" className={cn("space-y-4 pt-5", TAB_TRANSITION)}>
          <SectionCard
            icon="🔍"
            eyebrow="Step 1 · Select Prediction"
            title="Individual Prediction Reasoning"
            sub="Inspect exactly why a specific customer was classified as risky or safe — with real feature names."
          >
            <div className="flex flex-wrap items-end gap-3">
              <div className="w-full max-w-[220px]">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Customer Index (Row Number)
                </label>
                <input
                  type="number"
                  min={0}
                  max={Math.max(0, sampleCount - 1)}
                  value={sampleIdx}
                  onChange={(e) => void changeSample(Math.max(0, Number(e.target.value)))}
                  className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400"
                />
              </div>
              {shapComputed && sampleCount > 0 && (
                <span className="pb-2.5 text-xs text-slate-400">of {sampleCount} scored sample{sampleCount === 1 ? "" : "s"}</span>
              )}
            </div>

            {!shapComputed ? (
              <div className="mt-5">
                <EmptyState
                  icon={Lock}
                  title="Compute SHAP values to unlock prediction reasoning"
                  description="This view is derived from SHAP values for the selected customer. Head to SHAP Analysis and compute them first."
                  action={
                    <Button size="sm" onClick={() => setActiveTab("shap")} className="gap-2">
                      Go to SHAP Analysis
                      <ArrowRight className="h-3.5 w-3.5" />
                    </Button>
                  }
                />
              </div>
            ) : parsedReasoning ? (
              <div className="mt-5 border-t border-slate-100 pt-5">
                <div className="text-[10px] font-bold uppercase tracking-wider text-blue-600">Step 2 · Prediction Result</div>
                <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <div
                    className={cn(
                      "relative overflow-hidden rounded-2xl p-4 shadow-md",
                      parsedReasoning.isHighRisk
                        ? "bg-gradient-to-br from-rose-950 via-rose-900 to-rose-800"
                        : "bg-gradient-to-br from-emerald-950 via-emerald-900 to-emerald-800",
                    )}
                  >
                    <span className="pointer-events-none absolute -right-2 -top-3 text-5xl opacity-10">{parsedReasoning.isHighRisk ? "⚠" : "✓"}</span>
                    <div className="flex items-center gap-2">
                      <span className="relative flex h-2 w-2 shrink-0">
                        <span className={cn("absolute inline-flex h-full w-full animate-ping rounded-full opacity-75", parsedReasoning.isHighRisk ? "bg-rose-400" : "bg-emerald-400")} />
                        <span className={cn("relative inline-flex h-2 w-2 rounded-full", parsedReasoning.isHighRisk ? "bg-rose-400" : "bg-emerald-400")} />
                      </span>
                      <span className="text-[10px] font-bold uppercase tracking-wider text-white/60">Prediction</span>
                    </div>
                    <div className="mt-2 text-lg font-extrabold text-white">
                      {parsedReasoning.isHighRisk ? "🔴" : "🟢"} {parsedReasoning.riskLabel}
                    </div>
                    <div className="mt-1 text-[11px] text-white/60">{parsedReasoning.detail}</div>
                  </div>
                  <div className="flex flex-col items-center justify-center rounded-2xl bg-gradient-to-br from-slate-900 to-blue-900 p-4 shadow-md">
                    <Gauge pct={parsedReasoning.probability} color={parsedReasoning.isHighRisk ? "#fb7185" : "#60a5fa"} />
                    <div className="text-xl font-extrabold text-white">
                      <AnimatedNumber value={parsedReasoning.probability} formatter={(n) => `${n.toFixed(1)}%`} />
                    </div>
                    <div className="mt-0.5 text-[10px] font-bold uppercase tracking-wider text-white/50">Default Probability</div>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Decision Threshold</div>
                    <div className="mt-2 text-2xl font-extrabold tabular-nums text-slate-900">{parsedReasoning.threshold}%</div>
                    <div className="mt-2.5 h-1.5 rounded-full bg-slate-100">
                      <div
                        className={cn("h-full rounded-full transition-all duration-700", parsedReasoning.isHighRisk ? "bg-rose-500" : "bg-blue-500")}
                        style={{ width: `${Math.min(100, parsedReasoning.probability)}%` }}
                      />
                    </div>
                    <div className="mt-2 text-[11px] text-slate-500">
                      Prob {parsedReasoning.probability}% —{" "}
                      <strong className={parsedReasoning.isHighRisk ? "text-rose-600" : "text-emerald-600"}>
                        {parsedReasoning.probability < parsedReasoning.threshold ? "well below" : "at or above"}
                      </strong>{" "}
                      threshold
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              data?.shap.sample_reasoning && (
                <div className="mt-5 whitespace-pre-line rounded-xl border border-slate-200 bg-slate-50/60 p-4 text-sm text-slate-700">
                  {plainMarkdown(data.shap.sample_reasoning)}
                </div>
              )
            )}
          </SectionCard>

          {shapComputed && (topDriversReduces.length > 0 || topDriversIncreases.length > 0) && (
            <SectionCard
              icon="🎯"
              eyebrow="Step 3 · Why This Prediction"
              title="Top Prediction Drivers"
              sub="SHAP attribution for each feature — positive values increase default risk"
            >
              <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                <div>
                  <div className="mb-3 flex items-center gap-2">
                    <span className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-100 text-emerald-700">
                      <TrendingDown className="h-3.5 w-3.5" />
                    </span>
                    <span className="text-xs font-bold text-emerald-700">Reduces Default Risk</span>
                  </div>
                  <div className="space-y-2">
                    {topDriversReduces.map((d) => (
                      <div key={d.Feature} className="flex items-center justify-between gap-3 rounded-lg border border-emerald-200 bg-emerald-50 px-3.5 py-2.5">
                        <div className="min-w-0">
                          <div className="truncate text-xs font-semibold text-slate-900">{d.Feature}</div>
                          <div className="text-[11px] text-slate-500">value: {formatValue(d.Value)}</div>
                        </div>
                        <span className="shrink-0 rounded-md border border-emerald-300 bg-white px-2 py-0.5 text-xs font-bold tabular-nums text-emerald-700">
                          {d.SHAP.toFixed(4)}
                        </span>
                      </div>
                    ))}
                    {topDriversReduces.length === 0 && <p className="text-xs text-slate-400">No risk-reducing drivers found.</p>}
                  </div>
                </div>
                <div>
                  <div className="mb-3 flex items-center gap-2">
                    <span className="flex h-6 w-6 items-center justify-center rounded-md bg-rose-100 text-rose-700">
                      <TrendingUp className="h-3.5 w-3.5" />
                    </span>
                    <span className="text-xs font-bold text-rose-700">Increases Default Risk</span>
                  </div>
                  <div className="space-y-2">
                    {topDriversIncreases.map((d) => (
                      <div key={d.Feature} className="flex items-center justify-between gap-3 rounded-lg border border-rose-200 bg-rose-50 px-3.5 py-2.5">
                        <div className="min-w-0">
                          <div className="truncate text-xs font-semibold text-slate-900">{d.Feature}</div>
                          <div className="text-[11px] text-slate-500">value: {formatValue(d.Value)}</div>
                        </div>
                        <span className="shrink-0 rounded-md border border-rose-300 bg-white px-2 py-0.5 text-xs font-bold tabular-nums text-rose-700">
                          +{d.SHAP.toFixed(4)}
                        </span>
                      </div>
                    ))}
                    {topDriversIncreases.length === 0 && <p className="text-xs text-slate-400">No risk-increasing drivers found.</p>}
                  </div>
                </div>
              </div>
            </SectionCard>
          )}

          {shapComputed && sampleShapRows.length > 0 && (
            <SectionCard
              icon="📉"
              eyebrow="Step 4 · Contribution Visualization"
              title="SHAP Contribution Chart"
              sub="How each feature shifts the prediction — sorted by magnitude of impact"
            >
              <div className="overflow-hidden">
                <PlotlyChart figure={sampleShapFigure} style={{ minHeight: 320 }} config={{ displayModeBar: false }} />
              </div>
            </SectionCard>
          )}

          {shapComputed && Object.keys(data?.shap.sample_features ?? {}).length > 0 && (
            <SectionCard icon="🔎" eyebrow="Step 5 · Reference" title="Raw Feature Values" sub={`Full feature vector for customer #${sampleIdx}`}>
              <details className="group">
                <summary className="flex cursor-pointer list-none items-center gap-2 text-sm font-semibold text-slate-700">
                  <ChevronRight className="h-3.5 w-3.5 shrink-0 text-blue-600 transition-transform group-open:rotate-90" />
                  🔎 Raw Feature Values for This Customer
                </summary>
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">
                        <th className="py-2 pr-3">#</th>
                        <th className="py-2 pr-3">Feature</th>
                        <th className="py-2 pr-3">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(data?.shap.sample_features ?? {}).map(([feat, val], rowIndex) => (
                        <tr key={feat} className="border-t border-slate-100">
                          <td className="py-2 pr-3 font-mono text-xs text-slate-400">{rowIndex + 1}</td>
                          <td className="py-2 pr-3 font-mono text-xs text-slate-700">{feat}</td>
                          <td className="py-2 pr-3 font-mono text-xs font-semibold text-slate-900">{formatValue(val)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </SectionCard>
          )}
        </TabsContent>

        {/* ── Summary — executive overview: KPI strip, then paired 2-column
             grids for related sections so the report reads as a dashboard
             rather than a linear stack of full-width printouts. ── */}
        <TabsContent value="summary" className={cn("space-y-4 pt-5", TAB_TRANSITION)}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-slate-500">Report generated {reportGeneratedAt}</p>
            <Button onClick={downloadReport} className="no-print gap-2 bg-gradient-to-br from-slate-900 via-blue-800 to-blue-600 shadow-[0_6px_20px_rgba(37,99,235,0.35)] transition-opacity hover:opacity-90">
              <Printer className="h-4 w-4" />
              Download Full Report
            </Button>
          </div>

          {importanceRows.length > 0 && (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { icon: "🏆", label: "Top Feature", value: summaryTopFeature, sub: summaryTopFeatureSub, grad: "from-slate-900 to-blue-900", accent: "bg-blue-400" },
                { icon: "⚡", label: "Key Risk Driver", value: summaryKeyRiskDriver, sub: "2nd-ranked feature", grad: "from-amber-950 to-amber-800", accent: "bg-amber-400" },
                { icon: "🔢", label: "Features Analyzed", value: summaryFeaturesAnalyzed, sub: "real column names", grad: "from-emerald-950 to-emerald-800", accent: "bg-emerald-400" },
                { icon: "🧮", label: "Explainability Method", value: summaryShapAvailable ? "SHAP" : "Not yet computed", sub: summaryShapAvailable ? "TreeExplainer" : undefined, grad: "from-violet-950 to-violet-800", accent: "bg-violet-400" },
              ].map((k) => (
                <div key={k.label} className={cn("relative overflow-hidden rounded-2xl bg-gradient-to-br p-4 shadow-md", k.grad)}>
                  <span className="pointer-events-none absolute -right-2 -top-3 text-5xl opacity-10">{k.icon}</span>
                  <div className="text-[10px] font-bold uppercase tracking-wider text-white/50">{k.label}</div>
                  <div className="mt-2 truncate text-base font-bold text-white" title={k.value}>{k.value}</div>
                  <div className="mt-2 flex items-center gap-1.5">
                    <span className={cn("h-[3px] w-5 rounded-full", k.accent)} />
                    {k.sub && <span className="truncate text-[11px] text-white/60">{k.sub}</span>}
                  </div>
                </div>
              ))}
            </div>
          )}

          <div id="full-report-content" className="space-y-4">
            {/* Model Identity + Dataset Summary — both light, pair well */}
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <SectionCard icon="🪪" title="1. Model Identity">
                <DefList
                  items={[
                    { label: "Model", value: trainingResult?.model_name ?? "—", highlight: true },
                    { label: "Task type", value: trainingResult?.task_type ?? "—" },
                    { label: "Target column", value: targetColumn, mono: true },
                    { label: "Report generated", value: reportGeneratedAt },
                  ]}
                />
              </SectionCard>

              <SectionCard icon="🗄️" title="2. Dataset Summary">
                <div className="grid grid-cols-2 gap-2">
                  <StatTile label="Rows" value={profile?.shape?.[0]?.toLocaleString() ?? "—"} />
                  <StatTile label="Columns" value={profile?.shape?.[1]?.toLocaleString() ?? "—"} />
                  <StatTile label="Missing" value={profile?.missing_percentage !== undefined ? `${profile.missing_percentage}%` : "—"} />
                  <StatTile label="Duplicates" value={profile?.duplicate_rows ?? "—"} />
                </div>
                {Object.keys(classDistribution).length > 0 && (
                  <div className="mt-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(classDistribution).map(([cls, count]) => (
                          <span key={cls} className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-[11px] font-medium text-slate-700">
                            {cls}: {count.toLocaleString()}
                          </span>
                        ))}
                      </div>
                      {profile?.target_summary?.imbalance_ratio !== undefined && profile.target_summary.imbalance_ratio !== null && (
                        <span className="text-[11px] text-slate-500">Imbalance {profile.target_summary.imbalance_ratio}:1</span>
                      )}
                    </div>
                    <div className="mt-3 h-40 overflow-hidden">
                      <PlotlyChart figure={classDistributionFigure} style={{ minHeight: 144 }} config={{ displayModeBar: false }} />
                    </div>
                  </div>
                )}
              </SectionCard>
            </div>

            {/* Preprocessing + Feature Engineering — pair well, both stat-tile driven */}
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <SectionCard icon="⚙️" title="3. Preprocessing Decisions">
                <div className="grid grid-cols-3 gap-2">
                  <StatTile label="Train" value={splitStats.train_n?.toLocaleString() ?? "—"} />
                  <StatTile label="Validation" value={splitStats.val_n?.toLocaleString() ?? "—"} />
                  <StatTile label="Test" value={splitStats.test_n?.toLocaleString() ?? "—"} />
                </div>
                {treatmentCounts.length > 0 && (
                  <ul className="mt-4 space-y-1 text-sm text-slate-700">
                    {treatmentCounts.map((t) => (
                      <li key={t.treatment} className="flex items-center justify-between border-b border-slate-100 pb-1.5 last:border-0">
                        <span className="text-slate-500">{t.label}</span>
                        <strong className="text-slate-900">{t.count} column{t.count === 1 ? "" : "s"}</strong>
                      </li>
                    ))}
                  </ul>
                )}
              </SectionCard>

              <SectionCard icon="🧩" title="4. Feature Engineering Summary">
                <div className="grid grid-cols-4 gap-2">
                  <StatTile label="Before" value={feOriginalFeatures ?? "—"} />
                  <StatTile label="After" value={feFinalFeatures ?? "—"} />
                  <StatTile label="Added" value={feAdded.length} />
                  <StatTile label="Removed" value={feRemoved.length} />
                </div>
                {(interactionRows.length > 0 || woeRows.length > 0 || rescuedFeatures.length > 0) && (
                  <div className="mt-4 space-y-3 text-sm text-slate-700">
                    {interactionRows.length > 0 && (
                      <div>
                        <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Top interaction terms</div>
                        <ul className="mt-1.5 space-y-1">
                          {interactionRows.slice(0, 3).map((f: any) => (
                            <li key={f.name}><code className="font-mono text-xs text-slate-900">{f.name}</code></li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {woeRows.length > 0 && (
                      <div>
                        <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">WOE-encoded features</div>
                        <p className="mt-1.5 text-xs text-slate-600">{woeRows.length} feature{woeRows.length === 1 ? "" : "s"} bucketed via weight-of-evidence.</p>
                      </div>
                    )}
                  </div>
                )}
              </SectionCard>
            </div>

            {/* Wide preprocessing strategy table — needs the full row width */}
            {strategyRows.length > 0 && (
              <SectionCard icon="📐" title="Preprocessing Strategy by Column">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">
                        <th className="py-2 pr-3">Column</th>
                        <th className="py-2 pr-3">Type</th>
                        <th className="py-2 pr-3">Scaler</th>
                        <th className="py-2 pr-3">Imputer</th>
                        <th className="py-2 pr-3">Encoding</th>
                        <th className="py-2 pr-3">Transform</th>
                      </tr>
                    </thead>
                    <tbody>
                      {strategyRows.map((row: any) => (
                        <tr key={row.feature} className="border-t border-slate-100">
                          <td className="py-2 pr-3 font-mono text-xs text-slate-700">{row.feature}</td>
                          <td className="py-2 pr-3 text-xs text-slate-600">{row.type}</td>
                          <td className="py-2 pr-3 text-xs text-slate-600">{row.scaler}</td>
                          <td className="py-2 pr-3 text-xs text-slate-600">{row.imputer}</td>
                          <td className="py-2 pr-3 text-xs text-slate-600">{row.encoding}</td>
                          <td className="py-2 pr-3 text-xs text-slate-600">{row.transform}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </SectionCard>
            )}

            {/* Training configuration */}
            <SectionCard icon="🎛️" title="5. Model Training Configuration">
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <DefList
                  items={[
                    { label: "Model type", value: trainingResult?.model_name ?? "—", highlight: true },
                    { label: "Class balancing", value: trainingInfo.class_weighting?.method ?? (trainingInfo.class_weighting?.applied ? "class_weight='balanced'" : "None") },
                    { label: "Cross-validation", value: trainingConfig2.use_cv ? `${trainingConfig2.cv_folds}-fold (mean ${trainingInfo.cv_mean ?? "—"}, std ${trainingInfo.cv_std ?? "—"})` : "Not used" },
                    { label: "OOT holdout", value: trainingConfig2.use_oot ? `Cutoff ${trainingInfo.oot?.cutoff_date ?? "—"}, ${trainingInfo.oot?.oot_n ?? 0} rows` : "Not used" },
                  ]}
                />
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Hyperparameters — {hyperparams.source}</div>
                  {Object.keys(hyperparams.params).length > 0 ? (
                    <ul className="mt-2 grid grid-cols-2 gap-1.5 text-sm text-slate-700">
                      {Object.entries(hyperparams.params).map(([k, v]) => (
                        <li key={k}><code className="font-mono text-xs text-slate-900">{k}</code>: {String(v)}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-2 text-sm text-slate-500">No overrides recorded — the model used its library defaults.</p>
                  )}
                </div>
              </div>
            </SectionCard>

            {/* Evaluation metrics — heavy card: tiles, chart + report table side by side */}
            <SectionCard icon="🧠" title="6. Evaluation Metrics">
              <div className="grid grid-cols-3 gap-2 sm:grid-cols-5">
                <StatTile label="Accuracy" value={metrics.accuracy ?? "—"} />
                <StatTile label="Precision" value={metrics.precision ?? "—"} />
                <StatTile label="Recall" value={metrics.recall ?? "—"} />
                <StatTile label="F1" value={metrics.f1 ?? "—"} />
                <StatTile label="ROC-AUC" value={metrics.roc_auc ?? "—"} />
                <StatTile label="PR-AUC" value={metrics.pr_auc ?? "—"} />
                <StatTile label="KS" value={metrics.ks_statistic ?? "—"} />
                <StatTile label="Brier" value={metrics.brier_score ?? "—"} />
                <StatTile label="Gini" value={gini ?? "—"} />
              </div>

              {metrics.threshold_used !== undefined && (
                <p className="mt-4 text-sm text-slate-700">
                  <strong className="text-slate-900">Decision threshold:</strong> {metrics.threshold_used}
                  {metrics.threshold_selection && (
                    <> — auto-selected to maximize {metrics.threshold_selection.metric} (F1 {metrics.threshold_selection.f1}, precision {metrics.threshold_selection.precision}, recall {metrics.threshold_selection.recall})</>
                  )}
                </p>
              )}

              {(metricsChartData.length > 0 || Object.keys(classificationReport).length > 0) && (
                <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
                  {metricsChartData.length > 0 && (
                    <div className="overflow-hidden">
                      <PlotlyChart figure={metricsFigure} style={{ minHeight: 240 }} config={{ displayModeBar: false }} />
                    </div>
                  )}
                  {Object.keys(classificationReport).length > 0 && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">
                            <th className="py-2 pr-3">Class</th>
                            <th className="py-2 pr-3">Precision</th>
                            <th className="py-2 pr-3">Recall</th>
                            <th className="py-2 pr-3">F1</th>
                            <th className="py-2 pr-3">Support</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(classificationReport)
                            .filter(([key]) => key !== "accuracy")
                            .map(([key, row]: [string, any]) => (
                              <tr key={key} className="border-t border-slate-100">
                                <td className="py-2 pr-3 font-mono text-xs text-slate-700">{key}</td>
                                <td className="py-2 pr-3 text-xs text-slate-600">{row.precision?.toFixed?.(4) ?? "—"}</td>
                                <td className="py-2 pr-3 text-xs text-slate-600">{row.recall?.toFixed?.(4) ?? "—"}</td>
                                <td className="py-2 pr-3 text-xs text-slate-600">{row["f1-score"]?.toFixed?.(4) ?? "—"}</td>
                                <td className="py-2 pr-3 text-xs text-slate-600">{row.support ?? "—"}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </SectionCard>

            {/* Explainability findings — ranked list + chart side by side */}
            <SectionCard icon="📝" title="7. Explainability Findings">
              {importanceRows.length > 0 && (
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <div className="space-y-2.5">
                    {importanceRows.slice(0, 10).map((r, i) => {
                      const barPct = Math.max(2, (r.Importance / maxImportanceValue) * 100);
                      return (
                        <div key={r.Feature}>
                          <div className="flex items-center justify-between gap-2 text-sm">
                            <div className="flex min-w-0 items-center gap-2">
                              <span
                                className={cn(
                                  "flex h-5 w-5 shrink-0 items-center justify-center rounded-md text-[10px] font-bold",
                                  i < 3 ? "bg-gradient-to-br from-blue-700 to-blue-500 text-white" : "bg-slate-100 text-slate-400",
                                )}
                              >
                                {i + 1}
                              </span>
                              <code className="truncate font-mono text-xs text-slate-900">{r.Feature}</code>
                            </div>
                            <span className="shrink-0 font-mono text-xs font-bold text-blue-700">{(r.Importance * 100).toFixed(2)}%</span>
                          </div>
                          <div className="mt-1 ml-7 h-1 rounded-full bg-slate-100">
                            <div className="h-full rounded-full bg-gradient-to-r from-blue-600 to-blue-400" style={{ width: `${barPct}%` }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="overflow-hidden">
                    <PlotlyChart figure={importanceTop10Figure} style={{ minHeight: 260 }} config={{ displayModeBar: false }} />
                  </div>
                </div>
              )}
              {data?.summary ? (
                <div className={cn("whitespace-pre-line text-sm text-slate-700", importanceRows.length > 0 && "mt-4 border-t border-slate-100 pt-4")}>{plainMarkdown(data.summary)}</div>
              ) : (
                <p className={cn("text-sm text-slate-500", importanceRows.length > 0 && "mt-4 border-t border-slate-100 pt-4")}>Model performance summary is not available yet.</p>
              )}
            </SectionCard>
          </div>

          {/* Document Download Hub */}
          <SectionCard icon="📥" title="Document Download Hub" sub="Every downloadable artifact generated across the Model Development pipeline, in one place." className="no-print">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
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
          </SectionCard>
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
