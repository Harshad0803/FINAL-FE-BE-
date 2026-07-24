import { createFileRoute, Link } from "@tanstack/react-router";
import React from "react";
import { PageHeader } from "@/components/app-shell";
import { ArrowRight, AlertTriangle, RefreshCw, Printer, ChevronDown, ChevronRight } from "lucide-react";
import { ApiError, formUpload } from "@/lib/api";
import { useDataset } from "@/lib/app-context";

export const Route = createFileRoute("/validation/findings")({
  head: () => ({ meta: [{ title: "Stage 7 — Findings & Final Report — Aegis Credit" }] }),
  component: Findings,
});

type Finding = {
  stage: string;
  check: string;
  severity: string;
  status: string;
  finding: string;
  recommendation: string;
  regulation: string;
};

type Stage8Response = {
  findings: Finding[];
  verdict: "APPROVED" | "CONDITIONALLY APPROVED" | "REJECTED" | string;
  verdict_desc: string;
  high_count: number;
  medium_count: number;
  low_count: number;
  total_count: number;
  monitoring_frequency: string;
  revalidation_trigger: string;
  model_tier: string;
  stated_auc: number | null;
  replicated_auc: number | null;
};

type RemediationRow = {
  finding: string;
  severity: string;
  detail: string;
  owner: string;
  targetDate: string;
  status: string;
};

const VERDICT_STYLES: Record<string, { border: string; bg: string; text: string; icon: string }> = {
  APPROVED: { border: "border-emerald-500", bg: "bg-emerald-500/10", text: "text-emerald-600 dark:text-emerald-300", icon: "✅" },
  "CONDITIONALLY APPROVED": { border: "border-amber-500", bg: "bg-amber-500/10", text: "text-amber-600 dark:text-amber-300", icon: "⚠️" },
  REJECTED: { border: "border-red-500", bg: "bg-red-500/10", text: "text-red-600 dark:text-red-300", icon: "❌" },
};

const SEVERITY_STYLES: Record<string, { border: string; bg: string; badge: string }> = {
  HIGH: { border: "border-red-500/40", bg: "bg-red-500/10", badge: "bg-red-500 text-red-950" },
  MEDIUM: { border: "border-amber-500/40", bg: "bg-amber-500/10", badge: "bg-amber-500 text-amber-950" },
  LOW: { border: "border-emerald-500/40", bg: "bg-emerald-500/10", badge: "bg-emerald-500 text-emerald-950" },
};

function severityStyle(sev: string) {
  return SEVERITY_STYLES[sev?.toUpperCase()] ?? { border: "border-border", bg: "bg-card", badge: "bg-muted text-foreground" };
}

function csvEscape(value: unknown): string {
  const s = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function downloadCsv(filename: string, rows: Array<Record<string, unknown>>) {
  if (!rows.length) return;
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(","), ...rows.map((r) => headers.map((h) => csvEscape(r[h])).join(","))];
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function todayLabel() {
  return new Date().toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

// Plain-English justification for the verdict — mirrors the exact thresholds
// the /validation/stage8/findings endpoint uses to decide it (main.py: verdict
// is APPROVED iff 0 HIGH findings and <=2 MEDIUM; CONDITIONALLY APPROVED iff
// <=2 HIGH findings; REJECTED otherwise). Kept in sync with that logic rather
// than re-deriving it, since the API only returns the outcome, not the "why".
function verdictReasoning(data: Stage8Response): string {
  const { verdict, high_count, medium_count } = data;
  if (verdict === "APPROVED") {
    return `Approved because there are 0 unresolved HIGH findings and ${medium_count} MEDIUM finding(s) — both within the thresholds required for full approval (max 2 MEDIUM, 0 HIGH).`;
  }
  if (verdict === "CONDITIONALLY APPROVED") {
    const mediumNote = medium_count > 2
      ? ` (and ${medium_count} MEDIUM findings, which alone would exceed the 2 allowed for full approval)`
      : "";
    return `Conditionally approved because ${high_count} HIGH finding(s) were raised${mediumNote} — this is within the maximum of 2 HIGH findings allowed for conditional approval, so deployment may proceed once they are resolved within the agreed timeframe.`;
  }
  return `Rejected because ${high_count} unresolved HIGH findings exceed the maximum of 2 allowed for conditional approval — full remediation and resubmission is required before this model can be reconsidered.`;
}

// Same Stage 5 metric/threshold table the backend evaluates (main.py, Stage
// 5 section of /validation/stage8/findings) — reproduced here because the
// API only returns FAILING metrics as findings, not the full pass/fail
// picture needed to show "why something passed" as well as "why it failed".
type MetricThresholdRow = { metric: string; value: number | null; threshold: number; op: ">=" | "<="; regulation: string; pass: boolean | null };

function metricThresholdRows(repMetrics: Record<string, any>): MetricThresholdRow[] {
  const rocAuc = typeof repMetrics.roc_auc === "number" ? repMetrics.roc_auc : null;
  const gini = rocAuc !== null ? Math.round((2 * rocAuc - 1) * 10000) / 10000 : null;
  const rows: Array<[string, number | null, number, ">=" | "<=", string]> = [
    ["ROC-AUC", rocAuc, 0.70, ">=", "SS1/23 P4.1"],
    ["Recall", typeof repMetrics.recall === "number" ? repMetrics.recall : null, 0.60, ">=", "SS1/23 P4.4"],
    ["Gini", gini, 0.40, ">=", "SS11/13 §10.3"],
    ["Brier Score", typeof repMetrics.brier_score === "number" ? repMetrics.brier_score : null, 0.25, "<=", "SS11/13 §10.5"],
  ];
  return rows.map(([metric, value, threshold, op, regulation]) => ({
    metric,
    value,
    threshold,
    op,
    regulation,
    pass: value === null ? null : op === ">=" ? value >= threshold : value <= threshold,
  }));
}

const STAGE_ORDER = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5", "Stage 6"];

function Findings() {
  const ds = useDataset();
  const {
    file,
    profile,
    validationIntakeData,
    validationMddText,
    validationProfile,
    validationStage5Result,
    validationStage7BiasResult,
    validationStage8Result,
    setValidationStage8Result,
  } = ds;

  const datasetName = profile?.dataset_name ?? file?.name ?? "the active validation dataset";
  const datasetReady = Boolean(file || profile?.csv_text || profile?.dataset_name);

  const [loading, setLoading] = React.useState(!validationStage8Result);
  const [error, setError] = React.useState<string | null>(null);
  // "network" = the request never reached the backend at all (connection
  // refused, CORS block, backend down/restarting) — the fetch() promise
  // rejects directly rather than resolving with a non-2xx status, so it
  // reads as a transient infrastructure issue rather than a broken page.
  // "server" = the backend responded but with an error (4xx/5xx/bad body).
  const [errorKind, setErrorKind] = React.useState<"network" | "server" | null>(null);
  const [data, setData] = React.useState<Stage8Response | null>((validationStage8Result as Stage8Response | null) ?? null);
  const skipInitialAutoRun = React.useRef(validationStage8Result !== null && validationStage8Result !== undefined);
  const [retryToken, setRetryToken] = React.useState(0);

  React.useEffect(() => {
    if (skipInitialAutoRun.current) {
      skipInitialAutoRun.current = false;
      setLoading(false);
      return;
    }

    let active = true;
    setLoading(true);
    setError(null);
    setErrorKind(null);

    const repMetrics = (validationStage5Result as any)?.report?.metrics ?? {};
    const benchmarkMetrics = (validationStage5Result as any)?.report?.benchmark?.metrics ?? {};
    const biasRows = ((validationStage7BiasResult as any)?.rows ?? []) as Array<{ AUC: number | null }>;
    const biasAucVals = biasRows.map((r) => r.AUC).filter((v): v is number => v !== null);
    const biasAucGap = biasAucVals.length >= 2 ? Math.max(...biasAucVals) - Math.min(...biasAucVals) : null;

    const form = new FormData();
    form.append("intake_json", JSON.stringify(validationIntakeData ?? {}));
    form.append("mdd_text", validationMddText ?? "");
    form.append(
      "validation_profile_json",
      JSON.stringify({
        missing_by_column: validationProfile?.missing_by_column ?? {},
        duplicate_rate: validationProfile?.duplicate_rate ?? 0,
      }),
    );
    form.append("rep_metrics_json", JSON.stringify(repMetrics));
    form.append("benchmark_metrics_json", JSON.stringify(benchmarkMetrics));
    if (biasAucGap !== null) form.append("bias_auc_gap", String(biasAucGap));

    void formUpload<Stage8Response>("/validation/stage8/findings", form)
      .then((resp) => {
        if (!active) return;
        setData(resp);
        setValidationStage8Result(resp as unknown as Record<string, any>);
      })
      .catch((err) => {
        console.error("Stage8 fetch error", err);
        if (!active) return;
        if (err instanceof ApiError) {
          setErrorKind("server");
          const detail =
            err.body && typeof err.body === "object" && "detail" in (err.body as any)
              ? String((err.body as any).detail)
              : err.message;
          setError(detail);
        } else {
          // Not an ApiError means fetch() itself rejected before getting a
          // response — a real network-level failure (backend unreachable,
          // CORS, DNS), not a request the server actually processed.
          setErrorKind("network");
          setError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!active) return;
        setLoading(false);
      });

    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validationIntakeData, validationMddText, validationProfile, validationStage5Result, validationStage7BiasResult, retryToken]);

  const findings = data?.findings ?? [];

  const [remediation, setRemediation] = React.useState<RemediationRow[]>([]);
  React.useEffect(() => {
    setRemediation((prev) => {
      if (prev.length === findings.length) return prev;
      return findings.map((f) => ({
        finding: `${f.stage} — ${f.check}`,
        severity: f.severity,
        detail: f.finding,
        owner: "",
        targetDate: "",
        status: "Open",
      }));
    });
  }, [findings]);

  const updateRemediation = (idx: number, patch: Partial<RemediationRow>) => {
    setRemediation((prev) => prev.map((r, i) => (i === idx ? { ...r, ...patch } : r)));
  };

  // Grouped-by-stage, collapsed-by-default view of the findings tracker —
  // one row per finding (Stage / Finding / Severity / Status), full detail
  // only shown once a row is expanded. S.No numbers the flattened list so
  // it stays stable/unique across groups.
  const [expandedFindings, setExpandedFindings] = React.useState<Record<number, boolean>>({});
  const toggleFinding = (idx: number) => setExpandedFindings((prev) => ({ ...prev, [idx]: !prev[idx] }));

  const groupedFindings = React.useMemo(() => {
    const indexed = findings.map((f, i) => ({ ...f, _sno: i + 1 }));
    const byStage = new Map<string, typeof indexed>();
    for (const f of indexed) {
      if (!byStage.has(f.stage)) byStage.set(f.stage, []);
      byStage.get(f.stage)!.push(f);
    }
    const orderedStages = [
      ...STAGE_ORDER.filter((s) => byStage.has(s)),
      ...Array.from(byStage.keys()).filter((s) => !STAGE_ORDER.includes(s)),
    ];
    return orderedStages.map((stage) => ({ stage, items: byStage.get(stage)! }));
  }, [findings]);

  const repMetricsForReport = (validationStage5Result as any)?.report?.metrics ?? {};
  const metricRows = React.useMemo(() => metricThresholdRows(repMetricsForReport), [repMetricsForReport]);

  const downloadFullReportPdf = () => window.print();

  const ij = validationIntakeData ?? {};
  const verdictStyle = VERDICT_STYLES[data?.verdict ?? ""] ?? VERDICT_STYLES.REJECTED;

  const execSummaryDefault = React.useMemo(() => {
    if (!data) return "";
    const stated = data.stated_auc;
    const rep = data.replicated_auc;
    const gini = rep !== null && rep !== undefined ? (2 * rep - 1).toFixed(4) : "N/A";
    const bmMetrics = (validationStage5Result as any)?.report?.benchmark?.metrics ?? {};
    const benchmarkLine =
      Object.keys(bmMetrics).length && rep !== null && rep !== undefined
        ? rep >= (bmMetrics.roc_auc ?? 0) - 0.02
          ? "Champion outperforms baseline"
          : "Champion underperforms baseline — see findings"
        : "N/A";

    return `Model: ${ij.model_name ?? "N/A"} (${ij.model_type ?? "N/A"})
Validation Date: ${todayLabel()}
Risk Tier: ${data.model_tier ?? "N/A"}
Verdict: ${data.verdict}

Performance Summary:
- Replicated AUC: ${rep !== null && rep !== undefined ? rep.toFixed(4) : "N/A"} (Stated: ${stated !== null && stated !== undefined ? stated.toFixed(4) : "N/A"})
- Gini: ${gini} | Recall: ${(validationStage5Result as any)?.report?.metrics?.recall ?? "N/A"}
- Benchmark: ${benchmarkLine}

Key Findings: ${data.total_count} total (${data.high_count} HIGH, ${data.medium_count} MEDIUM)
${data.high_count === 0 ? "No HIGH findings raised." : `${data.high_count} HIGH finding(s) require remediation before deployment.`}

Recommended Actions:
${
  data.verdict === "APPROVED"
    ? "Model approved for deployment subject to ongoing monitoring."
    : data.verdict === "CONDITIONALLY APPROVED"
      ? `Complete remediation of ${data.high_count} HIGH finding(s) by agreed deadlines.`
      : "Model rejected. Full remediation and resubmission required."
}
Monitoring frequency: ${data.monitoring_frequency}
Revalidation trigger: ${data.revalidation_trigger}`.trim();
  }, [data, ij, validationStage5Result]);

  const [execSummary, setExecSummary] = React.useState("");
  React.useEffect(() => {
    if (execSummaryDefault) setExecSummary(execSummaryDefault);
  }, [execSummaryDefault]);

  const [signOff, setSignOff] = React.useState({ validator: "", modelOwner: "", committee: "" });

  return (
    <div className="space-y-8">
      <PageHeader
        title="Stage 7 — Findings & Final Validation Report"
        description="SS1/23 P4.1/P5 · SS11/13 §13 — consolidated findings, verdict, and sign-off for the Model Risk Committee."
      />

      <section className="rounded-xl border border-border bg-background p-4 text-sm text-muted-foreground">
        {datasetReady ? (
          <>Using the shared dataset from Stage 1 / Stage 2: <span className="font-semibold text-foreground">{datasetName}</span>.</>
        ) : (
          <>No active dataset is available in shared state yet. Complete Stage 1 Intake and Stage 2 Data Validation first.</>
        )}
      </section>

      {loading ? (
        <div className="rounded-xl border border-border bg-card p-6 text-center">Compiling Stage 7 findings...</div>
      ) : error ? (
        <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
            <div className="flex-1">
              <div className="text-sm font-semibold text-destructive">
                {errorKind === "network"
                  ? "Backend unreachable — this looks like a transient network issue"
                  : "Error loading Stage 7 findings"}
              </div>
              <p className="mt-1 text-sm text-foreground/80">
                {errorKind === "network"
                  ? "The findings request never reached the server (connection refused, CORS, or the backend is still starting up). It's likely temporary — try again in a moment."
                  : error}
              </p>
              {errorKind === "network" ? <p className="mt-1 text-xs text-muted-foreground">{error}</p> : null}
              <button
                type="button"
                onClick={() => setRetryToken((t) => t + 1)}
                className="mt-3 inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold hover:border-primary/40"
              >
                <RefreshCw className="h-3.5 w-3.5" /> Retry
              </button>
            </div>
          </div>
        </div>
      ) : data ? (
        <>
          <div className="flex items-center justify-end">
            <button
              type="button"
              onClick={downloadFullReportPdf}
              className="no-print inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90"
            >
              <Printer className="h-4 w-4" />
              Download Full Report (PDF)
            </button>
          </div>

          <div id="full-report-content" className="space-y-8">
          {/* Model identity */}
          <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <h3 className="text-sm font-semibold">1. Model Identity</h3>
            <div className="mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
              <div><strong>Model:</strong> {ij.model_name ?? "N/A"}</div>
              <div><strong>Type:</strong> {ij.model_type ?? "N/A"}</div>
              <div><strong>Risk tier:</strong> {data.model_tier}</div>
              <div><strong>Validator:</strong> {signOff.validator || "—"}</div>
              <div><strong>Validation date:</strong> {todayLabel()}</div>
            </div>
          </section>

          {/* Overall verdict banner */}
          <section className={`rounded-xl border-2 ${verdictStyle.border} ${verdictStyle.bg} p-6 shadow-elegant`}>
            <h3 className="text-sm font-semibold">2. Overall Verdict</h3>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
              <div className={`text-2xl font-extrabold ${verdictStyle.text}`}>
                {verdictStyle.icon} {data.verdict}
              </div>
              <div className="text-right text-sm">
                <span className="font-bold text-red-600 dark:text-red-400">HIGH: {data.high_count}</span>{" "}
                <span className="font-bold text-amber-600 dark:text-amber-400">MEDIUM: {data.medium_count}</span>{" "}
                <span className="text-muted-foreground">Total findings: {data.total_count}</span>
              </div>
            </div>
            <p className="mt-3 text-sm text-foreground/80">{data.verdict_desc}</p>
            <p className="mt-2 text-sm font-medium text-foreground">{verdictReasoning(data)}</p>
            <p className="mt-2 text-xs text-muted-foreground">
              Model: {ij.model_name ?? "N/A"} · Type: {ij.model_type ?? "N/A"} · Tier: {data.model_tier} · Date: {todayLabel()}
            </p>
          </section>

          {/* Findings tracker */}
          <section className="space-y-4">
            <div>
              <h3 className="text-sm font-semibold">3. Findings Summary</h3>
              <p className="text-xs text-muted-foreground">
                Auto-compiled from Stages 1–7. All HIGH findings must be resolved before model deployment.
              </p>
            </div>

            {findings.length === 0 ? (
              <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-700 dark:text-emerald-300">
                ✅ No findings raised across all validation stages — model is fully compliant.
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div className="rounded-xl border border-border bg-card p-4">
                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Total Findings</div>
                    <div className="mt-2 text-2xl font-semibold">{data.total_count}</div>
                  </div>
                  <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4">
                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground">🔴 HIGH</div>
                    <div className="mt-2 text-2xl font-semibold text-red-600 dark:text-red-300">{data.high_count}</div>
                  </div>
                  <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground">🟡 MEDIUM</div>
                    <div className="mt-2 text-2xl font-semibold text-amber-600 dark:text-amber-300">{data.medium_count}</div>
                  </div>
                  <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4">
                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground">🟢 LOW</div>
                    <div className="mt-2 text-2xl font-semibold text-emerald-600 dark:text-emerald-300">{data.low_count}</div>
                  </div>
                </div>

                <div className="no-print space-y-6">
                  {groupedFindings.map(({ stage, items }) => (
                    <div key={stage}>
                      <div className="mb-2 flex items-center gap-2">
                        <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">{stage}</h4>
                        <span className="rounded-full border border-border bg-background px-2 py-0.5 text-[10px] text-muted-foreground">
                          {items.length} finding{items.length === 1 ? "" : "s"}
                        </span>
                      </div>
                      <div className="overflow-hidden rounded-lg border border-border">
                        <table className="w-full text-sm">
                          <thead className="bg-background text-[10px] uppercase tracking-wider text-muted-foreground">
                            <tr>
                              <th className="w-10 px-3 py-2 text-left">S.No</th>
                              <th className="px-3 py-2 text-left">Finding</th>
                              <th className="px-3 py-2 text-left">Severity</th>
                              <th className="px-3 py-2 text-left">Status</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border">
                            {items.map((f) => {
                              const s = severityStyle(f.severity);
                              const expanded = Boolean(expandedFindings[f._sno]);
                              return (
                                <React.Fragment key={f._sno}>
                                  <tr
                                    className="cursor-pointer hover:bg-background/60"
                                    onClick={() => toggleFinding(f._sno)}
                                  >
                                    <td className="px-3 py-2 align-top text-muted-foreground">{f._sno}</td>
                                    <td className="px-3 py-2 align-top">
                                      <div className="flex items-center gap-1.5 font-medium text-foreground">
                                        {expanded ? <ChevronDown className="h-3.5 w-3.5 shrink-0" /> : <ChevronRight className="h-3.5 w-3.5 shrink-0" />}
                                        {f.check}
                                      </div>
                                    </td>
                                    <td className="px-3 py-2 align-top">
                                      <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide ${s.badge}`}>
                                        {f.severity}
                                      </span>
                                    </td>
                                    <td className="px-3 py-2 align-top text-xs font-bold text-muted-foreground">{f.status}</td>
                                  </tr>
                                  {expanded && (
                                    <tr className={`${s.bg}`}>
                                      <td className="px-3 py-3" />
                                      <td colSpan={3} className="px-3 py-3">
                                        <div className="text-sm text-foreground">📌 {f.finding}</div>
                                        <div className="mt-2 text-sm text-muted-foreground">💡 {f.recommendation}</div>
                                        <div className="mt-1 text-xs text-muted-foreground">📋 {f.regulation}</div>
                                      </td>
                                    </tr>
                                  )}
                                </React.Fragment>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Print-only: full findings detail, always expanded, since a
                    static PDF has no click-to-expand affordance. */}
                <div className="print-only space-y-6">
                  {groupedFindings.map(({ stage, items }) => (
                    <div key={`print-${stage}`}>
                      <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">{stage}</h4>
                      <div className="mt-2 space-y-2">
                        {items.map((f) => {
                          const s = severityStyle(f.severity);
                          return (
                            <div key={`print-${f._sno}`} className={`rounded-r-lg border-l-4 ${s.border} ${s.bg} p-3`}>
                              <div className="text-sm font-semibold text-foreground">
                                {f._sno}. {f.check} — <span className="uppercase">{f.severity}</span> ({f.status})
                              </div>
                              <div className="mt-1 text-sm text-foreground">📌 {f.finding}</div>
                              <div className="mt-1 text-sm text-muted-foreground">💡 {f.recommendation}</div>
                              <div className="mt-1 text-xs text-muted-foreground">📋 {f.regulation}</div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Remediation log — an internal working tool for tracking
                    owners/deadlines, not part of the formal PDF report. */}
                <div className="no-print">
                  <h3 className="text-sm font-semibold">📝 Remediation Action Log</h3>
                  <p className="text-xs text-muted-foreground">Add owners and deadlines for each finding before sign-off.</p>
                  <div className="mt-3 overflow-x-auto rounded-lg border border-border">
                    <table className="w-full text-sm">
                      <thead className="bg-background text-[10px] uppercase tracking-wider text-muted-foreground">
                        <tr>
                          <th className="px-3 py-2 text-left">#</th>
                          <th className="px-3 py-2 text-left">Finding</th>
                          <th className="px-3 py-2 text-left">Severity</th>
                          <th className="px-3 py-2 text-left">Owner</th>
                          <th className="px-3 py-2 text-left">Target Date</th>
                          <th className="px-3 py-2 text-left">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {remediation.map((r, i) => (
                          <tr key={r.finding + i}>
                            <td className="px-3 py-2 align-top text-muted-foreground">{i + 1}</td>
                            <td className="px-3 py-2 align-top">
                              <div className="font-medium">{r.finding}</div>
                              <div className="text-xs text-muted-foreground">{r.detail}</div>
                            </td>
                            <td className="px-3 py-2 align-top">
                              <select
                                value={r.severity}
                                onChange={(e) => updateRemediation(i, { severity: e.target.value })}
                                className="rounded-md border border-border bg-background px-2 py-1 text-xs"
                              >
                                <option>HIGH</option>
                                <option>MEDIUM</option>
                                <option>LOW</option>
                              </select>
                            </td>
                            <td className="px-3 py-2 align-top">
                              <input
                                value={r.owner}
                                onChange={(e) => updateRemediation(i, { owner: e.target.value })}
                                placeholder="Owner"
                                className="w-32 rounded-md border border-border bg-background px-2 py-1 text-xs"
                              />
                            </td>
                            <td className="px-3 py-2 align-top">
                              <input
                                value={r.targetDate}
                                onChange={(e) => updateRemediation(i, { targetDate: e.target.value })}
                                placeholder="e.g. 31 Aug 2026"
                                className="w-32 rounded-md border border-border bg-background px-2 py-1 text-xs"
                              />
                            </td>
                            <td className="px-3 py-2 align-top">
                              <select
                                value={r.status}
                                onChange={(e) => updateRemediation(i, { status: e.target.value })}
                                className="rounded-md border border-border bg-background px-2 py-1 text-xs"
                              >
                                <option>Open</option>
                                <option>In Progress</option>
                                <option>Resolved</option>
                                <option>Risk Accepted</option>
                              </select>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </section>

          {/* Performance metrics vs regulatory thresholds */}
          <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <h3 className="text-sm font-semibold">4. Performance Metrics vs Regulatory Thresholds</h3>
            <p className="mt-1 text-xs text-muted-foreground">Replicated (Stage 3/4) metrics evaluated against the minimum required by regulation.</p>
            <div className="mt-3 overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="bg-background text-[10px] uppercase tracking-wider text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 text-left">Metric</th>
                    <th className="px-3 py-2 text-left">Value</th>
                    <th className="px-3 py-2 text-left">Threshold</th>
                    <th className="px-3 py-2 text-left">Regulation</th>
                    <th className="px-3 py-2 text-left">Result</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {metricRows.map((row) => (
                    <tr key={row.metric}>
                      <td className="px-3 py-2 align-top font-medium">{row.metric}</td>
                      <td className="px-3 py-2 align-top">{row.value !== null ? row.value.toFixed(4) : "N/A"}</td>
                      <td className="px-3 py-2 align-top text-muted-foreground">{row.op} {row.threshold}</td>
                      <td className="px-3 py-2 align-top text-muted-foreground">{row.regulation}</td>
                      <td className="px-3 py-2 align-top">
                        {row.pass === null ? (
                          <span className="text-muted-foreground">N/A</span>
                        ) : row.pass ? (
                          <span className="rounded-full bg-emerald-500 px-2 py-0.5 text-[10px] font-bold uppercase text-emerald-950">Pass</span>
                        ) : (
                          <span className="rounded-full bg-red-500 px-2 py-0.5 text-[10px] font-bold uppercase text-red-950">Fail</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Monitoring & revalidation */}
          <section>
            <h3 className="text-sm font-semibold">5. Monitoring & Revalidation Recommendations</h3>
            <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="rounded-lg border border-border bg-card p-4">
                <div className="text-xs font-bold text-primary">MONITORING FREQUENCY</div>
                <div className="mt-1 text-xl font-bold">{data.monitoring_frequency}</div>
                <div className="mt-1 text-xs text-muted-foreground">Based on {data.model_tier}</div>
              </div>
              <div className="rounded-lg border border-border bg-card p-4">
                <div className="text-xs font-bold text-primary">REVALIDATION TRIGGER</div>
                <div className="mt-1 text-base font-bold">{data.revalidation_trigger}</div>
                <div className="mt-1 text-xs text-muted-foreground">SS1/23 P4.4</div>
              </div>
            </div>
          </section>

          {/* Executive summary */}
          <section>
            <h3 className="text-sm font-semibold">6. Executive Summary</h3>
            <p className="text-xs text-muted-foreground no-print">Auto-generated from validation findings. Edit before final sign-off.</p>
            <textarea
              value={execSummary}
              onChange={(e) => setExecSummary(e.target.value)}
              rows={14}
              className="no-print mt-3 w-full rounded-lg border border-border bg-background px-3 py-2 font-mono text-xs leading-relaxed"
            />
            <pre className="print-only mt-3 whitespace-pre-wrap font-mono text-xs leading-relaxed">{execSummary}</pre>
          </section>

          {/* Downloads */}
          <section className="no-print">
            <h3 className="text-sm font-semibold">📥 Download Evidence Pack</h3>
            <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
              <button
                type="button"
                disabled={findings.length === 0}
                onClick={() =>
                  downloadCsv(
                    `validation_findings_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}.csv`,
                    findings,
                  )
                }
                className="rounded-lg border border-border bg-card px-4 py-2 text-sm font-semibold hover:border-primary/40 disabled:cursor-not-allowed disabled:opacity-60"
              >
                📋 Download Findings Report (CSV)
              </button>
              <button
                type="button"
                onClick={() =>
                  downloadCsv(`validation_report_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}.csv`, [
                    { Section: "Verdict", Value: data.verdict },
                    { Section: "Model Name", Value: ij.model_name ?? "N/A" },
                    { Section: "Model Type", Value: ij.model_type ?? "N/A" },
                    { Section: "Risk Tier", Value: data.model_tier },
                    { Section: "Validation Date", Value: todayLabel() },
                    { Section: "Replicated AUC", Value: data.replicated_auc ?? "N/A" },
                    { Section: "Stated AUC", Value: data.stated_auc ?? "N/A" },
                    {
                      Section: "AUC Gap",
                      Value:
                        data.replicated_auc !== null && data.stated_auc !== null
                          ? Math.abs((data.replicated_auc ?? 0) - (data.stated_auc ?? 0)).toFixed(4)
                          : "N/A",
                    },
                    { Section: "Total Findings", Value: data.total_count },
                    { Section: "HIGH Findings", Value: data.high_count },
                    { Section: "MEDIUM Findings", Value: data.medium_count },
                    { Section: "Monitoring Frequency", Value: data.monitoring_frequency },
                    { Section: "Revalidation Trigger", Value: data.revalidation_trigger },
                    { Section: "Executive Summary", Value: execSummary },
                  ])
                }
                className="rounded-lg border border-border bg-card px-4 py-2 text-sm font-semibold hover:border-primary/40"
              >
                📄 Download Full Validation Report (CSV)
              </button>
            </div>
          </section>

          {/* Sign-off */}
          <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <h3 className="text-sm font-semibold">7. Sign-off</h3>
            <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
              {(
                [
                  ["Validator", "validator", "Risk Validation"],
                  ["Model Owner", "modelOwner", "Credit Risk Modelling"],
                  ["Committee", "committee", "Model Risk Committee"],
                ] as const
              ).map(([role, key, sub]) => (
                <div key={role} className="rounded-lg border border-border bg-background p-4">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{role}</div>
                  <input
                    value={signOff[key]}
                    onChange={(e) => setSignOff((prev) => ({ ...prev, [key]: e.target.value }))}
                    placeholder="Name / status"
                    className="no-print mt-1 w-full rounded-md border border-border bg-card px-2 py-1 text-sm font-semibold"
                  />
                  <div className="print-only mt-1 border-b border-foreground/40 pb-1 text-sm font-semibold">
                    {signOff[key] || " "}
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">{sub}</div>
                </div>
              ))}
            </div>
          </section>
          </div>
        </>
      ) : null}

      <div className="flex items-center justify-between">
        <Link
          to="/validation/regulatory"
          className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-semibold hover:border-primary/40"
        >
          ← Back to Stage 6: Explainability and Fairness
        </Link>
        <Link
          to="/validation"
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90"
        >
          Finish
          <ArrowRight className="h-4 w-4" />
        </Link>
      </div>
    </div>
  );
}
