import { createFileRoute, Link } from "@tanstack/react-router";
import { PageHeader } from "@/components/app-shell";
import { ArrowRight } from "lucide-react";
import PlotlyChart from "@/components/plotly-chart";
import { CheckSummaryTiles } from "@/components/check-summary";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, formUpload } from "@/lib/api";
import { useDataset } from "@/lib/app-context";

export const Route = createFileRoute("/validation/stress")({
  head: () => ({ meta: [{ title: "Stress & Backtesting — Aegis Credit" }] }),
  component: Stress,
});

const FREQ_OPTIONS: { key: string; label: string }[] = [
  { key: "monthly", label: "Monthly" },
  { key: "quarterly", label: "Quarterly" },
  { key: "half_yearly", label: "Half-Yearly" },
  { key: "yearly", label: "Yearly" },
];

function statusTone(status: string | undefined): "pass" | "warn" | "fail" | "pending" {
  switch (status) {
    case "PASS":
      return "pass";
    case "WARN":
      return "warn";
    case "FAIL":
      return "fail";
    default:
      return "pending";
  }
}

const TONE_CLASSES: Record<string, string> = {
  pass: "border-emerald-500/40 bg-emerald-500/10 text-emerald-400",
  warn: "border-amber-500/40 bg-amber-500/10 text-amber-400",
  fail: "border-red-500/40 bg-red-500/10 text-red-400",
  pending: "border-slate-500/40 bg-slate-500/10 text-slate-400",
  na: "border-indigo-500/30 bg-indigo-500/5 text-indigo-400",
};

const TONE_ICON: Record<string, string> = {
  pass: "✅",
  warn: "🟡",
  fail: "🔴",
  pending: "⏳",
  na: "⏳",
};

// Every check rendered through this card (sensitivity AUC drop, PSI, backtest
// gap) is a quantitative statistical check — the cited regulation requires
// this kind of check to be performed, not the specific numeric cutoff, which
// is an industry-standard convention. Split the two so the citation isn't
// misread as the source of the number itself.
function CheckCard({ check }: { check: { id: string; title: string; status: string; observed: string; threshold: string; source: string } }) {
  const tone = statusTone(check.status);
  return (
    <div className={`rounded-lg border-l-4 p-4 ${TONE_CLASSES[tone]} bg-card border border-border`}>
      <div className="text-sm font-semibold text-foreground">
        {TONE_ICON[tone]} [{check.id}] {check.title}
      </div>
      <div className="mt-1 text-xs text-muted-foreground">{check.observed}</div>
      <div className="mt-1 text-[11px] text-muted-foreground/80">Regulatory basis: {check.source} — requires this to be assessed/documented</div>
      <div className="mt-0.5 text-[11px] text-muted-foreground/80">Threshold: {check.threshold} — industry-standard convention</div>
    </div>
  );
}

function Stress() {
  const { file, profile } = useDataset();
  const datasetName = profile?.dataset_name ?? file?.name ?? "the active validation dataset";
  const datasetReady = Boolean(file || profile?.csv_text || profile?.dataset_name);
  const columns: string[] = useMemo(() => (profile?.columns ?? profile?.col_types?.all ?? []) as string[], [profile]);

  const [algorithms, setAlgorithms] = useState<string[]>([]);
  const [targetCol, setTargetCol] = useState<string>("");
  const [algorithm, setAlgorithm] = useState<string>("");
  const [freq, setFreq] = useState<string>("quarterly");

  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [report, setReport] = useState<any | null>(null);

  const [shockFeature, setShockFeature] = useState<string>("");
  const [shockDirection, setShockDirection] = useState<"increase" | "decrease">("increase");
  const [shockMagnitude, setShockMagnitude] = useState<number>(20);
  const [shockRunning, setShockRunning] = useState(false);
  const [shockError, setShockError] = useState<string | null>(null);
  const [shockResult, setShockResult] = useState<any | null>(null);

  useEffect(() => {
    void api<{ models: string[] }>("/models/list")
      .then((res) => {
        setAlgorithms(res.models ?? []);
        setAlgorithm((prev) => prev || res.models?.[0] || "");
      })
      .catch(() => setAlgorithms([]));
  }, []);

  useEffect(() => {
    if (!targetCol && columns.length > 0) {
      // Best-effort default: prefer an obvious target-sounding column, else the last column.
      const guess = columns.find((c) => /default|target|label|bad_flag/i.test(c)) ?? columns[columns.length - 1];
      setTargetCol(guess);
    }
  }, [columns, targetCol]);

  useEffect(() => {
    if (!shockFeature && report?.available && report.numeric_features?.length) {
      setShockFeature(report.numeric_features[0]);
    }
  }, [report, shockFeature]);

  const buildForm = () => {
    const form = new FormData();
    if (file) {
      form.append("file", file);
    } else if (profile?.csv_text) {
      form.append("csv_text", profile.csv_text);
    }
    form.append("target_col", targetCol);
    form.append("algorithm", algorithm);
    return form;
  };

  const runStressSuite = async () => {
    if (!datasetReady || !targetCol || !algorithm) return;
    setRunError(null);
    setRunning(true);
    setShockResult(null);
    try {
      const form = buildForm();
      form.append("freq", freq);
      const resp = await formUpload<{ stage: string; report: any }>("/validation/stress/run", form);
      setReport(resp.report ?? null);
      if (!resp.report?.available) {
        setRunError(resp.report?.reason ?? "Stress suite did not return a usable result.");
      }
    } catch (error: any) {
      setReport(null);
      setRunError(error?.message ?? "Failed to run stress & backtesting checks.");
    } finally {
      setRunning(false);
    }
  };

  const hasAutoRun = useRef(false);
  useEffect(() => {
    if (hasAutoRun.current) return;
    if (!datasetReady || !targetCol || !algorithm || running) return;
    hasAutoRun.current = true;
    void runStressSuite();
  }, [datasetReady, targetCol, algorithm]);

  useEffect(() => {
    if (!hasAutoRun.current) return;
    void runStressSuite();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [freq]);


  const applyShock = async () => {
    if (!shockFeature || !targetCol || !algorithm) return;
    setShockError(null);
    setShockRunning(true);
    try {
      const form = buildForm();
      form.append("shock_feature", shockFeature);
      form.append("shock_direction", shockDirection);
      form.append("shock_magnitude_pct", String(shockMagnitude));
      const resp = await formUpload<{ stage: string; result: any }>("/validation/stress/shock", form);
      setShockResult(resp.result ?? null);
    } catch (error: any) {
      setShockResult(null);
      setShockError(error?.message ?? "Shock failed.");
    } finally {
      setShockRunning(false);
    }
  };

  const psiChartData = useMemo(() => {
    const bins = report?.psi?.bins ?? [];
    return bins.map((b: any) => ({ bin: b.bin, "Train %": b.train_pct, "Test %": b.test_pct }));
  }, [report]);

  const psiFigure = useMemo(() => {
    if (!psiChartData.length) return null;
    return {
      data: [
        {
          type: "bar",
          x: psiChartData.map((row) => row.bin),
          y: psiChartData.map((row) => row["Train %"]),
          name: "Train %",
          marker: { color: "oklch(0.6 0.16 260)" },
        },
        {
          type: "bar",
          x: psiChartData.map((row) => row.bin),
          y: psiChartData.map((row) => row["Test %"]),
          name: "Test %",
          marker: { color: "oklch(0.6 0.18 135)" },
        },
      ],
      layout: {
        barmode: "group",
        margin: { l: 40, r: 20, t: 20, b: 60 },
        xaxis: { tickfont: { size: 9 }, automargin: true, tickangle: -30 },
        yaxis: { title: { text: "%" }, tickfont: { size: 11 }, automargin: true },
        height: 256,
      },
    };
  }, [psiChartData]);

  const sensitivityChartData = useMemo(() => {
    const rows = report?.sensitivity?.rows ?? [];
    return rows.map((r: any) => ({ feature: r.feature, "AUC drop": r.auc_drop }));
  }, [report]);

  const sensitivityFigure = useMemo(() => {
    if (!sensitivityChartData.length) return null;
    return {
      data: [
        {
          type: "bar",
          x: sensitivityChartData.map((row) => row.feature),
          y: sensitivityChartData.map((row) => row["AUC drop"]),
          name: "AUC drop",
          marker: { color: "oklch(0.6 0.16 260)" },
        },
      ],
      layout: {
        margin: { l: 40, r: 20, t: 20, b: 60 },
        xaxis: { tickfont: { size: 9 }, automargin: true, tickangle: -30 },
        yaxis: { title: { text: "AUC drop" }, tickfont: { size: 11 }, automargin: true },
        height: 256,
      },
    };
  }, [sensitivityChartData]);

  const macroChartData = useMemo(() => {
    const scenarios = report?.macro_scenarios?.scenarios ?? [];
    return scenarios.map((s: any) => ({
      name: s.name,
      "Base PD": +(s.base_pd * 100).toFixed(2),
      "Scenario PD": +(s.scn_pd * 100).toFixed(2),
    }));
  }, [report]);

  const macroFigure = useMemo(() => {
    if (!macroChartData.length) return null;
    return {
      data: [
        {
          type: "bar",
          x: macroChartData.map((row) => row.name),
          y: macroChartData.map((row) => row["Base PD"]),
          name: "Base PD",
          marker: { color: "oklch(0.6 0.16 260)" },
        },
        {
          type: "bar",
          x: macroChartData.map((row) => row.name),
          y: macroChartData.map((row) => row["Scenario PD"]),
          name: "Scenario PD",
          marker: { color: "oklch(0.6 0.22 27)" },
        },
      ],
      layout: {
        barmode: "group",
        margin: { l: 40, r: 20, t: 20, b: 60 },
        xaxis: { tickfont: { size: 11 }, automargin: true },
        yaxis: { title: { text: "%" }, tickfont: { size: 11 }, automargin: true },
        height: 256,
      },
    };
  }, [macroChartData]);

  const backtestChartData = useMemo(() => {
    const periods = report?.backtest?.periods ?? [];
    return periods.map((p: any) => ({
      period: p.period,
      "Actual default rate": +(p.actual_dr * 100).toFixed(2),
      "Avg predicted PD": +(p.avg_pred_pd * 100).toFixed(2),
    }));
  }, [report]);

  const backtestFigure = useMemo(() => {
    if (!backtestChartData.length) return null;
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: backtestChartData.map((row) => row.period),
          y: backtestChartData.map((row) => row["Avg predicted PD"]),
          name: "Avg predicted PD",
          line: { color: "oklch(0.6 0.18 135)", width: 2.5 },
        },
        {
          type: "scatter",
          mode: "lines",
          x: backtestChartData.map((row) => row.period),
          y: backtestChartData.map((row) => row["Actual default rate"]),
          name: "Actual default rate",
          line: { color: "oklch(0.6 0.22 27)", width: 2.5 },
        },
      ],
      layout: {
        margin: { l: 40, r: 20, t: 20, b: 40 },
        xaxis: { tickfont: { size: 11 }, automargin: true },
        yaxis: { title: { text: "%" }, tickfont: { size: 11 }, automargin: true },
        height: 280,
      },
    };
  }, [backtestChartData]);

  const summary = report?.summary;

  return (
    <div className="space-y-8">
      <PageHeader
        title="Stage 5 — Stress & Backtesting"
        description="Scenario simulations, model stability over time, and back-tested predictions vs realised outcomes."
      />

      <section className="rounded-xl border border-border bg-background p-4 text-sm text-muted-foreground">
        {datasetReady ? (
          <>Using the shared dataset from Stage 1 / Stage 2: <span className="font-semibold text-foreground">{datasetName}</span>.</>
        ) : (
          <>No active dataset is available in shared state yet. Complete Stage 1 Intake and Stage 2 Data Validation first.</>
        )}
      </section>

      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <h3 className="text-sm font-semibold">Run configuration</h3>
        <p className="mt-1 text-xs text-muted-foreground">
          Stress testing retrains the replicated model within this run (same approach as Stage 3 Model Replication —
          nothing is cached between requests), then applies sensitivity, macro-scenario, stability, backtesting, and
          directional checks against it.
        </p>
        <div className="mt-4 text-xs text-muted-foreground">
          {running ? "Running stress suite…" : report ? "Stress suite complete." : !datasetReady ? "Waiting for an active dataset…" : null}
        </div>
        {runError ? <div className="mt-3 text-xs text-red-500">{runError}</div> : null}
      </section>

      {summary ? <CheckSummaryTiles summary={summary} /> : null}

      <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
          <h3 className="text-sm font-semibold">Sensitivity — AUC drop on feature removal</h3>
          <p className="text-xs text-muted-foreground">From Stage 3 ablation. SS1/23 P4.3.</p>
          <div className="mt-4 h-64">
            {sensitivityFigure ? (
              <PlotlyChart figure={sensitivityFigure} style={{ height: "100%" }} />
            ) : (
              <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                Run the stress suite to see ablation results.
              </div>
            )}
          </div>
          {report?.sensitivity?.check ? <div className="mt-4"><CheckCard check={report.sensitivity.check} /></div> : null}

          <div className="mt-6 border-t border-border pt-4">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Manual feature shock</h4>
            <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-3">
              <select
                className="rounded-md border border-border bg-background p-2 text-sm"
                value={shockFeature}
                onChange={(e) => setShockFeature(e.target.value)}
              >
                {(report?.numeric_features ?? []).map((f: string) => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
              <select
                className="rounded-md border border-border bg-background p-2 text-sm"
                value={shockDirection}
                onChange={(e) => setShockDirection(e.target.value as "increase" | "decrease")}
              >
                <option value="increase">Increase (+)</option>
                <option value="decrease">Decrease (-)</option>
              </select>
              <input
                type="number"
                min={5}
                max={100}
                step={5}
                value={shockMagnitude}
                onChange={(e) => setShockMagnitude(Number(e.target.value))}
                className="rounded-md border border-border bg-background p-2 text-sm"
              />
            </div>
            <button
              type="button"
              disabled={!shockFeature || shockRunning}
              onClick={() => void applyShock()}
              className="mt-3 w-full rounded-lg bg-primary/90 px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary disabled:opacity-50"
            >
              {shockRunning ? "Applying…" : "⚡ Apply Shock"}
            </button>
            {shockError ? <div className="mt-2 text-xs text-red-500">{shockError}</div> : null}
            {shockResult ? (
              <div className="mt-3 rounded-lg border border-border bg-background p-3 text-xs text-muted-foreground">
                <span className="font-semibold text-foreground">{shockResult.feature}</span> shocked {shockResult.direction} {shockResult.magnitude_pct}%:
                base avg PD <b>{shockResult.base_pd.toFixed(4)}</b> → shocked avg PD <b>{shockResult.shock_pd.toFixed(4)}</b>{" "}
                (change <b>{shockResult.pd_change >= 0 ? "+" : ""}{shockResult.pd_change.toFixed(4)}, {shockResult.pd_change_pct.toFixed(1)}%</b>)
              </div>
            ) : null}
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
          <h3 className="text-sm font-semibold">Score stability (PSI) — train vs test</h3>
          <p className="text-xs text-muted-foreground">SS11/13 §10.6. PSI &lt; 0.10 stable, 0.10–0.25 minor shift, &gt; 0.25 major shift.</p>
          <div className="mt-4 h-64">
            {psiFigure ? (
              <PlotlyChart figure={psiFigure} style={{ height: "100%" }} />
            ) : (
              <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                Run the stress suite to see the score distribution.
              </div>
            )}
          </div>
          {report?.psi?.check ? <div className="mt-4"><CheckCard check={report.psi.check} /></div> : null}
        </div>
      </section>

      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <h3 className="text-sm font-semibold">Macro stress scenarios — average predicted PD</h3>
        <p className="text-xs text-muted-foreground">
          SS3/18 §2.1.{" "}
          {report?.macro_scenarios?.detected_drivers
            ? Object.entries(report.macro_scenarios.detected_drivers).map(([k, v]) => `${k} → ${v}`).join(", ")
            : ""}
        </p>
        <div className="mt-4 h-64">
          {macroFigure ? (
            <PlotlyChart figure={macroFigure} style={{ height: "100%" }} />
          ) : (
            <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
              Run the stress suite to see scenario results.
            </div>
          )}
        </div>
        {report?.macro_scenarios?.scenarios?.length ? (
          <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
            {report.macro_scenarios.scenarios.map((s: any) => (
              <div key={s.id} className="rounded-lg border border-border bg-background p-3 text-xs">
                <div className="font-semibold text-foreground">{s.name}</div>
                <div className="mt-1 text-muted-foreground">{s.desc}</div>
                <div className="mt-2 text-muted-foreground">
                  Base <b>{s.base_pd.toFixed(4)}</b> → Scenario <b>{s.scn_pd.toFixed(4)}</b> ({s.pd_change_pct >= 0 ? "+" : ""}{s.pd_change_pct.toFixed(1)}%)
                </div>
                <div className="mt-1 text-[11px] text-muted-foreground/70">
                  {s.applied?.length ? `Applied: ${s.applied.join("; ")}` : "No matching driver columns found"}
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </section>

      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold">Backtesting — predicted vs actual default rate</h3>
            <p className="text-xs text-muted-foreground">
              {report?.backtest?.available
                ? `Grouped by ${report.backtest.freq} · date column: ${report.backtest.date_col}`
                : report?.backtest?.reason ?? "Run the stress suite to see backtesting results."}
            </p>
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Backtest period grouping</label>
            <select
              className="mt-1 w-full rounded-md border border-border bg-background p-2 text-sm"
              value={freq}
              disabled={running}
              onChange={(e) => setFreq(e.target.value)}
            >
              {FREQ_OPTIONS.map((f) => (
                <option key={f.key} value={f.key}>{f.label}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="mt-4 h-72">
          {backtestFigure ? (
            <PlotlyChart figure={backtestFigure} style={{ height: "100%" }} />
          ) : (
            <div className="flex h-full items-center justify-center text-xs text-muted-foreground">No backtesting data yet.</div>
          )}
        </div>
        {report?.backtest?.check ? <div className="mt-4"><CheckCard check={report.backtest.check} /></div> : null}
      </section>

      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <h3 className="text-sm font-semibold">Directional testing — economic intuition check</h3>
        <p className="text-xs text-muted-foreground">
          SS1/23 P4.3 · SS3/18 §2.1. Each driver is shocked ±10% in the adverse direction; average predicted PD should
          move as basic credit-risk intuition expects.
        </p>
        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
          {(report?.directional ?? []).map((r: any) => {
            if (r.status === "SKIP") {
              return (
                <div key={r.id} className="rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-4 text-xs">
                  <div className="font-semibold text-foreground">⏳ [{r.id}] {r.driver} → {r.expected}</div>
                  <div className="mt-1 text-muted-foreground">{r.note}</div>
                </div>
              );
            }
            const tone = statusTone(r.status);
            return (
              <div key={r.id} className={`rounded-lg border p-4 text-xs ${TONE_CLASSES[tone]}`}>
                <div className="font-semibold text-foreground">{TONE_ICON[tone]} [{r.id}] {r.driver} → {r.expected}</div>
                {r.status === "ERROR" ? (
                  <div className="mt-1 text-muted-foreground">{r.error}</div>
                ) : (
                  <div className="mt-1 text-muted-foreground">
                    {r.column} shocked {r.shock_desc}: avg PD {r.base_pd.toFixed(4)} → {r.new_pd.toFixed(4)} ({r.delta >= 0 ? "+" : ""}{r.delta.toFixed(4)})
                  </div>
                )}
              </div>
            );
          })}
          {!report?.directional?.length ? (
            <div className="text-xs text-muted-foreground">Run the stress suite to see directional test results.</div>
          ) : null}
        </div>
      </section>

      <div className="text-right">
        <Link
          to="/validation/regulatory"
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90"
        >
          Continue to Stage 6
          <ArrowRight className="h-4 w-4" />
        </Link>
      </div>
    </div>
  );
}
