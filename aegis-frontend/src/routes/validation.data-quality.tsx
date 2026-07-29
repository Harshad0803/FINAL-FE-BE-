import { createFileRoute, Link } from "@tanstack/react-router";
import { PageHeader } from "@/components/app-shell";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CheckSummaryTiles, deriveCheckTotal } from "@/components/check-summary";
import { useDataset } from "@/lib/app-context";
import { formUpload } from "@/lib/api";
import { ArrowRight, AlertTriangle, AlertCircle, Clock, Check } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

export const Route = createFileRoute("/validation/data-quality")({
  head: () => ({ meta: [{ title: "Stage 2 — Data & Model Soundness — Aegis Credit" }] }),
  component: DataQuality,
});

type Status = "PASS" | "WARN" | "FAIL" | string;

type ThresholdCheck = {
  check_id: string;
  title: string;
  severity: string;
  status: Status;
  source: string;
  principle: string;
  observed: string;
  threshold: string;
  detail: string;
  check_type?: string;
};

// check_type "data"/"cross_reference" checks (e.g. VIF > 10, missing data
// < 20%, duplicate rate < 1%) compute a number against an industry-standard
// statistical convention — the regulation cited requires that kind of check
// to exist and be documented, not that specific cutoff value. check_type
// "doc"/"manual" checks (methodology justification, bias documentation,
// etc.) genuinely reference regulatory language, so their citation stays
// combined with the threshold as before.
function isQuantitativeConventionCheck(checkType: string | undefined): boolean {
  return checkType === "data" || checkType === "cross_reference";
}

type RagRule = {
  rule_id: string;
  flag: string;
  suggestion: string;
  severity: string;
  status: Status;
  source: string;
  principle: string;
  observed_value?: string | string[] | null;
  not_verifiable?: boolean;
  check_source?: "llm" | "llm_error" | "quantitative" | "keyword_search" | "" | string;
  reasoning?: string;
};

// Shared shape for both /validation/stage2/run and /validation/stage3/run —
// the two sub-tabs below (Data Validation, Conceptual Soundness) are
// otherwise-identical RAG + threshold-check panels against different backend
// stages, so one response type covers both.
type StageCheckResponse = {
  thresholdChecks: ThresholdCheck[];
  ragRules: RagRule[];
  summary: { total: number; pass: number; warn: number; fail: number; pending?: number; na?: number };
  regulatoryAlignment: {
    verdict: "PASS" | "CONDITIONAL" | "FAIL" | string;
    counts: { pass: number; warn: number; fail: number; pending: number };
    remediation_summary: string;
    regulatory_references: string[];
    high_severity_fails: unknown[];
  };
  featureRelevance?: { importance_df?: unknown[]; top_drivers?: unknown[] };
  llm_pending?: boolean;
};

function buildValidationCsv(thresholdChecks: ThresholdCheck[], ragRules: RagRule[]) {
  const header = ["Check ID", "Title", "Source", "Principle", "Severity", "Status", "Observed", "Threshold/Suggestion"];
  const escape = (value: string | number | null | undefined) => {
    const text = value == null ? "" : String(value);
    return `"${text.replace(/"/g, '""')}"`;
  };
  const lines = [header.map(escape).join(",")];
  for (const c of thresholdChecks) {
    lines.push([c.check_id, c.title, c.source, c.principle, c.severity, c.status, c.observed, c.threshold].map(escape).join(","));
  }
  for (const r of ragRules) {
    const observed = Array.isArray(r.observed_value) ? r.observed_value.join(", ") : r.observed_value ?? "";
    lines.push([r.rule_id, r.flag, r.source, r.principle, r.severity, r.status, observed, r.suggestion].map(escape).join(","));
  }
  return lines.join("\n");
}

function DataQuality() {
  const {
    file,
    profile,
    trainingResult,
    validationIntakeData,
    validationMddText,
    validationMddMetrics,
    validationProfile: sharedValidationProfile,
    validationResults: sharedValidationResults,
    setValidationProfile,
    setValidationResults,
    validationStage3Result,
    setValidationStage3Result,
  } = useDataset();

  const datasetLoaded = Boolean(file || profile?.csv_text || profile?.dataset_name);

  // Tracks which sub-tab is active so the bottom button can tell whether
  // the reviewer is still on the first sub-tab (Data Validation — button
  // just advances to the second sub-tab) or the last one (Conceptual
  // Soundness — button actually navigates to Stage 3).
  const [activeTab, setActiveTab] = useState<string>("data-validation");

  // ── Profile (charts: missing values, distribution, leakage) — unrelated to
  // the RAG check pipeline, kept as its own call against /data/profile. Shared
  // by both sub-tabs below (both threshold-detail panels use it for the
  // missing-values chart). ──
  const [validationProfile, setValidationProfileState] = useState<any | null>(sharedValidationProfile ?? null);
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const runDataProfile = async () => {
    if (!datasetLoaded) return;
    setRunError(null);
    setIsRunning(true);
    try {
      const form = new FormData();
      if (file) {
        form.append("file", file);
      } else if (profile?.csv_text) {
        form.append("csv_text", profile.csv_text);
      } else {
        throw new Error("No dataset available to validate.");
      }

      const result = await formUpload("/data/profile", form);
      const nextProfile = result as any;
      setValidationProfileState(nextProfile);
      setValidationProfile(nextProfile);
      setValidationResults({
        profile: nextProfile,
        intake: validationIntakeData ?? null,
        mddText: validationMddText ?? null,
        mddMetrics: validationMddMetrics ?? null,
      });
    } catch (error: any) {
      setRunError(error?.message ?? "Failed to profile the dataset.");
    } finally {
      setIsRunning(false);
    }
  };

  useEffect(() => {
    if (!datasetLoaded || validationProfile || isRunning) return;
    void runDataProfile();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetLoaded, validationProfile, isRunning, file, profile?.csv_text, profile?.dataset_name]);

  useEffect(() => {
    if (sharedValidationProfile && !validationProfile) {
      setValidationProfileState(sharedValidationProfile);
    }
  }, [sharedValidationProfile, validationProfile]);

  // ── Sub-tab 1: Data Validation — RAG threshold checks + rules against
  // /validation/stage2/run (checks 2.1-2.10). ──
  const [checksLoading, setChecksLoading] = useState(true);
  const [checksError, setChecksError] = useState<string | null>(null);
  const [data, setData] = useState<StageCheckResponse | null>(null);
  // Separate from checksLoading: quantitative checks + PENDING RAG stubs
  // come back fast; this tracks the slower follow-up LLM call so it never
  // blocks the rest of the page.
  const [llmLoading, setLlmLoading] = useState(false);

  useEffect(() => {
    if (!datasetLoaded) {
      setChecksLoading(false);
      return;
    }

    let active = true;
    setChecksLoading(true);
    setChecksError(null);

    const form = new FormData();
    if (file) {
      form.append("file", file);
    } else if (profile?.csv_text) {
      form.append("csv_text", profile.csv_text);
    }
    form.append("intake_json", JSON.stringify(validationIntakeData ?? {}));
    if (validationMddText) {
      const mddBlob = new Blob([validationMddText], { type: "text/plain" });
      form.append("mdd_file", new File([mddBlob], "mdd.txt", { type: "text/plain" }));
    }

    void formUpload<StageCheckResponse>("/validation/stage2/run", form)
      .then((resp) => {
        if (!active) return;
        setData(resp);
        setChecksLoading(false);

        if (resp.llm_pending) {
          setLlmLoading(true);
          const llmForm = new FormData();
          llmForm.append("intake_json", JSON.stringify(validationIntakeData ?? {}));
          if (validationMddText) {
            const mddBlob = new Blob([validationMddText], { type: "text/plain" });
            llmForm.append("mdd_file", new File([mddBlob], "mdd.txt", { type: "text/plain" }));
          }

          void formUpload<{ llm_results: RagRule[] }>("/validation/stage2/llm-check", llmForm)
            .then((llmResp) => {
              if (!active) return;
              setData((prev) => {
                if (!prev) return prev;
                const byId = new Map(llmResp.llm_results.map((r) => [r.rule_id, r]));
                const mergedRules = prev.ragRules.map((r) => byId.get(r.rule_id) ?? r);
                return { ...prev, ragRules: mergedRules };
              });
            })
            .catch((err) => {
              console.error("Stage2 LLM check error", err);
            })
            .finally(() => {
              if (!active) return;
              setLlmLoading(false);
            });
        }
      })
      .catch((err) => {
        console.error("Stage2 fetch error", err);
        if (!active) return;
        setChecksError(err?.message ?? String(err));
        setChecksLoading(false);
      });

    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetLoaded, file, profile?.csv_text]);

  const summary = data?.summary ?? { total: 0, pass: 0, warn: 0, fail: 0 };

  const validationCsv = useMemo(() => buildValidationCsv(data?.thresholdChecks ?? [], data?.ragRules ?? []), [data]);
  const downloadValidationReport = () => {
    const blob = new Blob([validationCsv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "data_validation_report.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // ── Sub-tab 2: Conceptual Soundness — RAG threshold checks + rules against
  // /validation/stage3/run (checks 3.1-3.x). Independent loading/error/data
  // state from Data Validation above, so one tab never blocks the other. ──
  const [conceptualLoading, setConceptualLoading] = useState(!validationStage3Result);
  const [conceptualError, setConceptualError] = useState<string | null>(null);
  const [conceptualData, setConceptualData] = useState<StageCheckResponse | null>(
    (validationStage3Result as StageCheckResponse | null) ?? null,
  );
  const [conceptualLlmLoading, setConceptualLlmLoading] = useState(false);

  const skipInitialConceptualAutoRun = useRef(validationStage3Result !== null && validationStage3Result !== undefined);

  useEffect(() => {
    if (!datasetLoaded) {
      setConceptualLoading(false);
      return;
    }
    if (skipInitialConceptualAutoRun.current) {
      skipInitialConceptualAutoRun.current = false;
      setConceptualLoading(false);
      return;
    }

    let active = true;
    setConceptualLoading(true);
    setConceptualError(null);

    const form = new FormData();
    // Mirrors the Data Validation tab: use the real working dataset from
    // context rather than posting an empty intake, which is why the RAG
    // Agent Rules column previously came back empty — check_for_validation
    // needs dataset metrics and check_mdd_keywords needs the MDD text to
    // match against, neither of which were ever being sent.
    if (file) {
      form.append("file", file);
    } else if (profile?.csv_text) {
      form.append("csv_text", profile.csv_text);
    }
    // check_conceptual_soundness() reads intake_json.methodology (check 3.1),
    // but the Intake form has no dedicated methodology field — the actual
    // algorithm choice lives on Training's trainingResult. Fill it in here
    // when the Intake form didn't already provide one.
    const intakePayload: Record<string, any> = { ...(validationIntakeData ?? {}) };
    if (!intakePayload.methodology && trainingResult?.model_name) {
      intakePayload.methodology = trainingResult.model_name;
    }
    form.append("intake_json", JSON.stringify(intakePayload));
    if (validationMddText) {
      const mddBlob = new Blob([validationMddText], { type: "text/plain" });
      form.append("mdd_file", new File([mddBlob], "mdd.txt", { type: "text/plain" }));
    }

    void formUpload<StageCheckResponse>("/validation/stage3/run", form)
      .then((resp) => {
        if (!active) return;
        setConceptualData(resp);
        setValidationStage3Result(resp as unknown as Record<string, any>);
        setConceptualLoading(false);

        // Quantitative results are already in `resp`. If there's an MDD to
        // review, kick off the slower LLM conceptual check as a separate,
        // non-blocking call and merge its verdicts into ragRules by
        // rule_id as soon as they land, replacing the PENDING stubs.
        if (resp.llm_pending) {
          setConceptualLlmLoading(true);
          const llmForm = new FormData();
          llmForm.append("intake_json", JSON.stringify(intakePayload));
          if (validationMddText) {
            const mddBlob = new Blob([validationMddText], { type: "text/plain" });
            llmForm.append("mdd_file", new File([mddBlob], "mdd.txt", { type: "text/plain" }));
          }

          void formUpload<{ llm_results: RagRule[] }>("/validation/stage3/llm-check", llmForm)
            .then((llmResp) => {
              if (!active) return;
              setConceptualData((prev) => {
                if (!prev) return prev;
                const byId = new Map(llmResp.llm_results.map((r) => [r.rule_id, r]));
                const mergedRules = prev.ragRules.map((r) => byId.get(r.rule_id) ?? r);
                const merged = { ...prev, ragRules: mergedRules };
                setValidationStage3Result(merged as unknown as Record<string, any>);
                return merged;
              });
            })
            .catch((err) => {
              console.error("Stage3 LLM check error", err);
            })
            .finally(() => {
              if (!active) return;
              setConceptualLlmLoading(false);
            });
        }
      })
      .catch((err) => {
        console.error("Stage3 fetch error", err);
        if (!active) return;
        setConceptualError(err?.message ?? String(err));
        setConceptualLoading(false);
      });

    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetLoaded, file, profile?.csv_text]);

  const conceptualSummary = conceptualData?.summary ?? { total: 0, pass: 0, warn: 0, fail: 0 };

  return (
    <div className="space-y-8">
      <PageHeader
        title="Stage 2 — Data & Model Soundness"
        description="Is the dataset complete and representative, and are the chosen features, methodology, and assumptions appropriate for the stated business objective and regulatory context?"
      />

      {!datasetLoaded ? (
        <div className="rounded-xl border border-border bg-card p-6 text-center">
          <h3 className="text-lg font-semibold">No dataset available</h3>
          <p className="mt-2 text-sm text-muted-foreground">
            Upload a dataset and complete Intake before these checks can run.
          </p>
        </div>
      ) : (
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList>
            <TabsTrigger value="data-validation">Data Validation</TabsTrigger>
            <TabsTrigger value="conceptual">Conceptual Soundness</TabsTrigger>
          </TabsList>

          <TabsContent value="data-validation" className="space-y-8 pt-4">
            {checksLoading ? (
              <div className="rounded-xl border border-border bg-card p-6 text-center">Loading Data Validation...</div>
            ) : checksError ? (
              <div className="rounded-xl border border-border bg-card p-6 text-destructive">Error loading Stage 2: {checksError}</div>
            ) : (
              <>
                <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-sm font-semibold">Data Validation Results</h3>
                    <button
                      type="button"
                      onClick={downloadValidationReport}
                      className="rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-foreground hover:bg-muted"
                    >
                      Download CSV
                    </button>
                  </div>
                  <ComplianceSummaryRow summary={summary} />
                </section>

                <section className="grid min-w-0 grid-cols-1 gap-6 lg:grid-cols-2">
                  <div className="min-w-0">
                    <div className="mb-3 rounded-lg border border-border bg-muted/40 px-4 py-3">
                      <div className="text-sm font-bold text-primary">📐 Threshold checks</div>
                      <div className="text-xs text-muted-foreground">Quantitative checks against regulatory thresholds</div>
                    </div>
                    {data?.thresholdChecks && data.thresholdChecks.length > 0 ? (
                      <ThresholdPanel checks={data.thresholdChecks} profileSource={validationProfile ?? profile} />
                    ) : (
                      <div className="rounded-xl border border-border bg-card p-4 text-sm text-muted-foreground">
                        No threshold checks were returned for this stage.
                      </div>
                    )}
                  </div>

                  <div className="min-w-0">
                    <div className="mb-3 rounded-lg border border-border bg-muted/40 px-4 py-3">
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-sm font-bold text-violet-500">🤖 RAG agent rules</div>
                        {llmLoading && (
                          <span className="flex items-center gap-1 text-xs font-medium text-violet-500">
                            <Clock className="h-3 w-3 animate-pulse" /> AI reviewing documentation…
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Regulatory rules fetched from knowledge store (IFRS 9, SS11/13, SS1/23)
                      </div>
                    </div>
                    {data?.ragRules && data.ragRules.length > 0 ? (
                      <RagRulesPanel rules={data.ragRules} />
                    ) : (
                      <div className="rounded-xl border border-border bg-card p-4 text-sm text-muted-foreground">
                        No RAG agent flags generated for this stage.
                      </div>
                    )}
                  </div>
                </section>

                <section className={`rounded-xl border p-6 ${regulatoryVerdictStyle(data?.regulatoryAlignment?.verdict).border} ${regulatoryVerdictStyle(data?.regulatoryAlignment?.verdict).bg}`}>
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-xs font-semibold uppercase tracking-wider text-foreground/70">Regulatory alignment</div>
                    <span className={`text-xs font-bold uppercase tracking-wide ${regulatoryVerdictStyle(data?.regulatoryAlignment?.verdict).text}`}>
                      {regulatoryVerdictStyle(data?.regulatoryAlignment?.verdict).icon} {data?.regulatoryAlignment?.verdict ?? "—"}
                    </span>
                  </div>
                  <p className="mt-2 text-sm">
                    Pass/Warn/Fail: {data?.regulatoryAlignment?.counts?.pass ?? 0}/{data?.regulatoryAlignment?.counts?.warn ?? 0}/
                    {data?.regulatoryAlignment?.counts?.fail ?? 0}
                  </p>
                  {data?.regulatoryAlignment?.remediation_summary && (
                    <p className="mt-3 text-sm text-muted-foreground">{data.regulatoryAlignment.remediation_summary}</p>
                  )}
                  {data?.regulatoryAlignment?.regulatory_references?.length ? (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {data.regulatoryAlignment.regulatory_references.map((ref: string) => (
                        <span
                          key={ref}
                          className="rounded-full border border-primary/20 bg-background px-3 py-1 text-xs font-medium text-foreground/80"
                        >
                          {ref}
                        </span>
                      ))}
                    </div>
                  ) : null}
                  {data?.regulatoryAlignment?.high_severity_fails?.length ? (
                    <div className="mt-3 text-sm text-destructive">
                      High severity fails: {data.regulatoryAlignment.high_severity_fails.length}
                    </div>
                  ) : null}
                </section>

                {runError ? (
                  <div className="rounded-xl border border-red-300 bg-red-50 p-3 text-sm text-red-900">{runError}</div>
                ) : null}

                <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
                  <h3 className="text-sm font-semibold">Data leakage detection</h3>
                  <p className="mt-2 text-sm text-foreground/80">
                    {validationProfile?.leakage_risk_cols && validationProfile.leakage_risk_cols.length > 0
                      ? `Potential leakage detected: ${validationProfile.leakage_risk_cols.join(", ")}`
                      : "No potential target leakage detected in the current dataset."}
                  </p>
                </div>
              </>
            )}
          </TabsContent>

          <TabsContent value="conceptual" className="space-y-8 pt-4">
            {conceptualLoading ? (
              <div className="rounded-xl border border-border bg-card p-6 text-center">Loading Conceptual Soundness...</div>
            ) : conceptualError ? (
              <div className="rounded-xl border border-border bg-card p-6 text-destructive">Error loading Stage 2: {conceptualError}</div>
            ) : (
              <>
                <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
                  <h3 className="mb-4 text-sm font-semibold">Conceptual Soundness Results</h3>
                  <ComplianceSummaryRow summary={conceptualSummary} />
                </section>

                <section className="grid min-w-0 grid-cols-1 gap-6 lg:grid-cols-2">
                  <div className="min-w-0">
                    <div className="mb-3 rounded-lg border border-border bg-muted/40 px-4 py-3">
                      <div className="text-sm font-bold text-primary">📐 Threshold checks</div>
                      <div className="text-xs text-muted-foreground">Quantitative checks against regulatory thresholds</div>
                    </div>
                    {conceptualData?.thresholdChecks && conceptualData.thresholdChecks.length > 0 ? (
                      <ThresholdPanel checks={conceptualData.thresholdChecks} profileSource={validationProfile ?? profile} />
                    ) : (
                      <div className="rounded-xl border border-border bg-card p-4 text-sm text-muted-foreground">
                        No threshold checks generated for this stage.
                      </div>
                    )}
                  </div>

                  <div className="min-w-0">
                    <div className="mb-3 rounded-lg border border-border bg-muted/40 px-4 py-3">
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-sm font-bold text-violet-500">🤖 RAG agent rules</div>
                        {conceptualLlmLoading && (
                          <span className="flex items-center gap-1 text-xs font-medium text-violet-500">
                            <Clock className="h-3 w-3 animate-pulse" /> AI reviewing documentation…
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Regulatory rules fetched from knowledge store (SS1/23, SS11/13, IFRS 9)
                      </div>
                    </div>
                    {conceptualData?.ragRules && conceptualData.ragRules.length > 0 ? (
                      <RagRulesPanel rules={conceptualData.ragRules} />
                    ) : (
                      <div className="rounded-xl border border-border bg-card p-4 text-sm text-muted-foreground">
                        No RAG agent flags generated for this stage.
                      </div>
                    )}
                  </div>
                </section>

                <section className={`rounded-xl border p-6 ${regulatoryVerdictStyle(conceptualData?.regulatoryAlignment?.verdict).border} ${regulatoryVerdictStyle(conceptualData?.regulatoryAlignment?.verdict).bg}`}>
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-xs font-semibold uppercase tracking-wider text-foreground/70">Regulatory alignment</div>
                    <span className={`text-xs font-bold uppercase tracking-wide ${regulatoryVerdictStyle(conceptualData?.regulatoryAlignment?.verdict).text}`}>
                      {regulatoryVerdictStyle(conceptualData?.regulatoryAlignment?.verdict).icon} {conceptualData?.regulatoryAlignment?.verdict ?? "—"}
                    </span>
                  </div>
                  <p className="mt-2 text-sm">
                    Pass/Warn/Fail: {conceptualData?.regulatoryAlignment?.counts?.pass ?? 0}/{conceptualData?.regulatoryAlignment?.counts?.warn ?? 0}/
                    {conceptualData?.regulatoryAlignment?.counts?.fail ?? 0}
                  </p>
                  {conceptualData?.regulatoryAlignment?.remediation_summary && (
                    <p className="mt-3 text-sm text-muted-foreground">{conceptualData.regulatoryAlignment.remediation_summary}</p>
                  )}
                  {conceptualData?.regulatoryAlignment?.regulatory_references?.length ? (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {conceptualData.regulatoryAlignment.regulatory_references.map((ref: string) => (
                        <span
                          key={ref}
                          className="rounded-full border border-primary/20 bg-background px-3 py-1 text-xs font-medium text-foreground/80"
                        >
                          {ref}
                        </span>
                      ))}
                    </div>
                  ) : null}
                  {conceptualData?.regulatoryAlignment?.high_severity_fails?.length ? (
                    <div className="mt-3 text-sm text-destructive">
                      High severity fails: {conceptualData.regulatoryAlignment.high_severity_fails.length}
                    </div>
                  ) : null}
                </section>
              </>
            )}
          </TabsContent>
        </Tabs>
      )}

      <div className="text-right">
        {activeTab === "data-validation" ? (
          <button
            type="button"
            onClick={() => setActiveTab("conceptual")}
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90"
          >
            Continue
            <ArrowRight className="h-4 w-4" />
          </button>
        ) : (
          <Link
            to="/validation/challenger"
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90"
          >
            Continue to Stage 3
            <ArrowRight className="h-4 w-4" />
          </Link>
        )}
      </div>
    </div>
  );
}

const STATUS_STYLES: Record<string, { border: string; bg: string; badge: string; icon: string }> = {
  PASS: { border: "border-emerald-500/40", bg: "bg-emerald-500/10", badge: "bg-emerald-500 text-emerald-950", icon: "✅" },
  WARN: { border: "border-amber-500/40", bg: "bg-amber-500/10", badge: "bg-amber-500 text-amber-950", icon: "🟡" },
  FAIL: { border: "border-red-500/40", bg: "bg-red-500/10", badge: "bg-red-500 text-red-950", icon: "🔴" },
  PENDING: { border: "border-slate-400/40", bg: "bg-slate-400/10", badge: "bg-slate-400 text-slate-950", icon: "⏱️" },
};

// Mirrors the Findings page's VERDICT_STYLES convention — the regulatory
// alignment card should reflect the actual verdict, not always render as a
// green "all good" card regardless of whether it's PASS/CONDITIONAL/FAIL.
const REGULATORY_VERDICT_STYLES: Record<string, { border: string; bg: string; text: string; icon: string }> = {
  PASS: { border: "border-emerald-500/40", bg: "bg-emerald-500/10", text: "text-emerald-700 dark:text-emerald-300", icon: "✅" },
  CONDITIONAL: { border: "border-amber-500/40", bg: "bg-amber-500/10", text: "text-amber-700 dark:text-amber-300", icon: "⚠️" },
  FAIL: { border: "border-red-500/40", bg: "bg-red-500/10", text: "text-red-700 dark:text-red-300", icon: "❌" },
};

function regulatoryVerdictStyle(verdict: string | undefined) {
  return REGULATORY_VERDICT_STYLES[verdict ?? ""] ?? { border: "border-border", bg: "bg-card", text: "text-foreground", icon: "⚪" };
}

const CHECK_SOURCE_LABELS: Record<string, { label: string; classes: string }> = {
  llm: { label: "🤖 LLM", classes: "bg-violet-500 text-violet-950" },
  llm_error: { label: "⚠️ LLM (call failed)", classes: "bg-amber-500 text-amber-950" },
  quantitative: { label: "📐 Rule-based", classes: "bg-sky-500 text-sky-950" },
  keyword_search: { label: "🔤 Keyword search", classes: "bg-slate-400 text-slate-950" },
};

function statusStyle(status: string | undefined) {
  return STATUS_STYLES[status ?? ""] ?? { border: "border-border", bg: "bg-card", badge: "bg-muted text-foreground", icon: "⚪" };
}

function statusIcon(status: string | undefined, className = "h-4 w-4") {
  switch (status) {
    case "FAIL":
      return <AlertTriangle className={`${className} text-red-600 dark:text-red-400`} />;
    case "WARN":
      return <AlertCircle className={`${className} text-amber-600 dark:text-amber-400`} />;
    case "PENDING":
      return <Clock className={`${className} text-slate-500 dark:text-slate-400`} />;
    default:
      return <Check className={`${className} text-emerald-600 dark:text-emerald-400`} />;
  }
}

// ── Checks-passed donut + pass/warn/fail count cards ─────────────────────

function ComplianceDonut({ pass, warn, fail, size = 88 }: { pass: number; warn: number; fail: number; size?: number }) {
  const total = pass + warn + fail;
  const failPct = total > 0 ? (fail / total) * 100 : 0;
  const warnPct = total > 0 ? (warn / total) * 100 : 0;
  const score = total > 0 ? Math.round((pass / total) * 100) : 0;
  const gradient =
    total > 0
      ? `conic-gradient(#dc2626 0% ${failPct}%, #d97706 ${failPct}% ${failPct + warnPct}%, #16a34a ${failPct + warnPct}% 100%)`
      : "#e5e7eb";

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <div className="h-full w-full rounded-full" style={{ background: gradient }} />
      <div
        className="absolute flex items-center justify-center rounded-full bg-card"
        style={{ inset: Math.round(size * 0.16) }}
      >
        <span className="text-lg font-bold text-foreground">{score}%</span>
      </div>
    </div>
  );
}

function ComplianceSummaryRow({
  summary,
}: {
  summary: { total: number; pass: number; warn: number; fail: number; pending?: number; na?: number };
}) {
  // The backend's own `total` (e.g. len(combined) in main.py's
  // _group_stage2/_group_stage3) previously included "pending" findings
  // that were never rendered as a tile below, so the "X/Y passed" header
  // and the tile row could show different totals. Derive the total here
  // from the exact same buckets CheckSummaryTiles renders instead.
  const total = deriveCheckTotal(summary);
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 rounded-xl border border-border bg-background p-4">
        <ComplianceDonut pass={summary.pass} warn={summary.warn} fail={summary.fail} />
        <div className="min-w-0">
          <div className="text-xs text-muted-foreground">Checks passed</div>
          <div className="text-base font-bold leading-tight text-foreground">
            {summary.pass}/{total} passed
          </div>
        </div>
      </div>
      <CheckSummaryTiles summary={summary} checksLabel="Checks" />
    </div>
  );
}

// ── Dataset insight (e.g. missing values by column) shown inside a threshold
// check's detail panel when the check text suggests it's relevant. Purely
// additive UI — falls back to nothing if the profile doesn't have usable
// fields, so it never breaks rendering of the underlying backend data. ──

function extractMissingness(profileSource: any): { column: string; pct: number }[] | null {
  if (!profileSource || typeof profileSource !== "object") return null;

  const mapCandidates = [
    profileSource.missing_by_column,
    profileSource.missing_pct_by_column,
    profileSource.column_missing_pct,
    profileSource.missingness,
    profileSource.null_pct_by_column,
  ];
  for (const candidate of mapCandidates) {
    if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
      const entries = Object.entries(candidate)
        .map(([column, raw]) => {
          const num = Number(raw);
          if (Number.isNaN(num)) return null;
          return { column, pct: num <= 1 ? num * 100 : num };
        })
        .filter((e): e is { column: string; pct: number } => e !== null);
      if (entries.length) return entries.sort((a, b) => b.pct - a.pct);
    }
  }

  const arrayCandidates = [
    profileSource.columns,
    profileSource.column_stats,
    profileSource.column_summary,
    profileSource.column_profiles,
  ];
  for (const arr of arrayCandidates) {
    if (Array.isArray(arr) && arr.length) {
      const entries = arr
        .map((c: any) => {
          const column = c?.name ?? c?.column ?? c?.column_name;
          const raw = c?.missing_pct ?? c?.missing_percentage ?? c?.null_pct ?? c?.pct_missing ?? c?.missing_rate;
          if (column == null || raw == null) return null;
          const num = Number(raw);
          if (Number.isNaN(num)) return null;
          return { column: String(column), pct: num <= 1 ? num * 100 : num };
        })
        .filter((e: any): e is { column: string; pct: number } => e !== null);
      if (entries.length) return entries.sort((a, b) => b.pct - a.pct);
    }
  }

  return null;
}

function MissingnessChart({ rows }: { rows: { column: string; pct: number }[] }) {
  const top = rows.filter((r) => r.pct > 0).slice(0, 8);
  if (!top.length) return null;
  const max = Math.max(...top.map((r) => r.pct), 1);

  return (
    <div className="mt-4 rounded-lg border border-border/60 bg-background/60 p-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Missing values by column
      </div>
      <div className="space-y-1.5">
        {top.map((r) => (
          <div key={r.column} className="flex items-center gap-2 text-xs">
            <span className="w-28 shrink-0 truncate text-foreground/80" title={r.column}>
              {r.column}
            </span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-amber-500"
                style={{ width: `${Math.min(100, (r.pct / max) * 100)}%` }}
              />
            </div>
            <span className="w-12 shrink-0 text-right text-muted-foreground">{r.pct.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DatasetInsight({ check, profileSource }: { check: ThresholdCheck; profileSource: any }) {
  const text = `${check.title} ${check.detail} ${check.observed}`.toLowerCase();
  const looksLikeMissingness = /missing|null value|completeness/.test(text);
  if (!looksLikeMissingness) return null;

  const rows = extractMissingness(profileSource);
  if (!rows) return null;

  return <MissingnessChart rows={rows} />;
}

// ── Threshold checks: compact tile grid, tap a tile for detail ──────────────

function shortLabel(title: string, maxLen = 20): string {
  const words = title.trim().split(/\s+/).filter(Boolean);
  if (!words.length) return title.slice(0, maxLen);

  let label = "";
  for (const word of words) {
    const candidate = label ? `${label} ${word}` : word;
    if (candidate.length > maxLen && label) break;
    label = candidate;
    if (label.length >= maxLen) break;
  }
  return label.length > maxLen ? `${label.slice(0, maxLen - 1)}…` : label;
}

function ThresholdTile({
  check,
  active,
  onClick,
}: {
  check: ThresholdCheck;
  active: boolean;
  onClick: () => void;
}) {
  const palette =
    check.status === "FAIL"
      ? "border-red-300 bg-red-50 text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-300"
      : check.status === "WARN"
      ? "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-300"
      : "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-500/40 dark:bg-emerald-500/10 dark:text-emerald-300";

  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex min-h-[64px] flex-col items-center justify-center gap-1 rounded-lg border p-2.5 text-center transition-colors ${palette} ${
        active ? "ring-2 ring-primary ring-offset-1 ring-offset-background" : ""
      }`}
    >
      {statusIcon(check.status)}
      <span className="text-[11px] font-semibold leading-tight">{shortLabel(check.title)}</span>
    </button>
  );
}

function ThresholdDetailPanel({ check, profileSource }: { check: ThresholdCheck; profileSource: any }) {
  const s = statusStyle(check.status);
  const isConvention = isQuantitativeConventionCheck(check.check_type);
  return (
    <div className={`mt-4 rounded-xl border ${s.border} ${s.bg} p-5`}>
      <div className="flex items-start justify-between gap-3">
        <h4 className="text-base font-bold text-foreground">
          [{check.check_id}] {check.title}
        </h4>
        <span className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-bold ${s.badge}`}>{check.status}</span>
      </div>
      {isConvention ? (
        <p className="mt-1 text-xs text-muted-foreground">
          Regulatory basis: {check.source} {check.principle} — requires this to be assessed/documented
        </p>
      ) : (
        <p className="mt-1 text-xs text-muted-foreground">
          {check.source} — {check.principle}
        </p>
      )}
      <dl className="mt-3 space-y-1.5 text-sm">
        <div>
          <dt className="inline font-semibold text-foreground">Observed </dt>
          <dd className="inline text-foreground/90">{check.observed}</dd>
        </div>
        <div>
          <dt className="inline font-semibold text-foreground">Threshold </dt>
          <dd className="inline text-foreground/90">
            {check.threshold}
            {isConvention ? <span className="text-muted-foreground"> — industry-standard convention</span> : null}
          </dd>
        </div>
      </dl>
      <p className="mt-3 text-sm text-muted-foreground">{check.detail}</p>
      <DatasetInsight check={check} profileSource={profileSource} />
    </div>
  );
}

function ThresholdPanel({ checks, profileSource }: { checks: ThresholdCheck[]; profileSource: any }) {
  const statusRank: Record<string, number> = { FAIL: 0, WARN: 1, PASS: 2 };
  const sorted = [...checks].sort((a, b) => (statusRank[a.status] ?? 3) - (statusRank[b.status] ?? 3));
  const [selectedId, setSelectedId] = useState<string | null>(sorted[0]?.check_id ?? null);
  const selected = checks.find((c) => c.check_id === selectedId) ?? null;

  return (
    <div>
      <div className="mb-2 text-xs text-muted-foreground">Tap a tile for detail</div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {sorted.map((c) => (
          <ThresholdTile key={c.check_id} check={c} active={selectedId === c.check_id} onClick={() => setSelectedId(c.check_id)} />
        ))}
      </div>
      {selected ? <ThresholdDetailPanel check={selected} profileSource={profileSource} /> : null}
    </div>
  );
}

// ── RAG agent rules: filterable list, tap a row for detail ──────────────────

type StatusFilter = "ALL" | "FAIL" | "WARN" | "PENDING" | "PASS";

function ruleEffectiveStatus(rule: RagRule): string {
  if (rule.not_verifiable) return "PENDING";
  return (rule.status || "").toUpperCase();
}

function RagRuleRow({ rule, active, onClick }: { rule: RagRule; active: boolean; onClick: () => void }) {
  const eff = ruleEffectiveStatus(rule);
  const s = statusStyle(eff);
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex w-full min-w-0 items-center gap-2 rounded-lg border px-3 py-2.5 text-left transition-colors ${
        active ? `${s.border} ${s.bg}` : "border-border bg-card hover:bg-muted/50"
      }`}
    >
      <span className="shrink-0">{statusIcon(eff)}</span>
      <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">{rule.flag}</span>
    </button>
  );
}

function RagRuleDetailPanel({ rule }: { rule: RagRule }) {
  const eff = ruleEffectiveStatus(rule);
  const s = statusStyle(eff);
  const observed = Array.isArray(rule.observed_value) ? rule.observed_value.join(", ") : rule.observed_value;
  const csrc = CHECK_SOURCE_LABELS[rule.check_source ?? ""];

  return (
    <div className={`mt-4 rounded-xl border ${s.border} ${s.bg} p-5`}>
      <div className="flex items-start justify-between gap-3">
        <h4 className="text-base font-bold text-foreground">{rule.flag}</h4>
        <span className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-bold ${s.badge}`}>{eff}</span>
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {rule.source} — {rule.principle}
        {csrc ? ` · ${csrc.label.replace(/^\S+\s/, "")}` : ""}
        {rule.not_verifiable ? " · not verifiable with current data" : ""}
      </p>
      {observed != null && observed !== "" ? (
        <p className="mt-3 text-sm text-foreground">
          Observed: <span className="text-foreground/90">{observed}</span>
        </p>
      ) : null}
      {rule.reasoning ? <p className="mt-2 text-xs italic text-violet-600 dark:text-violet-300">{rule.reasoning}</p> : null}
      <p className="mt-3 text-sm text-muted-foreground">{rule.suggestion}</p>
    </div>
  );
}

function RagRulesPanel({ rules }: { rules: RagRule[] }) {
  const [filter, setFilter] = useState<StatusFilter>("ALL");
  const [selectedId, setSelectedId] = useState<string | null>(rules[0]?.rule_id ?? null);

  const filtered = filter === "ALL" ? rules : rules.filter((r) => ruleEffectiveStatus(r) === filter);
  const selected = filtered.find((r) => r.rule_id === selectedId) ?? filtered[0] ?? null;

  const tabs: { key: StatusFilter; label: string }[] = [
    { key: "ALL", label: "All" },
    { key: "FAIL", label: "Fail" },
    { key: "WARN", label: "Warn" },
    { key: "PENDING", label: "Pending" },
    { key: "PASS", label: "Pass" },
  ];

  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {tabs.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setFilter(t.key)}
            className={`rounded-full border px-3 py-1 text-xs font-semibold transition-colors ${
              filter === t.key
                ? "border-primary bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:bg-muted"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="space-y-1.5">
        {filtered.length ? (
          filtered.map((r) => (
            <RagRuleRow key={r.rule_id} rule={r} active={selected?.rule_id === r.rule_id} onClick={() => setSelectedId(r.rule_id)} />
          ))
        ) : (
          <div className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
            No rules match this filter.
          </div>
        )}
      </div>

      {selected ? <RagRuleDetailPanel rule={selected} /> : null}
    </div>
  );
}
