import { createFileRoute, Link } from "@tanstack/react-router";
import { PageHeader } from "@/components/app-shell";
import { useDataset } from "@/lib/app-context";
import { useResumeState } from "@/hooks/use-resume-state";
import {
  FileText, Database, GitCompareArrows, BarChart3, Activity, ShieldCheck, ClipboardCheck,
  ArrowRight, Gauge, ClipboardList, ScrollText,
} from "lucide-react";
import { StageHero, HeroChip, KpiStrip, ValidationPipeline, StatusPill, type PipelineNodeStatus } from "@/components/validation-ui";

export const Route = createFileRoute("/validation/")({
  head: () => ({
    meta: [
      { title: "Model Validation — Aegis Credit" },
      { name: "description", content: "Independent validation: intake, data quality, conceptual soundness, challenger, performance, stress, regulatory, findings." },
    ],
  }),
  component: ValidationHome,
});

type StageStatus = "complete" | "active" | "pending";

function ValidationHome() {
  const {
    profile,
    trainingResult,
    validationIntakeData,
    validationStage3Result,
    validationStage4Result,
    validationStage5Result,
    validationStage7Result,
    validationStage8Result,
  } = useDataset();

  // Stress & Backtesting doesn't persist a result into shared context (its
  // page keeps that state locally) — the only real signal of whether it's
  // been run is the same backend-saved run log every stage page resumes
  // from, so ask it the same way validation.stress.tsx does.
  const { data: resumedStress } = useResumeState<{ stage: string; report: any }>(
    "validation_pipeline_log.csv",
    "stress_testing",
  );

  const stage3Done = Boolean(validationStage3Result);
  const stage4Done = Boolean(validationStage4Result?.replication);
  const stage5Done = Boolean(validationStage5Result);
  const stressDone = Boolean(resumedStress?.report);
  const stage7Done = Boolean(validationStage7Result);
  const stage8Done = Boolean(validationStage8Result);
  const intakeDone = Boolean(validationIntakeData);

  const stages: Array<{
    stage: number;
    to: string;
    icon: typeof FileText;
    title: string;
    desc: string;
    done: boolean;
    sub?: string;
  }> = [
    {
      stage: 1, to: "/validation/intake", icon: FileText, title: "Intake & Governance",
      desc: "Model metadata, artifacts, risk tier, and governance attestation.",
      done: intakeDone,
      sub: validationIntakeData?.model_name ? `Model: ${validationIntakeData.model_name}` : undefined,
    },
    {
      stage: 2, to: "/validation/data-quality", icon: Database, title: "Data & Model Soundness",
      desc: "Automated dataset checks, leakage scan, sample representativeness, feature relevance, methodology, and assumptions.",
      done: stage3Done,
      sub: stage3Done ? `${validationStage3Result?.summary?.pass ?? 0}/${validationStage3Result?.summary?.total ?? 0} checks passed` : undefined,
    },
    {
      stage: 3, to: "/validation/challenger", icon: GitCompareArrows, title: "Model Replication",
      desc: "Independently reproduce developer outputs and verify the replication checks.",
      done: stage4Done,
      sub: stage4Done ? `${(validationStage4Result?.flags ?? []).length} flag${(validationStage4Result?.flags ?? []).length === 1 ? "" : "s"} raised` : undefined,
    },
    {
      stage: 4, to: "/validation/performance", icon: BarChart3, title: "Benchmarking",
      desc: "AUC, KS, calibration, threshold analysis, hold-out validation, and champion vs challenger benchmarking.",
      done: stage5Done,
      sub: stage5Done && typeof validationStage5Result?.report?.roc_curve?.auc === "number"
        ? `Champion ROC-AUC ${validationStage5Result.report.roc_curve.auc.toFixed(3)}`
        : undefined,
    },
    {
      stage: 5, to: "/validation/stress", icon: Activity, title: "Stress & Backtesting",
      desc: "Scenario simulations, stability, backtests, and stress results.",
      done: stressDone,
    },
    {
      stage: 6, to: "/validation/regulatory", icon: ShieldCheck, title: "Explainability and Fairness",
      desc: "SHAP feature importance, fair lending bias checks, IFRS 9 / IFRS 7 / SS1/23 review, and remediation.",
      done: stage7Done,
      sub: stage7Done ? `${validationStage7Result?.summary?.pass ?? 0}/${validationStage7Result?.summary?.total ?? 0} compliance checks passed` : undefined,
    },
    {
      stage: 7, to: "/validation/findings", icon: ClipboardCheck, title: "Findings & Final Report",
      desc: "Final observations, risk grading, recommendation, and sign-off.",
      done: stage8Done,
      sub: stage8Done ? `Verdict: ${validationStage8Result?.verdict}` : undefined,
    },
  ];

  const completedCount = stages.filter((s) => s.done).length;
  const firstIncompleteIndex = stages.findIndex((s) => !s.done);

  const pipelineNodes: Array<{ label: string; status: PipelineNodeStatus; sub?: string }> = stages.map((s, i) => ({
    label: s.title,
    status: s.done ? "complete" : i === firstIncompleteIndex ? "active" : "pending",
  }));

  const modelName = validationIntakeData?.model_name ?? trainingResult?.model_name ?? null;
  const compliancePct = stage7Done && validationStage7Result?.summary?.total
    ? Math.round((validationStage7Result.summary.pass / validationStage7Result.summary.total) * 100)
    : null;

  const kpiTiles = [
    {
      icon: Gauge, label: "Validation Progress", tone: "primary" as const,
      value: `${completedCount}/${stages.length}`,
      sub: completedCount === stages.length ? "All stages complete" : `Stage ${Math.min(firstIncompleteIndex + 1, stages.length)} in progress`,
    },
    {
      icon: FileText, label: "Model Under Review", tone: "slate" as const,
      value: modelName ?? "—",
      sub: modelName ? (validationIntakeData?.model_version ? `v${validationIntakeData.model_version}` : "Real intake record") : "Not yet submitted",
    },
    {
      icon: ShieldCheck, label: "Compliance", tone: "emerald" as const,
      value: compliancePct !== null ? `${compliancePct}%` : "Pending",
      sub: stage7Done ? `${validationStage7Result?.summary?.pass}/${validationStage7Result?.summary?.total} checks passed` : "Stage 6 not run yet",
    },
    {
      icon: ClipboardList, label: "Findings", tone: "amber" as const,
      value: stage8Done ? String(validationStage8Result?.total_count ?? 0) : "—",
      sub: stage8Done
        ? `${validationStage8Result?.high_count ?? 0} high · ${validationStage8Result?.medium_count ?? 0} medium`
        : "Stage 7 not run yet",
    },
    {
      icon: ScrollText, label: "Overall Decision", tone: "violet" as const,
      value: stage8Done ? String(validationStage8Result?.verdict) : "Pending",
      sub: stage8Done ? "From Findings & Final Report" : "Awaiting final report",
    },
  ];

  return (
    <div className="space-y-6">
      <PageHeader title="Model Validation" description="Independent review of an existing credit risk model across performance, conceptual soundness, regulatory compliance, and governance." />

      <StageHero
        eyebrow="MODEL RISK MANAGEMENT · VALIDATION WORKSPACE"
        title={modelName ? `Validating ${modelName}` : "Validation Workspace"}
        description={
          completedCount === 0
            ? "No validation stages have been run yet. Start with Intake & Governance to register the model under review."
            : `${completedCount} of ${stages.length} validation stages complete. Continue the workflow below to reach a final decision.`
        }
        chips={
          <>
            <HeroChip tone={completedCount === stages.length ? "success" : "neutral"}>
              {completedCount}/{stages.length} stages complete
            </HeroChip>
            {stage8Done && validationStage8Result && (
              <HeroChip tone={validationStage8Result.verdict === "APPROVED" ? "success" : validationStage8Result.verdict === "REJECTED" ? "warning" : "neutral"}>
                {validationStage8Result.verdict}
              </HeroChip>
            )}
          </>
        }
      />

      <KpiStrip tiles={kpiTiles} />

      <ValidationPipeline nodes={pipelineNodes} />

      <section>
        <h2 className="mb-3 text-sm font-semibold text-slate-900">Validation Stages</h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {stages.map((s) => {
            const Icon = s.icon;
            const status: StageStatus = s.done ? "complete" : stages.indexOf(s) === firstIncompleteIndex ? "active" : "pending";
            return (
              <Link key={s.to} to={s.to} className="group block">
                <div className="flex items-start gap-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition-all hover:-translate-y-0.5 hover:border-blue-300 hover:shadow-md">
                  <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${s.done ? "bg-emerald-50 text-emerald-600" : "bg-blue-50 text-blue-600"}`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-slate-400">
                        Stage {s.stage}
                      </span>
                      <h3 className="text-sm font-semibold text-slate-900">{s.title}</h3>
                      <StatusPill tone={s.done ? "pass" : status === "active" ? "pending" : "na"}>
                        {s.done ? "Complete" : status === "active" ? "In progress" : "Not started"}
                      </StatusPill>
                    </div>
                    <p className="mt-1 text-xs text-slate-500">{s.desc}</p>
                    {s.sub && <p className="mt-1.5 text-xs font-semibold text-slate-700">{s.sub}</p>}
                  </div>
                  <ArrowRight className="h-4 w-4 shrink-0 text-slate-400 transition-transform group-hover:translate-x-0.5 group-hover:text-blue-600" />
                </div>
              </Link>
            );
          })}
        </div>
      </section>
    </div>
  );
}
