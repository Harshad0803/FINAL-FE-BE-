import { createFileRoute, Link } from "@tanstack/react-router";
import { PageHeader } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { ArrowRight, FileText, Database, GitCompareArrows, BarChart3, Activity, ShieldCheck, ClipboardCheck } from "lucide-react";

export const Route = createFileRoute("/validation/")({
  head: () => ({
    meta: [
      { title: "Model Validation — Aegis Credit" },
      { name: "description", content: "Independent validation: intake, data quality, conceptual soundness, challenger, performance, stress, regulatory, findings." },
    ],
  }),
  component: ValidationHome,
});

const stages = [
  { stage: 1, to: "/validation/intake", icon: FileText, title: "Intake & Governance", desc: "Model metadata, artifacts, risk tier, and governance attestation." },
  { stage: 2, to: "/validation/data-quality", icon: Database, title: "Data & Model Soundness", desc: "Automated dataset checks, leakage scan, sample representativeness, feature relevance, methodology, and assumptions." },
  { stage: 3, to: "/validation/challenger", icon: GitCompareArrows, title: "Model Replication", desc: "Independently reproduce developer outputs and verify the R4.1-R4.8 replication checks." },
  { stage: 4, to: "/validation/performance", icon: BarChart3, title: "Benchmarking", desc: "AUC, KS, calibration, threshold analysis, hold-out validation, and champion vs challenger benchmarking." },
  { stage: 5, to: "/validation/stress", icon: Activity, title: "Stress & Backtesting", desc: "Scenario simulations, stability, backtests, and stress results." },
  { stage: 6, to: "/validation/regulatory", icon: ShieldCheck, title: "Explainability and Fairness", desc: "SHAP feature importance, fair lending bias checks, IFRS 9 / IFRS 7 / SS1/23 review, and remediation." },
  { stage: 7, to: "/validation/findings", icon: ClipboardCheck, title: "Findings & Final Report", desc: "Final observations, risk grading, recommendation, and sign-off." },
] as const;

function ValidationHome() {
  return (
    <div className="space-y-8">
      <PageHeader
        title="Model Validation"
        description="Independent review of an existing credit risk model across performance, conceptual soundness, regulatory compliance, and governance."
      />

      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["RAG status", "AMBER", "1 high, 2 medium findings"],
          ["Compliance", "92.4%", "IFRS 9 / IFRS 7 / SS1/23"],
          ["Champion AUC", "0.873", "vs challenger 0.869"],
          ["Validation cycle", "Q2 · 2026", "Quarterly, Tier 2"],
        ].map(([l, v, s]) => (
          <Card key={l} className="p-5 shadow-elegant">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{l}</div>
            <div className="mt-2 text-2xl font-semibold tracking-tight">{v}</div>
            <div className="mt-1 text-xs text-muted-foreground">{s}</div>
          </Card>
        ))}
      </section>

      <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {stages.map((s) => {
          const Icon = s.icon;
          return (
            <Link
              key={s.to}
              to={s.to}
              className="group block"
            >
              <Card className="flex items-start gap-4 p-5 shadow-elegant transition-all hover:-translate-y-0.5 hover:border-primary/40">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary-soft text-primary">
                  <Icon className="h-5 w-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="border-border bg-background px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
                      Stage {s.stage}
                    </Badge>
                    <h3 className="text-sm font-semibold text-foreground">{s.title}</h3>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">{s.desc}</p>
                </div>
                <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-primary" />
              </Card>
            </Link>
          );
        })}
      </section>
    </div>
  );
}
