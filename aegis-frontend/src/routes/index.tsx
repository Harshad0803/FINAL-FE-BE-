import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRight, Boxes, ShieldCheck, Sparkles, BarChart3, FileCheck2, GitCompareArrows, Database } from "lucide-react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Aegis Credit — Model Development & Validation" },
      { name: "description", content: "Choose a workspace: build credit risk models or independently validate them for regulatory compliance and governance." },
    ],
  }),
  component: Landing,
});

function Landing() {
  return (
    <div className="mx-auto flex max-w-6xl flex-col">
      <div className="mb-10 md:mb-14">
        <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
          <span className="h-1.5 w-1.5 rounded-full bg-primary" />
          Enterprise AI Platform
        </div>
        <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-5xl">
          Choose your workspace
        </h1>
        <p className="mt-3 max-w-2xl text-sm text-muted-foreground md:text-base">
          Aegis Credit unifies model development and independent validation in a single, regulator-grade platform.
          Select a workspace to begin.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <WorkspaceCard
          to="/development"
          accent="from-primary/15 to-transparent"
          icon={<Boxes className="h-6 w-6 text-primary" />}
          eyebrow="Workspace 01"
          title="Model Development"
          description="Build, train, evaluate, and explain credit risk models with an end-to-end ML workflow."
          bullets={[
            { icon: <Sparkles className="h-3.5 w-3.5" />, label: "Data → Features → Training → Explainability" },
            { icon: <BarChart3 className="h-3.5 w-3.5" />, label: "Live model metrics & SHAP attribution" },
          ]}
          cta="Open Model Development"
        />

        <WorkspaceCard
          to="/validation"
          accent="from-foreground/10 to-transparent"
          icon={<ShieldCheck className="h-6 w-6 text-primary" />}
          eyebrow="Workspace 02"
          title="Model Validation"
          description="Independently validate existing models for performance, conceptual soundness, regulatory compliance, and governance."
          bullets={[
            { icon: <GitCompareArrows className="h-3.5 w-3.5" />, label: "Champion vs challenger benchmarking" },
            { icon: <FileCheck2 className="h-3.5 w-3.5" />, label: "IFRS 9 / IFRS 7 / SS1/23 evidence pack" },
          ]}
          cta="Open Model Validation"
        />

        <WorkspaceCard
          to="/model-inventory"
          accent="from-primary/10 to-transparent"
          icon={<Database className="h-6 w-6 text-primary" />}
          eyebrow="Workspace 03"
          title="Model Inventory"
          description="Register models, attach data sources, track validation status, and export a governance-ready inventory."
          bullets={[
            { icon: <Database className="h-3.5 w-3.5" />, label: "Persistent model registry" },
            { icon: <FileCheck2 className="h-3.5 w-3.5" />, label: "Excel export for consultant reviews" },
          ]}
          cta="Open Model Inventory"
        />
      </div>
    </div>
  );
}

function WorkspaceCard({
  to, icon, eyebrow, title, description, bullets, cta, accent,
}: {
  to: string;
  icon: React.ReactNode;
  eyebrow: string;
  title: string;
  description: string;
  bullets: { icon: React.ReactNode; label: string }[];
  cta: string;
  accent: string;
}) {
  return (
    <Link
      to={to}
      className="group relative flex flex-col overflow-hidden rounded-2xl border border-border bg-card p-8 shadow-elegant transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-lg"
    >
      <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${accent} opacity-60`} />
      <div className="relative">
        <div className="flex items-center justify-between">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary-soft">
            {icon}
          </div>
          <span className="text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
            {eyebrow}
          </span>
        </div>
        <h2 className="mt-6 text-2xl font-semibold tracking-tight">{title}</h2>
        <p className="mt-2 text-sm text-muted-foreground">{description}</p>

        <ul className="mt-6 space-y-2">
          {bullets.map((b) => (
            <li key={b.label} className="flex items-center gap-2 text-sm text-foreground/80">
              <span className="flex h-5 w-5 items-center justify-center rounded-md bg-primary-soft text-primary">
                {b.icon}
              </span>
              {b.label}
            </li>
          ))}
        </ul>

        <div className="mt-8 inline-flex items-center gap-2 rounded-lg gradient-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground shadow-elegant transition-transform group-hover:translate-x-0.5">
          {cta} <ArrowRight className="h-4 w-4" />
        </div>
      </div>
    </Link>
  );
}
