import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { PageHeader } from "@/components/app-shell";
import { api, formUpload } from "@/lib/api";
import { useDataset } from "@/lib/app-context";
import { ArrowRight, FileCheck, FileText, Upload, CheckCircle2, Circle } from "lucide-react";
import { useResumeState } from "@/hooks/use-resume-state";

export const Route = createFileRoute("/validation/intake")({
  head: () => ({ meta: [{ title: "Model Intake — Aegis Credit" }] }),
  component: Intake,
});

type IntakeDisplay = {
  title: string;
  description: string;
  modelMetadata: {
    title: string;
    description: string;
    registeredLabel: string;
    items: [string, string][];
  };
  targetDefinition: {
    title: string;
    expression: string;
    detail: string;
    baseRateLabel: string;
    baseRate: string;
    sampleSizeLabel: string;
    sampleSize: string;
  };
  riskTier: {
    title: string;
    value: string;
    description: string;
  };
  artifactTitle: string;
  artifactDescription: string;
  artifactSummary: string;
  artifacts: {
    fileName: string;
    status: string;
    timestamp: string;
    required: boolean;
  }[];
  governance: {
    title: string;
    description: string;
    status: string;
    checklist: string[];
  };
  nextStep: {
    description: string;
    label: string;
    path: string;
  };
};

type IntakeResponse = {
  demo_mode?: string;
  demo_label?: string;
  display: IntakeDisplay;
  val_intake_data?: {
    model_name?: string;
    owning_team?: string;
    model_owner?: string;
    lead_validator?: string;
    model_type?: string;
    model_version?: string;
    model_tier?: string;
    model_purpose?: string;
    mdd_text?: string;
    frameworks?: string[];
  };
  val_mdd_reported_metrics?: Record<string, any>;
  chk_inventory?: boolean;
  chk_tier?: boolean;
  chk_artifacts?: boolean;
  chk_prev_findings?: boolean;
  chk_reg_scope?: boolean;
  chk_independence?: boolean;
  chk_plan_approved?: boolean;
  chk_attestation?: boolean;
};

// Risk tier is a classification (how much oversight this model needs), not
// a pass/fail judgment — deliberately NOT the red/amber/green severity scale
// used elsewhere for HIGH/MEDIUM/LOW findings, since that would wrongly
// imply a tier itself is "bad." Single blue hue at three intensities instead:
// darker/bolder = more oversight, lighter = routine, same hue throughout
// signals "different degree of the same thing," not "different verdict."
function riskTierTone(value: string): { border: string; bg: string; text: string } {
  if (value.includes("1")) return { border: "border-blue-700/40", bg: "bg-blue-700/10", text: "text-blue-800" };
  if (value.includes("2")) return { border: "border-blue-400/40", bg: "bg-blue-400/10", text: "text-blue-600" };
  if (value.includes("3")) return { border: "border-blue-200/60", bg: "bg-blue-100/40", text: "text-blue-500" };
  return { border: "border-border", bg: "bg-muted", text: "text-foreground" };
}

// Neutral placeholder shown until the reviewer actually loads a demo (or
// completes a real intake) — deliberately has no plausible-looking values,
// since a prior version of this screen fetched a canned "clean" demo
// snapshot unconditionally on mount and displayed it as if it were live data.
const emptyIntake: IntakeDisplay = {
  title: "Stage 1 — Intake & Governance",
  description:
    "Capture model metadata, upload all required artifacts, and complete the governance attestation checklist before proceeding to automated validation stages.",
  modelMetadata: {
    title: "Model metadata",
    description: "Key registration details supplied by the development team.",
    registeredLabel: "Not yet registered",
    items: [],
  },
  targetDefinition: {
    title: "Target definition",
    expression: "Not yet determined",
    detail: "Load a demo or upload a dataset to populate the target definition",
    baseRateLabel: "Base rate",
    baseRate: "—",
    sampleSizeLabel: "Sample size",
    sampleSize: "—",
  },
  riskTier: {
    title: "Risk tier",
    value: "—",
    description: "Upload a dataset or load a demo to determine risk tier.",
  },
  artifactTitle: "Artifact inventory",
  artifactDescription: "Uploaded evidence to support subsequent validation stages.",
  artifactSummary: "No artifacts yet",
  artifacts: [],
  governance: {
    title: "Governance attestation",
    description: "Confirm the model and validation plan are ready to proceed.",
    status: "Not started",
    checklist: [
      "Model is registered in the model inventory",
      "Risk tier assignment has been documented",
      "Submitted artifacts cover dataset, MDD, and training code",
      "Previous validation findings (if any) have been reviewed",
      "Regulatory scope (IFRS 9 / SS1/23 / SS11/13) is identified",
      "Independent validation team has no conflict of interest",
      "Validation plan has been approved by the Head of Model Risk",
    ],
  },
  nextStep: {
    description: "Once intake is confirmed, proceed to Stage 2 data validation and automated checks.",
    label: "Proceed to Stage 2",
    path: "/validation/data-quality",
  },
};

function Intake() {
  const [intake, setIntake] = useState<IntakeDisplay>(emptyIntake);
  // True only once a demo has actually been loaded (or, in future, a real
  // intake has been saved) — gates the model-metadata/risk-tier/artifact
  // cards so they show a neutral prompt instead of pre-filled-looking data
  // before the reviewer has done anything.
  const [intakeLoaded, setIntakeLoaded] = useState(false);
  const navigate = useNavigate();
  const {
    setUploadResult,
    profile,
    setValidationIntakeData,
    setValidationMddText,
    setValidationMddMetrics,
    setValidationProfile,
    setValidationResults,
  } = useDataset();

  // Form state mirroring the Streamlit intake
  const [modelName, setModelName] = useState("");
  const [owningTeam, setOwningTeam] = useState("");
  const [modelOwner, setModelOwner] = useState("");
  const [leadValidator, setLeadValidator] = useState("");
  const [modelType, setModelType] = useState("PD (Probability of Default)");
  const [tier, setTier] = useState("Tier 2 — Medium Risk");
  const [version, setVersion] = useState("");
  const [purpose, setPurpose] = useState("");
  const [frameworks, setFrameworks] = useState<string[]>(["IFRS9", "SS1/23", "RBI"]);
  const [demoMode, setDemoMode] = useState<string | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoError, setDemoError] = useState<string | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [draftSavedAt, setDraftSavedAt] = useState<string | null>(null);
  const [proceedError, setProceedError] = useState<string | null>(null);

  // Resume where the reviewer left off: only pre-fills the (currently empty)
  // model name field from the most recently saved intake run — it doesn't
  // silently overwrite in-progress form data. The existing draft-lookup
  // effect below then surfaces its own opt-in "restore this draft?" prompt
  // once the name is filled in, same as if the reviewer had typed it.
  const { data: resumedIntake } = useResumeState<{ model_name?: string }>("validation_pipeline_log.csv", "intake");
  useEffect(() => {
    if (!modelName.trim() && resumedIntake?.model_name) {
      setModelName(resumedIntake.model_name);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumedIntake]);

  const toggleFramework = (fw: string) => {
    setFrameworks((prev) =>
      prev.includes(fw) ? prev.filter((f) => f !== fw) : [...prev, fw]
    );
  };

  // Artifacts state
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [mddFileName, setMddFileName] = useState<string | null>(null);
  const [mddText, setMddText] = useState<string | null>(null);
  const [mddMetrics, setMddMetrics] = useState<Record<string, any> | null>(null);
  const [mddDocumentPath, setMddDocumentPath] = useState<string | null>(null);
  const [trainingCodeFileName, setTrainingCodeFileName] = useState<string | null>(null);
  const [perfFileName, setPerfFileName] = useState<string | null>(null);
  const [profileFileName, setProfileFileName] = useState<string | null>(null);
  const [assumptionsFileName, setAssumptionsFileName] = useState<string | null>(null);
  const [hyperparamsFileName, setHyperparamsFileName] = useState<string | null>(null);

  // Checkboxes
  const [chkInventory, setChkInventory] = useState(false);
  const [chkTier, setChkTier] = useState(false);
  const [chkArtifacts, setChkArtifacts] = useState(false);
  const [chkPrevFindings, setChkPrevFindings] = useState(false);
  const [chkRegScope, setChkRegScope] = useState(false);
  const [chkIndependence, setChkIndependence] = useState(false);
  const [chkPlanApproved, setChkPlanApproved] = useState(false);
  const [chkAttestation, setChkAttestation] = useState(false);

  // This page's own form/checklist state lives in local useState (not
  // DatasetContext), so it needs its own small persistence layer to survive
  // a refresh — same localStorage pattern app-context.tsx uses for the
  // shared dataset/validation state. Restored once on mount; every change
  // after that re-saves the whole snapshot.
  // `isRestored` must be React state, not a ref: the write-effect below
  // needs to skip a render entirely between "restore finished" and "write
  // effect re-checks its guard" — a ref flip doesn't force that extra
  // render, so the write effect would still fire once with pre-restore
  // (default) values and clobber the just-read localStorage entry before
  // the restored state ever painted. Mirrors app-context.tsx's isHydrated.
  const [isRestored, setIsRestored] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") {
      setIsRestored(true);
      return;
    }
    try {
      const stored = window.localStorage.getItem("aegis_intake_form_state");
      if (stored) {
        const s = JSON.parse(stored);
        setIntake(s.intake ?? emptyIntake);
        setIntakeLoaded(Boolean(s.intakeLoaded));
        setModelName(s.modelName ?? "");
        setOwningTeam(s.owningTeam ?? "");
        setModelOwner(s.modelOwner ?? "");
        setLeadValidator(s.leadValidator ?? "");
        setModelType(s.modelType ?? "PD (Probability of Default)");
        setTier(s.tier ?? "Tier 2 — Medium Risk");
        setVersion(s.version ?? "");
        setPurpose(s.purpose ?? "");
        setFrameworks(Array.isArray(s.frameworks) && s.frameworks.length > 0 ? s.frameworks : ["IFRS9", "SS1/23", "RBI"]);
        setDemoMode(s.demoMode ?? null);
        setMddFileName(s.mddFileName ?? null);
        setMddText(s.mddText ?? null);
        setMddMetrics(s.mddMetrics ?? null);
        setMddDocumentPath(s.mddDocumentPath ?? null);
        setTrainingCodeFileName(s.trainingCodeFileName ?? null);
        setPerfFileName(s.perfFileName ?? null);
        setProfileFileName(s.profileFileName ?? null);
        setAssumptionsFileName(s.assumptionsFileName ?? null);
        setHyperparamsFileName(s.hyperparamsFileName ?? null);
        setChkInventory(Boolean(s.chkInventory));
        setChkTier(Boolean(s.chkTier));
        setChkArtifacts(Boolean(s.chkArtifacts));
        setChkPrevFindings(Boolean(s.chkPrevFindings));
        setChkRegScope(Boolean(s.chkRegScope));
        setChkIndependence(Boolean(s.chkIndependence));
        setChkPlanApproved(Boolean(s.chkPlanApproved));
        setChkAttestation(Boolean(s.chkAttestation));
      }
    } catch {
      // Ignore invalid stored state
    } finally {
      setIsRestored(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !isRestored) return;
    const snapshot = {
      intake, intakeLoaded, modelName, owningTeam, modelOwner, leadValidator, modelType, tier, version, purpose, frameworks,
      demoMode, mddFileName, mddText, mddMetrics, mddDocumentPath, trainingCodeFileName, perfFileName, profileFileName,
      assumptionsFileName, hyperparamsFileName, chkInventory, chkTier, chkArtifacts, chkPrevFindings, chkRegScope,
      chkIndependence, chkPlanApproved, chkAttestation,
    };
    try {
      window.localStorage.setItem("aegis_intake_form_state", JSON.stringify(snapshot));
    } catch (err) {
      console.warn("Failed to persist intake form state to localStorage:", err);
    }
  }, [
    intake, intakeLoaded, modelName, owningTeam, modelOwner, leadValidator, modelType, tier, version, purpose,
    demoMode, mddFileName, mddText, mddMetrics, trainingCodeFileName, perfFileName, profileFileName,
    assumptionsFileName, hyperparamsFileName, chkInventory, chkTier, chkArtifacts, chkPrevFindings, chkRegScope,
    chkIndependence, chkPlanApproved, chkAttestation, isRestored,
  ]);

  const datasetUploaded = Boolean(datasetFile || profile?.dataset_name);
  const mddUploaded = Boolean(mddFileName || mddText);
  const trainingCodeUploaded = Boolean(trainingCodeFileName);
  const profileUploaded = Boolean(profileFileName);
  const assumptionsUploaded = Boolean(assumptionsFileName);
  const perfUploaded = Boolean(perfFileName);
  const hyperparamsUploaded = Boolean(hyperparamsFileName);
  const readyCount = [datasetUploaded, mddUploaded, trainingCodeUploaded, profileUploaded, assumptionsUploaded, perfUploaded, hyperparamsUploaded].filter(Boolean).length;

  // Governance checklist items are intentionally NOT gated — they're an
  // honest record of what's actually true at this point in the process
  // (e.g. "no conflict of interest" may genuinely be unresolved yet), not a
  // set of boxes to force-tick. Only the attestation checkbox below — the
  // user's own confirmation that whatever state the checklist is actually
  // in is honestly represented — gates moving to the next stage.
  const handleProceed = () => {
    if (!chkAttestation) {
      setProceedError("Please confirm the information above is accurate before proceeding.");
      return;
    }
    setProceedError(null);
    setValidationIntakeData({
      model_name: modelName,
      owning_team: owningTeam,
      model_owner: modelOwner,
      lead_validator: leadValidator,
      model_type: modelType,
      model_tier: tier,
      model_version: version,
      model_purpose: purpose,
      mdd_text: mddText ?? null,
      mdd_document_path: mddDocumentPath ?? null,
      frameworks,
    });
    navigate({ to: intake.nextStep.path });
  };

  // Backend-persisted draft (keyed by model_name — no real user/session
  // system exists yet, so this is the closest thing to "this submission").
  // Deliberately opt-in rather than auto-restoring: silently overwriting
  // fields the user is mid-typing (or that a demo just populated) as soon as
  // a matching model_name is typed would be a worse surprise than just
  // asking first.
  const [draftAvailable, setDraftAvailable] = useState<{ savedAt: string; data: Record<string, any> } | null>(null);
  const [draftDismissedFor, setDraftDismissedFor] = useState<string | null>(null);
  const [draftSaving, setDraftSaving] = useState(false);
  const [draftLoadError, setDraftLoadError] = useState<string | null>(null);

  useEffect(() => {
    const trimmed = modelName.trim();
    if (!trimmed || intakeLoaded || draftDismissedFor === trimmed) {
      setDraftAvailable(null);
      return;
    }
    let active = true;
    const timer = setTimeout(() => {
      void api<{ found: boolean; saved_at?: string; data?: Record<string, any> }>(
        `/validation/intake/draft?model_name=${encodeURIComponent(trimmed)}`,
      )
        .then((resp) => {
          if (!active) return;
          if (resp.found && resp.data) {
            setDraftAvailable({ savedAt: resp.saved_at ?? "", data: resp.data });
          } else {
            setDraftAvailable(null);
          }
        })
        .catch(() => {
          if (active) setDraftAvailable(null);
        });
    }, 600);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [modelName, intakeLoaded, draftDismissedFor]);

  const applyDraft = (draft: Record<string, any>) => {
    if (draft.owningTeam) setOwningTeam(draft.owningTeam);
    if (draft.modelOwner) setModelOwner(draft.modelOwner);
    if (draft.leadValidator) setLeadValidator(draft.leadValidator);
    if (draft.modelType) setModelType(draft.modelType);
    if (draft.tier) setTier(draft.tier);
    if (draft.version) setVersion(draft.version);
    if (draft.purpose) setPurpose(draft.purpose);
    if (draft.mddFileName) setMddFileName(draft.mddFileName);
    if (draft.trainingCodeFileName) setTrainingCodeFileName(draft.trainingCodeFileName);
    if (draft.profileFileName) setProfileFileName(draft.profileFileName);
    if (draft.assumptionsFileName) setAssumptionsFileName(draft.assumptionsFileName);
    if (draft.perfFileName) setPerfFileName(draft.perfFileName);
    if (draft.hyperparamsFileName) setHyperparamsFileName(draft.hyperparamsFileName);
    setChkInventory(Boolean(draft.chkInventory));
    setChkTier(Boolean(draft.chkTier));
    setChkArtifacts(Boolean(draft.chkArtifacts));
    setChkPrevFindings(Boolean(draft.chkPrevFindings));
    setChkRegScope(Boolean(draft.chkRegScope));
    setChkIndependence(Boolean(draft.chkIndependence));
    setChkPlanApproved(Boolean(draft.chkPlanApproved));
    setChkAttestation(Boolean(draft.chkAttestation));
  };

  const loadAvailableDraft = () => {
    if (!draftAvailable) return;
    applyDraft(draftAvailable.data);
    setDraftSavedAt(draftAvailable.savedAt);
    setDraftAvailable(null);
  };

  const dismissAvailableDraft = () => {
    setDraftDismissedFor(modelName.trim());
    setDraftAvailable(null);
  };

  const saveDraft = async () => {
    const trimmed = modelName.trim();
    if (!trimmed) return;
    setDraftSaving(true);
    setDraftLoadError(null);
    try {
      const data = {
        owningTeam, modelOwner, leadValidator, modelType, tier, version, purpose,
        mddFileName, trainingCodeFileName, profileFileName, assumptionsFileName, perfFileName, hyperparamsFileName,
        chkInventory, chkTier, chkArtifacts, chkPrevFindings, chkRegScope, chkIndependence, chkPlanApproved, chkAttestation,
      };
      const resp = await api<{ saved: boolean; saved_at: string }>("/validation/intake/draft", {
        method: "POST",
        body: JSON.stringify({ model_name: trimmed, data }),
      });
      setDraftSavedAt(new Date(resp.saved_at).toLocaleString());
    } catch (err) {
      setDraftLoadError(err instanceof Error ? err.message : "Failed to save draft.");
    } finally {
      setDraftSaving(false);
    }
  };

  const demoOptions = [
    { key: "Demo A — Gold Standard", mode: "clean" },
    { key: "Demo B — Flawed Submission", mode: "flawed" },
  ];

  const loadDemo = async (mode: string) => {
    setDemoError(null);
    setDemoLoading(true);
    try {
      const response = await api<IntakeResponse>(`/validation/intake?mode=${mode}`);
      if (!response.display) {
        throw new Error("Invalid demo response from backend.");
      }
      setIntake(response.display);
      setIntakeLoaded(true);
      const snapshot = response.val_intake_data;
      if (snapshot) {
        const selectedFrameworks = Array.isArray(snapshot.frameworks) && snapshot.frameworks.length > 0
          ? snapshot.frameworks
          : frameworks;
        const intakeSnapshot = {
          ...snapshot,
          mdd_text: snapshot.mdd_text ?? null,
          mdd_document_path: snapshot.mdd_document_path ?? null,
          frameworks: selectedFrameworks,
        };
        setValidationIntakeData(intakeSnapshot);
        setFrameworks(selectedFrameworks);
        setModelName(snapshot.model_name ?? "");
        setOwningTeam(snapshot.owning_team ?? "");
        setModelOwner(snapshot.model_owner ?? "");
        setLeadValidator(snapshot.lead_validator ?? "");
        setModelType(snapshot.model_type ?? "PD (Probability of Default)");
        setTier(snapshot.model_tier ?? "Tier 2 — Medium Risk");
        setVersion(snapshot.model_version ?? "");
        setPurpose(snapshot.model_purpose ?? "");
        if (snapshot.mdd_text) {
          setMddText(snapshot.mdd_text);
          setMddFileName("Parsed MDD from backend");
          setMddDocumentPath(snapshot.mdd_document_path ?? null);
          setValidationMddText(snapshot.mdd_text);
        }
      }
      setChkInventory(response.chk_inventory ?? false);
      setChkTier(response.chk_tier ?? false);
      setChkArtifacts(response.chk_artifacts ?? false);
      setChkPrevFindings(response.chk_prev_findings ?? false);
      setChkRegScope(response.chk_reg_scope ?? false);
      setChkIndependence(response.chk_independence ?? false);
      setChkPlanApproved(response.chk_plan_approved ?? false);
      setChkAttestation(response.chk_attestation ?? false);
      if (response.val_mdd_reported_metrics) {
        setMddMetrics(response.val_mdd_reported_metrics);
        setValidationMddMetrics(response.val_mdd_reported_metrics);
      }
      setDemoMode(response.demo_label ?? response.demo_mode ?? null);

      const demoForm = new FormData();
      demoForm.append("demo_mode", mode);
      const demoProfile = await formUpload("/data/upload", demoForm);
      setUploadResult(null, demoProfile as any);
      setDatasetFile(null);
      setValidationProfile(demoProfile as any);
      setValidationResults(null);
    } catch (error) {
      setDemoError(error instanceof Error ? error.message : "Unable to load demo submission.");
    } finally {
      setDemoLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title={intake.title}
        description={intake.description}
      />

      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="text-sm font-semibold text-foreground">Demo mode</div>
            <p className="mt-2 max-w-xl text-sm text-muted-foreground">Load a pre-configured demo intake submission instead of uploading every artifact manually.</p>
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {demoOptions.map((demo) => (
              <button
                key={demo.key}
                className="group inline-flex min-h-[110px] w-full items-center justify-center rounded-3xl border border-border bg-background px-4 py-4 text-center text-sm font-semibold text-foreground transition hover:border-primary hover:text-primary"
                onClick={() => loadDemo(demo.mode)}
                disabled={demoLoading}
              >
                <span className="inline-flex flex-col items-center gap-2 text-sm font-semibold leading-tight">
                  {demo.key.split(" — ").map((line, index) => (
                    <span key={index} className={index === 1 ? "text-primary" : "text-foreground"}>
                      {line}
                    </span>
                  ))}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <div className="rounded-full border border-border bg-background px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            {demoMode ?? "No demo loaded"}
          </div>
          {demoLoading ? <div className="text-xs text-muted-foreground">Loading demo...</div> : null}
          {demoError ? <div className="text-xs text-amber-300">{demoError}</div> : null}
        </div>
      </section>

      <section className="grid grid-cols-1 gap-6 xl:grid-cols-[1.55fr_0.95fr]">
        <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-foreground">{intake.modelMetadata.title}</h3>
              <p className="mt-1 text-xs text-muted-foreground">{intake.modelMetadata.description}</p>
            </div>
            <div
              className={
                intake.modelMetadata.registeredLabel === "Registered"
                  ? "inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary-soft px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-primary"
                  : "inline-flex items-center gap-2 rounded-full border border-border bg-muted px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground"
              }
            >
              <FileText className="h-3.5 w-3.5" /> {intake.modelMetadata.registeredLabel}
            </div>
          </div>

          <div className="mt-5 grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="rounded-lg border border-border bg-background px-3 py-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Model name</div>
              <input value={modelName} onChange={(e) => setModelName(e.target.value)} className="mt-1 w-full bg-background px-2 py-2 text-sm font-semibold text-foreground" />
            </div>
            <div className="rounded-lg border border-border bg-background px-3 py-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Owning team / business unit</div>
              <input value={owningTeam} onChange={(e) => setOwningTeam(e.target.value)} className="mt-1 w-full bg-background px-2 py-2 text-sm font-semibold text-foreground" />
            </div>
            <div className="rounded-lg border border-border bg-background px-3 py-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Model owner (name)</div>
              <input value={modelOwner} onChange={(e) => setModelOwner(e.target.value)} className="mt-1 w-full bg-background px-2 py-2 text-sm font-semibold text-foreground" />
            </div>
            <div className="rounded-lg border border-border bg-background px-3 py-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Lead validator (name)</div>
              <input value={leadValidator} onChange={(e) => setLeadValidator(e.target.value)} className="mt-1 w-full bg-background px-2 py-2 text-sm font-semibold text-foreground" />
            </div>
            <div className="rounded-lg border border-border bg-background px-3 py-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Model type</div>
              <select value={modelType} onChange={(e) => setModelType(e.target.value)} className="mt-1 w-full bg-background px-2 py-2 text-sm font-semibold text-foreground">
                <option>PD (Probability of Default)</option>
                <option>LGD (Loss Given Default)</option>
                <option>EAD (Exposure at Default)</option>
                <option>Scorecard / Rating</option>
              </select>
            </div>
            <div className="rounded-lg border border-border bg-background px-3 py-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Model version</div>
              <input value={version} onChange={(e) => setVersion(e.target.value)} className="mt-1 w-full bg-background px-2 py-2 text-sm font-semibold text-foreground" />
            </div>
            <div className="rounded-lg border border-border bg-background px-3 py-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Risk tier</div>
              <select value={tier} onChange={(e) => setTier(e.target.value)} className="mt-1 w-full bg-background px-2 py-2 text-sm font-semibold text-foreground">
                <option>Tier 1 — High Risk</option>
                <option>Tier 2 — Medium Risk</option>
                <option>Tier 3 — Low Risk</option>
              </select>
            </div>
            <div className="rounded-lg border border-border bg-background px-3 py-3 md:col-span-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Model purpose</div>
              <textarea value={purpose} onChange={(e) => setPurpose(e.target.value)} rows={3} className="mt-1 w-full bg-background px-2 py-2 text-sm font-semibold text-foreground" />
            </div>
          </div>

          {draftAvailable ? (
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-primary/30 bg-primary-soft px-4 py-3 text-sm">
              <span>
                A saved draft exists for <strong>{modelName.trim()}</strong>
                {draftAvailable.savedAt ? ` (saved ${new Date(draftAvailable.savedAt).toLocaleString()})` : ""}.
              </span>
              <span className="flex gap-2">
                <button type="button" onClick={loadAvailableDraft} className="rounded-lg border border-primary bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground">
                  Load draft
                </button>
                <button type="button" onClick={dismissAvailableDraft} className="rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold text-foreground">
                  Dismiss
                </button>
              </span>
            </div>
          ) : null}
        </div>

        <div className="space-y-4">
          <div className="rounded-xl border border-border bg-card p-5 shadow-elegant">
            <div className="text-sm font-semibold text-foreground">{intake.targetDefinition.title}</div>
            <div className="mt-3 rounded-lg border border-border bg-background p-3 font-mono text-[12px] leading-6 text-foreground">
              {intake.targetDefinition.expression}<br />
              {intake.targetDefinition.detail}
            </div>
            <div className="mt-3 text-xs text-muted-foreground">
              {intake.targetDefinition.baseRateLabel}: <span className="font-semibold text-foreground">{intake.targetDefinition.baseRate}</span> · {intake.targetDefinition.sampleSizeLabel}: {intake.targetDefinition.sampleSize}
            </div>
          </div>

          {(() => {
            const tone = riskTierTone(intake.riskTier.value);
            return (
              <div className={`rounded-xl border ${tone.border} ${tone.bg} p-5 shadow-elegant`}>
                <div className={`text-sm font-semibold ${tone.text}`}>{intake.riskTier.title}</div>
                <div className={`mt-2 text-3xl font-semibold ${tone.text}`}>{intake.riskTier.value}</div>
                <div className="mt-1 text-xs text-muted-foreground">{intake.riskTier.description}</div>
              </div>
            );
          })()}
        </div>
      </section>

      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <h3 className="text-sm font-semibold text-foreground">{intake.artifactTitle}</h3>
            <p className="mt-1 text-xs text-muted-foreground">{intake.artifactDescription}</p>
          </div>
          <div
            className={
              intake.artifacts.length > 0
                ? "inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary-soft px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-primary"
                : "inline-flex items-center gap-2 rounded-full border border-border bg-muted px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground"
            }
          >
            <FileCheck className="h-3.5 w-3.5" /> {intake.artifactSummary}
          </div>
        </div>

        {intake.artifacts.length > 0 ? (
          <div className="mt-6 grid gap-3 lg:grid-cols-3">
            {intake.artifacts.map((artifact) => (
              <div key={artifact.fileName} className="rounded-lg border border-border bg-background p-4">
                <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{artifact.fileName}</div>
                <div className="mt-3 text-sm font-semibold text-foreground">{artifact.status}</div>
                <div className="mt-2 text-[11px] text-muted-foreground">{artifact.timestamp}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="mt-6 rounded-lg border border-dashed border-border bg-background p-6 text-center text-sm text-muted-foreground">
            No artifacts on file yet — load a demo above or upload files below.
          </div>
        )}

        <div className="mt-5 grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-3">
          {/* Dataset upload */}
          <div className="rounded-lg border border-border bg-background p-4">
            <div className="text-sm font-semibold text-foreground">Validation dataset (CSV / XLSX) *</div>
            <div className="mt-2 flex items-center justify-between">
              <div className="text-xs text-muted-foreground">{datasetFile ? datasetFile.name : profile?.dataset_name ?? "No file uploaded"}</div>
              <label className="inline-flex items-center gap-2 cursor-pointer rounded bg-card px-2 py-1">
                <Upload className="h-4 w-4" />
                <input type="file" accept=".csv,.xlsx" className="hidden" onChange={async (e) => {
                  const f = e.target.files?.[0] ?? null;
                  if (!f) return;
                  setDatasetFile(f);
                  try {
                    const form = new FormData();
                    form.append("file", f);
                    const resp = await formUpload("/data/upload", form);
                    // store into dataset context
                    setUploadResult(f, resp as any);
                  } catch (err) {
                    console.error("Dataset upload failed", err);
                  }
                }} />
                <span className="text-xs text-primary">Upload</span>
              </label>
            </div>
          </div>

          {/* MDD upload */}
          <div className="rounded-lg border border-border bg-background p-4">
            <div className="text-sm font-semibold text-foreground">Model Development Document (PDF / DOCX / TXT) *</div>
            <div className="mt-2 flex items-center justify-between">
              <div className="text-xs text-muted-foreground">{mddFileName ?? (mddText ? `Parsed ${mddText.length} chars` : "No MDD uploaded")}</div>
              <label className="inline-flex items-center gap-2 cursor-pointer rounded bg-card px-2 py-1">
                <Upload className="h-4 w-4" />
                <input type="file" accept=".pdf,.docx,.txt" className="hidden" onChange={async (e) => {
                  const f = e.target.files?.[0] ?? null;
                  if (!f) return;
                  setMddFileName(f.name);
                  setParseError(null);
                  try {
                    const form = new FormData();
                    form.append("mdd_file", f);
                    const resp = await formUpload<Record<string, any>>("/validation/parse-mdd", form);
                    setMddText(resp?.mdd_text ?? null);
                    setMddMetrics(resp?.metrics ?? null);
                    setMddDocumentPath(resp?.mdd_document_path ?? null);
                    // Publish to shared context — Stage 3's RAG keyword-search
                    // check (check_mdd_keywords) reads validationMddText from
                    // here. Without this, an MDD uploaded via this input never
                    // reaches Stage 3 and its RAG Agent Rules column stays empty.
                    setValidationMddText(resp?.mdd_text ?? null);
                    setValidationMddMetrics(resp?.metrics ?? null);
                  } catch (err) {
                    console.error("MDD parse failed", err);
                    setMddText(null);
                    setMddMetrics(null);
                    setParseError(err instanceof Error ? err.message : "Failed to parse MDD file.");
                  }
                }} />
                <span className="text-xs text-primary">Upload & Parse</span>
              </label>
            </div>
            {parseError ? <div className="mt-2 text-xs text-red-500">{parseError}</div> : null}
            {mddMetrics ? (
              <div className="mt-4 rounded-lg border border-border bg-slate-950/50 p-3 text-sm text-foreground">
                <div className="font-semibold">Extracted MDD metrics</div>
                <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {Object.entries(mddMetrics).map(([label, value]) => (
                    <div key={label} className="rounded-lg bg-background/70 px-3 py-2">
                      <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{label.replace(/_/g, " ")}</div>
                      <div className="mt-1 text-sm font-semibold text-foreground">{value ?? "—"}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : mddText ? (
              <div className="mt-4 text-xs text-muted-foreground">No reported metrics were detected in the uploaded MDD.</div>
            ) : null}
          </div>

          {/* Training code */}
          <div className="rounded-lg border border-border bg-background p-4">
            <div className="text-sm font-semibold text-foreground">Training code / scripts (ZIP / PY / IPYNB) *</div>
            <div className="mt-2 flex items-center justify-between">
              <div className="text-xs text-muted-foreground">{trainingCodeFileName ?? "No file uploaded"}</div>
              <label className="inline-flex items-center gap-2 cursor-pointer rounded bg-card px-2 py-1">
                <Upload className="h-4 w-4" />
                <input type="file" accept=".zip,.py,.ipynb" className="hidden" onChange={(e) => { const f = e.target.files?.[0] ?? null; if (f) setTrainingCodeFileName(f.name); }} />
                <span className="text-xs text-primary">Attach</span>
              </label>
            </div>
          </div>

          {/* Data profile (optional) */}
          <div className="rounded-lg border border-border bg-background p-4">
            <div className="text-sm font-semibold text-foreground">Data profile (CSV / XLSX / PDF)</div>
            <div className="mt-2 flex items-center justify-between">
              <div className="text-xs text-muted-foreground">{profileFileName ?? "No file uploaded"}</div>
              <label className="inline-flex items-center gap-2 cursor-pointer rounded bg-card px-2 py-1">
                <Upload className="h-4 w-4" />
                <input type="file" accept=".csv,.xlsx,.pdf" className="hidden" onChange={(e) => { const f = e.target.files?.[0] ?? null; if (f) setProfileFileName(f.name); }} />
                <span className="text-xs text-primary">Attach</span>
              </label>
            </div>
          </div>

          {/* Assumptions & limitations (optional) */}
          <div className="rounded-lg border border-border bg-background p-4">
            <div className="text-sm font-semibold text-foreground">Assumptions &amp; limitations (PDF / DOCX / TXT)</div>
            <div className="mt-2 flex items-center justify-between">
              <div className="text-xs text-muted-foreground">{assumptionsFileName ?? "No file uploaded"}</div>
              <label className="inline-flex items-center gap-2 cursor-pointer rounded bg-card px-2 py-1">
                <Upload className="h-4 w-4" />
                <input type="file" accept=".pdf,.docx,.txt" className="hidden" onChange={(e) => { const f = e.target.files?.[0] ?? null; if (f) setAssumptionsFileName(f.name); }} />
                <span className="text-xs text-primary">Attach</span>
              </label>
            </div>
          </div>

          {/* Performance report (optional) */}
          <div className="rounded-lg border border-border bg-background p-4">
            <div className="text-sm font-semibold text-foreground">Performance report (PDF / DOCX / XLSX)</div>
            <div className="mt-2 flex items-center justify-between">
              <div className="text-xs text-muted-foreground">{perfFileName ?? "No file uploaded"}</div>
              <label className="inline-flex items-center gap-2 cursor-pointer rounded bg-card px-2 py-1">
                <Upload className="h-4 w-4" />
                <input type="file" accept=".pdf,.docx,.xlsx" className="hidden" onChange={(e) => { const f = e.target.files?.[0] ?? null; if (f) setPerfFileName(f.name); }} />
                <span className="text-xs text-primary">Attach</span>
              </label>
            </div>
          </div>

          {/* Hyperparameters (optional) */}
          <div className="rounded-lg border border-border bg-background p-4">
            <div className="text-sm font-semibold text-foreground">Hyperparameters (JSON)</div>
            <div className="mt-2 flex items-center justify-between">
              <div className="text-xs text-muted-foreground">{hyperparamsFileName ?? "No file uploaded"}</div>
              <label className="inline-flex items-center gap-2 cursor-pointer rounded bg-card px-2 py-1">
                <Upload className="h-4 w-4" />
                <input type="file" accept=".json" className="hidden" onChange={(e) => { const f = e.target.files?.[0] ?? null; if (f) setHyperparamsFileName(f.name); }} />
                <span className="text-xs text-primary">Attach</span>
              </label>
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="text-sm font-semibold text-foreground">Artifact completeness</div>
            <p className="mt-2 text-xs text-muted-foreground">Required items must be present before intake submission. Optional artifacts provide better detail for later validation stages.</p>
          </div>
          <div className="rounded-full border border-border bg-muted px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
            {readyCount}/7 uploaded
          </div>
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {[
            { label: "Submitted Dataset", uploaded: datasetUploaded, required: true, detail: "Needed for Stage 2 data checks" },
            { label: "Model Dev Document", uploaded: mddUploaded, required: true, detail: "Governance evidence for Stage 1" },
            { label: "Training Code", uploaded: trainingCodeUploaded, required: true, detail: "Required for replication and review" },
            { label: "Data Profile", uploaded: profileUploaded, required: false, detail: "Helps check feature coverage" },
            { label: "Assumptions", uploaded: assumptionsUploaded, required: false, detail: "Model limitations and assumptions" },
            { label: "Performance Report", uploaded: perfUploaded, required: false, detail: "Model accuracy and stability" },
            { label: "Hyperparameters", uploaded: hyperparamsUploaded, required: false, detail: "Training configuration details" },
          ].map((item) => {
            // Uploaded = same "good/complete" treatment used for PASS states
            // elsewhere in the app (bg-primary-soft/border-primary). Pending
            // is deliberately NOT urgency-coded by Required/Optional — both
            // look identical; only the caption text ("REQUIRED"/"OPTIONAL")
            // differentiates them, same as before this pass.
            const tone = item.uploaded
              ? { border: "border-primary/30", bg: "bg-primary-soft", status: "text-primary", Icon: CheckCircle2 }
              : { border: "border-border", bg: "bg-background", status: "text-muted-foreground", Icon: Circle };
            return (
              <div key={item.label} className={`rounded-lg border ${tone.border} ${tone.bg} p-4`}>
                <div className="flex items-center justify-between gap-2">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{item.label}</div>
                  <tone.Icon className={`h-3.5 w-3.5 shrink-0 ${tone.status}`} />
                </div>
                <div className={`mt-2 text-base font-medium ${tone.status}`}>{item.uploaded ? "Uploaded" : "Pending"}</div>
                <div className="mt-2 text-xs text-muted-foreground">
                  {item.required ? "REQUIRED" : "OPTIONAL"} · {item.detail}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="rounded-xl border border-border p-4">
          <div className="mb-2 text-sm font-medium">Applicable Regulatory Frameworks</div>
          {[
            { key: "IFRS9", label: "IFRS 9" },
            { key: "SS1/23", label: "SS1/23" },
            { key: "RBI", label: "RBI Model Risk Management" },
          ].map((fw) => (
            <label key={fw.key} className="flex items-center gap-2 py-1 text-sm">
              <input
                type="checkbox"
                checked={frameworks.includes(fw.key)}
                onChange={() => toggleFramework(fw.key)}
              />
              {fw.label}
            </label>
          ))}
        </div>

        <div className="mt-5">
          <h3 className="text-sm font-semibold text-foreground">{intake.governance.title}</h3>
          <p className="mt-1 text-xs text-muted-foreground">{intake.governance.description}</p>
        </div>

        <div className="mt-5 grid gap-3 md:grid-cols-2">
          <label className="flex items-center gap-3 rounded-lg border border-border bg-background px-3 py-3 text-sm">
            <input type="checkbox" className="h-4 w-4 accent-primary" checked={chkInventory} onChange={(e) => setChkInventory(e.target.checked)} />
            <span>{intake.governance.checklist[0]}</span>
          </label>
          <label className="flex items-center gap-3 rounded-lg border border-border bg-background px-3 py-3 text-sm">
            <input type="checkbox" className="h-4 w-4 accent-primary" checked={chkTier} onChange={(e) => setChkTier(e.target.checked)} />
            <span>{intake.governance.checklist[1]}</span>
          </label>
          <label className="flex items-center gap-3 rounded-lg border border-border bg-background px-3 py-3 text-sm">
            <input type="checkbox" className="h-4 w-4 accent-primary" checked={chkArtifacts} onChange={(e) => setChkArtifacts(e.target.checked)} />
            <span>{intake.governance.checklist[2]}</span>
          </label>
          <label className="flex items-center gap-3 rounded-lg border border-border bg-background px-3 py-3 text-sm">
            <input type="checkbox" className="h-4 w-4 accent-primary" checked={chkPrevFindings} onChange={(e) => setChkPrevFindings(e.target.checked)} />
            <span>{intake.governance.checklist[3]}</span>
          </label>
          <label className="flex items-center gap-3 rounded-lg border border-border bg-background px-3 py-3 text-sm">
            <input type="checkbox" className="h-4 w-4 accent-primary" checked={chkRegScope} onChange={(e) => setChkRegScope(e.target.checked)} />
            <span>{intake.governance.checklist[4]}</span>
          </label>
          <label className="flex items-center gap-3 rounded-lg border border-border bg-background px-3 py-3 text-sm">
            <input type="checkbox" className="h-4 w-4 accent-primary" checked={chkIndependence} onChange={(e) => setChkIndependence(e.target.checked)} />
            <span>{intake.governance.checklist[5]}</span>
          </label>
          <label className="flex items-center gap-3 rounded-lg border border-border bg-background px-3 py-3 text-sm">
            <input type="checkbox" className="h-4 w-4 accent-primary" checked={chkPlanApproved} onChange={(e) => setChkPlanApproved(e.target.checked)} />
            <span>{intake.governance.checklist[6]}</span>
          </label>
        </div>
      </section>

      {/* Attestation and Submit */}
      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="mb-4 text-sm text-muted-foreground">Confirm that all required artifacts are uploaded and the validation readiness checklist is complete. Use the button below to save progress without submitting.</div>
        <label className="flex items-center gap-3 mb-4">
          <input type="checkbox" className="h-4 w-4 accent-primary" checked={chkAttestation} onChange={(e) => setChkAttestation(e.target.checked)} />
          <span className="text-sm">I confirm the above information is accurate and complete</span>
        </label>
        {submitError ? <div className="mb-4 text-sm text-red-500">{submitError}</div> : null}
        <div className="flex flex-wrap items-center gap-3">
          <button
            className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => void saveDraft()}
            disabled={!modelName.trim() || draftSaving}
            title={!modelName.trim() ? "Enter a model name to save a draft." : undefined}
          >
            {draftSaving ? "Saving…" : "Save draft"}
          </button>
          {!modelName.trim() ? (
            <span className="text-xs text-muted-foreground">Enter a model name to save a draft.</span>
          ) : draftSavedAt ? (
            <span className="text-xs text-muted-foreground">Draft saved {draftSavedAt} — keyed to model name "{modelName.trim()}"</span>
          ) : null}
          {draftLoadError ? <span className="text-xs text-destructive">{draftLoadError}</span> : null}
        </div>
      </section>

      <section className="rounded-xl border border-border bg-card p-6 shadow-elegant">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="text-sm text-muted-foreground">{intake.nextStep.description}</div>
          <button
            type="button"
            onClick={handleProceed}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-sm hover:bg-primary/90"
          >
            <span>{intake.nextStep.label}</span>
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
        {proceedError ? (
          <div className="mt-4 rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
            {proceedError}
          </div>
        ) : null}
      </section>
    </div>
  );
}
